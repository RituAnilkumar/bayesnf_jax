"""
src/ensemble_uncertainty.py

Combines per-run predictions from a Hydra multirun sweep into a single ensemble
uncertainty estimate, decomposed into structural, epistemic, and aleatoric components.

Uncertainty decomposition via the law of total variance:

    Var[y] = E_m[Var[y|m]]  +  Var_m[E[y|m]]
             └─ epistemic ──┘   └─ structural ─┘

    std_epistemic  = sqrt( Σ_k w_k * epistemic_std_k² )
    std_structural = sqrt( Σ_k w_k * (mean_k - mu_ensemble)² )
    std_aleatoric  = sqrt( Σ_k w_k * aleatoric_std_k² )
                     (NaN when preds_full.csv lacks aleatoric_std column,
                      i.e. when runs were trained without heteroscedastic=true)
    std_total      = sqrt( std_epistemic² + std_structural² [+ std_aleatoric²] )

where:
  - mean_k, std_k   are the per-model predicted mean and MC std from preds_full.csv
  - mu_ensemble     is the weighted ensemble mean prediction
  - w_k             are model weights (performance-based softmax or equal)

Model weights come from the composite score computed by hyperparam_tuning.py
(lower composite = better generalisation = higher weight).

Inputs per run directory:
  preds_full.csv             — rgi_id, year, p2_5, p50, p97_5, mean, std
  regional_annual_mwe.csv   — year, total_area_km2, p2_5, p50, p97_5, mean, std

Outputs written to {output_dir}/:
  ensemble_glacier.csv       — per (rgi_id, year): median_mwe, std_structural,
                               std_epistemic, std_aleatoric, std_total
  ensemble_regional_mwe.csv — per year, MWE/yr: same uncertainty columns
  ensemble_regional_gt.csv  — per year, Gt/yr: same uncertainty columns
  model_weights.csv          — run_id, composite, weight (for audit)
  regional_uncertainty.png  — time series of ensemble median ± uncertainty bands

Usage:
  python src/ensemble_uncertainty.py
  python src/ensemble_uncertainty.py --config conf/config_ensemble_uncertainty.yaml
  python src/ensemble_uncertainty.py --multirun_root /path/to/r06_3645680 \\
                                     --output_dir outputs/ensemble/r06

Batch usage:
for i in $(seq -w 1 19); do
  python src/ensemble_uncertainty_split.py \
    --multirun_root /scratch/b5at/ranil.b5at/bayesnf_jax/multirun/r${i}_397*/ \
    --output_dir outputs/all_regs/r${i}
done
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

# Ensure the project root is on sys.path so this script can be run directly
# as `python src/ensemble_uncertainty.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.hyperparam_tuning import (
    build_results_df,
    add_composite_score,
)


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_weights(composite_scores: np.ndarray, weighting: str, temperature: float) -> np.ndarray:
    """
    Compute normalised model weights from composite scores (lower = better).

    weighting='equal'      : uniform 1/K weights
    weighting='performance': softmax over -composite / temperature
    """
    K = len(composite_scores)
    if weighting == "equal":
        return np.ones(K) / K
    if weighting == "performance":
        neg = -composite_scores / temperature
        neg -= neg.max()          # numerical stability before exp
        w = np.exp(neg)
        return w / w.sum()
    raise ValueError(f"Unknown weighting scheme: '{weighting}'. Choose 'equal' or 'performance'.")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _read_run_model_cfg(run_dirs: list[Path]) -> dict:
    """
    Read model config paths from the first run that has a .hydra/config.yaml.

    Extracts model.glambie_path, model.inp_dir, model.reg_subdir — the same
    values the Hydra predict run used — so no manual duplication in the
    ensemble config is needed.

    Returns an empty dict if no run has a readable config.
    """
    for run_dir in run_dirs:
        cfg_path = run_dir / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path) as fh:
                full_cfg = yaml.safe_load(fh)
            model = full_cfg.get("model", {})
            return {
                "glambie_path": model.get("glambie_path", ""),
                "inp_dir":      model.get("inp_dir", ""),
                "reg_subdir":   model.get("reg_subdir", ""),
            }
        except Exception as exc:
            warnings.warn(f"Could not read {cfg_path}: {exc}")
            continue
    warnings.warn("No readable .hydra/config.yaml found — auxiliary data paths unavailable.")
    return {}


def _load_glacier_preds(run_dir: Path) -> pd.DataFrame | None:
    """Load preds_full.csv from a run directory. Returns None if absent."""
    path = run_dir / "preds_full.csv"
    if not path.exists():
        warnings.warn(f"preds_full.csv not found in {run_dir} — run excluded.")
        return None
    return pd.read_csv(path)


def _load_regional_mwe(run_dir: Path) -> pd.DataFrame | None:
    """Load regional_annual_mwe.csv from a run directory. Returns None if absent."""
    path = run_dir / "regional_annual_mwe.csv"
    if not path.exists():
        warnings.warn(f"regional_annual_mwe.csv not found in {run_dir} — run excluded from regional.")
        return None
    return pd.read_csv(path).sort_values("year").reset_index(drop=True)


def _assert_grid_alignment(dfs: list[pd.DataFrame], key_cols: list[str], label: str) -> None:
    """Assert that all DataFrames share identical key_col values in the same order."""
    ref = dfs[0][key_cols].values
    for i, df in enumerate(dfs[1:], start=1):
        if not (df[key_cols].values == ref).all():
            raise ValueError(
                f"{label}: grid mismatch between run 0 and run {i} on columns {key_cols}. "
                "All runs must be predictions over the same (rgi_id, year) grid."
            )


# ---------------------------------------------------------------------------
# Ensemble computation
# ---------------------------------------------------------------------------

def _ensemble_components(
    means_mat: np.ndarray,
    stds_mat: np.ndarray,
    weights: np.ndarray,
    aleatoric_mat: np.ndarray | None = None,
) -> dict:
    """
    Compute ensemble uncertainty components from stacked per-model arrays.

    Args:
        means_mat:     (K, N) array of per-model predicted means
        stds_mat:      (K, N) array of per-model epistemic MC stds
        weights:       (K,) normalised model weights
        aleatoric_mat: (K, N) array of per-model predicted aleatoric stds,
                       or None when runs were trained homoscedastically.

    Returns dict with keys: median_mwe, std_structural, std_epistemic,
                             std_aleatoric (NaN if aleatoric_mat is None), std_total
    """
    w = weights[:, np.newaxis]                        # (K, 1) for broadcasting

    mu_ensemble    = (w * means_mat).sum(axis=0)      # (N,)
    std_structural = np.sqrt(
        (w * (means_mat - mu_ensemble[np.newaxis]) ** 2).sum(axis=0)
    )
    std_epistemic  = np.sqrt((w * stds_mat ** 2).sum(axis=0))

    if aleatoric_mat is not None:
        std_aleatoric = np.sqrt((w * aleatoric_mat ** 2).sum(axis=0))
        std_total     = np.sqrt(std_structural ** 2 + std_epistemic ** 2 + std_aleatoric ** 2)
    else:
        std_aleatoric = np.full_like(mu_ensemble, np.nan)
        std_total     = np.sqrt(std_structural ** 2 + std_epistemic ** 2)

    return {
        "median_mwe":     mu_ensemble,
        "std_structural": std_structural,
        "std_epistemic":  std_epistemic,
        "std_aleatoric":  std_aleatoric,
        "std_total":      std_total,
    }


# ---------------------------------------------------------------------------
# Auxiliary data helpers (self-contained — no import from predict.py)
# ---------------------------------------------------------------------------

def _load_glambie_wide(glambie_path: str):
    """Load GLaMBIE wide-format CSV. Returns None if path is empty or missing."""
    if not glambie_path or not os.path.exists(glambie_path):
        return None
    return pd.read_csv(glambie_path)


def _glambie_combined_gt(glambie_wide_df, total_area_km2: float) -> pd.DataFrame:
    """Extract GLaMBIE combined as Gt/yr per year."""
    if glambie_wide_df is None or "combined_mwe" not in glambie_wide_df.columns:
        return pd.DataFrame(columns=["year", "gt", "gt_err"])
    gb = glambie_wide_df.dropna(subset=["combined_mwe", "combined_mwe_errors"]).copy()
    if gb.empty:
        return pd.DataFrame(columns=["year", "gt", "gt_err"])
    gb["year"]   = np.floor(gb["end_date"]).astype(int)
    gb["gt"]     = gb["combined_mwe"].values     * total_area_km2 * 1e-3
    gb["gt_err"] = gb["combined_mwe_errors"].values * total_area_km2 * 1e-3
    return gb[["year", "gt", "gt_err"]].sort_values("year").reset_index(drop=True)


def _glambie_sources_mwe(glambie_wide_df) -> dict:
    """Extract altimetry and gravimetry from GLaMBIE wide format as MWE/yr per year."""
    empty = pd.DataFrame(columns=["year", "mwe", "mwe_err"])
    if glambie_wide_df is None:
        return {"altimetry": empty, "gravimetry": empty}
    result = {}
    for source in ("altimetry", "gravimetry"):
        val_col = f"{source}_mwe"
        err_col = f"{source}_mwe_errors"
        if val_col not in glambie_wide_df.columns or err_col not in glambie_wide_df.columns:
            result[source] = empty
            continue
        df = glambie_wide_df.dropna(subset=[val_col, err_col]).copy()
        if df.empty:
            result[source] = empty
            continue
        df["year"]    = np.floor(df["end_date"]).astype(int)
        df["mwe"]     = df[val_col].values
        df["mwe_err"] = df[err_col].values
        result[source] = df[["year", "mwe", "mwe_err"]].sort_values("year").reset_index(drop=True)
    return result


def _load_oggm_regional(inp_dir: str, reg_subdir: str) -> pd.DataFrame:
    """
    Load OGGM targets and compute area-weighted regional Gt/yr and MWE/yr per year.

    Returns DataFrame with columns: year, gt, mwe.
    Returns empty DataFrame if files are missing.
    """
    if not inp_dir or not reg_subdir:
        return pd.DataFrame(columns=["year", "gt", "mwe"])

    oggm_path = os.path.join(inp_dir, reg_subdir, f"oggm_targets_{reg_subdir}.csv")
    feat_path = os.path.join(inp_dir, reg_subdir, f"main_features_{reg_subdir}.csv")

    if not os.path.exists(oggm_path) or not os.path.exists(feat_path):
        warnings.warn(f"OGGM files not found under {inp_dir}/{reg_subdir} — OGGM series skipped.")
        return pd.DataFrame(columns=["year", "gt", "mwe"])

    oggm = pd.read_csv(oggm_path)
    # Handle both raw mm/yr and pre-converted MWE/yr column names
    if "mass_balance_mwe" not in oggm.columns:
        if "mass_balance" in oggm.columns:
            oggm["mass_balance_mwe"] = oggm["mass_balance"] / 1000.0
        else:
            warnings.warn("OGGM targets CSV has no recognised mass balance column — skipped.")
            return pd.DataFrame(columns=["year", "gt", "mwe"])

    feats  = pd.read_csv(feat_path, usecols=["rgi_id", "year", "Area"])
    merged = oggm.merge(feats, on=["rgi_id", "year"], how="inner")

    rows = []
    for yr, grp in merged.groupby("year"):
        total_area = grp["Area"].sum()
        if total_area > 0:
            w_mwe = (grp["mass_balance_mwe"] * grp["Area"]).sum() / total_area
            rows.append({"year": yr, "gt": w_mwe * total_area * 1e-3, "mwe": w_mwe})

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Regional aleatoric aggregation
# ---------------------------------------------------------------------------

def _aggregate_aleatoric_to_regional(
    ensemble_glacier_df: pd.DataFrame,
    inp_dir: str,
    reg_subdir: str,
) -> pd.DataFrame | None:
    """
    Propagate per-glacier aleatoric sigma to regional sigma via area-weighted aggregation.

    sigma_regional_t = sqrt( Σ_i (area_i/total_area_t)² * sigma_i² )

    This is the correct propagation for a weighted mean:
        regional_mwe_t = Σ_i w_i * pred_i,  w_i = area_i / total_area_t
    Under independence: Var[regional_t] = Σ_i w_i² * sigma_i²

    Args:
        ensemble_glacier_df: ensemble_glacier.csv with columns rgi_id, year, std_aleatoric
        inp_dir:             OGGM feature file base directory
        reg_subdir:          Region subdirectory (e.g. 'r06')

    Returns:
        DataFrame with columns year, std_aleatoric (regional), or None if unavailable.
    """
    if "std_aleatoric" not in ensemble_glacier_df.columns:
        return None
    if ensemble_glacier_df["std_aleatoric"].isna().all():
        return None

    feat_path = os.path.join(inp_dir, reg_subdir, f"main_features_{reg_subdir}.csv")
    if not os.path.exists(feat_path):
        warnings.warn(f"Feature file not found at {feat_path} — regional aleatoric skipped.")
        return None

    try:
        feats = pd.read_csv(feat_path, usecols=["rgi_id", "year", "Area"])
    except Exception as exc:
        warnings.warn(f"Could not load {feat_path}: {exc} — regional aleatoric skipped.")
        return None

    merged = ensemble_glacier_df[["rgi_id", "year", "std_aleatoric"]].merge(
        feats, on=["rgi_id", "year"], how="inner"
    )

    rows = []
    for yr, grp in merged.groupby("year"):
        total_area = grp["Area"].sum()
        if total_area <= 0:
            continue
        w = grp["Area"].values / total_area
        sigma_regional = np.sqrt((w ** 2 * grp["std_aleatoric"].values ** 2).sum())
        rows.append({"year": yr, "std_aleatoric": sigma_regional})

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True) if rows else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _shade_ensemble(ax, years, mu, s_tot):
    """Draw ±2σ total uncertainty band and ensemble median."""
    ax.fill_between(years, mu - 2 * s_tot, mu + 2 * s_tot,
                    alpha=0.25, color="steelblue", label="±2σ total")
    ax.plot(years, mu, color="steelblue", lw=1.8, label="ensemble median")


def plot_ensemble_gt(
    ensemble_gt: pd.DataFrame,
    glambie_wide_df,
    oggm_df: pd.DataFrame,
    total_area_km2: np.ndarray,
    output_dir: Path,
) -> None:
    """Regional Gt/yr: ensemble uncertainty bands + OGGM + GLaMBIE sources."""
    years           = ensemble_gt["year"].values
    mu              = ensemble_gt["median_gt"].values
    total_area_mean = float(total_area_km2.mean())

    gb_combined = _glambie_combined_gt(glambie_wide_df, total_area_mean)
    gb_sources  = _glambie_sources_mwe(glambie_wide_df)

    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_ensemble(ax, years, mu, ensemble_gt["std_total"].values)

    if not gb_combined.empty:
        ax.errorbar(gb_combined["year"].values, gb_combined["gt"].values,
                    yerr=gb_combined["gt_err"].values,
                    fmt="o", color="black", ms=4, lw=1.2, capsize=3, label="GLaMBIE combined")

    for source, color, marker in [("altimetry", "forestgreen", "s"), ("gravimetry", "purple", "^")]:
        df = gb_sources[source]
        if not df.empty:
            gt_vals = df["mwe"].values * total_area_mean * 1e-3
            gt_errs = df["mwe_err"].values * total_area_mean * 1e-3
            ax.errorbar(df["year"].values, gt_vals, yerr=gt_errs,
                        fmt=marker, color=color, ms=4, lw=1.2, capsize=3,
                        label=f"GLaMBIE {source}")

    if not oggm_df.empty:
        ax.plot(oggm_df["year"].values, oggm_df["gt"].values,
                color="red", lw=1.2, ls="--", label="OGGM")

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year"); ax.set_ylabel("Gt/yr")
    ax.set_title("Regional mass balance — ensemble (Gt/yr)")
    ax.legend(fontsize=8); fig.tight_layout()
    _savefig(fig, output_dir / "ensemble_regional_gt.png")


def plot_ensemble_mwe(
    ensemble_mwe: pd.DataFrame,
    glambie_wide_df,
    oggm_df: pd.DataFrame,
    total_area_km2: np.ndarray,
    output_dir: Path,
) -> None:
    """Regional MWE/yr: ensemble uncertainty bands + OGGM + GLaMBIE sources."""
    years           = ensemble_mwe["year"].values
    mu              = ensemble_mwe["median_mwe"].values
    total_area_mean = float(total_area_km2.mean())

    gb_sources     = _glambie_sources_mwe(glambie_wide_df)
    gb_combined_gt = _glambie_combined_gt(glambie_wide_df, total_area_mean)

    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_ensemble(ax, years, mu, ensemble_mwe["std_total"].values)

    # GLaMBIE combined converted back to MWE
    if not gb_combined_gt.empty and total_area_mean > 0:
        gb_mwe     = gb_combined_gt["gt"].values / (total_area_mean * 1e-3)
        gb_mwe_err = gb_combined_gt["gt_err"].values / (total_area_mean * 1e-3)
        ax.errorbar(gb_combined_gt["year"].values, gb_mwe, yerr=gb_mwe_err,
                    fmt="o", color="black", ms=4, lw=1.0, capsize=3, label="GLaMBIE combined")

    for source, color, marker in [("altimetry", "forestgreen", "s"), ("gravimetry", "purple", "^")]:
        df = gb_sources[source]
        if not df.empty:
            ax.errorbar(df["year"].values, df["mwe"].values, yerr=df["mwe_err"].values,
                        fmt=marker, color=color, ms=4, lw=1.0, capsize=3,
                        label=f"GLaMBIE {source}")

    if not oggm_df.empty:
        ax.plot(oggm_df["year"].values, oggm_df["mwe"].values,
                color="red", lw=1.2, ls="--", label="OGGM")

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year"); ax.set_ylabel("MWE/yr")
    ax.set_title("Regional mass balance — ensemble (MWE/yr)")
    ax.legend(fontsize=8); fig.tight_layout()
    _savefig(fig, output_dir / "ensemble_regional_mwe.png")


def plot_ensemble_cumulative_gt(
    ensemble_gt: pd.DataFrame,
    glambie_wide_df,
    oggm_df: pd.DataFrame,
    total_area_km2: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Cumulative Gt from the first year GLaMBIE combined data is available
    (or the first prediction year if GLaMBIE is absent).

    Ensemble cumulative uncertainty propagated in quadrature assuming
    year-to-year independence: cum_std_t = sqrt(Σ_{s≤t} std_total_s²).
    """
    total_area_mean = float(total_area_km2.mean())
    gb_combined = _glambie_combined_gt(glambie_wide_df, total_area_mean)

    start_year = int(gb_combined["year"].min()) if not gb_combined.empty \
        else int(ensemble_gt["year"].min())

    mask       = ensemble_gt["year"].values >= start_year
    years      = ensemble_gt["year"].values[mask]
    cum_median = np.cumsum(ensemble_gt["median_gt"].values[mask])
    cum_std    = np.sqrt(np.cumsum(ensemble_gt["std_total"].values[mask] ** 2))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(years, cum_median - 2 * cum_std, cum_median + 2 * cum_std,
                    alpha=0.15, color="steelblue", label="±2σ total")
    ax.plot(years, cum_median, color="steelblue", lw=1.8, label="Ensemble median")

    if not oggm_df.empty:
        og = oggm_df[oggm_df["year"] >= start_year].copy()
        if not og.empty:
            ax.plot(og["year"].values, np.cumsum(og["gt"].values),
                    color="seagreen", lw=1.3, ls="--", label="OGGM")

    if not gb_combined.empty:
        gb_from = gb_combined[gb_combined["year"] >= start_year].copy()
        if not gb_from.empty:
            gb_cum     = gb_from["gt"].cumsum().values
            gb_cum_err = np.sqrt(np.cumsum(gb_from["gt_err"].values ** 2))
            ax.fill_between(gb_from["year"].values,
                            gb_cum - 1.96 * gb_cum_err, gb_cum + 1.96 * gb_cum_err,
                            alpha=0.20, color="black")
            ax.plot(gb_from["year"].values, gb_cum, "k-", lw=1.5, label="GLaMBIE combined")

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year"); ax.set_ylabel("Cumulative mass balance (Gt)")
    ax.set_title(f"Cumulative regional mass balance from {start_year} — ensemble")
    ax.legend(fontsize=8); fig.tight_layout()
    _savefig(fig, output_dir / "ensemble_cumulative_gt.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ensemble_uncertainty(cfg: dict) -> None:
    multirun_root = Path(cfg["multirun_root"])
    output_dir    = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    weighting    = cfg.get("weighting", "performance")
    temperature  = float(cfg.get("softmax_temperature", 0.1))
    loyo_w       = float(cfg.get("loyo_weight", 0.0))
    glambie_w    = float(cfg.get("glambie_weight", 1.0))
    test_years   = list(cfg.get("glambie_test_years", [2021, 2022, 2023]))
    min_runs     = int(cfg.get("min_runs_per_region", 5))

    # ------------------------------------------------------------------
    # 1. Load metrics and compute model weights
    # ------------------------------------------------------------------
    print(f"\n=== Ensemble uncertainty: {multirun_root.name} ===")
    results_df = build_results_df(multirun_root, test_years, min_runs_per_region=min_runs)
    results_df = add_composite_score(results_df, loyo_weight=loyo_w, glambie_weight=glambie_w)

    valid = results_df.dropna(subset=["composite"]).copy().reset_index(drop=True)
    if len(valid) < min_runs:
        raise RuntimeError(
            f"Only {len(valid)} runs with complete metrics (threshold={min_runs}). "
            "Lower min_runs_per_region or check that runs completed successfully."
        )

    weights = compute_weights(valid["composite"].values, weighting, temperature)
    valid["weight"] = weights
    valid[["run_id", "composite", "weight"]].to_csv(output_dir / "model_weights.csv", index=False)
    print(f"  {len(valid)} valid runs, weighting='{weighting}'")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    run_dirs = [Path(d) for d in valid["run_dir"].values]

    # ------------------------------------------------------------------
    # 1b. Read model config paths from first run (needed for aux data + aleatoric)
    # ------------------------------------------------------------------
    run_model_cfg = _read_run_model_cfg(run_dirs)
    glambie_path  = run_model_cfg.get("glambie_path", "")
    inp_dir       = run_model_cfg.get("inp_dir", "")
    reg_subdir    = run_model_cfg.get("reg_subdir", "")

    # ------------------------------------------------------------------
    # 2. Per-glacier ensemble
    # ------------------------------------------------------------------
    print("\n--- Per-glacier ensemble ---")
    glacier_dfs, glacier_weights = [], []
    for run_dir, w in zip(run_dirs, weights):
        df = _load_glacier_preds(run_dir)
        if df is not None:
            glacier_dfs.append(df)
            glacier_weights.append(w)

    if len(glacier_dfs) < min_runs:
        warnings.warn(f"Only {len(glacier_dfs)} runs have preds_full.csv — proceeding with {len(glacier_dfs)}.")

    _assert_grid_alignment(glacier_dfs, ["rgi_id", "year"], "preds_full.csv")

    glacier_weights_arr = np.array(glacier_weights)
    glacier_weights_arr /= glacier_weights_arr.sum()   # renormalise after any exclusions

    means_mat = np.stack([df["mean"].values for df in glacier_dfs])   # (K, N)

    # Use epistemic_std when available (heteroscedastic runs); fall back to std
    # (which equals epistemic std in homoscedastic runs).
    epistemic_col = "epistemic_std" if all("epistemic_std" in df.columns for df in glacier_dfs) else "std"
    stds_mat = np.stack([df[epistemic_col].values for df in glacier_dfs])  # (K, N)

    # Aleatoric: available only when all runs were trained with heteroscedastic=true
    has_aleatoric = all("aleatoric_std" in df.columns for df in glacier_dfs)
    aleatoric_mat = (
        np.stack([df["aleatoric_std"].values for df in glacier_dfs])
        if has_aleatoric else None
    )
    if has_aleatoric:
        print(f"  Heteroscedastic runs detected — aleatoric uncertainty will be included.")

    comps = _ensemble_components(means_mat, stds_mat, glacier_weights_arr, aleatoric_mat)

    ref = glacier_dfs[0]
    ensemble_glacier = pd.DataFrame({
        "rgi_id":         ref["rgi_id"].values,
        "year":           ref["year"].values,
        "median_mwe":     comps["median_mwe"],
        "std_structural": comps["std_structural"],
        "std_epistemic":  comps["std_epistemic"],
        "std_aleatoric":  comps["std_aleatoric"],
        "std_total":      comps["std_total"],
    })
    ensemble_glacier.to_csv(output_dir / "ensemble_glacier.csv", index=False)
    print(f"  Saved ensemble_glacier.csv  ({len(ensemble_glacier)} rows)")

    # ------------------------------------------------------------------
    # 3. Regional MWE ensemble
    # ------------------------------------------------------------------
    print("\n--- Regional MWE ensemble ---")
    regional_dfs, regional_weights, regional_run_dirs = [], [], []
    for run_dir, w in zip(run_dirs, weights):
        df = _load_regional_mwe(run_dir)
        if df is not None:
            regional_dfs.append(df)
            regional_weights.append(w)
            regional_run_dirs.append(run_dir)

    _assert_grid_alignment(regional_dfs, ["year"], "regional_annual_mwe.csv")

    regional_weights_arr = np.array(regional_weights)
    regional_weights_arr /= regional_weights_arr.sum()

    reg_means_mat = np.stack([df["mean"].values for df in regional_dfs])   # (K, n_years)
    reg_stds_mat  = np.stack([df["std"].values  for df in regional_dfs])   # (K, n_years)
    # regional_annual_mwe.csv std is always epistemic (computed from mu samples only)

    reg_comps = _ensemble_components(reg_means_mat, reg_stds_mat, regional_weights_arr)
    # Note: regional aleatoric is computed separately below via area-weighted propagation.

    ref_regional = regional_dfs[0]
    years        = ref_regional["year"].values

    if "total_area_km2" in ref_regional.columns:
        total_area_km2 = ref_regional["total_area_km2"].values
    else:
        # Older predict outputs didn't write total_area_km2 to regional_annual_mwe.csv.
        # Derive it from regional_annual_gt.csv: total_area = gt_mean / (mwe_mean * 1e-3).
        gt_path = regional_run_dirs[0] / "regional_annual_gt.csv"
        gt_df = pd.read_csv(gt_path).sort_values("year").reset_index(drop=True)
        mwe_mean = ref_regional["mean"].values
        gt_mean  = gt_df["mean"].values
        total_area_km2 = np.where(
            np.abs(mwe_mean) > 1e-10,
            gt_mean / (mwe_mean * 1e-3),
            0.0,
        )
        print("  Note: total_area_km2 derived from regional_annual_gt.csv (older run format)")

    # Aggregate per-glacier aleatoric sigma to regional via area-weighting.
    # This requires the ensemble_glacier_df (already computed above) and the
    # glacier area data from the OGGM feature file.
    reg_aleatoric_df = _aggregate_aleatoric_to_regional(ensemble_glacier, inp_dir, reg_subdir)
    if reg_aleatoric_df is not None:
        year_to_reg_aleatoric = dict(zip(reg_aleatoric_df["year"], reg_aleatoric_df["std_aleatoric"]))
        reg_aleatoric_arr = np.array([year_to_reg_aleatoric.get(y, np.nan) for y in years])
        # Recompute std_total for regional to include aleatoric
        valid_mask = ~np.isnan(reg_aleatoric_arr)
        reg_std_total = np.sqrt(
            reg_comps["std_structural"] ** 2 + reg_comps["std_epistemic"] ** 2
            + np.where(valid_mask, reg_aleatoric_arr ** 2, 0.0)
        )
        print(f"  Regional aleatoric propagated from per-glacier ensemble ({valid_mask.sum()}/{len(years)} years).")
    else:
        reg_aleatoric_arr = np.full(len(years), np.nan)
        reg_std_total     = reg_comps["std_total"]

    ensemble_regional_mwe = pd.DataFrame({
        "year":           years,
        "median_mwe":     reg_comps["median_mwe"],
        "std_structural": reg_comps["std_structural"],
        "std_epistemic":  reg_comps["std_epistemic"],
        "std_aleatoric":  reg_aleatoric_arr,
        "std_total":      reg_std_total,
    })
    ensemble_regional_mwe.to_csv(output_dir / "ensemble_regional_mwe.csv", index=False)
    print(f"  Saved ensemble_regional_mwe.csv  ({len(years)} years)")

    # ------------------------------------------------------------------
    # 4. Regional Gt ensemble  (MWE × total_area × 1e-3, exact unit conversion)
    # ------------------------------------------------------------------
    scale = total_area_km2 * 1e-3   # (n_years,) — converts MWE/yr to Gt/yr

    ensemble_regional_gt = pd.DataFrame({
        "year":           years,
        "median_gt":      reg_comps["median_mwe"]     * scale,
        "std_structural": reg_comps["std_structural"] * scale,
        "std_epistemic":  reg_comps["std_epistemic"]  * scale,
        "std_aleatoric":  np.where(~np.isnan(reg_aleatoric_arr), reg_aleatoric_arr * scale, np.nan),
        "std_total":      reg_std_total                          * scale,
    })
    ensemble_regional_gt.to_csv(output_dir / "ensemble_regional_gt.csv", index=False)
    print(f"  Saved ensemble_regional_gt.csv")

    # ------------------------------------------------------------------
    # 5. Load auxiliary data for plots
    # Paths already read into run_model_cfg above (step 1b).
    # ------------------------------------------------------------------

    glambie_wide_df = _load_glambie_wide(glambie_path)
    if glambie_wide_df is not None:
        print(f"  Loaded GLaMBIE data from {glambie_path}")
    else:
        print("  GLaMBIE path not set or file missing — GLaMBIE series skipped in plots.")

    oggm_df = _load_oggm_regional(inp_dir, reg_subdir)
    if not oggm_df.empty:
        print(f"  Loaded OGGM regional series ({len(oggm_df)} years).")
    else:
        print("  OGGM data not loaded — OGGM series skipped in plots.")

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")
    plot_ensemble_gt(ensemble_regional_gt, glambie_wide_df, oggm_df, total_area_km2, output_dir)
    plot_ensemble_mwe(ensemble_regional_mwe, glambie_wide_df, oggm_df, total_area_km2, output_dir)
    plot_ensemble_cumulative_gt(ensemble_regional_gt, glambie_wide_df, oggm_df,
                                total_area_km2, output_dir)
    print(f"\nDone. Outputs written to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble uncertainty estimation from Hydra multirun.")
    parser.add_argument("--config",       default="conf/config_ensemble_uncertainty.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--multirun_root", default=None,
                        help="Override multirun_root from config.")
    parser.add_argument("--output_dir",   default=None,
                        help="Override output_dir from config.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if args.multirun_root is not None:
        cfg["multirun_root"] = args.multirun_root
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    run_ensemble_uncertainty(cfg)


if __name__ == "__main__":
    main()
