"""
src/ensemble_uncertainty.py

Combines per-run predictions from a Hydra multirun sweep into a single ensemble
uncertainty estimate, decomposed into structural and epistemic components.

Uncertainty decomposition via the law of total variance:

    Var[y] = E_m[Var[y|m]]  +  Var_m[E[y|m]]
             └─ epistemic ──┘   └─ structural ─┘

    std_epistemic  = sqrt( Σ_k w_k * std_k² )
    std_structural = sqrt( Σ_k w_k * (mean_k - mu_ensemble)² )
    std_aleatoric  = NaN  (placeholder — requires heteroscedastic output head)
    std_total      = sqrt( std_epistemic² + std_structural² )

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
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

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
) -> dict:
    """
    Compute ensemble uncertainty components from stacked per-model arrays.

    Args:
        means_mat: (K, N) array of per-model predicted means
        stds_mat:  (K, N) array of per-model within-model MC stds
        weights:   (K,) normalised model weights

    Returns dict with keys: median_mwe, std_structural, std_epistemic,
                             std_aleatoric, std_total
    """
    w = weights[:, np.newaxis]                        # (K, 1) for broadcasting

    mu_ensemble    = (w * means_mat).sum(axis=0)      # (N,)
    std_structural = np.sqrt(
        (w * (means_mat - mu_ensemble[np.newaxis]) ** 2).sum(axis=0)
    )
    std_epistemic  = np.sqrt((w * stds_mat ** 2).sum(axis=0))
    std_aleatoric  = np.full_like(mu_ensemble, np.nan)
    std_total      = np.sqrt(std_structural ** 2 + std_epistemic ** 2)

    return {
        "median_mwe":     mu_ensemble,
        "std_structural": std_structural,
        "std_epistemic":  std_epistemic,
        "std_aleatoric":  std_aleatoric,
        "std_total":      std_total,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_regional_uncertainty(
    regional_mwe: pd.DataFrame,
    regional_gt: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Two-panel time series: regional MWE/yr (top) and Gt/yr (bottom).
    Shaded bands show ±1 std for each uncertainty component.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for ax, df, ylabel, title in [
        (axes[0], regional_mwe, "MWE/yr", "Regional ensemble — MWE/yr"),
        (axes[1], regional_gt,  "Gt/yr",  "Regional ensemble — Gt/yr"),
    ]:
        years = df["year"].values
        mu    = df["median_mwe" if "median_mwe" in df.columns else "median_gt"].values
        s_str = df["std_structural"].values
        s_eps = df["std_epistemic"].values
        s_tot = df["std_total"].values

        ax.fill_between(years, mu - s_tot, mu + s_tot,
                        alpha=0.15, color="steelblue", label="±1σ total")
        ax.fill_between(years, mu - s_str, mu + s_str,
                        alpha=0.25, color="darkorange", label="±1σ structural")
        ax.fill_between(years, mu - s_eps, mu + s_eps,
                        alpha=0.30, color="steelblue", label="±1σ epistemic")
        ax.plot(years, mu, color="steelblue", lw=1.8, label="ensemble median")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    axes[1].set_xlabel("Year")
    fig.tight_layout()
    _savefig(fig, output_dir / "regional_uncertainty.png")


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
    stds_mat  = np.stack([df["std"].values  for df in glacier_dfs])   # (K, N)

    comps = _ensemble_components(means_mat, stds_mat, glacier_weights_arr)

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
    regional_dfs, regional_weights = [], []
    for run_dir, w in zip(run_dirs, weights):
        df = _load_regional_mwe(run_dir)
        if df is not None:
            regional_dfs.append(df)
            regional_weights.append(w)

    _assert_grid_alignment(regional_dfs, ["year"], "regional_annual_mwe.csv")

    regional_weights_arr = np.array(regional_weights)
    regional_weights_arr /= regional_weights_arr.sum()

    reg_means_mat = np.stack([df["mean"].values for df in regional_dfs])   # (K, n_years)
    reg_stds_mat  = np.stack([df["std"].values  for df in regional_dfs])   # (K, n_years)

    reg_comps = _ensemble_components(reg_means_mat, reg_stds_mat, regional_weights_arr)

    ref_regional   = regional_dfs[0]
    years          = ref_regional["year"].values
    total_area_km2 = ref_regional["total_area_km2"].values  # same prediction grid across runs

    ensemble_regional_mwe = pd.DataFrame({
        "year":           years,
        "median_mwe":     reg_comps["median_mwe"],
        "std_structural": reg_comps["std_structural"],
        "std_epistemic":  reg_comps["std_epistemic"],
        "std_aleatoric":  reg_comps["std_aleatoric"],
        "std_total":      reg_comps["std_total"],
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
        "std_aleatoric":  reg_comps["std_aleatoric"],         # NaN — no scaling applied
        "std_total":      reg_comps["std_total"]      * scale,
    })
    ensemble_regional_gt.to_csv(output_dir / "ensemble_regional_gt.csv", index=False)
    print(f"  Saved ensemble_regional_gt.csv")

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    plot_regional_uncertainty(ensemble_regional_mwe, ensemble_regional_gt, output_dir)
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
