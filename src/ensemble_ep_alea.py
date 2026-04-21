"""
src/ensemble_ep_alea.py

Selects the top-N models by GLaMBIE test RMSE and decomposes their combined
predictive uncertainty into epistemic, aleatoric, and structural components.

Uncertainty decomposition (law of total variance across top-N models):

    std_epistemic  = sqrt( Σ_k w_k * epistemic_std_k² )
                     weighted MC-parameter uncertainty within each model
    std_aleatoric  = sqrt( Σ_k w_k * aleatoric_std_k² )
                     weighted predicted noise sigma (heteroscedastic=true only)
    std_structural = sqrt( Σ_k w_k * (mean_k - mu_ensemble)² )
                     spread of model means around the ensemble mean
    std_total      = sqrt( std_epistemic² + std_aleatoric² + std_structural² )

Only heteroscedastic runs (model.heteroscedastic=true) are considered.
Model selection criterion: lowest glambie_rmse over test_years (default 2020-2024).
Equal weights are used across the top-N models.

Inputs (read from each selected run directory):
    preds_full.csv            — rgi_id, year, mean, std, aleatoric_std,
                                epistemic_std, total_std
    regional_annual_mwe.csv   — year, total_area_km2, mean, std

Outputs written to {output_dir}/:
    top_models_info.csv          — selected runs and their scores
    top_models_glacier.csv       — per (rgi_id, year): mean_mwe, epistemic_std,
                                   aleatoric_std, structural_std, total_std
    top_models_regional_mwe.csv  — per year MWE/yr: same uncertainty columns
    top_models_regional_gt.csv   — per year Gt/yr:  same uncertainty columns
    top_models_regional_gt.png
    top_models_regional_mwe.png
    top_models_cumulative_gt.png

Usage:
    python src/ensemble_ep_alea.py
    python src/ensemble_ep_alea.py --config conf/config_ensemble_uncertainty.yaml
    python src/ensemble_ep_alea.py --multirun_root /path/to/r06_run \\
                                   --output_dir outputs/best_model/r06
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.hyperparam_tuning import build_results_df
from src.ensemble_uncertainty import (
    _read_run_model_cfg,
    _load_glacier_preds,
    _load_glambie_wide,
    _glambie_combined_gt,
    _glambie_sources_mwe,
    _load_oggm_regional,
    _aggregate_aleatoric_to_regional,
    _ensemble_components,
    _savefig,
)


# ---------------------------------------------------------------------------
# Best-model selection
# ---------------------------------------------------------------------------

def pick_top_runs(
    multirun_root: Path,
    test_years: list[int],
    min_runs: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return the top-N heteroscedastic runs sorted by lowest GLaMBIE test RMSE.

    Selection criterion: glambie_rmse over test_years (default 2020-2024).
    Only runs with heteroscedastic=True in their Hydra overrides are considered.
    Raises if no such runs are found or none have a valid glambie_rmse.
    Returns a DataFrame of up to top_n rows sorted ascending by glambie_rmse.
    """
    results_df = build_results_df(multirun_root, test_years, min_runs_per_region=min_runs)

    # Filter to heteroscedastic runs only
    if "heteroscedastic" in results_df.columns:
        hetero_df = results_df[results_df["heteroscedastic"] == True].reset_index(drop=True)
        n_total   = len(results_df)
        n_hetero  = len(hetero_df)
        if hetero_df.empty:
            raise RuntimeError(
                f"No runs with heteroscedastic=True found among {n_total} runs in "
                f"{multirun_root}. This script requires heteroscedastic=True training. "
                "Use ensemble_uncertainty.py for homoscedastic runs."
            )
        if n_hetero < n_total:
            print(f"  Filtered to {n_hetero}/{n_total} heteroscedastic runs.")
        results_df = hetero_df
    else:
        print("  WARNING: 'heteroscedastic' column not found in overrides — "
              "cannot pre-filter. Will validate after loading preds_full.csv.")

    valid = results_df.dropna(subset=["glambie_rmse"]).sort_values("glambie_rmse").reset_index(drop=True)
    if valid.empty:
        raise RuntimeError(
            "No heteroscedastic runs have a valid GLaMBIE test RMSE. "
            "Check that finetune completed without NaN loss and that "
            "metrics_glambie_test.csv exists in each run directory."
        )

    top = valid.head(top_n).reset_index(drop=True)
    if len(top) < top_n:
        print(f"  WARNING: only {len(top)} valid runs available (requested top {top_n}).")
    print(f"  Top {len(top)} runs by GLaMBIE RMSE:")
    for _, row in top.iterrows():
        print(f"    {row['run_id']}  glambie_rmse={row['glambie_rmse']:.4f}  "
              f"loyo_rmse={row.get('loyo_rmse', float('nan')):.4f}")
    return top


# ---------------------------------------------------------------------------
# Per-glacier ensemble uncertainty (top-N models)
# ---------------------------------------------------------------------------

def _load_and_validate_preds(run_dir: Path) -> pd.DataFrame:
    """Load preds_full.csv and validate that aleatoric_std is present."""
    df = _load_glacier_preds(run_dir)
    if df is None:
        raise FileNotFoundError(f"preds_full.csv not found in {run_dir}")
    if "aleatoric_std" not in df.columns:
        raise ValueError(
            f"preds_full.csv in {run_dir} does not contain 'aleatoric_std'. "
            "Re-run with model.heteroscedastic=true or use ensemble_uncertainty.py."
        )
    return df


def assemble_glacier_ensemble(top_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble per-glacier uncertainty from the top-N models.

    Uses equal weights across models. Structural uncertainty comes from
    the spread of model means around the ensemble mean.

    Returns DataFrame with columns:
        rgi_id, year, mean_mwe, epistemic_std, aleatoric_std,
        structural_std, total_std
    """
    run_dirs = [Path(r["run_dir"]) for _, r in top_runs.iterrows()]
    dfs = [_load_and_validate_preds(d) for d in run_dirs]

    # All runs must cover the same (rgi_id, year) grid
    ref = dfs[0][["rgi_id", "year"]].reset_index(drop=True)
    for i, df in enumerate(dfs[1:], start=1):
        if not (df[["rgi_id", "year"]].reset_index(drop=True) == ref).all().all():
            raise ValueError(
                f"Grid mismatch between run 0 and run {i} on (rgi_id, year). "
                "All top-N runs must be predictions over the same grid."
            )

    epistemic_col = "epistemic_std" if all("epistemic_std" in df.columns for df in dfs) else "std"

    means_mat     = np.stack([df["mean"].values          for df in dfs])          # (K, N)
    stds_mat      = np.stack([df[epistemic_col].values   for df in dfs])          # (K, N)
    aleatoric_mat = np.stack([df["aleatoric_std"].values for df in dfs])          # (K, N)

    K = len(dfs)
    weights = np.ones(K) / K   # equal weights

    comp = _ensemble_components(means_mat, stds_mat, weights, aleatoric_mat)

    return pd.DataFrame({
        "rgi_id":        dfs[0]["rgi_id"].values,
        "year":          dfs[0]["year"].values,
        "mean_mwe":      comp["median_mwe"],
        "epistemic_std": comp["std_epistemic"],
        "aleatoric_std": comp["std_aleatoric"],
        "structural_std": comp["std_structural"],
        "total_std":     comp["std_total"],
    })


# ---------------------------------------------------------------------------
# Regional uncertainty aggregation (top-N ensemble)
# ---------------------------------------------------------------------------

def _get_total_area(run_dir: Path, mwe_df: pd.DataFrame) -> np.ndarray:
    """Extract total_area_km2 from regional_annual_mwe.csv, or derive from Gt file."""
    if "total_area_km2" in mwe_df.columns:
        return mwe_df["total_area_km2"].values
    gt_path = run_dir / "regional_annual_gt.csv"
    if gt_path.exists():
        gt_df    = pd.read_csv(gt_path).sort_values("year").reset_index(drop=True)
        mwe_mean = mwe_df["mean"].values
        area = np.where(
            np.abs(mwe_mean) > 1e-10,
            gt_df["mean"].values / (mwe_mean * 1e-3),
            0.0,
        )
        print("  Note: total_area_km2 derived from regional_annual_gt.csv.")
        return area
    warnings.warn("total_area_km2 unavailable — Gt conversion will be zero.")
    return np.zeros(len(mwe_df))


def compute_regional_uncertainties(
    glacier_df: pd.DataFrame,
    top_runs: pd.DataFrame,
    inp_dir: str,
    reg_subdir: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Aggregate per-glacier uncertainties to annual regional series for top-N ensemble.

    Epistemic:   weighted MC-parameter spread from each model's regional_annual_mwe.csv
    Aleatoric:   area-weighted propagation of ensemble aleatoric_std from glacier_df
    Structural:  spread of per-model regional means around the ensemble mean
    Total:       sqrt(epistemic² + aleatoric² + structural²)

    Returns:
        regional_df    — DataFrame: year, median_mwe, epistemic_std, aleatoric_std,
                         structural_std, total_std
        total_area_km2 — numpy array (for Gt conversion)
    """
    run_dirs = [Path(r["run_dir"]) for _, r in top_runs.iterrows()]

    reg_dfs     = []
    valid_dirs  = []
    for rd in run_dirs:
        path = rd / "regional_annual_mwe.csv"
        if not path.exists():
            warnings.warn(f"regional_annual_mwe.csv not found in {rd} — skipped for regional.")
            continue
        reg_dfs.append(pd.read_csv(path).sort_values("year").reset_index(drop=True))
        valid_dirs.append(rd)

    if not reg_dfs:
        raise FileNotFoundError("No regional_annual_mwe.csv found in any of the top runs.")

    years          = reg_dfs[0]["year"].values
    total_area_km2 = _get_total_area(valid_dirs[0], reg_dfs[0])

    K       = len(reg_dfs)
    weights = np.ones(K) / K
    means_mat = np.stack([df["mean"].values for df in reg_dfs])   # (K, T)
    stds_mat  = np.stack([df["std"].values  for df in reg_dfs])   # (K, T)

    # Regional aleatoric from ensemble per-glacier aleatoric via area-weighting.
    # _aggregate_aleatoric_to_regional expects column 'std_aleatoric', so rename.
    glacier_for_agg = glacier_df.rename(columns={"aleatoric_std": "std_aleatoric"})
    reg_alea_df = _aggregate_aleatoric_to_regional(glacier_for_agg, inp_dir, reg_subdir)
    if reg_alea_df is not None:
        year_to_alea  = dict(zip(reg_alea_df["year"], reg_alea_df["std_aleatoric"]))
        aleatoric_arr = np.array([year_to_alea.get(y, np.nan) for y in years])
        valid_ct = (~np.isnan(aleatoric_arr)).sum()
        print(f"  Regional aleatoric propagated from per-glacier ({valid_ct}/{len(years)} years).")
    else:
        aleatoric_arr = np.full(len(years), np.nan)

    # Structural + epistemic from ensemble components helper
    comp = _ensemble_components(means_mat, stds_mat, weights, aleatoric_mat=None)

    # Add aleatoric in quadrature
    alea_sq   = np.where(np.isnan(aleatoric_arr), 0.0, aleatoric_arr ** 2)
    total_std = np.sqrt(comp["std_structural"] ** 2 + comp["std_epistemic"] ** 2 + alea_sq)

    regional_df = pd.DataFrame({
        "year":           years,
        "median_mwe":     comp["median_mwe"],
        "epistemic_std":  comp["std_epistemic"],
        "aleatoric_std":  aleatoric_arr,
        "structural_std": comp["std_structural"],
        "total_std":      total_std,
    })
    return regional_df, total_area_km2


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _shade(ax, years, mu, s_epi, s_struct, s_tot):
    """
    Draw three uncertainty bands (outermost to innermost):
      - ±2σ total      (steelblue,   outer  — epistemic + aleatoric + structural)
      - ±2σ structural (mediumorchid, middle — model-choice spread only)
      - ±2σ epistemic  (darkorange,  inner  — within-model parameter uncertainty)
    """
    ax.fill_between(years, mu - 2 * s_tot, mu + 2 * s_tot,
                    alpha=0.15, color="steelblue",    label="±2σ total")
    ax.fill_between(years, mu - 2 * s_struct, mu + 2 * s_struct,
                    alpha=0.20, color="mediumorchid", label="±2σ structural")
    ax.fill_between(years, mu - 2 * s_epi, mu + 2 * s_epi,
                    alpha=0.30, color="darkorange",   label="±2σ epistemic")
    ax.plot(years, mu, color="steelblue", lw=1.8, label="ensemble mean")


def plot_top_models_gt(
    regional_gt: pd.DataFrame,
    glambie_wide_df,
    oggm_df: pd.DataFrame,
    total_area_km2: np.ndarray,
    output_dir: Path,
) -> None:
    years           = regional_gt["year"].values
    mu              = regional_gt["median_gt"].values
    total_area_mean = float(total_area_km2.mean())

    gb_combined = _glambie_combined_gt(glambie_wide_df, total_area_mean)
    gb_sources  = _glambie_sources_mwe(glambie_wide_df)

    fig, ax = plt.subplots(figsize=(12, 5))
    _shade(ax, years, mu,
           regional_gt["epistemic_std"].values,
           regional_gt["structural_std"].values,
           regional_gt["total_std"].values)

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
    ax.set_title("Regional mass balance — top-N ensemble (Gt/yr)\n"
                 "Orange: epistemic  Purple: structural  Blue: total")
    ax.legend(fontsize=8); fig.tight_layout()
    _savefig(fig, output_dir / "top_models_regional_gt.png")


def plot_top_models_mwe(
    regional_mwe: pd.DataFrame,
    glambie_wide_df,
    oggm_df: pd.DataFrame,
    total_area_km2: np.ndarray,
    output_dir: Path,
) -> None:
    years           = regional_mwe["year"].values
    mu              = regional_mwe["median_mwe"].values
    total_area_mean = float(total_area_km2.mean())

    gb_sources     = _glambie_sources_mwe(glambie_wide_df)
    gb_combined_gt = _glambie_combined_gt(glambie_wide_df, total_area_mean)

    fig, ax = plt.subplots(figsize=(12, 5))
    _shade(ax, years, mu,
           regional_mwe["epistemic_std"].values,
           regional_mwe["structural_std"].values,
           regional_mwe["total_std"].values)

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
    ax.set_title("Regional mass balance — top-N ensemble (MWE/yr)\n"
                 "Orange: epistemic  Purple: structural  Blue: total")
    ax.legend(fontsize=8); fig.tight_layout()
    _savefig(fig, output_dir / "top_models_regional_mwe.png")


def plot_top_models_cumulative_gt(
    regional_gt: pd.DataFrame,
    glambie_wide_df,
    oggm_df: pd.DataFrame,
    total_area_km2: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Cumulative Gt with epistemic, structural, and total uncertainty bands.

    Cumulative sigma propagated in quadrature (year-to-year independence):
        cum_sigma_t = sqrt( Σ_{s≤t} sigma_s² )
    """
    total_area_mean = float(total_area_km2.mean())
    gb_combined = _glambie_combined_gt(glambie_wide_df, total_area_mean)

    start_year = int(gb_combined["year"].min()) if not gb_combined.empty \
        else int(regional_gt["year"].min())

    mask   = regional_gt["year"].values >= start_year
    years  = regional_gt["year"].values[mask]
    cum_mu = np.cumsum(regional_gt["median_gt"].values[mask])

    cum_epi    = np.sqrt(np.cumsum(regional_gt["epistemic_std"].values[mask]  ** 2))
    cum_struct = np.sqrt(np.cumsum(regional_gt["structural_std"].values[mask] ** 2))
    cum_tot    = np.sqrt(np.cumsum(regional_gt["total_std"].values[mask]      ** 2))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(years, cum_mu - 2 * cum_tot,    cum_mu + 2 * cum_tot,
                    alpha=0.15, color="steelblue",    label="±2σ total")
    ax.fill_between(years, cum_mu - 2 * cum_struct, cum_mu + 2 * cum_struct,
                    alpha=0.20, color="mediumorchid", label="±2σ structural")
    ax.fill_between(years, cum_mu - 2 * cum_epi,    cum_mu + 2 * cum_epi,
                    alpha=0.25, color="darkorange",   label="±2σ epistemic")
    ax.plot(years, cum_mu, color="steelblue", lw=1.8, label="Ensemble cumulative mean")

    if not oggm_df.empty:
        og = oggm_df[oggm_df["year"] >= start_year].copy()
        if not og.empty:
            ax.plot(og["year"].values, np.cumsum(og["gt"].values),
                    color="red", lw=1.3, ls="--", label="OGGM")

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
    ax.set_title(f"Cumulative regional mass balance from {start_year} — top-N ensemble\n"
                 "Orange: epistemic  Purple: structural  Blue: total")
    ax.legend(fontsize=8); fig.tight_layout()
    _savefig(fig, output_dir / "top_models_cumulative_gt.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ep_alea(cfg: dict) -> None:
    multirun_root = Path(cfg["multirun_root"])
    output_dir    = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    test_years = list(cfg.get("glambie_test_years", [2020, 2021, 2022, 2023, 2024]))
    min_runs   = int(cfg.get("min_runs_per_region", 1))
    top_n      = int(cfg.get("top_n", 5))

    print(f"\n=== Top-{top_n} ensemble (epistemic + aleatoric + structural): {multirun_root.name} ===")
    print(f"  Selecting top {top_n} runs by GLaMBIE test RMSE over years {test_years}")

    # ------------------------------------------------------------------
    # 1. Select top-N runs
    # ------------------------------------------------------------------
    top_runs = pick_top_runs(multirun_root, test_years, min_runs, top_n=top_n)
    run_dirs = [Path(r["run_dir"]) for _, r in top_runs.iterrows()]

    top_runs.to_csv(output_dir / "top_models_info.csv", index=False)
    print(f"  Saved top_models_info.csv")

    # ------------------------------------------------------------------
    # 2. Per-glacier ensemble uncertainty
    # ------------------------------------------------------------------
    print("\n--- Per-glacier uncertainty ---")
    glacier_df = assemble_glacier_ensemble(top_runs)
    glacier_df.to_csv(output_dir / "top_models_glacier.csv", index=False)
    print(f"  Saved top_models_glacier.csv  ({len(glacier_df)} rows)")

    # ------------------------------------------------------------------
    # 3. Read model config paths (for aleatoric propagation + aux data)
    # ------------------------------------------------------------------
    run_model_cfg = _read_run_model_cfg(run_dirs)
    inp_dir      = run_model_cfg.get("inp_dir", "")
    reg_subdir   = run_model_cfg.get("reg_subdir", "")
    glambie_path = run_model_cfg.get("glambie_path", "")

    # ------------------------------------------------------------------
    # 4. Regional MWE uncertainty
    # ------------------------------------------------------------------
    print("\n--- Regional uncertainty ---")
    regional_mwe, total_area_km2 = compute_regional_uncertainties(
        glacier_df, top_runs, inp_dir, reg_subdir
    )
    regional_mwe.to_csv(output_dir / "top_models_regional_mwe.csv", index=False)
    print(f"  Saved top_models_regional_mwe.csv  ({len(regional_mwe)} years)")

    # ------------------------------------------------------------------
    # 5. Regional Gt (MWE × total_area × 1e-3)
    # ------------------------------------------------------------------
    scale = total_area_km2 * 1e-3

    regional_gt = pd.DataFrame({
        "year":           regional_mwe["year"].values,
        "median_gt":      regional_mwe["median_mwe"].values      * scale,
        "epistemic_std":  regional_mwe["epistemic_std"].values   * scale,
        "aleatoric_std":  np.where(
            regional_mwe["aleatoric_std"].isna(),
            np.nan,
            regional_mwe["aleatoric_std"].fillna(0).values * scale,
        ),
        "structural_std": regional_mwe["structural_std"].values  * scale,
        "total_std":      regional_mwe["total_std"].values       * scale,
    })
    regional_gt.to_csv(output_dir / "top_models_regional_gt.csv", index=False)
    print(f"  Saved top_models_regional_gt.csv")

    # ------------------------------------------------------------------
    # 6. Auxiliary data for plots
    # ------------------------------------------------------------------
    glambie_wide_df = _load_glambie_wide(glambie_path)
    if glambie_wide_df is not None:
        print(f"  Loaded GLaMBIE data from {glambie_path}")
    else:
        print("  GLaMBIE path not set or missing — GLaMBIE series skipped in plots.")

    oggm_df = _load_oggm_regional(inp_dir, reg_subdir)
    if not oggm_df.empty:
        print(f"  Loaded OGGM regional series ({len(oggm_df)} years).")
    else:
        print("  OGGM data not loaded — skipped in plots.")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")
    plot_top_models_gt(regional_gt, glambie_wide_df, oggm_df, total_area_km2, output_dir)
    plot_top_models_mwe(regional_mwe, glambie_wide_df, oggm_df, total_area_km2, output_dir)
    plot_top_models_cumulative_gt(regional_gt, glambie_wide_df, oggm_df, total_area_km2, output_dir)

    print(f"\nDone. Outputs written to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Best-model epistemic + aleatoric uncertainty from a Hydra multirun."
    )
    parser.add_argument("--config",        default="conf/config_ensemble_uncertainty.yaml",
                        help="Path to YAML config file (same format as ensemble_uncertainty).")
    parser.add_argument("--multirun_root", default=None,
                        help="Override multirun_root from config.")
    parser.add_argument("--output_dir",    default=None,
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

    run_ep_alea(cfg)


if __name__ == "__main__":
    main()
