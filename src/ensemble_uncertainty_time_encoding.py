"""
src/ensemble_uncertainty_time_encoding.py

Variant of ensemble_uncertainty_split.py for sweeps that include
model.use_time_encoding=true,false as a sweep axis, with model.heteroscedastic=true fixed.

Reads use_time_encoding from each run's .hydra/config.yaml, partitions all completed
runs into two groups, selects the top-N (default 5) within each group by a configurable
selection metric (default: glambie_rmse; alternatives: loyo_rmse, loyo_r2), and writes
separate ensemble outputs using equal weights across the top-N:

    {output_dir}/time_encoding/    — runs where model.use_time_encoding=true
    {output_dir}/no_time_encoding/ — runs where model.use_time_encoding=false

Within each group, full uncertainty decomposition is computed:
    std_epistemic  = sqrt( Σ_k (1/K) * epistemic_std_k² )
    std_aleatoric  = sqrt( Σ_k (1/K) * aleatoric_std_k² )   (always present; heteroscedastic=true)
    std_structural = sqrt( Σ_k (1/K) * (mean_k - mu_ensemble)² )
    std_total      = sqrt( std_epistemic² + std_aleatoric² + std_structural² )

Usage:
    python src/ensemble_uncertainty_time_encoding.py \\
        --multirun_root /scratch/.../multirun/r03_12345 \\
        --output_dir outputs/ensemble/r03

    python src/ensemble_uncertainty_time_encoding.py \\
        --config conf/config_ensemble_uncertainty.yaml \\
        --multirun_root /scratch/.../multirun/r03_12345 \\
        --output_dir outputs/ensemble/r03
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
import numpy as np
import pandas as pd
import yaml

from src.ensemble_uncertainty import (
    _read_run_model_cfg,
    _load_glacier_preds,
    _load_regional_mwe,
    _assert_grid_alignment,
    _ensemble_components,
    _aggregate_aleatoric_to_regional,
    _load_glambie_wide,
    _load_oggm_regional,
    plot_ensemble_gt,
    plot_ensemble_mwe,
    plot_ensemble_cumulative_gt,
    _savefig,
)
from src.hyperparam_tuning import build_results_df


# ---------------------------------------------------------------------------
# Read use_time_encoding flag from a run's Hydra config
# ---------------------------------------------------------------------------

def _read_use_time_encoding(run_dir: Path) -> bool | None:
    """
    Read model.use_time_encoding from .hydra/config.yaml for a single run.

    Returns True/False if found, None if config is missing or the key is absent
    (treated as True — time encoding on by default).
    """
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path) as fh:
            full_cfg = yaml.safe_load(fh)
        val = full_cfg.get("model", {}).get("use_time_encoding", True)
        return bool(val)
    except Exception as exc:
        warnings.warn(f"Could not read {cfg_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Top-N ensemble runner for a single group
# ---------------------------------------------------------------------------

def _minmax_norm_series(s: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]. Returns zeros if all values are equal."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def _run_top_n_group(
    group_df: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    min_runs: int,
    selection_metric: str = "glambie_rmse",
    loyo_r2_min: float | None = 0.1,
    logo_r2_min: float | None = 0.0,
    composite_loyo_r2_weight: float = 0.5,
) -> bool:
    """
    Run the full ensemble pipeline for one group.

    Selects top_n runs by selection_metric, uses equal weights.
    All runs are expected to be heteroscedastic=true; raises if aleatoric_std
    is missing from any preds_full.csv.

    Args:
        group_df:                 Subset of the results DataFrame for this group.
        output_dir:               Where to write this group's outputs.
        top_n:                    Number of top runs to include in ensemble.
        min_runs:                 Minimum runs required after gating to proceed.
        selection_metric:         Ranking criterion. One of:
                                    "glambie_rmse" — GLaMBIE RMSE (lower=better)
                                    "loyo_rmse"    — pretrain LOYO RMSE (lower=better)
                                    "loyo_r2"      — pretrain LOYO R² (higher=better)
                                    "composite"    — weighted combination of normalised
                                                     glambie_rmse and loyo_r2 (lower=better).
                                                     Weights: composite_loyo_r2_weight for
                                                     loyo_r2, remainder for glambie_rmse.
        loyo_r2_min:              Hard gate: minimum LOYO R² to be eligible. None = disabled.
        logo_r2_min:              Hard gate: minimum LOGO R² to be eligible. None = disabled.
        composite_loyo_r2_weight: Weight on loyo_r2 in the composite score (0–1).
                                  glambie_rmse weight = 1 - this. Only used when
                                  selection_metric="composite".

    Returns:
        True if the group was processed successfully, False if skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start with runs that have a valid glambie_rmse (always needed for print summary)
    valid = group_df.dropna(subset=["glambie_rmse"]).copy().reset_index(drop=True)

    # --- Hard gate: LOYO R² ---
    if loyo_r2_min is not None and "loyo_r2" in valid.columns:
        n_before = len(valid)
        valid = valid[valid["loyo_r2"] >= loyo_r2_min].reset_index(drop=True)
        n_rejected = n_before - len(valid)
        if n_rejected:
            print(f"  [loyo_r2 gate] Rejected {n_rejected}/{n_before} runs with "
                  f"loyo_r2 < {loyo_r2_min}")

    # --- Hard gate: LOGO R² ---
    if logo_r2_min is not None and "logo_r2" in valid.columns:
        n_before = len(valid)
        valid = valid[valid["logo_r2"] >= logo_r2_min].reset_index(drop=True)
        n_rejected = n_before - len(valid)
        if n_rejected:
            print(f"  [logo_r2 gate] Rejected {n_rejected}/{n_before} runs with "
                  f"logo_r2 < {logo_r2_min}")

    n_valid = len(valid)
    if n_valid < min_runs:
        gates = []
        if loyo_r2_min is not None:
            gates.append(f"loyo_r2 >= {loyo_r2_min}")
        if logo_r2_min is not None:
            gates.append(f"logo_r2 >= {logo_r2_min}")
        gate_str = " and ".join(gates) if gates else "no gate"
        print(f"  SKIPPED: only {n_valid} run(s) pass gates ({gate_str}) "
              f"(need {min_runs}). Adjust thresholds or lower --min_runs.")
        return False

    # --- Rank by selection_metric ---
    if selection_metric == "composite":
        w_loyo    = float(composite_loyo_r2_weight)
        w_glambie = 1.0 - w_loyo
        norm_glambie = _minmax_norm_series(valid["glambie_rmse"])
        norm_neg_r2  = 1.0 - _minmax_norm_series(valid["loyo_r2"])
        valid = valid.copy()
        valid["_composite"] = w_glambie * norm_glambie + w_loyo * norm_neg_r2
        valid = valid.sort_values("_composite", ascending=True).reset_index(drop=True)
        rank_label = f"composite (loyo_r2 w={w_loyo:.2f}, glambie_rmse w={w_glambie:.2f}) ↓ lower better"
    elif selection_metric == "loyo_r2":
        valid = valid.sort_values("loyo_r2", ascending=False).reset_index(drop=True)
        rank_label = "loyo_r2 ↑ higher better"
    else:
        valid = valid.sort_values(selection_metric, ascending=True).reset_index(drop=True)
        rank_label = f"{selection_metric} ↓ lower better"

    top = valid.head(top_n).reset_index(drop=True)
    if len(top) < top_n:
        print(f"  WARNING: only {len(top)} qualifying runs available (requested top {top_n}).")
    print(f"  Top {len(top)} runs by {rank_label}:")
    for _, row in top.iterrows():
        composite_str = (f"  composite={row['_composite']:.4f}" if "_composite" in row.index else "")
        print(f"    {row['run_id']}{composite_str}  "
              f"glambie_rmse={row.get('glambie_rmse', float('nan')):.4f}  "
              f"loyo_r2={row.get('loyo_r2', float('nan')):.4f}  "
              f"logo_r2={row.get('logo_r2', float('nan')):.4f}  "
              f"loyo_rmse={row.get('loyo_rmse', float('nan')):.4f}")

    top.to_csv(output_dir / "top_runs_info.csv", index=False)

    run_dirs = [Path(d) for d in top["run_dir"].values]
    K        = len(run_dirs)
    weights  = np.ones(K) / K    # equal weights

    # --- Read auxiliary paths from first readable run config ---
    run_model_cfg = _read_run_model_cfg(run_dirs)
    glambie_path  = run_model_cfg.get("glambie_path", "")
    inp_dir       = run_model_cfg.get("inp_dir", "")
    reg_subdir    = run_model_cfg.get("reg_subdir", "")

    # ----------------------------------------------------------------
    # Per-glacier ensemble
    # ----------------------------------------------------------------
    print("  --- Per-glacier ensemble ---")
    glacier_dfs, glacier_weights = [], []
    for run_dir, w in zip(run_dirs, weights):
        df = _load_glacier_preds(run_dir)
        if df is None:
            continue
        if "aleatoric_std" not in df.columns:
            raise ValueError(
                f"preds_full.csv in {run_dir} has no 'aleatoric_std' column. "
                "All runs in this sweep must use model.heteroscedastic=true."
            )
        glacier_dfs.append(df)
        glacier_weights.append(w)

    if len(glacier_dfs) < min_runs:
        warnings.warn(
            f"  Only {len(glacier_dfs)} runs have preds_full.csv "
            f"(threshold={min_runs}) — skipping."
        )
        return

    _assert_grid_alignment(glacier_dfs, ["rgi_id", "year"], "preds_full.csv")

    glacier_weights_arr = np.array(glacier_weights)
    glacier_weights_arr /= glacier_weights_arr.sum()

    epistemic_col = (
        "epistemic_std"
        if all("epistemic_std" in df.columns for df in glacier_dfs)
        else "std"
    )
    means_mat     = np.stack([df["mean"].values          for df in glacier_dfs])
    stds_mat      = np.stack([df[epistemic_col].values   for df in glacier_dfs])
    aleatoric_mat = np.stack([df["aleatoric_std"].values for df in glacier_dfs])

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

    # ----------------------------------------------------------------
    # Regional MWE ensemble
    # ----------------------------------------------------------------
    print("  --- Regional MWE ensemble ---")
    regional_dfs, regional_weights, regional_run_dirs = [], [], []
    for run_dir, w in zip(run_dirs, weights):
        df = _load_regional_mwe(run_dir)
        if df is not None:
            regional_dfs.append(df)
            regional_weights.append(w)
            regional_run_dirs.append(run_dir)

    if not regional_dfs:
        warnings.warn("  No regional_annual_mwe.csv found — skipping regional ensemble.")
        return

    _assert_grid_alignment(regional_dfs, ["year"], "regional_annual_mwe.csv")

    regional_weights_arr = np.array(regional_weights)
    regional_weights_arr /= regional_weights_arr.sum()

    reg_means_mat = np.stack([df["mean"].values for df in regional_dfs])
    reg_stds_mat  = np.stack([df["std"].values  for df in regional_dfs])

    reg_comps = _ensemble_components(reg_means_mat, reg_stds_mat, regional_weights_arr)

    ref_regional   = regional_dfs[0]
    years          = ref_regional["year"].values

    if "total_area_km2" in ref_regional.columns:
        total_area_km2 = ref_regional["total_area_km2"].values
    else:
        gt_path = regional_run_dirs[0] / "regional_annual_gt.csv"
        gt_df   = pd.read_csv(gt_path).sort_values("year").reset_index(drop=True)
        mwe_mean = ref_regional["mean"].values
        total_area_km2 = np.where(
            np.abs(mwe_mean) > 1e-10,
            gt_df["mean"].values / (mwe_mean * 1e-3),
            0.0,
        )
        print("  Note: total_area_km2 derived from regional_annual_gt.csv.")

    # Regional aleatoric via area-weighted propagation from per-glacier
    reg_aleatoric_df = _aggregate_aleatoric_to_regional(ensemble_glacier, inp_dir, reg_subdir)
    if reg_aleatoric_df is not None:
        year_to_reg_aleatoric = dict(zip(reg_aleatoric_df["year"], reg_aleatoric_df["std_aleatoric"]))
        reg_aleatoric_arr = np.array([year_to_reg_aleatoric.get(y, np.nan) for y in years])
        valid_mask = ~np.isnan(reg_aleatoric_arr)
        reg_std_total = np.sqrt(
            reg_comps["std_structural"] ** 2 + reg_comps["std_epistemic"] ** 2
            + np.where(valid_mask, reg_aleatoric_arr ** 2, 0.0)
        )
        print(f"  Regional aleatoric propagated ({valid_mask.sum()}/{len(years)} years).")
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

    # ----------------------------------------------------------------
    # Regional Gt ensemble
    # ----------------------------------------------------------------
    scale = total_area_km2 * 1e-3

    ensemble_regional_gt = pd.DataFrame({
        "year":           years,
        "median_gt":      reg_comps["median_mwe"]     * scale,
        "std_structural": reg_comps["std_structural"] * scale,
        "std_epistemic":  reg_comps["std_epistemic"]  * scale,
        "std_aleatoric":  np.where(~np.isnan(reg_aleatoric_arr), reg_aleatoric_arr * scale, np.nan),
        "std_total":      reg_std_total * scale,
    })
    ensemble_regional_gt.to_csv(output_dir / "ensemble_regional_gt.csv", index=False)
    print(f"  Saved ensemble_regional_gt.csv")

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    glambie_wide_df = _load_glambie_wide(glambie_path)
    oggm_df         = _load_oggm_regional(inp_dir, reg_subdir)

    print("  --- Generating plots ---")
    plot_ensemble_gt(ensemble_regional_gt, glambie_wide_df, oggm_df, total_area_km2, output_dir)
    plot_ensemble_mwe(ensemble_regional_mwe, glambie_wide_df, oggm_df, total_area_km2, output_dir)
    plot_ensemble_cumulative_gt(
        ensemble_regional_gt, glambie_wide_df, oggm_df, total_area_km2, output_dir
    )
    print(f"  Done → {output_dir}/")
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ensemble_time_encoding(cfg: dict) -> None:
    """
    Split a sweep by use_time_encoding and run top-N ensemble for each group.

    Reads model.use_time_encoding from each run's .hydra/config.yaml, partitions
    runs into two groups (True/False), selects top_n by glambie_rmse within each,
    and runs the full ensemble pipeline independently for each group.
    """
    multirun_root = Path(cfg["multirun_root"])
    output_dir    = Path(cfg["output_dir"])

    test_years                = list(cfg.get("glambie_test_years", [2020, 2021, 2022, 2023, 2024]))
    min_runs                  = int(cfg.get("min_runs_per_region", 1))
    top_n                     = int(cfg.get("top_n", 5))
    selection_metric          = str(cfg.get("selection_metric", "glambie_rmse"))
    loyo_r2_min               = cfg.get("loyo_r2_min", 0.1)
    logo_r2_min               = cfg.get("logo_r2_min", 0.0)
    composite_loyo_r2_weight  = float(cfg.get("composite_loyo_r2_weight", 0.5))
    if loyo_r2_min is not None:
        loyo_r2_min = float(loyo_r2_min)
    if logo_r2_min is not None:
        logo_r2_min = float(logo_r2_min)

    valid_metrics = {"glambie_rmse", "loyo_rmse", "loyo_r2", "composite"}
    if selection_metric not in valid_metrics:
        raise ValueError(f"selection_metric must be one of {valid_metrics}, got '{selection_metric}'")

    print(f"\n=== Ensemble split by use_time_encoding: {multirun_root.name} ===")
    print(f"  Selecting top {top_n} runs per group by {selection_metric}"
          + (f" over GLaMBIE test years {test_years}" if "glambie" in selection_metric else ""))
    print(f"  LOYO R² gate    : >= {loyo_r2_min}" if loyo_r2_min is not None else "  LOYO R² gate    : disabled")
    print(f"  LOGO R² gate    : >= {logo_r2_min}" if logo_r2_min is not None else "  LOGO R² gate    : disabled")

    results_df = build_results_df(multirun_root, test_years, min_runs_per_region=1)
    print(f"  {len(results_df)} total runs loaded.")

    # Read use_time_encoding for every run
    results_df["use_time_encoding"] = [
        _read_use_time_encoding(Path(d)) for d in results_df["run_dir"].values
    ]
    n_unknown = results_df["use_time_encoding"].isna().sum()
    if n_unknown:
        warnings.warn(
            f"  {n_unknown} runs had unreadable Hydra configs — "
            "use_time_encoding flag treated as True for those runs."
        )
        results_df["use_time_encoding"] = results_df["use_time_encoding"].fillna(True)

    groups = [
        (True,  "time_encoding",    output_dir / "time_encoding"),
        (False, "no_time_encoding", output_dir / "no_time_encoding"),
    ]

    skipped = []
    for flag_val, label, group_output_dir in groups:
        group = results_df[results_df["use_time_encoding"] == flag_val].copy().reset_index(drop=True)
        print(f"\n--- Group: {label} (use_time_encoding={flag_val})  —  {len(group)} runs ---")
        if group.empty:
            print("  No runs found — skipping.")
            skipped.append(label)
            continue
        ok = _run_top_n_group(
            group, group_output_dir, top_n, min_runs, selection_metric,
            loyo_r2_min, logo_r2_min, composite_loyo_r2_weight,
        )
        if not ok:
            skipped.append(label)

    if skipped:
        print(f"\n  WARNING: the following groups were skipped due to insufficient qualifying runs: {skipped}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble uncertainty split by model.use_time_encoding flag, top-N by GLaMBIE RMSE."
    )
    parser.add_argument("--config", default="conf/config_ensemble_uncertainty.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--multirun_root", default=None,
                        help="Override multirun_root from config.")
    parser.add_argument("--output_dir", default=None,
                        help="Override output_dir from config.")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Override top_n from config (default 5).")
    parser.add_argument("--loyo_r2_min", type=float, default=None,
                        help="Minimum LOYO R² a run must achieve to be eligible for the "
                             "ensemble (default 0.1). Pass --loyo_r2_min=-inf to disable.")
    parser.add_argument("--logo_r2_min", type=float, default=None,
                        help="Minimum LOGO R² a run must achieve (default 0.0). "
                             "Pass --logo_r2_min=-inf to disable.")
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
    if args.top_n is not None:
        cfg["top_n"] = args.top_n
    if args.loyo_r2_min is not None:
        cfg["loyo_r2_min"] = args.loyo_r2_min
    if args.logo_r2_min is not None:
        cfg["logo_r2_min"] = args.logo_r2_min

    run_ensemble_time_encoding(cfg)


if __name__ == "__main__":
    main()
