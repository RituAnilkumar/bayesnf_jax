"""
src/ensemble_uncertainty_split.py

Variant of ensemble_uncertainty.py for sweeps that include
model.heteroscedastic=true,false as a sweep axis.

Reads the heteroscedastic flag from each run's .hydra/config.yaml, partitions
all completed runs into two groups, and writes separate ensemble outputs:

    {output_dir}/aleatoric/             — runs where model.heteroscedastic=true
                                          uncertainty = structural + epistemic + aleatoric
    {output_dir}/epistemic_structural/  — runs where model.heteroscedastic=false
                                          uncertainty = structural + epistemic only

Within each group, model weighting and all ensemble/uncertainty decomposition
logic are identical to ensemble_uncertainty.py (law of total variance,
structural + epistemic + aleatoric components, performance-based softmax weights).

Usage:
    python src/ensemble_uncertainty_split.py \\
        --multirun_root /scratch/.../multirun/r06_12345 \\
        --output_dir outputs/ensemble/r06

    # or via YAML config:
    python src/ensemble_uncertainty_split.py \\
        --config conf/config_ensemble_uncertainty.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

# Ensure the project root is on sys.path so this script can be run directly
# (i.e. `python src/ensemble_uncertainty_split.py`) without `python -m`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

from src.ensemble_uncertainty import (
    compute_weights,
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
)
from src.hyperparam_tuning import build_results_df, add_composite_score


# ---------------------------------------------------------------------------
# Read heteroscedastic flag from a run's Hydra config
# ---------------------------------------------------------------------------

def _read_heteroscedastic(run_dir: Path) -> bool | None:
    """
    Read model.heteroscedastic from .hydra/config.yaml for a single run.

    Returns True/False if found, None if the config is missing or the key
    is absent (treated as False — homoscedastic).
    """
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path) as fh:
            full_cfg = yaml.safe_load(fh)
        val = full_cfg.get("model", {}).get("heteroscedastic", False)
        return bool(val)
    except Exception as exc:
        warnings.warn(f"Could not read {cfg_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Single-group ensemble runner (reuses ensemble_uncertainty helpers)
# ---------------------------------------------------------------------------

def _run_group_ensemble(
    valid_group: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    weighting: str,
    temperature: float,
    min_runs: int,
) -> None:
    """
    Run the full ensemble pipeline for one heteroscedastic group.

    Args:
        valid_group: Subset of the results DataFrame for this group.
                     Must have columns: run_dir, composite.
        cfg:         Full config dict (for glambie_weight etc. in aux data paths).
        output_dir:  Where to write this group's outputs.
        weighting:   'equal' or 'performance'.
        temperature: Softmax temperature for performance weighting.
        min_runs:    Minimum runs required to proceed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(valid_group)
    print(f"  {n} runs in group → {output_dir}")

    if n < min_runs:
        warnings.warn(
            f"  Only {n} runs (threshold={min_runs}) — skipping this group. "
            "Lower min_runs_per_region or check that runs completed successfully."
        )
        return

    run_dirs = [Path(d) for d in valid_group["run_dir"].values]
    weights  = compute_weights(valid_group["composite"].values, weighting, temperature)
    valid_group = valid_group.copy()
    valid_group["weight"] = weights
    valid_group[["run_id", "composite", "weight"]].to_csv(
        output_dir / "model_weights.csv", index=False
    )
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

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
        if df is not None:
            glacier_dfs.append(df)
            glacier_weights.append(w)

    if len(glacier_dfs) < min_runs:
        warnings.warn(
            f"  Only {len(glacier_dfs)} runs have preds_full.csv "
            f"(threshold={min_runs}) — skipping per-glacier ensemble."
        )
        return

    _assert_grid_alignment(glacier_dfs, ["rgi_id", "year"], "preds_full.csv")

    glacier_weights_arr = np.array(glacier_weights)
    glacier_weights_arr /= glacier_weights_arr.sum()

    means_mat = np.stack([df["mean"].values for df in glacier_dfs])

    epistemic_col = (
        "epistemic_std"
        if all("epistemic_std" in df.columns for df in glacier_dfs)
        else "std"
    )
    stds_mat = np.stack([df[epistemic_col].values for df in glacier_dfs])

    has_aleatoric = all("aleatoric_std" in df.columns for df in glacier_dfs)
    aleatoric_mat = (
        np.stack([df["aleatoric_std"].values for df in glacier_dfs])
        if has_aleatoric else None
    )
    if has_aleatoric:
        print("  Heteroscedastic runs detected — aleatoric uncertainty included.")

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
        gt_mean  = gt_df["mean"].values
        total_area_km2 = np.where(
            np.abs(mwe_mean) > 1e-10,
            gt_mean / (mwe_mean * 1e-3),
            0.0,
        )
        print("  Note: total_area_km2 derived from regional_annual_gt.csv (older run format)")

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
        "std_total":      reg_std_total                          * scale,
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ensemble_split(cfg: dict) -> None:
    """
    Split a heteroscedastic sweep into two ensemble runs and write separate outputs.

    Reads model.heteroscedastic from each run's .hydra/config.yaml, partitions
    runs into aleatoric (True) and deterministic (False) groups, then runs the
    full ensemble pipeline independently for each group.
    """
    multirun_root = Path(cfg["multirun_root"])
    output_dir    = Path(cfg["output_dir"])

    weighting   = cfg.get("weighting", "performance")
    temperature = float(cfg.get("softmax_temperature", 0.1))
    loyo_w      = float(cfg.get("loyo_weight", 0.0))
    glambie_w   = float(cfg.get("glambie_weight", 1.0))
    test_years  = list(cfg.get("glambie_test_years", [2021, 2022, 2023]))
    min_runs    = int(cfg.get("min_runs_per_region", 5))

    print(f"\n=== Ensemble split: {multirun_root.name} ===")

    # Build full results table and score all runs
    results_df = build_results_df(multirun_root, test_years, min_runs_per_region=1)
    results_df = add_composite_score(results_df, loyo_weight=loyo_w, glambie_weight=glambie_w)
    valid = results_df.dropna(subset=["composite"]).copy().reset_index(drop=True)
    print(f"  {len(valid)} runs with complete metrics across all groups.")

    # Read heteroscedastic flag for every run
    valid["heteroscedastic"] = [
        _read_heteroscedastic(Path(d)) for d in valid["run_dir"].values
    ]
    n_unknown = valid["heteroscedastic"].isna().sum()
    if n_unknown:
        warnings.warn(
            f"  {n_unknown} runs had unreadable Hydra configs — "
            "heteroscedastic flag treated as False for those runs."
        )
        valid["heteroscedastic"] = valid["heteroscedastic"].fillna(False)

    groups = [
        (True,  "aleatoric",          output_dir / "aleatoric"),
        (False, "epistemic_structural", output_dir / "epistemic_structural"),
    ]

    for hetero_val, label, group_output_dir in groups:
        group = valid[valid["heteroscedastic"] == hetero_val].copy().reset_index(drop=True)
        print(f"\n--- Group: {label} (heteroscedastic={hetero_val}) ---")
        if group.empty:
            print(f"  No runs found — skipping.")
            continue
        _run_group_ensemble(group, cfg, group_output_dir, weighting, temperature, min_runs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble uncertainty split by model.heteroscedastic flag."
    )
    parser.add_argument("--config", default="conf/config_ensemble_uncertainty.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--multirun_root", default=None,
                        help="Override multirun_root from config.")
    parser.add_argument("--output_dir", default=None,
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

    run_ensemble_split(cfg)


if __name__ == "__main__":
    main()
