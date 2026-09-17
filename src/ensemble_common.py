"""
src/ensemble_common.py

Shared helpers used by the top-N ensemble builders (ensemble_uncertainty_pretrain_year.py
and, historically, ensemble_uncertainty_time_encoding.py).

The correlation-aware cumulative-Gt uncertainty helpers (compute_cumulative_gt_variants,
plot_cumulative_sensitivity) live in src/cumulative_uncertainty.py — a dependency-free
module shared by this file and by ensemble_uncertainty.py directly, avoiding a circular
import (this module already imports from ensemble_uncertainty.py for the rest of the
ensemble-building helpers). See that module's docstring for the full reasoning trail
on the four cumulative-uncertainty scenarios.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from src.cumulative_uncertainty import (
    compute_cumulative_gt_variants,
    plot_cumulative_sensitivity,
)


# ---------------------------------------------------------------------------
# Small utility
# ---------------------------------------------------------------------------

def _minmax_norm_series(s: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]. Returns zeros if all values are equal."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Top-N ensemble runner for a single group
# ---------------------------------------------------------------------------

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
        return False

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
        return False

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
    # Cumulative Gt — four correlation-assumption scenarios
    # ----------------------------------------------------------------
    print("  --- Cumulative Gt (sensitivity to correlation assumptions) ---")
    alea_for_cum = np.where(np.isnan(ensemble_regional_gt["std_aleatoric"].values), 0.0,
                             ensemble_regional_gt["std_aleatoric"].values)
    cum_df = compute_cumulative_gt_variants(
        run_dirs=regional_run_dirs,
        weights=regional_weights_arr,
        years=years,
        median_gt=ensemble_regional_gt["median_gt"].values,
        std_structural=ensemble_regional_gt["std_structural"].values,
        std_epistemic=ensemble_regional_gt["std_epistemic"].values,
        std_aleatoric=alea_for_cum,
        file_name="regional_annual_gt.csv",
        value_col="mean",
    )
    cum_df.to_csv(output_dir / "ensemble_cumulative_gt.csv", index=False)
    print(f"  Saved ensemble_cumulative_gt.csv")
    plot_cumulative_sensitivity(cum_df, output_dir / "ensemble_cumulative_gt_sensitivity.png")
    print(f"  Saved ensemble_cumulative_gt_sensitivity.png")

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
