"""
src/build_exorheic_regional.py

Run this on the HPC where /scratch is mounted (top_runs_info.csv's run_dir
column points there). It rebuilds the regional MWE/Gt ensemble exactly as
_combine_runs() in src/ensemble_common.py does, EXCEPT each run's glacier set
is first restricted to the exorheic-only subset before that run's own
regional MWE is computed -- so the across-glacier aggregation (within a run)
and the across-run aggregation (via the existing _ensemble_components) happen
in the same order as the live pipeline, just on a different glacier set.

Why this can't be done from local data alone (see outputs/analysis_sle/ for
the full writeup): ensemble_glacier.csv already combines across runs at the
glacier level. Restricting it afterwards and re-aggregating across glaciers
gives the exact right answer for the MEAN (linear combination commutes with
subsetting), but NOT for epistemic/structural std, because
    sqrt(sum_k w_k * (sum_g a_g/A * sigma_{k,g})^2)   [combine-glaciers-first]
is not the same as
    sum_g a_g/A * sqrt(sum_k w_k * sigma_{k,g}^2)      [combine-runs-first]
unless every glacier's cross-run deviation pattern is proportional. The first
form is only computable from each run's OWN glacier-level preds_full.csv,
which lives only on /scratch.

Correlation assumptions used here (as given by the project):
  - aleatoric:          independent across glaciers -> quadrature, within each run
  - epistemic (param):  fully correlated across glaciers -> linear (area-weighted mean),
                         within each run
  - structural:         independent across ensemble members (runs) -> this is exactly
                         _ensemble_components' existing std_structural formula
                         (spread of each run's own regional mean around the ensemble
                         mean), unchanged from the live pipeline.
  - across runs: epistemic and aleatoric both combine via quadrature across runs
    (_ensemble_components' existing formula) -- runs are independently trained.

Usage (from the repo root, on the HPC):

    python -m src.build_exorheic_regional \\
        --flags-csv outputs/analysis_sle/glacier_basin_flags_full.csv \\
        --output-dir outputs/analysis_sle/exorheic_regional \\
        --top-runs-root outputs/ensemble_pretrain_year

For every region with a combined/top_runs_info.csv, writes TWO regional Gt/MWE
series: one with all glaciers (the reproduction check -- must match the
existing outputs/ensemble_pretrain_year/{region}/combined/ensemble_regional_gt.csv
to validate this script before trusting the masked numbers) and one restricted
to exorheic glaciers only.

Outputs per region, in --output-dir:
    {region}_unmasked_regional_mwe.csv   (reproduction check; columns as ensemble_regional_*.csv)
    {region}_unmasked_regional_gt.csv
    {region}_exorheic_regional_mwe.csv   (the actual result)
    {region}_exorheic_regional_gt.csv
Plus one console summary comparing the unmasked reproduction against the
live pipeline's own ensemble_regional_gt.csv for every region, printed and
saved to --output-dir/reproduction_check.csv.
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.ensemble_uncertainty import _ensemble_components, _load_glacier_preds

YEAR_MIN, YEAR_MAX = 1940, 2025
YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)


def _area_weighted_mean(df: pd.DataFrame, value_col: str, area_col: str = "Area") -> pd.Series:
    """Area-weighted mean of value_col over glaciers, grouped by year. Linear combination."""
    total_area = df.groupby("year")[area_col].sum()
    num = (df[value_col] * df[area_col]).groupby(df["year"]).sum()
    return (num / total_area).reindex(YEARS)


def _area_weighted_quadrature(df: pd.DataFrame, value_col: str, area_col: str = "Area") -> pd.Series:
    """Weighted quadrature combination over glaciers, grouped by year. Independence assumption.

    sigma_region = sqrt( sum_g (a_g/A)^2 * sigma_g^2 )
    """
    total_area = df.groupby("year")[area_col].transform("sum")
    w = df[area_col] / total_area
    term = (w * df[value_col]) ** 2
    out = term.groupby(df["year"]).sum().pow(0.5)
    return out.reindex(YEARS)


def _total_area_by_year(df: pd.DataFrame, area_col: str = "Area") -> pd.Series:
    return df.groupby("year")[area_col].sum().reindex(YEARS)


def _one_run_regional_series(preds_df: pd.DataFrame, area_df: pd.DataFrame, glacier_mask: pd.Series | None) -> dict:
    """
    Restrict one run's preds_full.csv to the glacier subset (if glacier_mask given,
    a boolean Series indexed by rgi_id that is True for glaciers to KEEP), then
    compute that run's own regional mean / epistemic / aleatoric series.

    preds_df columns expected: rgi_id, year, mean, epistemic_std (or std), aleatoric_std
    area_df columns expected:  rgi_id, Area  (static per glacier)
    """
    epistemic_col = "epistemic_std" if "epistemic_std" in preds_df.columns else "std"
    if "aleatoric_std" not in preds_df.columns:
        raise ValueError(
            "preds_full.csv has no 'aleatoric_std' column -- this run was not "
            "trained with model.heteroscedastic=true. All runs in a group must match."
        )

    df = preds_df.merge(area_df, on="rgi_id", how="inner")
    if glacier_mask is not None:
        keep_ids = glacier_mask[glacier_mask].index
        df = df[df["rgi_id"].isin(keep_ids)]

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]

    mean_region = _area_weighted_mean(df, "mean")
    epi_region = _area_weighted_mean(df, epistemic_col)          # linear: fully correlated within a run
    alea_region = _area_weighted_quadrature(df, "aleatoric_std")  # quadrature: independent within a run
    area_region = _total_area_by_year(df)

    return {
        "mean": mean_region.values,
        "epistemic_std": epi_region.values,
        "aleatoric_std": alea_region.values,
        "total_area_km2": area_region.values,
    }


def build_region(region: str, top_runs_csv: Path, area_df: pd.DataFrame,
                  glacier_mask: pd.Series | None) -> pd.DataFrame | None:
    """
    Rebuild one region's regional MWE ensemble (optionally masked to a glacier
    subset), using exactly the same _ensemble_components() call the live
    pipeline uses, with equal weights across the top-N runs (matching
    _combine_runs()'s weights = np.ones(K)/K).
    """
    top = pd.read_csv(top_runs_csv)
    run_dirs = [Path(d) for d in top["run_dir"].values]
    K = len(run_dirs)
    if K == 0:
        print(f"  {region}: no runs listed in {top_runs_csv} -- skipping.")
        return None

    means_mat = np.full((K, len(YEARS)), np.nan)
    epi_mat = np.full((K, len(YEARS)), np.nan)
    alea_mat = np.full((K, len(YEARS)), np.nan)
    area_mat = np.full((K, len(YEARS)), np.nan)
    n_ok = 0

    for i, run_dir in enumerate(run_dirs):
        preds = _load_glacier_preds(run_dir)
        if preds is None:
            continue
        try:
            series = _one_run_regional_series(preds, area_df, glacier_mask)
        except ValueError as exc:
            warnings.warn(f"  {region} run {run_dir}: {exc} -- excluded.")
            continue
        means_mat[i] = series["mean"]
        epi_mat[i] = series["epistemic_std"]
        alea_mat[i] = series["aleatoric_std"]
        area_mat[i] = series["total_area_km2"]
        n_ok += 1

    if n_ok == 0:
        print(f"  {region}: no runs had readable preds_full.csv -- skipping.")
        return None
    if n_ok < K:
        print(f"  {region}: WARNING only {n_ok}/{K} runs contributed (missing preds_full.csv or homoscedastic).")

    ok = ~np.isnan(means_mat).all(axis=1)
    means_mat, epi_mat, alea_mat, area_mat = means_mat[ok], epi_mat[ok], alea_mat[ok], area_mat[ok]
    weights = np.ones(ok.sum()) / ok.sum()

    comps = _ensemble_components(means_mat, epi_mat, weights, alea_mat)

    # total_area_km2: same glacier subset every run (static Area), so just take
    # the (non-nan) mean across runs for reporting -- should be identical across runs.
    total_area_km2 = np.nanmean(area_mat, axis=0)
    scale = total_area_km2 * 1e-3

    mwe_df = pd.DataFrame({
        "year": YEARS,
        "median_mwe": comps["median_mwe"],
        "std_structural": comps["std_structural"],
        "std_epistemic": comps["std_epistemic"],
        "std_aleatoric": comps["std_aleatoric"],
        "std_total": comps["std_total"],
        "total_area_km2": total_area_km2,
        "n_runs_used": ok.sum(),
    })
    # same column names as the live ensemble_regional_gt.csv, so existing draw code can read it
    gt_df = pd.DataFrame({
        "year": YEARS,
        "median_gt": mwe_df["median_mwe"] * scale,
        "std_structural": mwe_df["std_structural"] * scale,
        "std_epistemic": mwe_df["std_epistemic"] * scale,
        "std_aleatoric": mwe_df["std_aleatoric"] * scale,
        "std_total": mwe_df["std_total"] * scale,
        "total_area_km2": total_area_km2,
        "n_runs_used": ok.sum(),
    })

    return mwe_df, gt_df


def load_area(region: str, inp_dir: str) -> pd.DataFrame:
    path = os.path.join(inp_dir, region, f"main_features_{region}.csv")
    df = pd.read_csv(path, usecols=["rgi_id", "Area"]).drop_duplicates("rgi_id")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flags-csv", required=True, help="glacier_basin_flags_full.csv from outputs/analysis_sle/")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--top-runs-root", default="outputs/ensemble_pretrain_year")
    ap.add_argument("--group", default="combined", help="pretrain-year group subfolder name")
    ap.add_argument("--inp-dir", default="data_for_model", help="folder holding r{nn}/main_features_r{nn}.csv")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flags = pd.read_csv(args.flags_csv)[["rgi_id", "region", "endorheic_final"]]

    regions = sorted(
        d.parent.parent.name for d in Path(args.top_runs_root).glob(f"*/{args.group}/top_runs_info.csv")
    )
    print(f"Found {len(regions)} regions with {args.group}/top_runs_info.csv")

    repro_rows = []
    for region in regions:
        print(f"\n=== {region} ===")
        top_runs_csv = Path(args.top_runs_root) / region / args.group / "top_runs_info.csv"
        area_df = load_area(region, args.inp_dir)
        region_flags = flags[flags.region == region].set_index("rgi_id")["endorheic_final"]
        exorheic_mask = ~region_flags  # True = keep (exorheic)

        print("  Building UNMASKED (reproduction check)...")
        res_unmasked = build_region(region, top_runs_csv, area_df, glacier_mask=None)
        print("  Building EXORHEIC-only...")
        res_exo = build_region(region, top_runs_csv, area_df, glacier_mask=exorheic_mask)

        if res_unmasked is not None:
            mwe_u, gt_u = res_unmasked
            mwe_u.to_csv(out_dir / f"{region}_unmasked_regional_mwe.csv", index=False)
            gt_u.to_csv(out_dir / f"{region}_unmasked_regional_gt.csv", index=False)

            live_path = Path(args.top_runs_root) / region / args.group / "ensemble_regional_gt.csv"
            if live_path.exists():
                live = pd.read_csv(live_path)
                mg = gt_u.merge(live, on="year", suffixes=("_rebuilt", "_live"))
                cum_r, cum_l = mg["median_gt_rebuilt"].sum(), mg["median_gt_live"].sum()
                row = {"region": region, "rebuilt_cum_gt_2025": cum_r, "live_cum_gt_2025": cum_l,
                       "pct_diff_cum_gt": 100 * abs(cum_r - cum_l) / abs(cum_l)}
                # mean annual std per component, rebuilt / live (1.00 = reproduced)
                for c in ["std_structural", "std_epistemic", "std_aleatoric", "std_total"]:
                    row[f"ratio_{c}"] = np.nanmean(mg[f"{c}_rebuilt"]) / np.nanmean(mg[f"{c}_live"])
                repro_rows.append(row)
                print("  Reproduction: cum Gt diff {:.3f}% | std ratios struct {:.2f} epist {:.2f} "
                      "alea {:.2f} total {:.2f}".format(row["pct_diff_cum_gt"], row["ratio_std_structural"],
                      row["ratio_std_epistemic"], row["ratio_std_aleatoric"], row["ratio_std_total"]))

        if res_exo is not None:
            mwe_e, gt_e = res_exo
            mwe_e.to_csv(out_dir / f"{region}_exorheic_regional_mwe.csv", index=False)
            gt_e.to_csv(out_dir / f"{region}_exorheic_regional_gt.csv", index=False)
            print(f"  Saved exorheic-only series ({int(exorheic_mask.sum())} exorheic glaciers kept "
                  f"of {len(exorheic_mask)} total).")

    if repro_rows:
        repro_df = pd.DataFrame(repro_rows)
        repro_df.to_csv(out_dir / "reproduction_check.csv", index=False)
        print("\n=== Reproduction check summary (saved to reproduction_check.csv) ===")
        print(repro_df.to_string(index=False))
        print(
            "\nRead: pct_diff_cum_gt should be < ~1% for every region (mean reproduced); ratio_std_* close to 1.00 "
            "means that uncertainty component is reproduced. Trust the exorheic files only where they are.")


if __name__ == "__main__":
    main()
