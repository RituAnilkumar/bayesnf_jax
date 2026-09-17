"""
src/build_reference_timeseries.py

Second step of the revised (per-glacier WGMS) ensemble-selection validation
set. Builds on reference_benchmark_glaciers.csv (produced by
build_reference_glaciers.py) to:

  1. Pull each kept glacier's full annual_balance series from mass_balance.csv,
     convert mm w.e. -> MWE/yr.
  2. Decide, per RGI region, which "tier" of years to use:
       tier 1: only years <2000 or >2020 (avoids pretrain/finetune window)
       tier 2: fallback to the glacier's full record (incl. 2000-2020),
               used only if tier 1 has zero eligible glaciers in that region
       tier 3: no WGMS observations usable at all for that region (zero
               eligible glaciers even under tier 2, or zero kept glaciers to
               begin with) -> region gets no validation/test rows; ensemble
               selection for it must fall back to LOYO/LOGO R^2 only
               (handled downstream, not in this file).
  3. Split eligible glaciers per region into validation/testing, at the
     whole-glacier level (never split mid-series): 50/50, validation gets
     the extra glacier if odd, and ALL glaciers go to validation if fewer
     than 4 are eligible in that region. Assignment is random with a fixed
     seed for reproducibility.

Outputs (validation_data/per_gla/):
  reference_benchmark_mb_timeseries.csv — long format, one row per
      (glacier, year) observation, with split/tier/in_training_window columns
  reference_benchmark_region_summary.csv — one row per RGI region: tier used,
      n glaciers/entries in validation vs testing
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

GLACIERS_CSV     = Path("validation_data/per_gla/reference_benchmark_glaciers.csv")
MASS_BALANCE_CSV = Path("validation_data/per_gla/mass_balance.csv")
TIMESERIES_OUT   = Path("validation_data/per_gla/reference_benchmark_mb_timeseries.csv")
SUMMARY_OUT      = Path("validation_data/per_gla/reference_benchmark_region_summary.csv")

MM_TO_MWE   = 1e-3
TRAIN_YEAR_MIN = 2000
TRAIN_YEAR_MAX = 2020
SPLIT_SEED = 42


def _tier1_mask(year: pd.Series) -> pd.Series:
    return (year < TRAIN_YEAR_MIN) | (year > TRAIN_YEAR_MAX)


def load_obs(kept: pd.DataFrame, mb: pd.DataFrame) -> pd.DataFrame:
    """Return long-format obs for all kept glaciers: one row per
    (glacier_id, year) with a valid annual_balance."""
    ids = kept["glacier_id"].astype(int).tolist()
    sub = mb[mb["glacier_id"].isin(ids)].copy()
    sub = sub.dropna(subset=["annual_balance"])
    sub = sub[["glacier_id", "year", "annual_balance"]].rename(columns={"year": "YEAR"})
    sub["obs_mwe"] = sub["annual_balance"] * MM_TO_MWE
    sub["glacier_id"] = sub["glacier_id"].astype(int)
    sub["is_tier1"] = _tier1_mask(sub["YEAR"])
    sub["in_training_window"] = ~sub["is_tier1"]
    return sub


def assign_split(glacier_ids: list[int], rng: random.Random) -> dict[int, str]:
    """Whole-glacier 50/50 validation/testing split.

    Validation gets the extra glacier when the count is odd. If fewer than
    4 glaciers are eligible, all go to validation (nothing held out).
    """
    ids = list(glacier_ids)
    rng.shuffle(ids)
    n = len(ids)
    if n < 4:
        return {gid: "validation" for gid in ids}
    n_val = (n + 1) // 2  # ceil -> validation gets the extra on odd n
    return {gid: ("validation" if i < n_val else "testing") for i, gid in enumerate(ids)}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    glaciers = pd.read_csv(GLACIERS_CSV)
    kept = glaciers[glaciers["status"] == "kept"].copy()
    kept["glacier_id"] = kept["glacier_id"].astype(int)
    kept["rgi_region"] = kept["rgi_region"].astype(int)

    mb = pd.read_csv(MASS_BALANCE_CSV, low_memory=False)
    obs = load_obs(kept, mb)

    rng = random.Random(SPLIT_SEED)

    timeseries_rows = []
    summary_rows = []

    all_regions = sorted(kept["rgi_region"].unique())
    # Also report regions with zero kept glaciers at all (e.g. r04, r09),
    # discoverable only from a fixed RGI region list, not from `kept`.
    full_region_range = range(1, 20)

    for region in full_region_range:
        region_glaciers = kept[kept["rgi_region"] == region]
        if region_glaciers.empty:
            summary_rows.append({
                "rgi_region": region, "tier_used": 3, "n_glaciers_total": 0,
                "n_glaciers_validation": 0, "n_glaciers_testing": 0,
                "n_entries_validation": 0, "n_entries_testing": 0,
                "note": "no kept reference/benchmark glacier in this region "
                        "-> selection must fall back to LOYO/LOGO R2 only",
            })
            continue

        region_obs = obs[obs["glacier_id"].isin(region_glaciers["glacier_id"])]

        tier1_glacier_ids = sorted(region_obs.loc[region_obs["is_tier1"], "glacier_id"].unique())
        tier2_glacier_ids = sorted(region_obs["glacier_id"].unique())  # any valid obs, any year

        if tier1_glacier_ids:
            tier = 1
            eligible_ids = tier1_glacier_ids
            use_obs = region_obs[region_obs["is_tier1"]]
        elif tier2_glacier_ids:
            tier = 2
            eligible_ids = tier2_glacier_ids
            use_obs = region_obs  # full record, including 2000-2020
        else:
            tier = 3
            eligible_ids = []
            use_obs = region_obs.iloc[0:0]

        if tier == 3:
            summary_rows.append({
                "rgi_region": region, "tier_used": 3,
                "n_glaciers_total": len(region_glaciers),
                "n_glaciers_validation": 0, "n_glaciers_testing": 0,
                "n_entries_validation": 0, "n_entries_testing": 0,
                "note": "kept glacier(s) present but zero usable annual_balance "
                        "observations -> selection must fall back to LOYO/LOGO R2 only",
            })
            continue

        split_map = assign_split(eligible_ids, rng)

        merged = use_obs.merge(
            region_glaciers[["glacier_id", "input_name", "matched_wgms_name", "country",
                              "source_list", "rgi_id", "rgi_region"]],
            on="glacier_id", how="left",
        )
        merged["split"] = merged["glacier_id"].map(split_map)
        merged["tier_used"] = tier
        timeseries_rows.append(merged)

        n_val_gl  = sum(1 for v in split_map.values() if v == "validation")
        n_test_gl = sum(1 for v in split_map.values() if v == "testing")
        n_val_entries  = int((merged["split"] == "validation").sum())
        n_test_entries = int((merged["split"] == "testing").sum())

        summary_rows.append({
            "rgi_region": region, "tier_used": tier,
            "n_glaciers_total": len(region_glaciers),
            "n_glaciers_validation": n_val_gl, "n_glaciers_testing": n_test_gl,
            "n_entries_validation": n_val_entries, "n_entries_testing": n_test_entries,
            "note": "" if tier == 1 else
                    "tier-2 fallback: includes 2000-2020 (training-window overlap) "
                    "because no glacier had usable data outside that window",
        })

    timeseries_df = (pd.concat(timeseries_rows, ignore_index=True)
                      if timeseries_rows else pd.DataFrame())
    summary_df = pd.DataFrame(summary_rows)
    return timeseries_df, summary_df


def main() -> None:
    timeseries_df, summary_df = build()

    timeseries_df = timeseries_df.rename(columns={
        "input_name": "NAME", "matched_wgms_name": "WGMS_NAME",
        "country": "POLITICAL_UNIT", "rgi_id": "RGI_ID",
    })
    col_order = ["rgi_region", "RGI_ID", "glacier_id", "NAME", "WGMS_NAME",
                 "POLITICAL_UNIT", "source_list", "YEAR", "obs_mwe",
                 "in_training_window", "tier_used", "split"]
    timeseries_df = timeseries_df[[c for c in col_order if c in timeseries_df.columns]]
    timeseries_df.to_csv(TIMESERIES_OUT, index=False, float_format="%.6f")
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print(f"Saved {TIMESERIES_OUT} ({len(timeseries_df)} rows)")
    print(f"Saved {SUMMARY_OUT} ({len(summary_df)} rows)")

    print("\n=== Entries per region (validation / testing) ===")
    print(f"{'region':>6}  {'tier':>4}  {'gl_val':>6}  {'gl_test':>7}  "
          f"{'entries_val':>11}  {'entries_test':>12}  note")
    total_val_gl = total_test_gl = total_val_e = total_test_e = 0
    for _, r in summary_df.iterrows():
        print(f"  r{int(r['rgi_region']):02d}  {int(r['tier_used']):>4}  "
              f"{int(r['n_glaciers_validation']):>6}  {int(r['n_glaciers_testing']):>7}  "
              f"{int(r['n_entries_validation']):>11}  {int(r['n_entries_testing']):>12}  "
              f"{r['note']}")
        total_val_gl  += r["n_glaciers_validation"]
        total_test_gl += r["n_glaciers_testing"]
        total_val_e   += r["n_entries_validation"]
        total_test_e  += r["n_entries_testing"]
    print(f"\n  TOTAL glaciers   : validation={total_val_gl}  testing={total_test_gl}")
    print(f"  TOTAL obs entries: validation={total_val_e}  testing={total_test_e}")


if __name__ == "__main__":
    main()
