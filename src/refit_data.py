"""
src/refit_data.py

Loaders for the extra observations used only in the final refit
(model.final_refit=true, see src/model/finetune.py and src/refit_final.py).
These sources were previously held out for testing/validation; the final
refit folds the reliable parts of them into Stage 2 training.

Reliability filters (per instruction):
  - Dussaillant regional product: only years < duss_year_max (default 2000,
    i.e. before the Hugonnet window, where Dussaillant is tied to the same
    geodetic signal already used as a finetuning target) AND only
    region-years with at least duss_min_insitu glaciers measured in situ
    (WGMS FoG annual_balance, counted per GTN-G region exactly as in
    test_regional_duss.py). Region-years without in-situ support are
    extrapolated from neighbouring regions in the Dussaillant product and are
    not used. Post-2020 Dussaillant is not used either: GLaMBIE covers those
    years and its combined estimate already includes glaciological input.
  - WGMS glaciological series: only glaciers on the WGMS *reference* list
    (source_list=="reference", status=="kept" in reference_benchmark_glaciers.csv)
    — not benchmark or former-reference glaciers. Both validation- and
    testing-split reference glaciers are used. Optionally restricted to years
    outside the 2000-2020 Hugonnet window (wgms_exclude_window).

All values are MWE/yr; Dussaillant mwe/mwe_sigma and WGMS annual_balance /
annual_balance_unc are already in m w.e. (see build_reference_timeseries.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.test_regional_duss import load_glacier_counts, load_measured_by_region

HUGONNET_YEAR_MIN, HUGONNET_YEAR_MAX = 2000, 2020


def load_dussaillant_insitu(
    measured_dir: str | Path,
    per_gla_dir: str | Path,
    region_num: int,
    year_max: int = HUGONNET_YEAR_MIN,
    min_insitu: int = 1,
) -> pd.DataFrame | None:
    """
    Dussaillant regional annual MB for one RGI region, restricted to years
    < year_max with >= min_insitu in-situ glaciers measured in that region-year.

    Returns columns: year, value_mwe, error_mwe, n_insitu  (or None if empty).
    """
    measured = load_measured_by_region(Path(measured_dir)).get(region_num)
    if measured is None:
        return None
    counts = load_glacier_counts(Path(per_gla_dir))
    counts = counts[counts["region_key"] == f"{region_num:02d}"][["year", "n_glaciers"]]

    df = measured[measured["year"] < year_max].merge(counts, on="year", how="left")
    df["n_insitu"] = df["n_glaciers"].fillna(0).astype(int)
    df = df[(df["n_insitu"] >= min_insitu) & df["mwe"].notna()]
    if df.empty:
        return None
    return pd.DataFrame({
        "year":      df["year"].astype(int).values,
        "value_mwe": df["mwe"].astype(np.float32).values,
        "error_mwe": df["mwe_sigma"].astype(np.float32).values,
        "n_insitu":  df["n_insitu"].values,
    })


def load_wgms_reference(
    per_gla_dir: str | Path,
    region_num: int,
    default_unc: float = 0.3,
    exclude_window: bool = False,
) -> pd.DataFrame | None:
    """
    Glaciological annual MB for WGMS reference glaciers in one RGI region.

    annual_balance_unc is reported for only ~1/4 of rows; missing values are
    filled with default_unc (m w.e./yr).

    Returns columns: rgi_id, year, value_mwe, error_mwe, glacier_name  (or None).
    """
    per_gla_dir = Path(per_gla_dir)
    meta = pd.read_csv(per_gla_dir / "reference_benchmark_glaciers.csv")
    meta = meta[(meta["status"] == "kept") & (meta["source_list"] == "reference")
                & (meta["rgi_region"] == region_num)]
    if meta.empty:
        return None

    mb = pd.read_csv(per_gla_dir / "mass_balance.csv", low_memory=False,
                     usecols=["glacier_id", "year", "annual_balance", "annual_balance_unc"])
    mb = mb[mb["glacier_id"].isin(meta["glacier_id"])].dropna(subset=["annual_balance"])
    if exclude_window:
        mb = mb[(mb["year"] < HUGONNET_YEAR_MIN) | (mb["year"] > HUGONNET_YEAR_MAX)]
    if mb.empty:
        return None

    mb = mb.merge(meta[["glacier_id", "rgi_id", "matched_wgms_name"]], on="glacier_id")
    unc = mb["annual_balance_unc"]
    unc = unc.where(np.isfinite(unc) & (unc > 0), default_unc)
    return pd.DataFrame({
        "rgi_id":       mb["rgi_id"].values,
        "year":         mb["year"].astype(int).values,
        "value_mwe":    mb["annual_balance"].astype(np.float32).values,
        "error_mwe":    unc.astype(np.float32).values,
        "glacier_name": mb["matched_wgms_name"].values,
    })
