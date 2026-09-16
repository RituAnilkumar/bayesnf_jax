"""Feature importance analysis across all 19 regions (no_time_encoding group)."""
import pandas as pd
import numpy as np
import os
import sys

EXPLAIN_ROOT = "outputs/egu_fin/explain_te"
GROUP = "no_time_encoding"
FILE = "attributions_finetune_ensemble1_no_time_encoding.csv"

# Masked features (exclude CenLat, CenLon)
FEATURES = [
    "year",
    "log_Area", "Zmed", "Slope", "Aspect", "Zmin", "Zmax",
    "t2m_abl_mean", "t2m_abl_std", "t2m_acc_mean", "t2m_acc_std",
    "tp_abl_sum", "tp_acc_sum",
    "ssrd_abl_sum", "ssrd_acc_sum",
]

CLIMATE_FEATS = {"t2m_abl_mean", "t2m_abl_std", "t2m_acc_mean", "t2m_acc_std",
                 "tp_abl_sum", "tp_acc_sum", "ssrd_abl_sum", "ssrd_acc_sum"}
GEOM_FEATS   = {"log_Area", "Zmed", "Slope", "Aspect", "Zmin", "Zmax"}
YEAR_FEATS   = {"year"}

REGION_NAMES = {
    "r01": "Alaska",
    "r02": "Western Canada & US",
    "r03": "Arctic Canada North",
    "r04": "Arctic Canada South",
    "r05": "Greenland Periphery",
    "r06": "Iceland",
    "r07": "Svalbard",
    "r08": "Scandinavia",
    "r09": "Russian Arctic",
    "r10": "North Asia",
    "r11": "Central Europe",
    "r12": "Caucasus & Middle East",
    "r13": "Central Asia",
    "r14": "South Asia West",
    "r15": "South Asia East",
    "r16": "Low Latitudes",
    "r17": "Southern Andes",
    "r18": "New Zealand",
    "r19": "Antarctic & Subantarctic",
}

results = {}

for rkey in [f"r{i:02d}" for i in range(1, 20)]:
    path = os.path.join(EXPLAIN_ROOT, rkey, GROUP, FILE)
    if not os.path.exists(path):
        print(f"[MISSING] {rkey}")
        continue

    df = pd.read_csv(path)

    # Compute mean |attr| per feature
    mean_abs = {}
    for ft in FEATURES:
        col = f"attr_{ft}"
        if col not in df.columns:
            mean_abs[ft] = np.nan
        else:
            mean_abs[ft] = df[col].abs().mean()

    total = sum(v for v in mean_abs.values() if not np.isnan(v))
    if total == 0:
        total = 1.0

    # Ranked
    ranked = sorted(mean_abs.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)

    # Group percentages
    clim_pct = 100 * sum(mean_abs[f] for f in CLIMATE_FEATS if f in mean_abs and not np.isnan(mean_abs[f])) / total
    geom_pct = 100 * sum(mean_abs[f] for f in GEOM_FEATS  if f in mean_abs and not np.isnan(mean_abs[f])) / total
    year_pct = 100 * sum(mean_abs[f] for f in YEAR_FEATS  if f in mean_abs and not np.isnan(mean_abs[f])) / total

    # Year attribution trend (sign of slope over time)
    if "attr_year" in df.columns and "year" in df.columns:
        yr_mean = df.groupby("year")["attr_year"].mean().reset_index()
        yr_mean = yr_mean.sort_values("year")
        if len(yr_mean) >= 5:
            slope = np.polyfit(yr_mean["year"].values, yr_mean["attr_year"].values, 1)[0]
            mean_year_attr = yr_mean["attr_year"].mean()
        else:
            slope = np.nan
            mean_year_attr = np.nan
    else:
        slope = np.nan
        mean_year_attr = np.nan

    results[rkey] = {
        "name": REGION_NAMES.get(rkey, rkey),
        "mean_abs": mean_abs,
        "ranked": ranked,
        "clim_pct": clim_pct,
        "geom_pct": geom_pct,
        "year_pct": year_pct,
        "year_slope": slope,
        "year_mean_attr": mean_year_attr,
        "total": total,
    }

# Print analysis
print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS — NO TIME ENCODING, ALL REGIONS")
print("=" * 80)

for rkey in [f"r{i:02d}" for i in range(1, 20)]:
    if rkey not in results:
        print(f"\n{rkey}: [NO DATA]")
        continue
    r = results[rkey]
    print(f"\n{'─' * 70}")
    print(f"  {rkey.upper()}  {r['name']}")
    print(f"{'─' * 70}")
    print(f"  Group split:  Climate {r['clim_pct']:.0f}%  |  Geometry {r['geom_pct']:.0f}%  |  Year {r['year_pct']:.0f}%")

    year_dir = "positive (warming trend)" if r['year_mean_attr'] > 0 else "negative (cooling/declining trend)"
    slope_dir = "steepening" if r['year_slope'] > 0 else "flattening/reversing"
    print(f"  Year attr:    mean={r['year_mean_attr']:+.4f}, slope={r['year_slope']:+.5f}/yr  →  {slope_dir}")

    print(f"  Top 8 features (mean |attribution|, % of total):")
    for i, (ft, val) in enumerate(r["ranked"][:8], 1):
        pct = 100 * val / r["total"]
        # Direction for top features
        col = f"attr_{ft}"
        # not easily available here without df; skip direction
        print(f"    {i:2d}. {ft:<22s}  {val:.4f}  ({pct:.1f}%)")

print("\n\n" + "=" * 80)
print("SLIDE BULLET POINTS PER REGION")
print("=" * 80)

for rkey in [f"r{i:02d}" for i in range(1, 20)]:
    if rkey not in results:
        continue
    r = results[rkey]
    ranked = r["ranked"]
    top3 = [ft for ft, _ in ranked[:3]]
    top3_str = ", ".join(top3)

    clim = r["clim_pct"]
    geom = r["geom_pct"]
    yr   = r["year_pct"]
    ys   = r["year_slope"]
    ym   = r["year_mean_attr"]

    dominant = "climate" if clim > geom and clim > yr else ("geometry" if geom > clim and geom > yr else "temporal trend (year)")

    if abs(ym) < 0.005:
        year_note = "Year feature has near-zero net attribution — time trend weak"
    elif ym > 0:
        year_note = f"Year attribution is positive (mean {ym:+.3f}) — model captures accelerating mass loss trend"
        if ys > 0:
            year_note += ", slope steepening"
    else:
        year_note = f"Year attribution is negative (mean {ym:+.3f}) — temporal trend pulls toward less negative MB"
        if ys < 0:
            year_note += ", declining"

    print(f"\n{'─'*60}")
    print(f"  {rkey.upper()} — {r['name']}")
    print(f"{'─'*60}")
    print(f"  • Dominant driver group: {dominant} ({max(clim,geom,yr):.0f}% of attribution)")
    print(f"  • Top 3 features: {top3_str}")
    print(f"  • Climate / Geometry / Year split: {clim:.0f}% / {geom:.0f}% / {yr:.0f}%")
    print(f"  • {year_note}")
    if ranked[0][0] in CLIMATE_FEATS:
        ft0 = ranked[0][0]
        if "t2m" in ft0:
            print(f"  • Temperature ({ft0}) is the single largest driver")
        elif "tp" in ft0:
            print(f"  • Precipitation ({ft0}) is the single largest driver — accumulation-dominated regime")
        elif "ssrd" in ft0:
            print(f"  • Solar radiation ({ft0}) is the single largest driver")
    elif ranked[0][0] in GEOM_FEATS:
        print(f"  • Glacier geometry ({ranked[0][0]}) is the single largest driver — topographic control dominant")
