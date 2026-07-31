"""
src/plot_feature_pie.py

Generate feature importance pie charts from attribution CSVs.
Produces two pie charts per region:
  1. pie_importance_full.png        — all features (incl. CenLat, CenLon, year)
  2. pie_importance_masked.png      — masked (excludes CenLat, CenLon, year)

Features contributing < 2% of total are grouped into "Others".

Usage:
    python src/plot_feature_pie.py \
        --explain_root outputs/egu_fin/explain_te \
        --group        no_time_encoding
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

ALL_FEATURES = [
    "year",
    "CenLat", "CenLon",
    "log_Area", "Zmed", "Slope", "Aspect", "Zmin", "Zmax",
    "t2m_abl_mean", "t2m_abl_std", "t2m_acc_mean", "t2m_acc_std",
    "tp_abl_sum", "tp_acc_sum",
    "ssrd_abl_sum", "ssrd_acc_sum",
]

MASKED_FEATURES = [
    "log_Area", "Zmed", "Slope", "Aspect", "Zmin", "Zmax",
    "t2m_abl_mean", "t2m_abl_std", "t2m_acc_mean", "t2m_acc_std",
    "tp_abl_sum", "tp_acc_sum",
    "ssrd_abl_sum", "ssrd_acc_sum",
]

# Display labels (shorter for pie chart readability)
LABELS = {
    "year":         "Year",
    "CenLat":       "Latitude",
    "CenLon":       "Longitude",
    "log_Area":     "Area",
    "Zmed":         "Zmed",
    "Slope":        "Slope",
    "Aspect":       "Aspect",
    "Zmin":         "Zmin",
    "Zmax":         "Zmax",
    "t2m_abl_mean": "T abl (mean)",
    "t2m_abl_std":  "T abl (var)",
    "t2m_acc_mean": "T acc (mean)",
    "t2m_acc_std":  "T acc (var)",
    "tp_abl_sum":   "P abl",
    "tp_acc_sum":   "P acc",
    "ssrd_abl_sum": "Rad abl",
    "ssrd_acc_sum": "Rad acc",
}

# Color palette by feature group
_TEMP_COLORS  = ["#fc8d62", "#f1a4b5", "#e78ac3", "#fdbfcf"]   # temperature: orange-salmon → pink → orchid → light pink
_PREC_COLORS  = ["#8da0cb", "#a6c8e8"]                          # precipitation: periwinkle → powder blue
_RAD_COLORS   = ["#ffd92f", "#ffb347"]                          # radiation: yellow → light orange (distinct)
_GEOM_COLORS  = ["#66c2a5", "#a6d854", "#b3e2cd", "#abdda4",   # geometry: teal → yellow-green → mint → sage
                 "#80cdc1", "#d8f0a8"]                           #           turquoise → pale lime
_YEAR_COLOR   = "#b3b3e0"                                        # soft lavender
_COORD_COLORS = ["#c2a5cf", "#b2abd2"]                          # mauve → light purple (not grey)
_OTHER_COLOR  = "#e0e0e0"

FEATURE_COLORS = {
    "t2m_abl_mean": _TEMP_COLORS[0],
    "t2m_abl_std":  _TEMP_COLORS[1],
    "t2m_acc_mean": _TEMP_COLORS[2],
    "t2m_acc_std":  _TEMP_COLORS[3],
    "tp_abl_sum":   _PREC_COLORS[0],
    "tp_acc_sum":   _PREC_COLORS[1],
    "ssrd_abl_sum": _RAD_COLORS[0],
    "ssrd_acc_sum": _RAD_COLORS[1],
    "log_Area":     _GEOM_COLORS[0],
    "Zmed":         _GEOM_COLORS[1],
    "Slope":        _GEOM_COLORS[2],
    "Aspect":       _GEOM_COLORS[3],
    "Zmin":         _GEOM_COLORS[4],
    "Zmax":         _GEOM_COLORS[5],
    "year":         _YEAR_COLOR,
    "CenLat":       _COORD_COLORS[0],
    "CenLon":       _COORD_COLORS[1],
}

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

THRESHOLD = 0.05   # features below this % of total → "Others"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_mean_abs(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    out = {}
    for ft in features:
        col = f"attr_{ft}"
        out[ft] = float(df[col].abs().mean()) if col in df.columns else 0.0
    return out


def _make_pie_data(
    mean_abs: dict[str, float],
    features: list[str],
    threshold: float = THRESHOLD,
) -> tuple[list[float], list[str], list[str]]:
    """Return (sizes, labels, colors) for pie chart, grouping small slices as Others."""
    total = sum(mean_abs.get(ft, 0.0) for ft in features)
    if total == 0:
        total = 1.0

    fracs = {ft: mean_abs.get(ft, 0.0) / total for ft in features}

    main_fts   = [ft for ft in features if fracs[ft] >= threshold]
    other_fts  = [ft for ft in features if fracs[ft] < threshold]

    # Sort main features descending
    main_fts = sorted(main_fts, key=lambda ft: fracs[ft], reverse=True)

    sizes  = [fracs[ft] for ft in main_fts]
    labels = [LABELS.get(ft, ft) for ft in main_fts]
    colors = [FEATURE_COLORS.get(ft, _OTHER_COLOR) for ft in main_fts]

    if other_fts:
        sizes.append(sum(fracs[ft] for ft in other_fts))
        labels.append("Others")
        colors.append(_OTHER_COLOR)

    return sizes, labels, colors


def _plot_pie(
    sizes: list[float],
    labels: list[str],
    colors: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    # Explode slices >= 15% slightly for emphasis
    explode = [0.04 if s >= 0.15 else 0.0 for s in sizes]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        explode=explode,
        autopct=lambda p: f"{p:.1f}%" if p >= 2.0 else "",
        pctdistance=0.78,
        startangle=90,
        wedgeprops=dict(linewidth=0.6, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=c, edgecolor="white", label=f"{l} ({s*100:.1f}%)")
        for l, s, c in zip(labels, sizes, colors)
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        fontsize=7.5,
        frameon=False,
    )

    ax.set_title(title, fontsize=10, pad=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate feature importance pie charts from attribution CSVs."
    )
    parser.add_argument("--explain_root", required=True,
                        help="Root dir containing r01/, r02/, ... (e.g. outputs/egu_fin/explain_te)")
    parser.add_argument("--group", default="no_time_encoding",
                        help="Subdir within each region dir (default: no_time_encoding)")
    parser.add_argument("--filename", default=None,
                        help="Attribution CSV filename (auto-detected from group if omitted)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Fraction below which features go into Others (default {THRESHOLD:.0%})")
    args = parser.parse_args()

    explain_root = Path(args.explain_root)
    group        = args.group
    filename     = args.filename or f"attributions_finetune_ensemble1_{group}.csv"

    print(f"\n=== Feature pie charts: {explain_root.name}/{group} ===")

    for rkey in [f"r{i:02d}" for i in range(1, 20)]:
        csv_path = explain_root / rkey / group / filename
        if not csv_path.exists():
            print(f"  [{rkey}] MISSING {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        region_name = REGION_NAMES.get(rkey, rkey)
        out_dir = explain_root / rkey / group
        print(f"\n  {rkey} — {region_name}")

        # Full (all features incl. CenLat, CenLon, year)
        mean_abs_full = _compute_mean_abs(df, ALL_FEATURES)
        sizes, labels, colors = _make_pie_data(mean_abs_full, ALL_FEATURES, args.threshold)
        _plot_pie(
            sizes, labels, colors,
            title=f"{rkey.upper()} {region_name}\nFeature importance (all features)",
            output_path=out_dir / "pie_importance_full.png",
        )

        # Masked (no CenLat, CenLon, year)
        mean_abs_masked = _compute_mean_abs(df, MASKED_FEATURES)
        sizes, labels, colors = _make_pie_data(mean_abs_masked, MASKED_FEATURES, args.threshold)
        _plot_pie(
            sizes, labels, colors,
            title=f"{rkey.upper()} {region_name}\nFeature importance (masked: no lat/lon/year)",
            output_path=out_dir / "pie_importance_masked.png",
        )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
