"""
src/plot_model_estimates_complete.py

Plots per-region and global model ensemble estimates (median ± 2σ) from
no_time_encoding outputs.  No validation data overlaid.

Outputs (saved to --output_dir):
  timeseries_r01.png … timeseries_r19.png   one plot per region (MWE/yr)
  timeseries_global.png                      global sum (Gt/yr)

Usage:
    python src/plot_model_estimates_complete.py \\
        --ensemble_root outputs/egu_fin/ensemble_te_r2 \\
        --group         no_time_encoding \\
        --output_dir    outputs/egu_fin/model_estimates_complete
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Region metadata
# ---------------------------------------------------------------------------

REGION_NAMES = {
    "r01": "Alaska",
    "r02": "W. Canada & US",
    "r03": "Arctic Canada N",
    "r04": "Arctic Canada S",
    "r05": "Greenland periph.",
    "r06": "Iceland",
    "r07": "Svalbard & Jan Mayen",
    "r08": "Scandinavia",
    "r09": "Russian Arctic",
    "r10": "North Asia",
    "r11": "Central Europe",
    "r12": "Caucasus",
    "r13": "Central Asia",
    "r14": "S. Asia West",
    "r15": "S. Asia East",
    "r16": "Low Latitudes",
    "r17": "Southern Andes",
    "r18": "New Zealand",
    "r19": "Antarctic & Subantarctic",
}

# Font sizes — match val_reg_egu (fontscale = 2)
_FONTSCALE = 2.0
_LBL_FS  = round(10 * _FONTSCALE)
_TICK_FS = round(10 * _FONTSCALE)
_LEG_FS  = round(7  * _FONTSCALE)
_TTL_FS  = round(9  * _FONTSCALE)
_FIG_W   = 13 + 2 * (_FONTSCALE - 1)   # 15
_FIG_H   = 5  + 2 * (_FONTSCALE - 1)   # 7

_COLOR = "steelblue"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _load_mwe(ensemble_root: Path, region: str, group: str) -> pd.DataFrame | None:
    path = ensemble_root / region / group / "ensemble_regional_mwe.csv"
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    return pd.read_csv(path).sort_values("year").reset_index(drop=True)


def _load_gt(ensemble_root: Path, region: str, group: str) -> pd.DataFrame | None:
    path = ensemble_root / region / group / "ensemble_regional_gt.csv"
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    return pd.read_csv(path).sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-region MWE plot
# ---------------------------------------------------------------------------

def plot_region(df: pd.DataFrame, region: str, output_path: Path) -> None:
    name   = REGION_NAMES.get(region, region)
    years  = df["year"].values
    mu     = df["median_mwe"].values
    s_tot  = df["std_total"].values

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    ax.fill_between(years, mu - 2 * s_tot, mu + 2 * s_tot,
                    alpha=0.25, color=_COLOR, label="±2σ total")
    ax.plot(years, mu, color=_COLOR, lw=2.0, label="Ensemble median")

    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_title(f"{name}  ({region})", fontsize=_TTL_FS)
    ax.set_xlabel("Year", fontsize=_LBL_FS)
    ax.set_ylabel("Mass Balance (m.w.e/yr)", fontsize=_LBL_FS)
    ax.tick_params(labelsize=_TICK_FS)
    ax.legend(fontsize=_LEG_FS)
    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Global Gt plot
# ---------------------------------------------------------------------------

def plot_global(ensemble_root: Path, group: str, output_path: Path) -> None:
    regions = sorted(d.name for d in ensemble_root.iterdir()
                     if d.is_dir() and d.name.startswith("r"))

    all_dfs = []
    for r in regions:
        df = _load_gt(ensemble_root, r, group)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("  No Gt files found — skipping global plot.")
        return

    years = sorted(set().union(*[set(df["year"].values) for df in all_dfs]))
    years = np.array(years)
    median_sum = np.zeros(len(years))
    var_sum    = np.zeros(len(years))
    yr_idx     = {y: i for i, y in enumerate(years)}

    for df in all_dfs:
        for _, row in df.iterrows():
            yi = yr_idx.get(int(row["year"]))
            if yi is None:
                continue
            median_sum[yi] += row["median_gt"]
            var_sum[yi]    += row["std_total"] ** 2

    std_total = np.sqrt(var_sum)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.fill_between(years, median_sum - 2 * std_total, median_sum + 2 * std_total,
                    alpha=0.25, color=_COLOR, label="±2σ total")
    ax.plot(years, median_sum, color=_COLOR, lw=2.0, label="Ensemble median")

    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_title("Global (all regions)", fontsize=_TTL_FS)
    ax.set_xlabel("Year", fontsize=_LBL_FS)
    ax.set_ylabel("Mass Balance (Gt/yr)", fontsize=_LBL_FS)
    ax.tick_params(labelsize=_TICK_FS)
    ax.legend(fontsize=_LEG_FS)
    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot model-only ensemble estimates per region and globally."
    )
    parser.add_argument("--ensemble_root", default="outputs/egu_fin/ensemble_te_r2")
    parser.add_argument("--group",         default="no_time_encoding")
    parser.add_argument("--output_dir",    default="outputs/egu_fin/model_estimates_complete")
    args = parser.parse_args()

    ensemble_root = Path(args.ensemble_root)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Model estimates — {args.group} ===")
    print(f"  Ensemble root : {ensemble_root}")
    print(f"  Output dir    : {output_dir}\n")

    regions = sorted(d.name for d in ensemble_root.iterdir()
                     if d.is_dir() and d.name.startswith("r"))

    for region in regions:
        df = _load_mwe(ensemble_root, region, args.group)
        if df is None:
            continue
        plot_region(df, region, output_dir / f"timeseries_{region}.png")

    print()
    plot_global(ensemble_root, args.group, output_dir / "timeseries_global.png")
    print(f"\nDone. {len(regions)} regions + global → {output_dir}/")


if __name__ == "__main__":
    main()
