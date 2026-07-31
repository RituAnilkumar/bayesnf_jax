"""
src/plot_global_te_comparison.py

Compare time-encoding vs no-time-encoding global mass balance predictions.

Loads ensemble_regional_gt.csv for all regions under
    {ensemble_root}/r{nn}/time_encoding/
    {ensemble_root}/r{nn}/no_time_encoding/

sums median_gt across regions per year, combines std_total in quadrature
(regions treated as independent), then produces two standalone plots:

  1. global_annual_gt.png      — annual Gt/yr series with ±2σ bands + GLaMBIE
  2. global_cumulative_gt.png  — cumulative Gt from 2000 + GLaMBIE cumulative

Usage:
    python src/plot_global_te_comparison.py \\
        --ensemble_root outputs/egu_fin/ensemble_te_r2 \\
        --glambie      validation_data/glambie_global.csv \\
        --output_dir   outputs/egu_fin/global_te_comparison

    # restrict year range shown in annual plot:
    python src/plot_global_te_comparison.py \\
        --ensemble_root outputs/egu_fin/ensemble_te_r2 \\
        --glambie      validation_data/glambie_global.csv \\
        --output_dir   outputs/egu_fin/global_te_comparison \\
        --plot_from 1990
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_group_global(ensemble_root: Path, group: str) -> pd.DataFrame | None:
    """
    Sum ensemble_regional_gt.csv across all region subdirectories for one group.

    Returns DataFrame with columns: year, median_gt, std_total
    std_total is combined in quadrature across independent regions.
    Returns None if no region files are found.
    """
    region_dirs = sorted(
        d for d in ensemble_root.iterdir()
        if d.is_dir() and d.name.startswith("r")
    )

    all_dfs = []
    missing = []
    for rd in region_dirs:
        path = rd / group / "ensemble_regional_gt.csv"
        if not path.exists():
            missing.append(rd.name)
            continue
        df = pd.read_csv(path).sort_values("year").reset_index(drop=True)
        all_dfs.append((rd.name, df))

    if missing:
        warnings.warn(f"  [{group}] Missing regions: {missing}")
    if not all_dfs:
        return None

    # Align on the union of years
    years = sorted(set().union(*[set(df["year"].values) for _, df in all_dfs]))
    years = np.array(years)

    median_sum = np.zeros(len(years))
    var_sum    = np.zeros(len(years))

    for region, df in all_dfs:
        yr_to_idx = {y: i for i, y in enumerate(years)}
        for _, row in df.iterrows():
            yi = yr_to_idx.get(int(row["year"]))
            if yi is None:
                continue
            median_sum[yi] += row["median_gt"]
            var_sum[yi]    += row["std_total"] ** 2

    return pd.DataFrame({
        "year":      years,
        "median_gt": median_sum,
        "std_total": np.sqrt(var_sum),
    })


def _load_wgms_global(wgms_dir: str | Path) -> pd.DataFrame:
    """
    Sum regional WGMS/Dussaillant gt values across all available CSV files.

    Each file has columns: year, gt, gt_sigma.
    SA1 + SA2 are summed for r17; all other files contribute one region each.
    Returns DataFrame with columns: year, gt, gt_sigma (quadrature sum of sigmas).
    """
    wgms_dir = Path(wgms_dir)
    csvs = sorted(wgms_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {wgms_dir}")

    all_dfs = []
    for p in csvs:
        df = pd.read_csv(p)[["year", "gt", "gt_sigma"]].dropna()
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    grouped = (
        combined
        .groupby("year")
        .agg(gt=("gt", "sum"), gt_sigma=("gt_sigma", lambda x: np.sqrt((x**2).sum())))
        .reset_index()
        .sort_values("year")
    )
    return grouped


def _load_glambie_global(path: str | Path) -> pd.DataFrame:
    """
    Load GLaMBIE global CSV.

    Returns DataFrame with columns: year (int, from floor(end_dates)),
    gt, gt_err.
    """
    df = pd.read_csv(path)
    df["year"]   = np.floor(df["end_dates"]).astype(int)
    df["gt"]     = df["combined_gt"].values
    df["gt_err"] = df["combined_gt_errors"].values
    return df[["year", "gt", "gt_err"]].sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_COLOR_TE   = "darkorange"
_COLOR_NO   = "steelblue"
_COLOR_OBS  = "black"
_COLOR_WGMS = "forestgreen"

_LABEL_TE   = "Time encoding"
_LABEL_NO   = "No time encoding"
_LABEL_OBS  = "GLaMBIE global"
_LABEL_WGMS = "WGMS/Dussaillant regional sum"


def _shade(ax, years, mu, std, color, label, alpha_band=0.18):
    ax.fill_between(years, mu - 2 * std, mu + 2 * std, alpha=alpha_band, color=color)
    ax.plot(years, mu, color=color, lw=1.8, label=label)


def _plot_glambie_annual(ax, glambie_df: pd.DataFrame, plot_from: int) -> None:
    gb = glambie_df[glambie_df["year"] >= plot_from]
    if gb.empty:
        return
    ax.errorbar(
        gb["year"].values, gb["gt"].values, yerr=gb["gt_err"].values,
        fmt="-o", color=_COLOR_OBS, ms=4, lw=1.2, capsize=3,
        zorder=5, label=_LABEL_OBS,
    )


def _plot_glambie_cumulative(ax, glambie_df: pd.DataFrame, start_year: int) -> None:
    gb = glambie_df[glambie_df["year"] >= start_year].copy()
    if gb.empty:
        return
    gb_cum     = gb["gt"].cumsum().values
    gb_cum_err = np.sqrt(np.cumsum(gb["gt_err"].values ** 2))
    ax.fill_between(
        gb["year"].values,
        gb_cum - 1.96 * gb_cum_err,
        gb_cum + 1.96 * gb_cum_err,
        alpha=0.15, color=_COLOR_OBS,
    )
    ax.plot(gb["year"].values, gb_cum, "-o", color=_COLOR_OBS, lw=1.5,
            ms=4, zorder=5, label=_LABEL_OBS)


def _plot_wgms_annual(ax, wgms_df: pd.DataFrame, plot_from: int) -> None:
    w = wgms_df[wgms_df["year"] >= plot_from]
    if w.empty:
        return
    ax.errorbar(
        w["year"].values, w["gt"].values, yerr=w["gt_sigma"].values,
        fmt="-s", color=_COLOR_WGMS, ms=4, lw=1.2, capsize=3,
        zorder=5, label=_LABEL_WGMS,
    )


def _plot_wgms_cumulative(ax, wgms_df: pd.DataFrame, start_year: int) -> None:
    w = wgms_df[wgms_df["year"] >= start_year].copy()
    if w.empty:
        return
    w_cum     = w["gt"].cumsum().values
    w_cum_err = np.sqrt(np.cumsum(w["gt_sigma"].values ** 2))
    ax.fill_between(
        w["year"].values,
        w_cum - w_cum_err,
        w_cum + w_cum_err,
        alpha=0.15, color=_COLOR_WGMS,
    )
    ax.plot(w["year"].values, w_cum, "-s", color=_COLOR_WGMS, lw=1.5,
            ms=4, zorder=5, label=_LABEL_WGMS)


# ---------------------------------------------------------------------------
# Plot 1 — Annual series
# ---------------------------------------------------------------------------

def plot_annual(
    series: list[tuple[pd.DataFrame, str, str]],
    glambie_df: pd.DataFrame,
    output_path: Path,
    title: str,
    plot_from: int = 1940,
    wgms_df: pd.DataFrame | None = None,
) -> None:
    """
    Args:
        series:   list of (df, color, label) — one entry per model group to plot.
        wgms_df:  optional WGMS/Dussaillant regional-sum DataFrame.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    for df, color, label in series:
        mask = df["year"].values >= plot_from
        _shade(ax, df["year"].values[mask], df["median_gt"].values[mask],
               df["std_total"].values[mask], color, label)

    _plot_glambie_annual(ax, glambie_df, plot_from)
    if wgms_df is not None:
        _plot_wgms_annual(ax, wgms_df, plot_from)

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Global mass balance (Gt/yr)")
    ax.set_title(f"{title}\n±2σ total uncertainty")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 2 — Cumulative
# ---------------------------------------------------------------------------

def plot_cumulative(
    series: list[tuple[pd.DataFrame, str, str]],
    glambie_df: pd.DataFrame,
    output_path: Path,
    title: str,
    start_year: int = 2000,
    wgms_df: pd.DataFrame | None = None,
) -> None:
    """
    Args:
        series:   list of (df, color, label) — one entry per model group to plot.
        wgms_df:  optional WGMS/Dussaillant regional-sum DataFrame.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    for df, color, label in series:
        mask    = df["year"].values >= start_year
        years   = df["year"].values[mask]
        cum_mu  = np.cumsum(df["median_gt"].values[mask])
        cum_std = np.sqrt(np.cumsum(df["std_total"].values[mask] ** 2))
        _shade(ax, years, cum_mu, cum_std, color, label)

    _plot_glambie_cumulative(ax, glambie_df, start_year)
    if wgms_df is not None:
        _plot_wgms_cumulative(ax, wgms_df, start_year)

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative mass balance from {start_year} (Gt)")
    ax.set_title(f"{title}\n±2σ total uncertainty, cumulative from {start_year}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global annual + cumulative comparison: time encoding vs no time encoding."
    )
    parser.add_argument("--ensemble_root", required=True,
                        help="Root directory containing r01/, r02/, ... subdirs "
                             "(e.g. outputs/egu_fin/ensemble_te_r2).")
    parser.add_argument("--glambie", required=True,
                        help="Path to glambie_global.csv validation file.")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for plots. Defaults to {ensemble_root}/global_comparison/.")
    parser.add_argument("--wgms_dir", default=None,
                        help="Directory of per-region WGMS/Dussaillant CSVs "
                             "(e.g. validation_data/regional_wgms_duss). "
                             "When supplied, regional sums are overlaid on all plots.")
    parser.add_argument("--plot_from", type=int, default=1940,
                        help="First year shown in the annual plot (default 1940).")
    parser.add_argument("--cumulative_from", type=int, default=2000,
                        help="Start year for cumulative plot (default 2000).")
    args = parser.parse_args()

    ensemble_root = Path(args.ensemble_root)
    output_dir    = Path(args.output_dir) if args.output_dir else ensemble_root / "global_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Global TE comparison: {ensemble_root.name} ===")

    wgms_df = None
    if args.wgms_dir:
        print(f"\nLoading WGMS/Dussaillant regional sums from {args.wgms_dir}...")
        wgms_df = _load_wgms_global(args.wgms_dir)
        print(f"  {len(wgms_df)} years ({wgms_df['year'].min()}–{wgms_df['year'].max()}), "
              f"gt range [{wgms_df['gt'].min():.1f}, {wgms_df['gt'].max():.1f}] Gt/yr")

    print("\nLoading time_encoding group...")
    te_df = _load_group_global(ensemble_root, "time_encoding")
    if te_df is None:
        raise RuntimeError("No ensemble_regional_gt.csv found for time_encoding group.")
    print(f"  {len(te_df)} years, global median_gt range: "
          f"[{te_df['median_gt'].min():.1f}, {te_df['median_gt'].max():.1f}] Gt/yr")

    print("\nLoading no_time_encoding group...")
    no_df = _load_group_global(ensemble_root, "no_time_encoding")
    if no_df is None:
        raise RuntimeError("No ensemble_regional_gt.csv found for no_time_encoding group.")
    print(f"  {len(no_df)} years, global median_gt range: "
          f"[{no_df['median_gt'].min():.1f}, {no_df['median_gt'].max():.1f}] Gt/yr")

    print("\nLoading GLaMBIE global...")
    glambie_df = _load_glambie_global(args.glambie)
    print(f"  {len(glambie_df)} years ({glambie_df['year'].min()}–{glambie_df['year'].max()})")

    te_series = (te_df, _COLOR_TE, _LABEL_TE)
    no_series = (no_df, _COLOR_NO, _LABEL_NO)

    print("\nGenerating plots...")

    # Combined — both groups on one axes
    plot_annual(
        [te_series, no_series], glambie_df,
        output_dir / "global_annual_gt.png",
        title="Global glacier mass balance — time encoding comparison",
        plot_from=args.plot_from,
        wgms_df=wgms_df,
    )
    plot_cumulative(
        [te_series, no_series], glambie_df,
        output_dir / "global_cumulative_gt.png",
        title="Global glacier cumulative mass balance — time encoding comparison",
        start_year=args.cumulative_from,
        wgms_df=wgms_df,
    )

    # Time encoding only
    plot_annual(
        [te_series], glambie_df,
        output_dir / "global_annual_gt_time_encoding.png",
        title="Global glacier mass balance — with time encoding",
        plot_from=args.plot_from,
        wgms_df=wgms_df,
    )
    plot_cumulative(
        [te_series], glambie_df,
        output_dir / "global_cumulative_gt_time_encoding.png",
        title="Global glacier cumulative mass balance — with time encoding",
        start_year=args.cumulative_from,
        wgms_df=wgms_df,
    )

    # No time encoding only
    plot_annual(
        [no_series], glambie_df,
        output_dir / "global_annual_gt_no_time_encoding.png",
        title="Global glacier mass balance — no time encoding",
        plot_from=args.plot_from,
        wgms_df=wgms_df,
    )
    plot_cumulative(
        [no_series], glambie_df,
        output_dir / "global_cumulative_gt_no_time_encoding.png",
        title="Global glacier cumulative mass balance — no time encoding",
        start_year=args.cumulative_from,
        wgms_df=wgms_df,
    )

    print(f"\nDone. Outputs written to {output_dir}/")


if __name__ == "__main__":
    main()
