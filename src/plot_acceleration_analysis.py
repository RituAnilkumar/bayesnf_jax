"""
src/plot_acceleration_analysis.py

Produces three publication-ready figures characterising the 80-year record of
global and regional glacier mass balance from the no_time_encoding ensemble:

  1. acceleration_global_ts.png
       Global Gt/yr time series (annual + 10-yr rolling mean), zero-crossing
       marked, pre/post-1990 linear trend lines annotated with rates.

  2. acceleration_onset_timeline.png
       Horizontal lollipop chart: each region's onset of sustained negative
       mass balance, ordered chronologically, colour-coded by region group.

  3. acceleration_decade_bars.png
       Decade-mean global mass balance (Gt/yr), showing progressive
       acceleration across eight decades.

Usage:
    python src/plot_acceleration_analysis.py \\
        --ensemble_root outputs/egu_fin/ensemble_te_r2 \\
        --group         no_time_encoding \\
        --output_dir    outputs/egu_fin/acceleration_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGION_NAMES = {
    "r01": "Alaska",
    "r02": "W. Canada & US",
    "r03": "Arctic Canada N",
    "r04": "Arctic Canada S",
    "r05": "Greenland Periph.",
    "r06": "Iceland",
    "r07": "Svalbard & Jan Mayen",
    "r08": "Scandinavia",
    "r09": "Russian Arctic",
    "r10": "North Asia",
    "r11": "Central Europe",
    "r12": "Caucasus & Middle East",
    "r13": "Central Asia",
    "r14": "S. Asia West (Karakoram)",
    "r15": "S. Asia East",
    "r16": "Low Latitudes",
    "r17": "Southern Andes",
    "r18": "New Zealand",
    "r19": "Antarctic & Subantarctic",
}

# Region groups for colour coding on the onset chart
REGION_GROUP = {
    "r01": "North Pacific",
    "r02": "North Pacific",
    "r03": "Arctic",
    "r04": "Arctic",
    "r05": "Arctic",
    "r06": "Arctic",
    "r07": "Arctic",
    "r08": "Arctic",
    "r09": "Arctic",
    "r10": "High-Altitude Asia",
    "r11": "Temperate",
    "r12": "High-Altitude Asia",
    "r13": "High-Altitude Asia",
    "r14": "High-Altitude Asia",
    "r15": "High-Altitude Asia",
    "r16": "Tropics",
    "r17": "Southern Hemisphere",
    "r18": "Southern Hemisphere",
    "r19": "Southern Hemisphere",
}

GROUP_COLORS = {
    "North Pacific":       "#E87722",
    "Arctic":              "#4B9CD3",
    "High-Altitude Asia":  "#6A4C9C",
    "Temperate":           "#2CA02C",
    "Tropics":             "#D62728",
    "Southern Hemisphere": "#8C564B",
}

# Year range for historical analysis
HIST_MIN, HIST_MAX = 1940, 2020

# Linear-trend breakpoint
TREND_BREAK = 1990

# Font sizing
_FS = 13  # base font size


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_global_gt(ensemble_root: Path, group: str) -> pd.Series:
    """Sum regional median Gt/yr across all regions → global annual series."""
    regions = sorted(d.name for d in ensemble_root.iterdir()
                     if d.is_dir() and d.name.startswith("r"))
    all_series = []
    for r in regions:
        p = ensemble_root / r / group / "ensemble_regional_gt.csv"
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        df = pd.read_csv(p).sort_values("year")
        df = df[(df["year"] >= HIST_MIN) & (df["year"] <= HIST_MAX)]
        all_series.append(df.set_index("year")["median_gt"])
    global_gt = pd.concat(all_series, axis=1).sum(axis=1).sort_index()
    return global_gt


def load_global_gt_uncertainty(ensemble_root: Path, group: str) -> pd.DataFrame:
    """Sum regional median + quadrature-summed total std → global series."""
    regions = sorted(d.name for d in ensemble_root.iterdir()
                     if d.is_dir() and d.name.startswith("r"))
    med_list, var_list = [], []
    for r in regions:
        p = ensemble_root / r / group / "ensemble_regional_gt.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p).sort_values("year")
        df = df[(df["year"] >= HIST_MIN) & (df["year"] <= HIST_MAX)]
        df = df.set_index("year")
        med_list.append(df["median_gt"])
        var_list.append(df["std_total"] ** 2)
    global_med = pd.concat(med_list, axis=1).sum(axis=1).sort_index()
    global_std = pd.concat(var_list, axis=1).sum(axis=1).pipe(np.sqrt).sort_index()
    return pd.DataFrame({"median_gt": global_med, "std_total": global_std})


def load_regional_mwe(ensemble_root: Path, region: str, group: str) -> pd.DataFrame | None:
    p = ensemble_root / region / group / "ensemble_regional_mwe.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).sort_values("year")
    return df[(df["year"] >= HIST_MIN) & (df["year"] <= HIST_MAX)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

def detect_onset(
    df: pd.DataFrame,
    window: int = 10,
    min_years_from: int = 1950,
    future_frac: float = 0.80,
) -> int | None:
    """
    Return the first year where sustained mass loss is statistically detectable.

    Condition: the 10-yr centred rolling mean of median_mwe PLUS the rolling
    mean of std_total is still negative — i.e. even the upper (~1σ) confidence
    bound of the smoothed signal has crossed zero.  Additionally, ≥`future_frac`
    of all subsequent years must also satisfy this condition, ensuring the
    transition is sustained rather than a transient dip.

    This is more conservative than a simple zero-crossing of the median: it
    requires the loss to be unambiguous given the ensemble uncertainty.
    """
    df = df[df["year"] >= min_years_from].copy()
    idx = df.set_index("year")

    roll_med = idx["median_mwe"].rolling(window, center=True, min_periods=window // 2).mean()
    roll_std = idx["std_total"].rolling(window, center=True, min_periods=window // 2).mean()

    # Upper confidence bound of the smoothed signal
    upper_bound = roll_med + roll_std

    for y, v in upper_bound.items():
        if np.isnan(v) or v >= 0:
            continue
        future = upper_bound[upper_bound.index > y]
        if len(future) >= 5 and (future < 0).mean() >= future_frac:
            return int(y)
    return None


# ---------------------------------------------------------------------------
# Figure 1 — Global time series with acceleration diagnostics
# ---------------------------------------------------------------------------

def plot_global_ts(
    global_df: pd.DataFrame,
    output_path: Path,
) -> None:
    years = global_df.index.values
    med   = global_df["median_gt"].values
    std   = global_df["std_total"].values

    roll10 = (
        pd.Series(med, index=years)
        .rolling(10, center=True, min_periods=6)
        .mean()
    )

    # Zero crossing of rolling mean
    signs = np.sign(roll10.dropna().values)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    zero_cross_year = roll10.dropna().index[sign_changes[-1]] if len(sign_changes) else None

    # Linear trends: pre / post TREND_BREAK
    mask_pre  = (years >= 1950) & (years <= TREND_BREAK)
    mask_post = (years >= TREND_BREAK) & (years <= HIST_MAX)
    coef_pre  = np.polyfit(years[mask_pre],  med[mask_pre],  1)
    coef_post = np.polyfit(years[mask_post], med[mask_post], 1)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Uncertainty band
    ax.fill_between(years, med - 2 * std, med + 2 * std,
                    alpha=0.18, color="steelblue", label="±2σ ensemble spread")
    # Annual median
    ax.plot(years, med, color="steelblue", lw=1.0, alpha=0.5, label="Annual median")
    # 10-yr rolling mean
    ax.plot(roll10.index, roll10.values, color="steelblue", lw=2.5,
            label="10-yr rolling mean")

    # Trend lines
    yr_pre_range  = np.array([1950, TREND_BREAK])
    yr_post_range = np.array([TREND_BREAK, HIST_MAX])
    ax.plot(yr_pre_range,  np.polyval(coef_pre,  yr_pre_range),
            color="firebrick", lw=2.0, ls="--", label=f"Pre-{TREND_BREAK} trend")
    ax.plot(yr_post_range, np.polyval(coef_post, yr_post_range),
            color="darkred",  lw=2.0, ls="--", label=f"Post-{TREND_BREAK} trend")

    # Annotate trend rates
    ax.text(1965, np.polyval(coef_pre, 1965) - 55,
            f"{coef_pre[0]:+.1f} Gt yr⁻² ",
            color="firebrick", fontsize=_FS - 1, ha="center", style="italic")
    ax.text(2006, np.polyval(coef_post, 2006) - 70,
            f"{coef_post[0]:+.1f} Gt yr⁻²",
            color="darkred", fontsize=_FS - 1, ha="center", style="italic")

    # Zero crossing marker
    if zero_cross_year is not None:
        ax.axvline(zero_cross_year, color="gray", lw=1.2, ls=":", alpha=0.9)
        ax.text(zero_cross_year + 0.5, ax.get_ylim()[1] * 0.92,
                f"Zero crossing\n~{zero_cross_year}",
                color="gray", fontsize=_FS - 2, va="top")

    # Trend break marker
    ax.axvline(TREND_BREAK, color="black", lw=1.0, ls=":", alpha=0.6)
    ax.text(TREND_BREAK + 0.5, ax.get_ylim()[1] * 0.75,
            f"{TREND_BREAK}", color="black", fontsize=_FS - 2, va="top")

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlim(HIST_MIN, HIST_MAX)
    ax.set_xlabel("Year", fontsize=_FS)
    ax.set_ylabel("Mass Balance  (Gt yr⁻¹)", fontsize=_FS)
    ax.set_title("Global Glacier Mass Balance 1940–2020\n"
                 "(no-time-encoding BNF ensemble)", fontsize=_FS + 1, pad=8)
    ax.tick_params(labelsize=_FS - 1)
    ax.legend(fontsize=_FS - 2, loc="lower left")
    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2 — Regional onset timeline (lollipop)
# ---------------------------------------------------------------------------

def plot_onset_timeline(
    ensemble_root: Path,
    group: str,
    output_path: Path,
) -> None:
    regions = sorted(d.name for d in ensemble_root.iterdir()
                     if d.is_dir() and d.name.startswith("r"))

    onsets = {}
    for r in regions:
        df = load_regional_mwe(ensemble_root, r, group)
        if df is None:
            continue
        onset = detect_onset(df)
        onsets[r] = onset

    # Sort by onset year (None → put at end as "not detected")
    sorted_regions = sorted(onsets.keys(), key=lambda r: (onsets[r] is None, onsets[r] or 9999))

    fig, ax = plt.subplots(figsize=(10, 7.5))

    y_pos = np.arange(len(sorted_regions))
    reference = HIST_MIN  # bar starts from the left

    for i, r in enumerate(sorted_regions):
        onset = onsets[r]
        name  = REGION_NAMES.get(r, r)
        group_name = REGION_GROUP.get(r, "Other")
        color = GROUP_COLORS.get(group_name, "gray")

        if onset is not None:
            # Horizontal bar from HIST_MIN to onset
            ax.barh(i, onset - reference, left=reference,
                    height=0.55, color=color, alpha=0.25)
            # Lollipop stem
            ax.plot([onset, onset], [i - 0.35, i + 0.35],
                    color=color, lw=1.5, alpha=0.7)
            # Lollipop head
            ax.plot(onset, i, "o", color=color, ms=10, zorder=5)
            ax.text(onset + 1, i, str(onset),
                    va="center", ha="left", fontsize=_FS - 3, color="black")
        else:
            ax.text(HIST_MAX - 1, i, "not detected",
                    va="center", ha="right", fontsize=_FS - 3, color="gray",
                    style="italic")

        ax.text(reference - 1, i, f"  {name}",
                va="center", ha="right", fontsize=_FS - 2)

    ax.axvline(TREND_BREAK, color="black", lw=1.0, ls=":", alpha=0.5)
    ax.text(TREND_BREAK, len(sorted_regions) - 0.3, f"{TREND_BREAK}",
            ha="center", va="bottom", fontsize=_FS - 2, color="black")

    # Legend for groups
    legend_patches = [
        mpatches.Patch(color=c, label=g, alpha=0.7)
        for g, c in GROUP_COLORS.items()
        if g in REGION_GROUP.values()
    ]
    ax.legend(handles=legend_patches, fontsize=_FS - 3,
              loc="lower right", framealpha=0.9)

    ax.set_xlim(HIST_MIN - 2, HIST_MAX + 2)
    ax.set_yticks([])
    ax.set_xlabel("Year of onset of sustained negative mass balance", fontsize=_FS)
    ax.set_title("Regional Onset of Sustained Mass Loss\n"
                 "(10-yr rolling mean < −0.05 m.w.e. yr⁻¹, sustained)",
                 fontsize=_FS + 1, pad=8)
    ax.tick_params(labelsize=_FS - 1)
    ax.invert_yaxis()
    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 3 — 20-year block-average global mass balance
# ---------------------------------------------------------------------------

def plot_20yr_blocks(
    global_df: pd.DataFrame,
    output_path: Path,
    window: int = 20,
) -> None:
    """
    Split the historical record into non-overlapping 20-year windows and show
    the mean Gt/yr in each block, alongside the full annual time series.

    Two panels:
      Top — annual time series with block means drawn as horizontal steps,
             uncertainty (±1σ block-mean std) shaded per block.
      Bottom — bar chart of block means for direct comparison.
    """
    years = global_df.index.values
    med   = global_df["median_gt"].values
    std   = global_df["std_total"].values

    # Build non-overlapping blocks from HIST_MIN
    y_start = HIST_MIN
    blocks = []
    while y_start + window <= HIST_MAX + 1:
        y_end = y_start + window
        mask  = (years >= y_start) & (years < y_end)
        if mask.sum() == 0:
            y_start = y_end
            continue
        block_med  = med[mask].mean()
        # Uncertainty on the block mean: quadrature mean of annual variances / n
        block_std  = np.sqrt((std[mask] ** 2).mean()) / np.sqrt(mask.sum())
        blocks.append({
            "label":  f"{y_start}–{y_end - 1}",
            "y0":     y_start,
            "y1":     y_end,
            "mean":   block_med,
            "std":    block_std,
        })
        y_start = y_end

    fig, (ax_ts, ax_bar) = plt.subplots(
        2, 1, figsize=(11, 9),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.35},
    )

    # --- Top panel: time series + block steps ---
    ax_ts.fill_between(years, med - 2 * std, med + 2 * std,
                       alpha=0.15, color="steelblue", label="±2σ annual spread")
    ax_ts.plot(years, med, color="steelblue", lw=1.0, alpha=0.5, label="Annual median")

    colors_step = ["#4B9CD3" if b["mean"] >= 0 else "#D62728" for b in blocks]
    for b, c in zip(blocks, colors_step):
        x0, x1 = b["y0"], min(b["y1"], HIST_MAX + 1)
        bm, bs  = b["mean"], b["std"]
        # Block mean as horizontal line
        ax_ts.hlines(bm, x0, x1, colors=c, lw=2.5, zorder=4)
        # ±1σ band on the block mean
        ax_ts.fill_between([x0, x1], bm - bs, bm + bs,
                           alpha=0.25, color=c, zorder=3)
        # Annotate value
        ax_ts.text((x0 + x1) / 2, bm + (15 if bm >= 0 else -20),
                   f"{bm:+.0f}", ha="center", va="bottom" if bm >= 0 else "top",
                   fontsize=_FS - 1, fontweight="bold", color=c)

    ax_ts.axhline(0, color="black", lw=0.8, ls="--", alpha=0.7)
    ax_ts.set_xlim(HIST_MIN, HIST_MAX)
    ax_ts.set_xlabel("Year", fontsize=_FS)
    ax_ts.set_ylabel("Mass Balance  (Gt yr⁻¹)", fontsize=_FS)
    ax_ts.set_title("Global Glacier Mass Balance — 20-year block averages\n"
                    "(no-time-encoding BNF ensemble)", fontsize=_FS + 1, pad=8)
    ax_ts.tick_params(labelsize=_FS - 1)
    ax_ts.legend(fontsize=_FS - 2, loc="lower left")

    # --- Bottom panel: bar chart ---
    labels = [b["label"] for b in blocks]
    means  = np.array([b["mean"] for b in blocks])
    errs   = np.array([b["std"]  for b in blocks])
    colors_bar = ["#4B9CD3" if v >= 0 else "#D62728" for v in means]

    bars = ax_bar.bar(labels, means, color=colors_bar,
                      edgecolor="white", lw=0.5, width=0.65,
                      yerr=errs, capsize=5, error_kw={"lw": 1.5, "color": "black"})

    for bar, val in zip(bars, means):
        y_txt = val + 8 if val >= 0 else val - 8
        va = "bottom" if val >= 0 else "top"
        ax_bar.text(bar.get_x() + bar.get_width() / 2, y_txt,
                    f"{val:+.0f}", ha="center", va=va,
                    fontsize=_FS - 2, fontweight="bold")

    ax_bar.axhline(0, color="black", lw=0.8)
    ax_bar.set_ylabel("Mean (Gt yr⁻¹)", fontsize=_FS)
    ax_bar.set_xlabel("20-year period", fontsize=_FS)
    ax_bar.tick_params(labelsize=_FS - 1)
    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 4 (old Fig 3) — Decade-mean global mass balance bar chart
# ---------------------------------------------------------------------------

def plot_decade_bars(
    global_gt: pd.Series,
    output_path: Path,
) -> None:
    decades = [
        (1940, 1950), (1950, 1960), (1960, 1970), (1970, 1980),
        (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2020),
    ]
    labels, means = [], []
    for y0, y1 in decades:
        sub = global_gt[(global_gt.index >= y0) & (global_gt.index < y1)]
        labels.append(f"{y0}s")
        means.append(sub.mean())

    means = np.array(means)
    colors = ["#4B9CD3" if v >= 0 else "#D62728" for v in means]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, means, color=colors, edgecolor="white", lw=0.5, width=0.65)

    for bar, val in zip(bars, means):
        y_txt = val + 4 if val >= 0 else val - 4
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y_txt,
                f"{val:+.0f}", ha="center", va=va, fontsize=_FS - 2, fontweight="bold")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean Mass Balance  (Gt yr⁻¹)", fontsize=_FS)
    ax.set_xlabel("Decade", fontsize=_FS)
    ax.set_title("Global Glacier Mass Balance by Decade\n"
                 "(no-time-encoding BNF ensemble median)",
                 fontsize=_FS + 1, pad=8)
    ax.tick_params(labelsize=_FS - 1)

    # Annotate the post-1980 acceleration arrow
    ax.annotate("",
                xy=(labels[-1], means[-1]),
                xytext=(labels[4], means[4]),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(5.5, (means[4] + means[-1]) / 2,
            f"×{abs(means[-1]/means[4]):.1f}\nacceleration",
            ha="center", va="center", fontsize=_FS - 2)

    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 4 — Combined 4-panel summary (optional)
# ---------------------------------------------------------------------------

def plot_combined_summary(
    ensemble_root: Path,
    global_df: pd.DataFrame,
    global_gt: pd.Series,
    group: str,
    output_path: Path,
) -> None:
    """2-row, 2-col summary panel for paper SI or talk."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_ts    = fig.add_subplot(gs[0, :])   # full-width top
    ax_dec   = fig.add_subplot(gs[1, 0])
    ax_onset = fig.add_subplot(gs[1, 1])

    # --- Panel A: global time series ---
    years = global_df.index.values
    med   = global_df["median_gt"].values
    std   = global_df["std_total"].values
    roll10 = pd.Series(med, index=years).rolling(10, center=True, min_periods=6).mean()
    coef_pre  = np.polyfit(years[(years>=1950)&(years<=TREND_BREAK)],
                           med[(years>=1950)&(years<=TREND_BREAK)], 1)
    coef_post = np.polyfit(years[(years>=TREND_BREAK)&(years<=HIST_MAX)],
                           med[(years>=TREND_BREAK)&(years<=HIST_MAX)], 1)

    ax_ts.fill_between(years, med - 2*std, med + 2*std, alpha=0.18, color="steelblue")
    ax_ts.plot(years, med, color="steelblue", lw=1.0, alpha=0.45, label="Annual median")
    ax_ts.plot(roll10.index, roll10.values, color="steelblue", lw=2.5, label="10-yr rolling mean")
    ax_ts.plot([1950, TREND_BREAK], np.polyval(coef_pre,  [1950, TREND_BREAK]),
               color="firebrick", lw=2.0, ls="--",
               label=f"Pre-{TREND_BREAK}: {coef_pre[0]:+.1f} Gt yr⁻²")
    ax_ts.plot([TREND_BREAK, HIST_MAX], np.polyval(coef_post, [TREND_BREAK, HIST_MAX]),
               color="darkred",  lw=2.0, ls="--",
               label=f"Post-{TREND_BREAK}: {coef_post[0]:+.1f} Gt yr⁻²")
    ax_ts.axhline(0, color="black", lw=0.8, ls="--", alpha=0.7)
    ax_ts.axvline(TREND_BREAK, color="black", lw=1.0, ls=":", alpha=0.5)
    ax_ts.set_xlim(HIST_MIN, HIST_MAX)
    ax_ts.set_xlabel("Year", fontsize=_FS - 1)
    ax_ts.set_ylabel("Mass Balance  (Gt yr⁻¹)", fontsize=_FS - 1)
    ax_ts.set_title("(a) Global Glacier Mass Balance 1940–2020", fontsize=_FS, loc="left")
    ax_ts.tick_params(labelsize=_FS - 2)
    ax_ts.legend(fontsize=_FS - 3, loc="lower left", ncol=2)

    # --- Panel B: decade bars ---
    decades = [(1940,1950),(1950,1960),(1960,1970),(1970,1980),
               (1980,1990),(1990,2000),(2000,2010),(2010,2020)]
    dlabels, dmeans = [], []
    for y0, y1 in decades:
        sub = global_gt[(global_gt.index>=y0)&(global_gt.index<y1)]
        dlabels.append(f"{y0}s")
        dmeans.append(sub.mean())
    dmeans = np.array(dmeans)
    dcolors = ["#4B9CD3" if v >= 0 else "#D62728" for v in dmeans]
    bars = ax_dec.bar(dlabels, dmeans, color=dcolors, edgecolor="white", lw=0.5, width=0.65)
    for bar, val in zip(bars, dmeans):
        y_txt = val + 3 if val >= 0 else val - 3
        va = "bottom" if val >= 0 else "top"
        ax_dec.text(bar.get_x() + bar.get_width()/2, y_txt, f"{val:+.0f}",
                    ha="center", va=va, fontsize=_FS-4, fontweight="bold")
    ax_dec.axhline(0, color="black", lw=0.8)
    ax_dec.set_ylabel("Mean (Gt yr⁻¹)", fontsize=_FS - 2)
    ax_dec.set_title("(b) Decadal means", fontsize=_FS, loc="left")
    ax_dec.tick_params(labelsize=_FS - 3)
    ax_dec.tick_params(axis="x", rotation=45)

    # --- Panel C: onset timeline ---
    regions = sorted(d.name for d in ensemble_root.iterdir()
                     if d.is_dir() and d.name.startswith("r"))
    onsets = {}
    for r in regions:
        df = load_regional_mwe(ensemble_root, r, group)
        if df is None:
            continue
        onsets[r] = detect_onset(df)
    sorted_r = sorted(onsets.keys(), key=lambda r: (onsets[r] is None, onsets[r] or 9999))

    for i, r in enumerate(sorted_r):
        onset = onsets[r]
        name  = REGION_NAMES.get(r, r)
        grp   = REGION_GROUP.get(r, "Other")
        color = GROUP_COLORS.get(grp, "gray")
        if onset is not None:
            ax_onset.barh(i, onset - HIST_MIN, left=HIST_MIN, height=0.55,
                          color=color, alpha=0.20)
            ax_onset.plot(onset, i, "o", color=color, ms=7, zorder=5)
            ax_onset.plot([onset, onset], [i-0.3, i+0.3], color=color, lw=1.2)
            ax_onset.text(onset+0.8, i, str(onset), va="center", ha="left",
                          fontsize=_FS-5, color="black")
        ax_onset.text(HIST_MIN-1, i, f"  {name}", va="center", ha="right",
                      fontsize=_FS-5)

    ax_onset.axvline(TREND_BREAK, color="black", lw=1.0, ls=":", alpha=0.5)
    ax_onset.set_xlim(HIST_MIN-2, HIST_MAX+4)
    ax_onset.set_yticks([])
    ax_onset.set_xlabel("Year", fontsize=_FS-2)
    ax_onset.set_title("(c) Onset of sustained mass loss", fontsize=_FS, loc="left")
    ax_onset.tick_params(labelsize=_FS-3)
    ax_onset.invert_yaxis()
    legend_patches = [
        mpatches.Patch(color=c, label=g, alpha=0.7)
        for g, c in GROUP_COLORS.items()
        if g in REGION_GROUP.values()
    ]
    ax_onset.legend(handles=legend_patches, fontsize=_FS-5, loc="lower right", framealpha=0.9)

    fig.suptitle("80-Year Record of Glacier Mass Balance — BNF Ensemble\n"
                 "(no-time-encoding, climate-forced reconstruction)",
                 fontsize=_FS+1, y=0.995)
    _savefig(fig, output_path, dpi=200)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_root", default="outputs/egu_fin/ensemble_te_r2")
    parser.add_argument("--group",         default="no_time_encoding")
    parser.add_argument("--output_dir",    default="outputs/egu_fin/acceleration_analysis")
    args = parser.parse_args()

    ensemble_root = Path(args.ensemble_root)
    output_dir    = Path(args.output_dir)

    print(f"\n=== Acceleration analysis — {args.group} ===")

    global_df = load_global_gt_uncertainty(ensemble_root, args.group)
    global_gt = global_df["median_gt"]

    print("  Figure 1: global time series")
    plot_global_ts(global_df, output_dir / "acceleration_global_ts.png")

    print("  Figure 2: onset timeline")
    plot_onset_timeline(ensemble_root, args.group, output_dir / "acceleration_onset_timeline.png")

    print("  Figure 3: 20-year block averages")
    plot_20yr_blocks(global_df, output_dir / "acceleration_20yr_blocks.png")

    print("  Figure 4: decade bars")
    plot_decade_bars(global_gt, output_dir / "acceleration_decade_bars.png")

    print("  Figure 4: combined summary panel")
    plot_combined_summary(ensemble_root, global_df, global_gt,
                          args.group, output_dir / "acceleration_summary_panel.png")

    print(f"\nDone → {output_dir}/")


if __name__ == "__main__":
    main()
