"""
src/plot_model_animations.py

Three racing animations from no_time_encoding ensemble outputs.

Outputs (saved to --output_dir):
  racing_cumulative_gt.gif   cumulative Gt per region (left axis) +
                             global (right axis, different scale)
  racing_bar_gt.gif          annual Gt/yr per region (sorted bars, racing)
  racing_bar_mwe.gif         annual MWE/yr per region (sorted bars, racing)

Usage:
    python src/plot_model_animations.py \\
        --ensemble_root outputs/egu_fin/ensemble_te_r2 \\
        --output_dir    outputs/egu_fin/model_estimates_complete
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Region metadata — Paul Tol "Muted"/"Bright" blend (colorblind-aware)
# ---------------------------------------------------------------------------

REGION_META: dict[str, tuple[str, str]] = {
    "r01": ("Alaska",              "#0072B2"),
    "r02": ("W. Canada & US",      "#56B4E9"),
    "r03": ("Arctic Canada N",     "#009E73"),
    "r04": ("Arctic Canada S",     "#44AA99"),
    "r05": ("Greenland periph.",   "#117733"),
    "r06": ("Iceland",             "#88CCEE"),
    "r07": ("Svalbard & J.M.",     "#332288"),
    "r08": ("Scandinavia",         "#DDCC77"),
    "r09": ("Russian Arctic",      "#CC6677"),
    "r10": ("North Asia",          "#AA4499"),
    "r11": ("Central Europe",      "#E69F00"),
    "r12": ("Caucasus",            "#D55E00"),
    "r13": ("Central Asia",        "#882255"),
    "r14": ("S. Asia West",        "#CC79A7"),
    "r15": ("S. Asia East",        "#994455"),
    "r16": ("Low Latitudes",       "#AA3377"),
    "r17": ("Southern Andes",      "#BBCC33"),
    "r18": ("New Zealand",         "#66AAEE"),
    "r19": ("Antarctic & Sub.",    "#555555"),
}
REGIONS = list(REGION_META.keys())

# One distinct marker per region (19 total)
REGION_MARKERS: dict[str, str] = {
    "r01": "o",   # circle
    "r02": "s",   # square
    "r03": "^",   # triangle up
    "r04": "v",   # triangle down
    "r05": "<",   # triangle left
    "r06": ">",   # triangle right
    "r07": "D",   # diamond
    "r08": "d",   # thin diamond
    "r09": "p",   # pentagon
    "r10": "h",   # hexagon
    "r11": "H",   # rotated hexagon
    "r12": "*",   # star
    "r13": "P",   # filled plus
    "r14": "X",   # filled x
    "r15": "8",   # octagon
    "r16": "1",   # tri_down (Y shape)
    "r17": "2",   # tri_up
    "r18": "3",   # tri_left
    "r19": "4",   # tri_right
}

# Font sizes — match val_reg_egu (fontscale = 2)
LBL_FS  = 20
TICK_FS = 18
LEG_FS  = 13
TTL_FS  = 20
YEAR_FS = 42
VAL_FS  = 11

START_YEAR = 1940
END_HOLD   = 20        # frames held at end
FPS        = 5         # default fps (cumulative + % bar)
FPS_MWE    = 3         # slower fps for MWE bar (333 ms/frame)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all(ensemble_root: Path, group: str) -> tuple[dict, dict, dict]:
    gt, mwe, gt_std = {}, {}, {}
    for r in REGIONS:
        p_gt  = ensemble_root / r / group / "ensemble_regional_gt.csv"
        p_mwe = ensemble_root / r / group / "ensemble_regional_mwe.csv"
        if p_gt.exists():
            df = pd.read_csv(p_gt).set_index("year")
            gt[r]     = df["median_gt"]
            gt_std[r] = df["std_total"]
        if p_mwe.exists():
            mwe[r] = pd.read_csv(p_mwe).set_index("year")["median_mwe"]
    return gt, mwe, gt_std


# ---------------------------------------------------------------------------
# Save helper — MP4 if ffmpeg available, else GIF
# ---------------------------------------------------------------------------

def _save(anim: FuncAnimation, path: Path, fps: int, dpi: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = manim.FFMpegWriter(
            fps=fps, bitrate=2000,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        out = path.with_suffix(".mp4")
        anim.save(str(out), writer=writer, dpi=dpi)
        print(f"  Saved {out.name}")
    except Exception as e:
        warnings.warn(f"FFMpeg not available ({e}); saving as GIF.")
        out = path.with_suffix(".gif")
        anim.save(str(out), writer=manim.PillowWriter(fps=fps), dpi=dpi)
        print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# 1. Racing cumulative line plot — dual y-axis, zero-aligned
# ---------------------------------------------------------------------------

def anim_cumulative(
    gt: dict,
    output_path: Path,
    start_year: int = START_YEAR,
    static_lw: bool = True,
    gt_std: dict | None = None,
    year_txt_loc: tuple[float, float] = (0.015, 0.97),
) -> None:
    """
    Regional lines on left y-axis; global total on right y-axis.
    Both axes share the same zero level (proportional limits).

    start_year   : first year of cumulative sum (default START_YEAR = 1940).
    static_lw    : if True, line widths are fixed at 2025-proportional values.
    gt_std       : per-region std_total series; when supplied, a ±2σ band is
                   drawn around the global cumulative line.
    year_txt_loc : axes-fraction (x, y) for the year counter label.
    """
    years   = list(range(start_year, 2026))
    n_years = len(years)

    # Build cumulative from start_year for each region
    cum: dict[str, np.ndarray] = {}
    for r, series in gt.items():
        s = series.reindex(years, fill_value=0.0)
        cum[r] = s.cumsum().values
    global_cum = np.sum(list(cum.values()), axis=0)

    # Cumulative global ±2σ uncertainty (regions independent, years independent)
    global_std_cum = None
    if gt_std:
        annual_var = np.zeros(n_years)
        for r, std_series in gt_std.items():
            if r not in cum:
                continue
            s = std_series.reindex(years, fill_value=0.0)
            annual_var += s.values ** 2
        global_std_cum = np.sqrt(np.cumsum(annual_var))

    # Compute zero-aligned axis limits
    regional_vals = np.concatenate(list(cum.values()))
    reg_lo_raw = regional_vals.min()
    reg_hi_raw = max(regional_vals.max(), abs(reg_lo_raw) * 0.05)
    glo_lo_raw = global_cum.min()
    glo_hi_raw = max(global_cum.max(), abs(glo_lo_raw) * 0.05)

    pad = 1.12
    reg_lo = reg_lo_raw * pad
    glo_lo = glo_lo_raw * pad

    k = max(reg_hi_raw / abs(reg_lo),
            glo_hi_raw / abs(glo_lo)) * 1.18
    reg_hi = abs(reg_lo) * k
    glo_hi = abs(glo_lo) * k

    # 2025 proportional linewidths (used for both static and legend)
    abs_final = {r: abs(cum[r][-1]) for r in cum}
    max_final = max(abs_final.values()) or 1.0
    lw_for    = {r: 0.4 + (abs_final[r] / max_final) * 3.4 for r in cum}

    # Figure
    fig, ax = plt.subplots(figsize=(17, 9))
    fig.patch.set_facecolor("white")
    ax2 = ax.twinx()

    # 10-year tick positions for markers
    tick_years   = [y for y in years if y % 10 == 0]
    tick_indices = [years.index(y) for y in tick_years]   # index into cum arrays

    line_artists:   dict[str, plt.Line2D] = {}
    marker_artists: dict[str, plt.Line2D] = {}
    for r in REGIONS:
        if r not in cum:
            continue
        name, color = REGION_META[r]
        mkr_style = REGION_MARKERS[r]
        init_lw = lw_for[r] if static_lw else 1.8
        (ln,) = ax.plot([], [], color=color, lw=init_lw, alpha=0.88)
        (mkr,) = ax.plot([], [], color=color, marker=mkr_style, markersize=7,
                          linestyle="none", alpha=0.95, zorder=5)
        line_artists[r]   = ln
        marker_artists[r] = mkr

    (global_ln,)  = ax2.plot([], [], color="black", lw=4.5, zorder=10)
    (global_mkr,) = ax2.plot([], [], color="black", marker="D", markersize=7,
                              linestyle="none", zorder=11)

    year_txt = ax.text(
        year_txt_loc[0], year_txt_loc[1], "", transform=ax.transAxes,
        fontsize=YEAR_FS, fontweight="bold", va="top", ha="left", color="#222222",
    )

    # Mutable container so update() can remove and redraw the uncertainty band
    glo_band = [None]

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.45, zorder=0)
    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(reg_lo, reg_hi)
    ax2.set_ylim(glo_lo, glo_hi)

    ax.set_xlabel("Year", fontsize=LBL_FS)
    ax.set_ylabel("Regional cumulative mass change (Gt)", fontsize=LBL_FS,
                   color="#444444")
    ax2.set_ylabel("Global cumulative mass change (Gt)", fontsize=LBL_FS,
                    color="black", labelpad=12)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax2.tick_params(axis="y",   labelsize=TICK_FS, colors="black")
    ax2.spines["right"].set_color("black")
    ax.set_title(f"Cumulative glacier mass change from {start_year}", fontsize=TTL_FS)
    ax.spines["top"].set_visible(False)

    # Custom legend handles: line + region-specific marker for each region
    custom_handles = [
        Line2D([0], [0], color=REGION_META[r][1],
               lw=lw_for[r], marker=REGION_MARKERS[r],
               markersize=7, label=REGION_META[r][0])
        for r in REGIONS if r in line_artists
    ] + [
        Line2D([0], [0], color="black", lw=4.5, marker="D",
               markersize=7, label="Global total")
    ]
    leg = ax.legend(handles=custom_handles,
                    fontsize=LEG_FS, ncol=4, loc="lower left",
                    framealpha=0.93, edgecolor="lightgray")

    ax.annotate("Line width ∝ cumulative contribution (2025 values)",
                xy=(0.015, 0.025), xycoords="axes fraction",
                fontsize=LEG_FS - 4, color="#666666", style="italic")

    fig.tight_layout()

    years_arr = np.array(years)
    n_frames  = n_years + END_HOLD

    def update(frame):
        i = min(frame + 1, n_years)
        x = years_arr[:i]

        # Indices of 10-year ticks that have been reached so far
        vis_ticks = [(yi, y) for yi, y in zip(tick_indices, tick_years) if yi < i]
        tx = [y  for yi, y in vis_ticks]

        if i > 0:
            if not static_lw:
                abs_vals = {r: abs(cum[r][i - 1]) for r in line_artists}
                max_abs  = max(abs_vals.values()) or 1.0
            for r, ln in line_artists.items():
                if not static_lw:
                    frac = abs_vals[r] / max_abs
                    ln.set_linewidth(0.4 + frac * 3.4)
                ln.set_data(x, cum[r][:i])
                ty = [cum[r][yi] for yi, _ in vis_ticks]
                marker_artists[r].set_data(tx, ty)
        else:
            for r in line_artists:
                line_artists[r].set_data([], [])
                marker_artists[r].set_data([], [])

        global_ln.set_data(x, global_cum[:i])
        gty = [global_cum[yi] for yi, _ in vis_ticks]
        global_mkr.set_data(tx, gty)

        # Update ±2σ global uncertainty band
        if global_std_cum is not None and i > 1:
            if glo_band[0] is not None:
                glo_band[0].remove()
            glo_band[0] = ax2.fill_between(
                x,
                global_cum[:i] - 2 * global_std_cum[:i],
                global_cum[:i] + 2 * global_std_cum[:i],
                alpha=0.18, color="black", zorder=7,
            )

        year_txt.set_text(str(int(years_arr[min(frame, n_years - 1)])))

    anim = FuncAnimation(fig, update, frames=n_frames,
                          blit=False, interval=1000 // FPS)
    _save(anim, output_path, FPS)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Racing bar — annual Gt/yr mass balance per region
# ---------------------------------------------------------------------------

_COL_NEG = (0.80, 0.15, 0.15, 0.60)   # red,  40 % transparent
_COL_POS = (0.15, 0.35, 0.80, 0.60)   # blue, 40 % transparent


def _bar_col(v: float) -> tuple:
    return _COL_NEG if v < 0 else _COL_POS


def anim_gt_bar(gt: dict, output_path: Path) -> None:
    """
    Fixed-order horizontal bars: annual mass balance (Gt/yr) per region.
    Symlog x-axis; red = negative (loss), blue = positive (gain).
    Sort order fixed by average Gt (ascending → most loss at top).
    """
    years  = list(range(START_YEAR, 2026))
    active = [r for r in REGIONS if r in gt]

    gt_by_year: dict[int, dict[str, float]] = {
        y: {r: float(gt[r].get(y, 0.0)) for r in active}
        for y in years
    }

    avg_gt = {r: np.mean([gt_by_year[y][r] for y in years]) for r in active}
    sreg   = sorted(active, key=lambda r: avg_gt[r])
    names  = [REGION_META[r][0] for r in sreg]
    n      = len(sreg)

    all_vals = [v for yd in gt_by_year.values() for v in yd.values()]
    xlo      = min(all_vals) * 1.15
    xhi      = max(max(all_vals) * 1.5, abs(xlo) * 0.08)
    linthresh = max(abs(xlo), xhi) * 0.01

    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.92, bottom=0.09)

    init_vals = [gt_by_year[years[0]][r] for r in sreg]
    bars = ax.barh(
        range(n), [abs(v) for v in init_vals],
        left=[min(0.0, v) for v in init_vals],
        color=[_bar_col(v) for v in init_vals],
        edgecolor="white", linewidth=0.4, height=0.70,
    )
    val_texts = [
        ax.text(0.0, i, "", va="center", ha="left", fontsize=VAL_FS, color="#1a1a1a")
        for i in range(n)
    ]

    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=TICK_FS)
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Mass Balance (Gt/yr)", fontsize=LBL_FS)
    ax.set_title("Annual glacier mass balance by region (Gt/yr)", fontsize=TTL_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.axvline(0, color="black", lw=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    year_txt = ax.text(
        0.97, 0.03, str(years[0]), transform=ax.transAxes,
        fontsize=YEAR_FS, fontweight="bold", ha="right", va="bottom",
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cccccc", alpha=0.92),
    )

    n_frames = len(years) + END_HOLD

    def update(frame):
        y  = years[min(frame, len(years) - 1)]
        vd = gt_by_year[y]
        for i, r in enumerate(sreg):
            v = vd[r]
            bars[i].set_x(min(0.0, v))
            bars[i].set_width(abs(v))
            bars[i].set_facecolor(_bar_col(v))
            off = max(abs(v) * 0.08, linthresh * 0.5)
            if abs(v) > linthresh * 0.1:
                xpos = v + off if v >= 0 else v - off
                ha   = "left" if v >= 0 else "right"
                val_texts[i].set_position((xpos, i))
                val_texts[i].set_text(f"{v:.1f}")
                val_texts[i].set_ha(ha)
            else:
                val_texts[i].set_text("")
        year_txt.set_text(str(y))

    anim = FuncAnimation(fig, update, frames=n_frames,
                          blit=False, interval=1000 // FPS_MWE)
    _save(anim, output_path, FPS_MWE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Racing bar — MWE/yr per region
# ---------------------------------------------------------------------------

def anim_mwe_bar(mwe: dict, output_path: Path) -> None:
    """
    Fixed-order horizontal bars: annual mass balance rate (m.w.e/yr).
    Symlog x-axis; red = negative (loss), blue = positive (gain).
    Sort order fixed by average MWE (ascending → most loss at top).
    """
    years  = list(range(START_YEAR, 2026))
    active = [r for r in REGIONS if r in mwe]

    mwe_by_year: dict[int, dict[str, float]] = {}
    for y in years:
        mwe_by_year[y] = {r: float(mwe[r].get(y, 0.0)) for r in active}

    avg_mwe = {r: np.mean([mwe_by_year[y][r] for y in years]) for r in active}
    sreg    = sorted(active, key=lambda r: avg_mwe[r])
    names   = [REGION_META[r][0] for r in sreg]
    n       = len(sreg)

    all_vals  = [v for yd in mwe_by_year.values() for v in yd.values()]
    xlo       = min(all_vals) * 1.15
    xhi       = max(max(all_vals) * 1.5, abs(xlo) * 0.08)
    linthresh = max(abs(xlo), xhi) * 0.01

    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.92, bottom=0.09)

    init_vals = [mwe_by_year[years[0]].get(r, 0.0) for r in sreg]
    bars = ax.barh(
        range(n), [abs(v) for v in init_vals],
        left=[min(0.0, v) for v in init_vals],
        color=[_bar_col(v) for v in init_vals],
        edgecolor="white", linewidth=0.4, height=0.70,
    )
    val_texts = [
        ax.text(0.0, i, "", va="center", ha="left", fontsize=VAL_FS, color="#1a1a1a")
        for i in range(n)
    ]

    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=TICK_FS)
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Mass Balance (m.w.e/yr)", fontsize=LBL_FS)
    ax.set_title("Glacier mass balance rate by region (m.w.e/yr)", fontsize=TTL_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.axvline(0, color="black", lw=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    year_txt = ax.text(
        0.97, 0.03, str(years[0]), transform=ax.transAxes,
        fontsize=YEAR_FS, fontweight="bold", ha="right", va="bottom",
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cccccc", alpha=0.92),
    )

    n_frames = len(years) + END_HOLD

    def update(frame):
        y  = years[min(frame, len(years) - 1)]
        vd = mwe_by_year[y]
        for i, r in enumerate(sreg):
            v = vd.get(r, 0.0)
            bars[i].set_x(min(0.0, v))
            bars[i].set_width(abs(v))
            bars[i].set_facecolor(_bar_col(v))
            off = max(abs(v) * 0.08, linthresh * 0.5)
            if abs(v) > linthresh * 0.1:
                xpos = v + off if v >= 0 else v - off
                ha   = "left" if v >= 0 else "right"
                val_texts[i].set_position((xpos, i))
                val_texts[i].set_text(f"{v:.2f}")
                val_texts[i].set_ha(ha)
            else:
                val_texts[i].set_text("")
        year_txt.set_text(str(y))

    anim = FuncAnimation(fig, update, frames=n_frames,
                          blit=False, interval=1000 // FPS_MWE)
    _save(anim, output_path, FPS_MWE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Racing animated plots of glacier ensemble estimates."
    )
    parser.add_argument("--ensemble_root", default="outputs/egu_fin/ensemble_te_r2")
    parser.add_argument("--group",         default="no_time_encoding")
    parser.add_argument("--output_dir",    default="outputs/egu_fin/model_estimates_complete")
    args = parser.parse_args()

    ensemble_root = Path(args.ensemble_root)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Racing animations ({args.group}) ===")
    print(f"  Ensemble root : {ensemble_root}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Year range    : {START_YEAR}–2025  |  FPS: {FPS}  |  "
          f"End hold: {END_HOLD} frames (~{END_HOLD//FPS}s)\n")

    gt, mwe, gt_std = load_all(ensemble_root, args.group)
    print(f"  Loaded {len(gt)} GT series, {len(mwe)} MWE series\n")

    print("[1/4] Cumulative line animation — static lw, from 1940 ...")
    anim_cumulative(gt, output_dir / "racing_cumulative_gt",
                    start_year=START_YEAR, static_lw=True)

    print("[2/4] Cumulative line animation — static lw, from 1975 ...")
    anim_cumulative(gt, output_dir / "racing_cumulative_gt_from1975",
                    start_year=1975, static_lw=True,
                    year_txt_loc=(0.015, 0.34))

    print(f"[3/4] Annual Gt/yr bar animation ({FPS} fps) ...")
    anim_gt_bar(gt, output_dir / "racing_bar_gt")

    print(f"[4/4] MWE/yr bar animation ({FPS_MWE} fps) ...")
    anim_mwe_bar(mwe, output_dir / "racing_bar_mwe")

    print(f"\nDone. Files in {output_dir}/")


if __name__ == "__main__":
    main()
