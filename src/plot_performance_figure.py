"""
src/plot_performance_figure.py

Publication figure: model performance against held-out observations, laid out
to the Science Advances figure guide (2026): 2-column width (7.25 in), depth
<= 7.8 in, sans-serif text at 8 pt (6 pt minimum), 9 pt bold capital panel
labels at upper left, line weights >= 0.28 pt, no figure title, no minor ticks
or grid lines, no repeated right-hand y axes, axes limited to the data range,
and no red/green or dark-blue/yellow pairings.

Layout: three time series stacked on the left (shared year axis), three
scatter panels stacked on the right.
  A  Storglaciären (WGMS testing-split glacier) vs. WGMS in-situ
  B  Central Europe (RGI 11) vs. Dussaillant et al., with in-situ glacier counts
  C  Iceland (RGI 06) vs. Dussaillant et al., with in-situ glacier counts
  D  All WGMS testing-split reference/benchmark glaciers, years <2000 and >2020,
     one colour + marker per glacier (colour = RGI region, marker = glacier)
  E  GLaMBIE combined 2020-2024, one colour + marker per RGI region
  F  The same GLaMBIE points as E, coloured by year

Categorical encoding: 26 glaciers / 19 regions are far more than hue alone can
separate, so identity is colour + marker shape (composite encoding). Region
colours are shared between D and E, so a region keeps its colour across panels.

All panels are drawn from the Step 6 testing outputs (not from their PNGs):
  outputs/testing_wgms_glaciers/wgms_test_joined.csv, wgms_test_metrics_*.csv
  outputs/testing_regional_duss/regional_duss_joined.csv, regional_duss_metrics.csv
  outputs/testing_glambie_regional/glambie_test_pairs.csv, glambie_test_metrics.csv
plus validation_data/per_gla/{mass_balance,glacier}.csv for the in-situ counts.
The model value is the ensemble mean (median_mwe column) with ±2σ total
uncertainty, and the metrics are the ones the testing scripts wrote.

Usage:
    python src/plot_performance_figure.py --output_dir outputs/fig_paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.test_regional_duss import load_glacier_counts

WGMS_DIR    = Path("outputs/testing_wgms_glaciers")
DUSS_DIR    = Path("outputs/testing_regional_duss")
GLAMBIE_DIR = Path("outputs/testing_glambie_regional")
PER_GLA_DIR = Path("validation_data/per_gla")

GLACIER_RGI = "RGI60-08.00213"   # Storglaciären
TRAIN_YEAR_MIN, TRAIN_YEAR_MAX = 2000, 2020
YEAR_MIN, YEAR_MAX = 1940, 2024

# Colours: reference categorical palette (CVD-validated order), model/obs = slots 1-2
PALETTE  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
C_MODEL  = PALETTE[0]
C_OBS    = PALETTE[1]
C_WINDOW = "#efeee9"     # finetuning-window band, recessive
C_COUNT  = "#8a8984"     # in-situ count bars (context, not series identity)
C_ERR    = "#c3c2b7"
C_INK    = "#0b0b0b"
C_INK2   = "#52514e"
# Ordinal blue ramp, steps 250-650, for GLaMBIE test years 2020-2024
YEAR_COLORS = {2020: "#86b6ef", 2021: "#5598e7", 2022: "#2a78d6", 2023: "#1c5cab", 2024: "#104281"}

# Region identity = (palette slot, marker). Shape tier by region number
# (1-7 circle, 8-14 square, 15-19 triangle); within a tier every hue is unique,
# and the seven regions with WGMS testing glaciers (2, 6, 7, 8, 11, 13, 15)
# all get different hues so panel D can colour by region with hue alone.
REGION_STYLE = {
    1: (3, "o"), 2: (0, "o"), 3: (4, "o"), 4: (5, "o"), 5: (6, "o"), 6: (1, "o"), 7: (2, "o"),
    8: (3, "s"), 9: (0, "s"), 10: (1, "s"), 11: (4, "s"), 12: (2, "s"), 13: (5, "s"), 14: (6, "s"),
    15: (6, "^"), 16: (0, "^"), 17: (1, "^"), 18: (2, "^"), 19: (3, "^"),
}
GLACIER_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">"]
# Short legend names (two-column legends must fit beside a 2-in panel)
GLACIER_SHORT = {
    "AUSTRE BROEGGERBREEN": "Austre Brøgger", "MIDTRE LOVÉNBREEN": "Midtre Lovén",
    "LANGFJORDJOEKELEN": "Langfjordjøkel", "REMBESDALSKAAKA": "Rembesdalsk.",
    "MARMAGLACIÄREN": "Marmaglaciären", "HELLSTUGUBREEN": "Hellstugubreen",
    "STORGLACIÄREN": "Storglaciären", "RABOTS GLACIÄR": "Rabots glaciär",
    "SOUTH CASCADE": "South Cascade", "PARLUNG NO. 94": "Parlung No. 94",
    "HOFSJÖKULL N": "Hofsjökull N", "HOFSJÖKULL SW": "Hofsjökull SW",
}
REGION_SHORT = {
    1: "Alaska", 2: "W Canada & US", 3: "Arctic Can. N", 4: "Arctic Can. S", 5: "Greenland",
    6: "Iceland", 7: "Svalbard", 8: "Scandinavia", 9: "Russian Arctic", 10: "North Asia",
    11: "C Europe", 12: "Caucasus", 13: "C Asia", 14: "S Asia W", 15: "S Asia E",
    16: "Low lat.", 17: "S Andes", 18: "N Zealand", 19: "Antarctic",
}

UNIT = "m w.e. yr$^{-1}$"
FIG_W, FIG_H = 7.25, 7.7     # 2-column width; depth within the 7.8 in limit
HEADROOM = 0.30              # fraction of each time-series panel reserved for its text

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Myriad Pro", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": C_INK2,
    "axes.labelcolor": C_INK,
    "xtick.color": C_INK2,
    "ytick.color": C_INK2,
    "text.color": C_INK,
    "axes.grid": False,
    "axes.unicode_minus": True,
    "pdf.fonttype": 42,   # keep text editable, not outlined
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(x: float, spec: str = ".2f") -> str:
    """Leading-zero decimal with a true minus sign; '–' for NaN."""
    if x is None or not np.isfinite(x):
        return "–"
    return format(x, spec).replace("-", "−")


def _panel_label(ax, letter: str, x: float = -0.09) -> None:
    ax.text(x, 1.0, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="right")


def _window_band(ax) -> None:
    ax.axvspan(TRAIN_YEAR_MIN - 0.5, TRAIN_YEAR_MAX + 0.5, color=C_WINDOW, lw=0, zorder=0)


def _model_band(ax, yr, mean, std) -> None:
    ok = np.isfinite(mean)
    ax.fill_between(yr[ok], (mean - 2 * std)[ok], (mean + 2 * std)[ok],
                    color=C_MODEL, alpha=0.22, lw=0, zorder=2)
    ax.plot(yr[ok], mean[ok], color=C_MODEL, lw=1.0, zorder=4)


def _data_range(*arrays, pad_frac: float = 0.04) -> tuple[float, float]:
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    pad = pad_frac * (v.max() - v.min())
    return v.min() - pad, v.max() + pad


def _annotate(ax, name: str, metric_lines: list[str], ymin: float, ymax: float) -> None:
    """Name + metrics in a band reserved at the top of the panel, clear of the data."""
    span = ymax - ymin
    ax.set_ylim(ymin, ymax + span * HEADROOM / (1 - HEADROOM))
    ax.text(0.005, 0.985, name, transform=ax.transAxes, fontsize=8, fontweight="bold",
            color=C_INK, va="top", ha="left", zorder=6)
    ax.text(0.005, 0.865, "\n".join(metric_lines), transform=ax.transAxes, fontsize=6,
            color=C_INK, va="top", ha="left", linespacing=1.3, zorder=6)


def _metric_line(prefix: str, n: int, r: float, rmse: float, mae: float, cov: float) -> str:
    return (f"{prefix} (n = {n}): r {_f(r)} | RMSE {_f(rmse)} | "
            f"MAE {_f(mae)} | cov. {cov:.0f}%")


def _scatter_metrics(ax, n: int, r: float, rmse: float, mae: float, cov: float,
                     head: str = "") -> None:
    text = (f"{head}n = {n}\nr = {_f(r)}\nRMSE = {_f(rmse)}\nMAE = {_f(mae)}\n"
            f"cov. = {cov:.0f}%")
    ax.text(0.97, 0.03, text, transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
            linespacing=1.3, zorder=6)


def _square_limits(ax, x, y) -> None:
    lo, hi = _data_range(x, y, pad_frac=0.05)
    ax.plot([lo, hi], [lo, hi], color=C_INK2, lw=0.6, ls=(0, (4, 2)), zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def _side_legend(ax_leg, handles, labels, title: str, ncol: int = 1) -> None:
    ax_leg.axis("off")
    leg = ax_leg.legend(handles, labels, title=title, loc="upper left", frameon=False, ncol=ncol,
                        fontsize=6, title_fontsize=6.5, handletextpad=0.15, labelspacing=0.22,
                        columnspacing=0.5, borderaxespad=0, borderpad=0, handlelength=0.9,
                        markerscale=1.0)
    leg._legend_box.align = "left"


def _marker(color: str, marker: str, ms: float = 3.6) -> Line2D:
    return Line2D([], [], ls="none", marker=marker, ms=ms, mfc=color, mec="white", mew=0.35)


# ---------------------------------------------------------------------------
# Time-series panels (left column)
# ---------------------------------------------------------------------------

def panel_glacier(ax) -> None:
    j = pd.read_csv(WGMS_DIR / "wgms_test_joined.csv")
    g = j[(j["rgi_id"] == GLACIER_RGI) & j["YEAR"].between(YEAR_MIN, YEAR_MAX)].sort_values("YEAR")
    m = pd.read_csv(WGMS_DIR / "wgms_test_metrics_per_glacier.csv")
    m = m[m["rgi_id"] == GLACIER_RGI].iloc[0]

    yr = g["YEAR"].to_numpy()
    mean, std = g["model_mean_mwe"].to_numpy(), g["model_std_total"].to_numpy()
    _window_band(ax)
    _model_band(ax, yr, mean, std)
    ax.plot(yr, g["obs_mwe"], color=C_OBS, lw=0.8, zorder=3)
    ax.plot(yr, g["obs_mwe"], "o", ms=2.6, mfc=C_OBS, mec="white", mew=0.4, ls="none", zorder=5)
    ax.axhline(0, color=C_INK2, lw=0.4, ls=(0, (3, 2)), zorder=1)

    _annotate(ax, "Storglaciären (WGMS testing glacier)",
              [_metric_line("<2000, >2020", int(m["n_years"]), m["corr"],
                            m["rmse"], m["mae"], m["coverage_2sig_pct"])],
              *_data_range(g["obs_mwe"], mean - 2 * std, mean + 2 * std))


def _regional_metric_lines(metrics: pd.DataFrame, code: str) -> list[str]:
    lines = []
    for window, label in [("outside", "<2000, >2020"), ("inside", "2000–2020")]:
        r = metrics[(metrics["region_code"] == code) & (metrics["window"] == window)].iloc[0]
        lines.append(_metric_line(label, int(r["n_points"]), r["r"], r["rmse"], r["mae"],
                                  r["coverage_2sig_pct"]))
    return lines


def panel_region(ax, ax_count, joined: pd.DataFrame, metrics: pd.DataFrame,
                 counts: pd.DataFrame, region_num: int, code: str, name: str) -> None:
    sub = joined[(joined["region_num"] == region_num) &
                 joined["year"].between(YEAR_MIN, YEAR_MAX)].sort_values("year")
    yr = sub["year"].to_numpy()
    obs, obs_s = sub["mwe"].to_numpy(), sub["mwe_sigma"].to_numpy()
    mean, std = sub["model_mean_mwe"].to_numpy(), sub["model_std_total"].to_numpy()

    _window_band(ax)
    ax.fill_between(yr, obs - 2 * obs_s, obs + 2 * obs_s, color=C_OBS, alpha=0.18, lw=0, zorder=1)
    ax.plot(yr, obs, color=C_OBS, lw=0.9, zorder=3)
    _model_band(ax, yr, mean, std)
    ax.axhline(0, color=C_INK2, lw=0.4, ls=(0, (3, 2)), zorder=1)

    _annotate(ax, f"{name} (RGI {region_num:02d}), regional",
              _regional_metric_lines(metrics, code),
              *_data_range(obs - 2 * obs_s, obs + 2 * obs_s, mean - 2 * std, mean + 2 * std))

    cnt = counts[(counts["region_key"] == f"{region_num:02d}") &
                 counts["year"].between(YEAR_MIN, YEAR_MAX)]
    _window_band(ax_count)
    ax_count.bar(cnt["year"], cnt["n_glaciers"], width=0.8, color=C_COUNT, lw=0, zorder=2)
    top = int(np.ceil(cnt["n_glaciers"].max() / 10.0) * 10) if not cnt.empty else 1
    ax_count.set_ylim(0, top * 1.45)          # headroom for the strip's label
    ax_count.set_yticks([0, top])
    ax_count.text(0.005, 0.97, "Glaciers measured in situ", transform=ax_count.transAxes,
                  fontsize=6, color=C_INK, va="top", ha="left")


# ---------------------------------------------------------------------------
# Scatter panels (right column)
# ---------------------------------------------------------------------------

def panel_wgms_scatter(ax, ax_leg) -> None:
    """All WGMS testing-split glaciers, testing years only (<2000, >2020)."""
    j = pd.read_csv(WGMS_DIR / "wgms_test_joined.csv")
    t = j[(j["YEAR"] < TRAIN_YEAR_MIN) | (j["YEAR"] > TRAIN_YEAR_MAX)]
    t = t.dropna(subset=["obs_mwe", "model_mean_mwe"])
    m = pd.read_csv(WGMS_DIR / "wgms_test_metrics_overall.csv").iloc[0]
    assert len(t) == int(m["n_years"]), "testing-window rows differ from the metrics file"

    handles, labels = [], []
    glaciers = (t[["rgi_region", "input_name"]].drop_duplicates()
                .sort_values(["rgi_region", "input_name"]))
    for region, grp in glaciers.groupby("rgi_region", sort=True):
        color = PALETTE[REGION_STYLE[int(region)][0]]
        for k, name in enumerate(grp["input_name"]):
            s = t[t["input_name"] == name]
            mk = GLACIER_MARKERS[k]
            ax.plot(s["obs_mwe"], s["model_mean_mwe"], ls="none", marker=mk, ms=2.8, mfc=color,
                    mec="white", mew=0.3, alpha=0.9, zorder=3)
            handles.append(_marker(color, mk, ms=3.2))
            labels.append(GLACIER_SHORT.get(name, name.title()))

    _square_limits(ax, t["obs_mwe"], t["model_mean_mwe"])
    ax.set_xlabel(f"WGMS observed ({UNIT})")
    ax.set_ylabel(f"Modelled ({UNIT})", labelpad=1)
    ax.text(0.03, 0.97, "WGMS testing glaciers\n<2000, >2020", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", va="top", ha="left")
    _scatter_metrics(ax, int(m["n_years"]), m["corr"], m["rmse"], m["mae"], m["coverage_2sig_pct"])
    _side_legend(ax_leg, handles, labels, "Glacier (colour = region)", ncol=2)


def _glambie_base(ax, p: pd.DataFrame) -> None:
    ax.errorbar(p["glambie_mwe"], p["median_mwe"], xerr=2 * p["glambie_mwe_err"],
                yerr=2 * p["std_total"], fmt="none", ecolor=C_ERR, elinewidth=0.35, capsize=0,
                zorder=2)
    _square_limits(ax, p["glambie_mwe"], p["median_mwe"])
    ax.set_ylabel(f"Modelled ({UNIT})", labelpad=1)


def panel_glambie_region(ax, ax_leg, p: pd.DataFrame, m: pd.Series) -> None:
    _glambie_base(ax, p)
    handles, labels = [], []
    for region in sorted(p["region"].unique(), key=lambda r: int(r[1:])):
        num = int(region[1:])
        slot, mk = REGION_STYLE[num]
        s = p[p["region"] == region]
        ax.plot(s["glambie_mwe"], s["median_mwe"], ls="none", marker=mk, ms=3.4,
                mfc=PALETTE[slot], mec="white", mew=0.35, zorder=3)
        handles.append(_marker(PALETTE[slot], mk))
        labels.append(f"{num:02d} {REGION_SHORT[num]}")
    ax.text(0.03, 0.97, "GLaMBIE, 2020–2024\nby region", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", va="top", ha="left")
    ax.tick_params(labelbottom=False)
    _scatter_metrics(ax, int(m["n"]), m["corr"], m["rmse"], m["mae"], m["coverage_2sig_pct"])
    _side_legend(ax_leg, handles, labels, "RGI region", ncol=2)


def panel_glambie_year(ax, ax_leg, p: pd.DataFrame) -> None:
    _glambie_base(ax, p)
    handles, labels = [], []
    for year, colour in YEAR_COLORS.items():
        s = p[p["year"] == year]
        ax.plot(s["glambie_mwe"], s["median_mwe"], "o", ms=3.4, mfc=colour, mec="white",
                mew=0.35, ls="none", zorder=3)
        handles.append(_marker(colour, "o"))
        labels.append(str(year))
    ax.text(0.03, 0.97, "GLaMBIE, 2020–2024\nby year", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", va="top", ha="left")
    ax.set_xlabel(f"GLaMBIE observed ({UNIT})")
    _side_legend(ax_leg, handles, labels, "Year")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build(output_dir: Path) -> None:
    joined  = pd.read_csv(DUSS_DIR / "regional_duss_joined.csv")
    metrics = pd.read_csv(DUSS_DIR / "regional_duss_metrics.csv")
    counts  = load_glacier_counts(PER_GLA_DIR)
    gl_pairs = pd.read_csv(GLAMBIE_DIR / "glambie_test_pairs.csv")
    gl_metrics = pd.read_csv(GLAMBIE_DIR / "glambie_test_metrics.csv").iloc[0]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[0.8, 1.0], wspace=0.2,
                     left=0.075, right=0.995, top=0.965, bottom=0.075)
    left = GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[0],
                                   height_ratios=[1.0, 1.0, 0.3, 1.0, 0.3], hspace=0.1)
    right = GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[1], width_ratios=[1.0, 0.98],
                                    hspace=0.16, wspace=0.05)

    ax_a  = fig.add_subplot(left[0])
    ax_b  = fig.add_subplot(left[1], sharex=ax_a)
    ax_bc = fig.add_subplot(left[2], sharex=ax_a)
    ax_c  = fig.add_subplot(left[3], sharex=ax_a)
    ax_cc = fig.add_subplot(left[4], sharex=ax_a)
    ax_d, ax_dl = fig.add_subplot(right[0, 0]), fig.add_subplot(right[0, 1])
    ax_e, ax_el = fig.add_subplot(right[1, 0]), fig.add_subplot(right[1, 1])
    ax_f, ax_fl = fig.add_subplot(right[2, 0]), fig.add_subplot(right[2, 1])

    panel_glacier(ax_a)
    panel_region(ax_b, ax_bc, joined, metrics, counts, 11, "CEU", "Central Europe")
    panel_region(ax_c, ax_cc, joined, metrics, counts, 6, "ISL", "Iceland")
    panel_wgms_scatter(ax_d, ax_dl)
    panel_glambie_region(ax_e, ax_el, gl_pairs, gl_metrics)
    panel_glambie_year(ax_f, ax_fl, gl_pairs)

    for ax in (ax_a, ax_b, ax_bc, ax_c):
        ax.tick_params(labelbottom=False)
    ax_a.set_xlim(YEAR_MIN - 0.5, YEAR_MAX + 0.5)
    ax_cc.set_xlabel("Year")

    # Extra gap between B's count strip and C so the strip reads as part of B
    shift = 0.02
    for ax in (ax_c, ax_cc):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - shift, pos.width, pos.height])

    # One shared y label for the three mass-balance panels (not repeated per panel)
    y_mid = (ax_a.get_position().y1 + ax_c.get_position().y0) / 2
    fig.text(0.004, y_mid, f"Annual mass balance ({UNIT})", rotation=90, va="center",
             ha="left", fontsize=8)

    for ax, letter in [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")]:
        _panel_label(ax, letter)
    for ax, letter in [(ax_d, "D"), (ax_e, "E"), (ax_f, "F")]:
        _panel_label(ax, letter, x=-0.2)

    # Single key for panels A-C, above panel A
    handles = [
        (mpatches.Patch(facecolor=C_MODEL, alpha=0.22, lw=0), Line2D([], [], color=C_MODEL, lw=1.0)),
        (mpatches.Patch(facecolor=C_OBS, alpha=0.18, lw=0), Line2D([], [], color=C_OBS, lw=0.9)),
        Line2D([], [], color=C_OBS, lw=0.8, marker="o", ms=2.6, mfc=C_OBS, mec="white", mew=0.4),
        mpatches.Patch(facecolor=C_WINDOW, lw=0),
    ]
    labels = ["Ensemble mean ± 2σ", "Dussaillant et al. ± 2σ", "WGMS in situ", "Finetuning window"]
    a_pos = ax_a.get_position()
    fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               loc="lower left", bbox_to_anchor=(a_pos.x0, a_pos.y1 + 0.005),
               ncol=4, frameon=False, fontsize=6, handlelength=1.6, columnspacing=0.9,
               handletextpad=0.35, borderaxespad=0)

    _check_overflow(fig, [ax_a, ax_b, ax_c], [ax_d, ax_e, ax_f])

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in [("pdf", {}),
                    ("png", {"dpi": 600}),
                    ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        out = output_dir / f"performance_evaluation.{ext}"
        fig.savefig(out, **kw)
        print(f"  Saved {out}")
    plt.close(fig)


def _check_overflow(fig, left_axes, right_axes) -> None:
    """Warn if any text/legend leaves the figure or left-column text reaches the right column."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fig_bb = fig.bbox
    right_x0 = min(ax.get_tightbbox(r).x0 for ax in right_axes)
    for artist in fig.findobj(lambda a: hasattr(a, "get_window_extent") and
                              a.__class__.__name__ in ("Text", "Legend") and a.get_visible()):
        try:
            bb = artist.get_window_extent(r)
        except Exception:
            continue
        if bb.width == 0:
            continue
        label = getattr(artist, "get_text", lambda: "legend")()[:30]
        if bb.x1 > fig_bb.x1 + 0.5 or bb.y0 < fig_bb.y0 - 0.5 or bb.x0 < fig_bb.x0 - 0.5:
            print(f"  WARNING: outside figure: {label!r} x1={bb.x1 / fig.dpi:.2f} in")
    for ax in left_axes:
        for t in ax.texts:
            bb = t.get_window_extent(r)
            if bb.x1 > right_x0:
                print(f"  WARNING: left text reaches right column: {t.get_text()[:30]!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Publication performance-evaluation figure (A-F).")
    p.add_argument("--output_dir", default="outputs/fig_paper")
    build(Path(p.parse_args().output_dir))


if __name__ == "__main__":
    main()
