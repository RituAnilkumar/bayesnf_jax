"""
src/plot_performance_figure_v2.py

Version 2 of the performance-evaluation figure (src/plot_performance_figure.py is unchanged; panel A and the
helpers are imported from it). Same Science Advances layout rules (2-column width, 8 pt text, 6 pt minimum,
9 pt bold panel labels, lines >= 0.28 pt, no titles, no minor ticks).

  A    Storglaciären (WGMS testing glacier) vs. WGMS in situ, as in version 1
  B-C  Central Europe and Iceland vs. Dussaillant et al.; the number of glaciers measured in situ is drawn as light
       grey bars in the same panel, on a light-grey right-hand axis (version 1 used a separate strip)
  D    Heat map of the WGMS testing glaciers, testing years (<2000, >2020), glaciers with at least MIN_N
       glacier-years only (24 glaciers, 779 glacier-years): one row per glacier, numbered and grouped by RGI region
       with a bracket; columns r, coverage of the ±2σ interval, RMSE, MAE and n. One diverging colormap for every
       column (red = worse, blue = better); each column has its own scale, shown by a key bar per metric. Bottom
       row: the retained glaciers pooled (recomputed with test_wgms_glaciers._metrics).
  E    GLaMBIE 2020-2024: standardised residual z = (model - GLaMBIE) / sqrt(σ_model² + σ_GLaMBIE²) (1σ each),
       all years of a region at the same x; reference lines at 0 and ±2.

Inputs (Step 6 testing outputs):
  outputs/testing_wgms_glaciers/wgms_test_joined.csv, wgms_test_metrics_per_glacier.csv, wgms_test_metrics_overall.csv
  outputs/testing_glambie_regional/glambie_test_pairs.csv  (median_mwe, std_total, glambie_mwe, glambie_mwe_err; 1σ)
  outputs/testing_regional_duss/regional_duss_joined.csv, regional_duss_metrics.csv; validation_data/per_gla

Usage:
    python src/plot_performance_figure_v2.py --output_dir outputs/fig_paper
Writes performance_evaluation_v2.{pdf,png,tiff}, performance_evaluation_v2_glacier_key.csv (row -> glacier) and
performance_evaluation_v2_pooled_metrics.csv (pooled metrics of the retained glaciers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plot_performance_figure import (  # noqa: E402  (also applies its rcParams)
    WGMS_DIR, DUSS_DIR, GLAMBIE_DIR, PER_GLA_DIR, YEAR_MIN, YEAR_MAX, C_MODEL, C_OBS, C_WINDOW, C_INK, C_INK2,
    YEAR_COLORS, REGION_SHORT, FIG_W, FIG_H, _f, _panel_label, _marker, _model_band, _data_range,
    GLACIER_RGI,
)
from src.test_regional_duss import load_glacier_counts  # noqa: E402
from src.test_wgms_glaciers import _metrics  # noqa: E402
from src.test_regional_duss import _compute_metrics as _duss_metrics  # noqa: E402

# Test years. Fine-tuning uses Hugonnet 2000-2019 (period mask year < 2020) and GlaMBIE without 2000, 2001 and
# 2020+, so every year before 2000 and from 2020 on is independent of fine-tuning. The Step 6 testing scripts used
# <2000 and >2020; the metrics here are recomputed from their joined files with 2020 included, and the scripts'
# own window is asserted to reproduce their CSVs first.
FT_FIRST, FT_LAST = 2000, 2019
OLD_LAST = 2020


def _is_test(year, last=FT_LAST):
    return (year < FT_FIRST) | (year > last)


def _window_band(ax) -> None:
    ax.axvspan(FT_FIRST - 0.5, FT_LAST + 0.5, color=C_WINDOW, lw=0, zorder=0)


def _wgms_glacier_metrics(j: pd.DataFrame, rgi_id: str, last=FT_LAST) -> dict:
    g = j[(j["rgi_id"] == rgi_id) & _is_test(j["YEAR"], last)].dropna(subset=["obs_mwe", "model_mean_mwe"])
    return _metrics(g["obs_mwe"].to_numpy(), g["model_mean_mwe"].to_numpy(), g["model_std_total"].to_numpy())


C_COUNT, C_COUNT_AX = "#b3b2ab", "#8a8984"   # in-situ count bars and their axis: light grey, recessive
CMAP = plt.get_cmap("RdBu")                  # red = worse, blue = better; safe for colour-deficient vision
MIN_N = 10                                   # glaciers with fewer testing glacier-years are not shown in D

# Heat-map columns. Each scale runs from its worst value (red, left end of its key) to its best (blue, right end).
#   (column, header, cell format, worst, best, log scale, key label, key end labels)
COLS = [
    ("corr", "r", lambda v: _f(v), -1.0, 1.0, False, "r", ("−1", "1")),
    ("coverage_2sig_pct", "Cov. (%)", lambda v: f"{v:.0f}", 50.0, 100.0, False, "Coverage (%)", ("50", "100")),
    ("rmse", "RMSE", lambda v: _f(v), 1.4, 0.0, False, "RMSE", ("1.4", "0")),
    ("mae", "MAE", lambda v: _f(v), 1.2, 0.0, False, "MAE", ("1.2", "0")),
    ("n_years", "n", lambda v: f"{int(v)}", 10.0, 70.0, True, "n", ("10", "≥70")),
]


def _score(values: np.ndarray, worst: float, best: float, log: bool) -> np.ndarray:
    """0 at the worst end of the scale, 1 at the best, clipped."""
    v = np.asarray(values, dtype=float)
    if log:
        v, worst, best = np.log(v), np.log(worst), np.log(best)
    return np.clip((v - worst) / (best - worst), 0.0, 1.0)


def _colour(score):
    return CMAP(0.1 + 0.8 * np.asarray(score))     # avoid the darkest ends so printed values stay legible


# ---------------------------------------------------------------------------
# A-C: name, testing-window metrics and an in-panel key in a band reserved at the top of each panel
#   (only the years outside the 2000-2020 finetuning window are scored; n = number of those years)

def _metric_text(n: int, r: float, rmse: float, mae: float, cov: float) -> str:
    return f"n = {n} | r {_f(r)} | RMSE {_f(rmse)} | MAE {_f(mae)} | cov. {cov:.0f}%"


def _header(ax, name: str, metric: str, handles, labels, ncol: int, ymin: float, ymax: float,
            headroom: float) -> None:
    span = ymax - ymin
    ax.set_ylim(ymin, ymax + span * headroom / (1 - headroom))
    ax.text(0.005, 0.985, name, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left",
            zorder=6)
    ax.text(0.005, 0.885, metric, transform=ax.transAxes, fontsize=6, va="top", ha="left", zorder=6)
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="upper left",
              bbox_to_anchor=(0.0, 0.815), ncol=ncol, frameon=False, fontsize=6, handlelength=1.6,
              columnspacing=0.9, handletextpad=0.35, labelspacing=0.25, borderaxespad=0.2, borderpad=0)


H_MODEL = (mpatches.Patch(facecolor=C_MODEL, alpha=0.22, lw=0), Line2D([], [], color=C_MODEL, lw=1.0))
H_DUSS = (mpatches.Patch(facecolor=C_OBS, alpha=0.18, lw=0), Line2D([], [], color=C_OBS, lw=0.9))
H_WGMS = Line2D([], [], color=C_OBS, lw=0.8, marker="o", ms=2.6, mfc=C_OBS, mec="white", mew=0.4)
H_WINDOW = mpatches.Patch(facecolor=C_WINDOW, lw=0)
H_COUNT = mpatches.Patch(facecolor=C_COUNT, lw=0)


def panel_glacier_v2(ax) -> None:
    j = pd.read_csv(WGMS_DIR / "wgms_test_joined.csv")
    g = j[(j["rgi_id"] == GLACIER_RGI) & j["YEAR"].between(YEAR_MIN, YEAR_MAX)].sort_values("YEAR")
    m = _wgms_glacier_metrics(j, GLACIER_RGI)
    yr = g["YEAR"].to_numpy()
    mean, std = g["model_mean_mwe"].to_numpy(), g["model_std_total"].to_numpy()
    _window_band(ax)
    _model_band(ax, yr, mean, std)
    ax.plot(yr, g["obs_mwe"], color=C_OBS, lw=0.8, zorder=3)
    ax.plot(yr, g["obs_mwe"], "o", ms=2.6, mfc=C_OBS, mec="white", mew=0.4, ls="none", zorder=5)
    ax.axhline(0, color=C_INK2, lw=0.4, ls=(0, (3, 2)), zorder=1)
    _header(ax, "Storglaciären (WGMS testing glacier)",
            _metric_text(int(m["n_years"]), m["corr"], m["rmse"], m["mae"], m["coverage_2sig_pct"]),
            [H_MODEL, H_WGMS, H_WINDOW], ["Ensemble mean ± 2σ", "WGMS in situ", "Finetuning window"], 2,
            *_data_range(g["obs_mwe"], mean - 2 * std, mean + 2 * std), headroom=0.27)


# B, C: regional series with in-situ counts on a right-hand axis
# ---------------------------------------------------------------------------

def panel_region_counts(ax, joined, metrics, counts, region_num: int, code: str, name: str) -> None:
    sub = joined[(joined["region_num"] == region_num) &
                 joined["year"].between(YEAR_MIN, YEAR_MAX)].sort_values("year")
    yr = sub["year"].to_numpy()
    obs, obs_s = sub["mwe"].to_numpy(), sub["mwe_sigma"].to_numpy()
    mean, std = sub["model_mean_mwe"].to_numpy(), sub["model_std_total"].to_numpy()

    axc = ax.twinx()                          # counts behind the series
    ax.set_zorder(axc.get_zorder() + 1)
    ax.patch.set_visible(False)
    _window_band(axc)
    cnt = counts[(counts["region_key"] == f"{region_num:02d}") & counts["year"].between(YEAR_MIN, YEAR_MAX)]
    axc.bar(cnt["year"], cnt["n_glaciers"], width=0.8, color=C_COUNT, lw=0, zorder=2)
    top = int(np.ceil(cnt["n_glaciers"].max() / 10.0) * 10) if not cnt.empty else 1
    axc.set_ylim(0, top * 2.6)                # bars fill the lower part of the panel
    axc.set_yticks([0, top])
    axc.spines["right"].set_visible(True)
    axc.spines["right"].set_color(C_COUNT_AX)
    axc.tick_params(axis="y", colors=C_COUNT_AX, labelsize=7)
    axc.set_ylabel("In situ glaciers", color=C_COUNT_AX, fontsize=7, labelpad=2)
    axc.yaxis.set_label_coords(1.11, 0.25)

    ax.fill_between(yr, obs - 2 * obs_s, obs + 2 * obs_s, color=C_OBS, alpha=0.18, lw=0, zorder=1)
    ax.plot(yr, obs, color=C_OBS, lw=0.9, zorder=3)
    _model_band(ax, yr, mean, std)
    ax.axhline(0, color=C_INK2, lw=0.4, ls=(0, (3, 2)), zorder=1)
    old = metrics[(metrics["region_code"] == code) & (metrics["window"] == "outside")].iloc[0]
    allr = joined[joined["region_num"] == region_num]          # all years, as in the testing script
    chk = _duss_metrics(allr[_is_test(allr["year"], OLD_LAST)])
    assert chk["n_points"] == old["n_points"] and abs(chk["rmse"] - old["rmse"]) < 1e-3, (code, chk, old)
    mrow = _duss_metrics(allr[_is_test(allr["year"])])
    _header(ax, f"{name} (RGI {region_num:02d}), regional",
            _metric_text(int(mrow["n_points"]), mrow["r"], mrow["rmse"], mrow["mae"], mrow["coverage_2sig_pct"]),
            [H_MODEL, H_DUSS, H_COUNT, H_WINDOW],
            ["Ensemble mean ± 2σ", "Dussaillant et al. ± 2σ", "In situ glaciers (right axis)", "Finetuning window"],
            2, *_data_range(obs - 2 * obs_s, obs + 2 * obs_s, mean - 2 * std, mean + 2 * std), headroom=0.33)


# ---------------------------------------------------------------------------
# D: heat map
# ---------------------------------------------------------------------------

def panel_heatmap(ax) -> tuple[pd.DataFrame, dict]:
    csv = pd.read_csv(WGMS_DIR / "wgms_test_metrics_per_glacier.csv")
    j = pd.read_csv(WGMS_DIR / "wgms_test_joined.csv")
    rows = []
    for _, g in csv.iterrows():
        old = _wgms_glacier_metrics(j, g["rgi_id"], OLD_LAST)
        assert old["n_years"] == g["n_years"] and abs(old["rmse"] - g["rmse"]) < 1e-3, (g["name"], old)
        rows.append({**g[["name", "rgi_id", "rgi_region"]].to_dict(), **_wgms_glacier_metrics(j, g["rgi_id"])})
    m = pd.DataFrame(rows)
    m = m[m["n_years"] >= MIN_N].sort_values(["rgi_region", "name"]).reset_index(drop=True)
    m["row"] = np.arange(1, len(m) + 1)
    m.to_csv(Path("outputs/fig_paper/performance_evaluation_v2_per_glacier_metrics.csv"), index=False)

    t = j[_is_test(j["YEAR"]) & j["rgi_id"].isin(m["rgi_id"])]
    t = t.dropna(subset=["obs_mwe", "model_mean_mwe"])
    pooled = _metrics(t["obs_mwe"].to_numpy(), t["model_mean_mwe"].to_numpy(), t["model_std_total"].to_numpy())
    assert pooled["n_years"] == int(m["n_years"].sum()), "pooled n differs from the per-glacier sum"

    table = pd.concat([m[[c for c, *_ in COLS]], pd.DataFrame([{c: pooled[c] for c, *_ in COLS}])],
                      ignore_index=True)
    n_gl = len(m)
    y_all = n_gl + 0.6
    ys = np.r_[np.arange(n_gl), y_all]
    for jx, (col, head, fmt, worst, best, log, *_) in enumerate(COLS):
        sc = _score(table[col].to_numpy(), worst, best, log)
        for i, y in enumerate(ys):
            ax.add_patch(mpatches.Rectangle((jx - 0.5, y - 0.5), 1, 1, facecolor=_colour(sc[i]),
                                            edgecolor="white", lw=0.5))
            dark = abs(sc[i] - 0.5) > 0.4
            ax.text(jx, y, fmt(table[col].iloc[i]), ha="center", va="center", fontsize=6,
                    color="white" if dark else C_INK, fontweight="bold" if i == len(ys) - 1 else "normal")
    ax.set_xlim(-0.5, len(COLS) - 0.5)
    ax.set_ylim(y_all + 0.5, -0.5)
    ax.set_xticks(range(len(COLS)), [c[1] for c in COLS])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2, labelsize=7)
    ax.set_yticks(ys, [str(k) for k in m["row"]] + ["All"])
    ax.tick_params(axis="y", length=0, pad=2, labelsize=6)
    for sp in ax.spines.values():
        sp.set_visible(False)

    trans = ax.get_yaxis_transform()
    x_br = -0.11
    for reg, grp in m.groupby("rgi_region", sort=True):
        y0, y1 = grp.index.min() - 0.38, grp.index.max() + 0.38
        ax.plot([x_br + 0.025, x_br, x_br, x_br + 0.025], [y0, y0, y1, y1], color=C_INK2, lw=0.6,
                transform=trans, clip_on=False)
        ax.text(x_br - 0.012, (y0 + y1) / 2, f"{int(reg):02d}", transform=trans,   # RGI region number
                ha="right", va="center", fontsize=6.5)
    return m[["row", "name", "rgi_id", "rgi_region", "n_years"]], pooled


def heatmap_keys(fig, x0: float, x1: float, y_top: float, y_bot: float) -> None:
    """One key bar per metric (figure coordinates): label at left, worst/best values under the bar ends."""
    n = len(COLS)
    row_h = (y_top - y_bot) / n
    bar_x0 = x0 + 0.42 * (x1 - x0)
    bar_w = 0.50 * (x1 - x0)
    for k, (_, _, _, _, _, _, label, (lo, hi)) in enumerate(COLS):
        yb = y_top - (k + 1) * row_h + 0.55 * row_h
        cax = fig.add_axes([bar_x0, yb, bar_w, 0.28 * row_h])
        cax.imshow(np.linspace(0.1, 0.9, 256)[None, :], aspect="auto", cmap=CMAP, vmin=0, vmax=1)
        cax.set_xticks([0, 255], [lo, hi])
        cax.set_yticks([])
        cax.tick_params(length=0, labelsize=6, pad=1)
        for sp in cax.spines.values():
            sp.set_visible(False)
        fig.text(bar_x0 - 0.03, yb + 0.14 * row_h, label, ha="right", va="center", fontsize=6.5)


# ---------------------------------------------------------------------------
# E: GLaMBIE standardised residuals
# ---------------------------------------------------------------------------

def panel_glambie_z(ax) -> None:
    p = pd.read_csv(GLAMBIE_DIR / "glambie_test_pairs.csv")
    p["z"] = (p["median_mwe"] - p["glambie_mwe"]) / np.sqrt(p["std_total"] ** 2 + p["glambie_mwe_err"] ** 2)
    p["num"] = p["region"].str[1:].astype(int)
    ax.axhspan(-2, 2, color=C_WINDOW, lw=0, zorder=0)
    ax.axhline(0, color=C_INK2, lw=0.6, zorder=1)
    for y in (-2, 2):
        ax.axhline(y, color=C_INK2, lw=0.6, ls=(0, (4, 2)), zorder=1)
    handles, labels = [], []
    for year, colour in YEAR_COLORS.items():
        s = p[p["year"] == year]
        ax.plot(s["num"], s["z"], "o", ms=3.0, mfc=colour, mec="white", mew=0.35, ls="none", zorder=3)
        handles.append(_marker(colour, "o"))
        labels.append(str(year))
    ax.set_xlim(0.4, 19.6)
    ax.set_xticks(range(1, 20), [str(k) for k in range(1, 20)])
    ax.tick_params(axis="x", labelsize=6, pad=1.5)
    ax.set_xlabel("RGI region")
    ax.set_ylabel("Standardised residual, $z$", labelpad=1)
    inside = (p["z"].abs() <= 2).mean() * 100
    lo, hi = min(p["z"].min(), -2), max(p["z"].max(), 2)
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(handles, labels, title="Year", loc="upper right", ncol=1, frameon=False, fontsize=6,
              title_fontsize=6.5, handletextpad=0.1, labelspacing=0.2, borderaxespad=0.2, handlelength=0.9)
    print(f"  GLaMBIE z: {inside:.1f}% of {len(p)} inside ±2; min {p['z'].min():.2f}, max {p['z'].max():.2f}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _move(ax, dy: float) -> None:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])


def build(output_dir: Path) -> None:
    joined = pd.read_csv(DUSS_DIR / "regional_duss_joined.csv")
    metrics = pd.read_csv(DUSS_DIR / "regional_duss_metrics.csv")
    counts = load_glacier_counts(PER_GLA_DIR)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    TOP, BOT = 0.985, 0.06
    left = GridSpec(3, 1, figure=fig, left=0.075, right=0.445, top=TOP, bottom=BOT, hspace=0.12)
    right = GridSpec(3, 1, figure=fig, left=0.60, right=0.995, top=TOP, bottom=BOT,
                     height_ratios=[2.3, 0.55, 1.05], hspace=0.12)

    ax_a = fig.add_subplot(left[0])
    ax_b = fig.add_subplot(left[1], sharex=ax_a)
    ax_c = fig.add_subplot(left[2], sharex=ax_a)
    ax_d = fig.add_subplot(right[0])
    ax_e = fig.add_subplot(right[2])

    panel_glacier_v2(ax_a)
    panel_region_counts(ax_b, joined, metrics, counts, 11, "CEU", "Central Europe")
    panel_region_counts(ax_c, joined, metrics, counts, 6, "ISL", "Iceland")
    key, pooled = panel_heatmap(ax_d)
    panel_glambie_z(ax_e)

    for ax in (ax_a, ax_b):
        ax.tick_params(labelbottom=False)
    ax_a.set_xlim(YEAR_MIN - 0.5, YEAR_MAX + 0.5)
    ax_c.set_xlabel("Year")
    fig.text(0.004, (ax_a.get_position().y1 + ax_c.get_position().y0) / 2,
             "Annual mass balance (m w.e. yr$^{-1}$)", rotation=90, va="center", ha="left", fontsize=8)

    # E starts just right of the left column's widest element (legends, right axes); D's cells start where E's
    # axis starts; both end at the figure margin. D's top meets A's top; E's axis bottom meets C's.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    lc_x1 = max(max(ax.get_tightbbox(r).transformed(inv).x1 for ax in fig.axes if ax not in (ax_d, ax_e)),
                max(ax.get_legend().get_window_extent(r).transformed(inv).x1 for ax in (ax_a, ax_b, ax_c)))
    e_pos = ax_e.get_position()
    e_lab = e_pos.x0 - ax_e.get_tightbbox(r).transformed(inv).x0
    x0, x1 = lc_x1 + 0.02 + e_lab, ax_d.get_position().x1
    ax_e.set_position([x0, e_pos.y0, x1 - x0, e_pos.height])
    d_pos = ax_d.get_position()
    ax_d.set_position([x0, d_pos.y0, x1 - x0, d_pos.height])
    fig.canvas.draw()
    a_top = ax_a.get_position().y1
    _move(ax_d, a_top - ax_d.get_tightbbox(r).transformed(inv).y1)
    _move(ax_e, ax_c.get_position().y0 - ax_e.get_position().y0)
    fig.canvas.draw()
    d_bot = ax_d.get_tightbbox(r).transformed(inv).y0
    e_top = ax_e.get_tightbbox(r).transformed(inv).y1
    heat_x0 = min(ax_d.get_tightbbox(r).transformed(inv).x0, ax_e.get_tightbbox(r).transformed(inv).x0)
    heatmap_keys(fig, heat_x0, ax_d.get_position().x1, d_bot - 0.008, e_top + 0.012)

    for ax, letter in [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")]:
        _panel_label(ax, letter)
    fig.text(heat_x0, a_top, "D", fontsize=9, fontweight="bold", va="top", ha="left")
    fig.text(heat_x0, ax_e.get_position().y1, "E", fontsize=9, fontweight="bold", va="top", ha="left")

    _check_layout(fig, [ax_a, ax_b, ax_c], [ax_d, ax_e])

    output_dir.mkdir(parents=True, exist_ok=True)
    key.to_csv(output_dir / "performance_evaluation_v2_glacier_key.csv", index=False)
    pd.DataFrame([pooled]).to_csv(output_dir / "performance_evaluation_v2_pooled_metrics.csv", index=False)
    print(f"  Pooled (n >= {MIN_N}): {pooled}")
    for ext, kw in [("pdf", {}), ("png", {"dpi": 600}),
                    ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        out = output_dir / f"performance_evaluation_v2.{ext}"
        fig.savefig(out, **kw)
        print(f"  Saved {out}")
    plt.close(fig)


def _check_layout(fig, left_axes, right_axes) -> None:
    """Warn if text leaves the figure or the left column (incl. its legend) reaches the right column."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    right_x0 = min(ax.get_tightbbox(r).x0 for ax in right_axes)
    for leg in fig.legends:
        if leg.get_window_extent(r).x1 > right_x0:
            print("  WARNING: A-C legend reaches the right column")
    for ax in left_axes:
        if ax.get_tightbbox(r).x1 > right_x0:
            print("  WARNING: left-column axis text reaches the right column")
    for t in fig.findobj(lambda a: a.__class__.__name__ == "Text" and a.get_visible() and a.get_text()):
        bb = t.get_window_extent(r)
        if bb.x1 > fig.bbox.x1 + 0.5 or bb.x0 < fig.bbox.x0 - 0.5 or bb.y1 > fig.bbox.y1 + 0.5:
            print(f"  WARNING: outside figure: {t.get_text()[:30]!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Publication performance-evaluation figure, version 2 (A-E).")
    p.add_argument("--output_dir", default="outputs/fig_paper")
    build(Path(p.parse_args().output_dir))


if __name__ == "__main__":
    main()
