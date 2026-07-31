"""
Karakoram anomaly showcase figure — three panels:

  Left:   Year attribution over time (1980–2025), multiple regions overlaid.
          r14 (Karakoram) is highlighted: attribution stays near zero while
          all other regions trend strongly negative.

  Middle: Contrasting climate drivers over time.
          r03 (Arctic Canada N): t2m_abl_mean attribution drifts negative
            → warming summers increasingly drive mass loss.
          r14 (Karakoram):      tp_acc_sum attribution stays positive/neutral
            → precipitation accumulation is the persistent control.
          Only climate variables shown (no static geometry).

  Right:  tp_acc_sum dependence for r14.
          More winter snowfall → more positive mass balance.
          Clean linear relationship, time-stable (colours = year, no drift).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

BASE      = "outputs/explain_top5_lim_glamb_meanbaseline"
DATA_BASE = "data_for_model"
OUTPATH   = "outputs/explain_top5_lim_glamb_meanbaseline/karakoram_anomaly.png"

KARA_COLOR  = "#d73027"   # red
ACN_COLOR   = "#2166ac"   # blue
OTHER_COLOR = "#bbbbbb"

OVERLAY_REGIONS = [
    ("r01", "Alaska"),
    ("r03", "Arctic Canada N"),
    ("r06", "Iceland"),
    ("r07", "Svalbard"),
    ("r08", "Scandinavia"),
    ("r09", "Russian Arctic"),
    ("r12", "Caucasus"),
    ("r14", "Karakoram"),
]


def _attr_csv(rprefix):
    rdirs = sorted(glob.glob(os.path.join(BASE, rprefix + "*")))
    if not rdirs:
        return None
    p = os.path.join(rdirs[0], "attributions_finetune_ensemble1.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _yearly(df, col):
    """Return (years, mean, std) grouped by year for one attribution column."""
    grp = df.groupby("year")[col]
    return grp.mean().index.values, grp.mean().values, grp.std().fillna(0).values


def _smooth(arr, w=3):
    return uniform_filter1d(arr, size=w, mode="nearest")


# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.subplots_adjust(wspace=0.38)

# ────────────────────────────────────────────────────────────────────────────
# Panel 1 — year attribution over time, multi-region overlay
# ────────────────────────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.axhline(0, color="black", lw=0.9, ls="--", zorder=1)
ax1.axvspan(2000, 2026, color="#fff3f3", alpha=0.6, zorder=0)

for rprefix, label in OVERLAY_REGIONS:
    df = _attr_csv(rprefix)
    if df is None or "attr_year" not in df.columns:
        continue
    yrs, mean_a, _ = _yearly(df, "attr_year")
    mask = (yrs >= 1980) & (yrs <= 2025)
    if not mask.any():
        continue
    y_sm = _smooth(mean_a[mask], w=3)

    if rprefix == "r14":
        ax1.plot(yrs[mask], y_sm, color=KARA_COLOR, lw=2.5, zorder=5, label="Karakoram (r14)")
    else:
        ax1.plot(yrs[mask], y_sm, color=OTHER_COLOR, lw=1.0, zorder=2, alpha=0.7)

# dummy line for legend
from matplotlib.lines import Line2D
ax1.legend(handles=[
    Line2D([0], [0], color=KARA_COLOR, lw=2.5, label="Karakoram (r14)"),
    Line2D([0], [0], color=OTHER_COLOR, lw=1.5, label="Other regions"),
], fontsize=8, loc="lower left")

ax1.set_xlim(1980, 2026)
ax1.set_ylim(-4.5, 4.5)
ax1.set_xlabel("Year", fontsize=10)
ax1.set_ylabel("Mean year attribution (MWE/yr)", fontsize=10)
ax1.set_title("Year attribution over time\n(multi-region comparison)", fontsize=11, fontweight="bold")
ax1.text(2013, 4.1, "Post-2000 era", fontsize=7.5, color="#cc4444", ha="center")

# ────────────────────────────────────────────────────────────────────────────
# Panel 2 — contrasting CLIMATE drivers over time (r03 t2m vs r14 tp)
# ────────────────────────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.axhline(0, color="black", lw=0.9, ls="--", zorder=1)

for rprefix, attr_col, color, ls, label in [
    ("r03", "attr_t2m_abl_mean", ACN_COLOR,  "-",  "Arctic Canada N — ablation temp. (t2m_abl_mean)"),
    ("r14", "attr_tp_acc_sum",   KARA_COLOR,  "-",  "Karakoram — accum. precip. (tp_acc_sum)"),
]:
    df = _attr_csv(rprefix)
    if df is None or attr_col not in df.columns:
        continue
    yrs, mean_a, std_a = _yearly(df, attr_col)
    mask = (yrs >= 1980) & (yrs <= 2025)
    if not mask.any():
        continue
    y_sm   = _smooth(mean_a[mask], w=3)
    std_sm = _smooth(std_a[mask],  w=3)

    ax2.fill_between(yrs[mask], y_sm - std_sm, y_sm + std_sm, alpha=0.12, color=color)
    ax2.plot(yrs[mask], y_sm, color=color, lw=2.0, ls=ls, label=label)

ax2.set_xlim(1980, 2026)
ax2.set_xlabel("Year", fontsize=10)
ax2.set_ylabel("Mean feature attribution (MWE/yr)", fontsize=10)
ax2.set_title("Contrasting climate drivers\n(temperature vs. precipitation)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=7.5, loc="lower left")

# ────────────────────────────────────────────────────────────────────────────
# Panel 3 — tp_acc_sum dependence for r14 (feature value vs attribution)
# ────────────────────────────────────────────────────────────────────────────
ax3 = axes[2]
ax3.axhline(0, color="black", lw=0.9, ls="--", zorder=1)

df14 = _attr_csv("r14")
feat14 = pd.read_csv(os.path.join(DATA_BASE, "r14", "main_features_r14.csv"))
feat14 = feat14[feat14["year"].between(1940, 2025)]

if df14 is not None and "attr_tp_acc_sum" in df14.columns:
    merged = df14.merge(feat14[["rgi_id", "year", "tp_acc_sum"]],
                        on=["rgi_id", "year"], how="inner")
    x    = merged["tp_acc_sum"].values
    y    = merged["attr_tp_acc_sum"].values
    yr_c = merged["year"].values

    norm = plt.Normalize(yr_c.min(), yr_c.max())
    sc = ax3.scatter(x, y, c=yr_c, cmap="viridis", s=5, alpha=0.45, norm=norm, zorder=2)
    cbar = fig.colorbar(sc, ax=ax3, pad=0.02, fraction=0.04)
    cbar.set_label("Year", fontsize=8)

    # smoothed trend line
    order_idx = np.argsort(x)
    xs, ys = x[order_idx], y[order_idx]
    if len(xs) > 30:
        ys_sm = uniform_filter1d(ys, size=max(1, len(xs) // 40), mode="nearest")
        ax3.plot(xs, ys_sm, color="black", lw=1.8, zorder=5)

    ax3.set_xlabel("tp_acc_sum — winter precip. (m)", fontsize=10)
    ax3.set_ylabel("Attribution for tp_acc_sum (MWE/yr)", fontsize=10)
    ax3.set_title("Karakoram (r14):\nMore winter snowfall → less mass loss", fontsize=11, fontweight="bold")
else:
    ax3.text(0.5, 0.5, "tp_acc_sum attribution not found",
             ha="center", va="center", transform=ax3.transAxes, fontsize=10)

# ── save ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    "The Karakoram Anomaly — evidence from model-learned attributions\n"
    "Year trend is absent; winter precipitation drives mass balance",
    fontsize=12, fontweight="bold", y=1.02,
)
os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)
fig.savefig(OUTPATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPATH}")
