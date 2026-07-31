"""
src/plot_global_from_glaciers.py

Builds global mass balance estimates directly from per-glacier ensemble_glacier.csv
files, rather than from pre-aggregated regional CSVs.  This gives full control
over the uncertainty propagation and produces internally consistent global figures.

Conversion:  Gt/yr = MWE/yr × area_km² × 1e-3
  (1 MWE/yr over 1 km² = 10⁶ m² × 1 m × 10³ kg/m³ = 10⁹ kg/yr = 10⁻³ Gt/yr)

Uncertainty propagation
  Annual global sum
    σ²_component = Σ_i  (σ_component,i × area_i × 1e-3)²   [quadrature over glaciers]

  20-year block mean (T = 20 years)
    Aleatoric   — independent across years  →  σ_block = √(mean σ_t²) / √T
    Epistemic   — correlated across years   →  σ_block = mean(σ_t)
    Structural  — correlated across years   →  σ_block = mean(σ_t)
    Total block σ = √(σ_aleatoric_block² + σ_epistemic_block² + σ_structural_block²)

Outputs (saved to --output_dir):
    global_from_glaciers_ts.png     — annual time series + 20-yr block steps
    global_from_glaciers_bars.png   — 20-yr block bar chart

Usage:
    python src/plot_global_from_glaciers.py \\
        --ensemble_root outputs/egu_fin/ensemble_te_r2 \\
        --data_root     data_for_model \\
        --group         no_time_encoding \\
        --output_dir    outputs/egu_fin/global_from_glaciers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HIST_MIN, HIST_MAX = 1940, 2020
BLOCK_WIDTH = 20
MWE_TO_GT = 1e-3   # Gt per (MWE/yr · km²)
_FS = 13


# ---------------------------------------------------------------------------
# Step 1 — load per-glacier areas (one value per rgi_id)
# ---------------------------------------------------------------------------

def load_areas(data_root: Path, region: str) -> pd.Series:
    """Return Series: rgi_id → area_km²  (one row per glacier, deduplicated)."""
    p = data_root / region / f"main_features_{region}.csv"
    df = pd.read_csv(p, usecols=["rgi_id", "Area"])
    return df.drop_duplicates("rgi_id").set_index("rgi_id")["Area"]


# ---------------------------------------------------------------------------
# Step 2 — load per-glacier predictions and convert to Gt/yr
# ---------------------------------------------------------------------------

def load_glacier_gt(
    ensemble_root: Path,
    data_root: Path,
    region: str,
    group: str,
) -> pd.DataFrame | None:
    """
    Load ensemble_glacier.csv for one region, merge glacier areas, convert
    all MWE/yr columns to Gt/yr.  Returns a DataFrame indexed by year with
    columns: median_gt, var_aleatoric, var_epistemic, var_structural, var_total.
    """
    p = ensemble_root / region / group / "ensemble_glacier.csv"
    if not p.exists():
        print(f"  MISSING: {p}")
        return None

    areas = load_areas(data_root, region)

    df = pd.read_csv(p)
    df = df[(df["year"] >= HIST_MIN) & (df["year"] <= HIST_MAX)]

    # Attach area
    df = df.join(areas.rename("area_km2"), on="rgi_id")
    missing = df["area_km2"].isna().sum()
    if missing > 0:
        print(f"  WARNING {region}: {missing} rows with no area — dropping")
        df = df.dropna(subset=["area_km2"])

    # Scale factor per glacier-row: area_km2 × MWE_TO_GT
    scale = df["area_km2"] * MWE_TO_GT

    # Per-glacier Gt/yr contributions
    df["gt_median"]   = df["median_mwe"]       * scale
    df["gt2_alea"]    = (df["std_aleatoric"]   * scale) ** 2
    df["gt2_epist"]   = (df["std_epistemic"]   * scale) ** 2
    df["gt2_struct"]  = (df["std_structural"]  * scale) ** 2
    df["gt2_total"]   = (df["std_total"]       * scale) ** 2

    # Sum across glaciers within each year
    ann = (
        df.groupby("year")
        .agg(
            median_gt   = ("gt_median",  "sum"),
            var_alea    = ("gt2_alea",   "sum"),
            var_epist   = ("gt2_epist",  "sum"),
            var_struct  = ("gt2_struct", "sum"),
            var_total   = ("gt2_total",  "sum"),
        )
        .sort_index()
    )
    return ann


# ---------------------------------------------------------------------------
# Step 3 — aggregate across all regions
# ---------------------------------------------------------------------------

def load_regional_gt_fallback(
    ensemble_root: Path,
    region: str,
    group: str,
) -> pd.DataFrame | None:
    """
    Fallback for regions without a features file (e.g. r19).
    Reads ensemble_regional_gt.csv and returns the same column structure
    as load_glacier_gt, treating the regional uncertainty as fully correlated
    (i.e. stored in var_struct; var_alea and var_epist set to zero).
    """
    p = ensemble_root / region / group / "ensemble_regional_gt.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).sort_values("year")
    df = df[(df["year"] >= HIST_MIN) & (df["year"] <= HIST_MAX)].set_index("year")
    out = pd.DataFrame({
        "median_gt":  df["median_gt"],
        "var_alea":   0.0,
        "var_epist":  0.0,
        "var_struct": df["std_total"] ** 2,
        "var_total":  df["std_total"] ** 2,
    })
    return out


def build_global(
    ensemble_root: Path,
    data_root: Path,
    group: str,
) -> pd.DataFrame:
    """
    Sum regional annual Gt/yr estimates → global annual series.
    Regions with a main_features file are aggregated from per-glacier data.
    Regions without one (r19) fall back to the pre-aggregated regional CSV.
    """
    regions = sorted(
        d.name for d in ensemble_root.iterdir()
        if d.is_dir() and d.name.startswith("r")
    )

    global_acc = None
    for r in regions:
        features_path = data_root / r / f"main_features_{r}.csv"
        if features_path.exists():
            ann = load_glacier_gt(ensemble_root, data_root, r, group)
        else:
            print(f"  {r}: no features file — using pre-aggregated regional GT")
            ann = load_regional_gt_fallback(ensemble_root, r, group)
        if ann is None:
            continue
        global_acc = ann if global_acc is None else global_acc.add(ann, fill_value=0)

    if global_acc is None:
        raise RuntimeError("No data loaded — check ensemble_root and group.")

    global_acc["std_alea"]   = np.sqrt(global_acc["var_alea"])
    global_acc["std_epist"]  = np.sqrt(global_acc["var_epist"])
    global_acc["std_struct"] = np.sqrt(global_acc["var_struct"])
    global_acc["std_total"]  = np.sqrt(global_acc["var_total"])
    return global_acc


# ---------------------------------------------------------------------------
# Step 4 — compute 20-year block means with correct uncertainty propagation
# ---------------------------------------------------------------------------

def compute_blocks(global_df: pd.DataFrame, width: int = BLOCK_WIDTH) -> list[dict]:
    """
    Non-overlapping block averages.  Uncertainty treatment per block:
      - aleatoric: independent across years  → σ / √T   (mean variance / T)
      - epistemic:  correlated across years  → mean(σ)   (does not shrink)
      - structural: correlated across years  → mean(σ)   (does not shrink)
    """
    blocks = []
    y = HIST_MIN
    while y + width <= HIST_MAX + 1:
        sub = global_df[(global_df.index >= y) & (global_df.index < y + width)]
        if len(sub) == 0:
            y += width
            continue
        T = len(sub)
        block = {
            "label":      f"{y}–{y + width - 1}",
            "y0":         y,
            "y1":         y + width,
            "mean_gt":    sub["median_gt"].mean(),
            # Aleatoric: independent → variance averages, then /T again for mean
            "sigma_alea":   np.sqrt(sub["var_alea"].mean()  / T),
            # Epistemic & structural: correlated → just take mean σ
            "sigma_epist":  sub["std_epist"].mean(),
            "sigma_struct": sub["std_struct"].mean(),
        }
        block["sigma_total"] = np.sqrt(
            block["sigma_alea"]   ** 2
            + block["sigma_epist"]  ** 2
            + block["sigma_struct"] ** 2
        )
        blocks.append(block)
        y += width
    return blocks


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 1 — time series + block steps
# ---------------------------------------------------------------------------

def plot_ts_with_blocks(
    global_df: pd.DataFrame,
    blocks: list[dict],
    output_path: Path,
) -> None:
    years = global_df.index.values
    med   = global_df["median_gt"].values
    s_tot = global_df["std_total"].values

    fig, (ax_ts, ax_bar) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.38},
    )

    # --- top: annual time series ---
    ax_ts.fill_between(
        years, med - 2 * s_tot, med + 2 * s_tot,
        alpha=0.15, color="steelblue", label="±2σ annual (total)",
    )
    ax_ts.plot(years, med, color="steelblue", lw=1.0, alpha=0.55,
               label="Annual median")

    # Overlay uncertainty decomposition bands at ±1σ
    ax_ts.fill_between(
        years,
        med - global_df["std_alea"].values,
        med + global_df["std_alea"].values,
        alpha=0.20, color="seagreen", label="±1σ aleatoric",
    )

    # Block steps
    blk_colors = ["#2166ac" if b["mean_gt"] >= 0 else "#d6604d" for b in blocks]
    for b, c in zip(blocks, blk_colors):
        x0, x1 = b["y0"], min(b["y1"], HIST_MAX + 1)
        bm, bs  = b["mean_gt"], b["sigma_total"]
        ax_ts.hlines(bm, x0, x1, colors=c, lw=3.0, zorder=5)
        ax_ts.fill_between([x0, x1], bm - 2 * bs, bm + 2 * bs,
                           alpha=0.30, color=c, zorder=4)
        offset = 18 if bm >= 0 else -25
        ax_ts.text(
            (x0 + x1) / 2, bm + offset,
            f"{bm:+.0f}\n±{2*bs:.0f}",
            ha="center", va="bottom" if bm >= 0 else "top",
            fontsize=_FS - 2, fontweight="bold", color=c,
        )

    ax_ts.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax_ts.set_xlim(HIST_MIN, HIST_MAX)
    ax_ts.set_xlabel("Year", fontsize=_FS)
    ax_ts.set_ylabel("Mass Balance  (Gt yr⁻¹)", fontsize=_FS)
    ax_ts.set_title(
        "Global Glacier Mass Balance — from per-glacier ensemble\n"
        "20-year block averages with propagated uncertainty",
        fontsize=_FS + 1, pad=8,
    )
    ax_ts.tick_params(labelsize=_FS - 1)
    ax_ts.legend(fontsize=_FS - 2, loc="lower left")

    # --- bottom: bar chart ---
    labels = [b["label"] for b in blocks]
    means  = np.array([b["mean_gt"]    for b in blocks])
    errs   = np.array([b["sigma_total"] for b in blocks])
    # Stack uncertainty components visually
    e_alea   = np.array([b["sigma_alea"]   for b in blocks])
    e_epist  = np.array([b["sigma_epist"]  for b in blocks])
    e_struct = np.array([b["sigma_struct"] for b in blocks])

    x = np.arange(len(labels))
    bar_colors = ["#2166ac" if v >= 0 else "#d6604d" for v in means]
    ax_bar.bar(x, means, color=bar_colors, edgecolor="white", lw=0.5,
               width=0.6, zorder=3)

    # Error bar showing total ±2σ
    ax_bar.errorbar(x, means, yerr=2 * errs, fmt="none",
                    ecolor="black", elinewidth=1.5, capsize=6, capthick=1.5,
                    zorder=5, label="±2σ total")

    for xi, (val, err) in enumerate(zip(means, errs)):
        offset = 8 if val >= 0 else -8
        va = "bottom" if val >= 0 else "top"
        ax_bar.text(xi, val + offset, f"{val:+.0f}",
                    ha="center", va=va, fontsize=_FS - 2, fontweight="bold")

    ax_bar.axhline(0, color="black", lw=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=_FS - 2)
    ax_bar.set_ylabel("Mean (Gt yr⁻¹)", fontsize=_FS)
    ax_bar.set_xlabel("20-year period", fontsize=_FS)
    ax_bar.tick_params(labelsize=_FS - 1)
    ax_bar.legend(fontsize=_FS - 2, loc="upper right")

    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2 — uncertainty decomposition bar chart per block
# ---------------------------------------------------------------------------

def plot_uncertainty_decomposition(
    blocks: list[dict],
    output_path: Path,
) -> None:
    """
    Stacked bar showing how total block uncertainty breaks down into
    aleatoric, epistemic, and structural components.
    """
    labels   = [b["label"]        for b in blocks]
    e_alea   = np.array([b["sigma_alea"]   for b in blocks])
    e_epist  = np.array([b["sigma_epist"]  for b in blocks])
    e_struct = np.array([b["sigma_struct"] for b in blocks])
    e_total  = np.array([b["sigma_total"]  for b in blocks])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x, 2*e_struct, label="Structural (ensemble spread)", color="#d73027", alpha=0.85)
    ax.bar(x, 2*e_epist,  bottom=2*e_struct,
           label="Epistemic (Bayesian posterior)", color="#fc8d59", alpha=0.85)
    ax.bar(x, 2*e_alea,   bottom=2*e_struct + 2*e_epist,
           label="Aleatoric (irreducible)", color="#91bfdb", alpha=0.85)

    # Total (quadrature) as a line
    ax.plot(x, 2*e_total, "ko-", lw=1.5, ms=6, zorder=5, label="Total (quadrature)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=_FS - 2)
    ax.set_ylabel("2σ uncertainty  (Gt yr⁻¹)", fontsize=_FS)
    ax.set_xlabel("20-year period", fontsize=_FS)
    ax.set_title("Uncertainty decomposition — 20-year block means\n"
                 "(structural & epistemic are correlated across years; "
                 "aleatoric shrinks with averaging)",
                 fontsize=_FS, pad=8)
    ax.tick_params(labelsize=_FS - 1)
    ax.legend(fontsize=_FS - 2)
    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_root", default="outputs/egu_fin/ensemble_te_r2")
    parser.add_argument("--data_root",     default="data_for_model")
    parser.add_argument("--group",         default="no_time_encoding")
    parser.add_argument("--output_dir",    default="outputs/egu_fin/global_from_glaciers")
    args = parser.parse_args()

    ensemble_root = Path(args.ensemble_root)
    data_root     = Path(args.data_root)
    output_dir    = Path(args.output_dir)

    print(f"\n=== Global from glaciers — {args.group} ===\n")

    print("Loading per-glacier data and aggregating...")
    global_df = build_global(ensemble_root, data_root, args.group)

    print(f"\nGlobal series: {len(global_df)} years, "
          f"{global_df.index.min()}–{global_df.index.max()}")
    print(f"2000-2019 mean: {global_df.loc[2000:2019, 'median_gt'].mean():+.1f} Gt/yr")

    blocks = compute_blocks(global_df)
    print("\n20-year block means (Gt/yr):")
    print(f"  {'Period':<12} {'Mean':>8} {'±2σ_alea':>10} {'±2σ_epist':>10} "
          f"{'±2σ_struct':>12} {'±2σ_total':>11}")
    for b in blocks:
        print(f"  {b['label']:<12} {b['mean_gt']:>+8.1f} "
              f"{2*b['sigma_alea']:>10.1f} {2*b['sigma_epist']:>10.1f} "
              f"{2*b['sigma_struct']:>12.1f} {2*b['sigma_total']:>11.1f}")

    print("\n  Figure 1: time series + block steps")
    plot_ts_with_blocks(
        global_df, blocks,
        output_dir / "global_from_glaciers_ts.png",
    )

    print("  Figure 2: uncertainty decomposition")
    plot_uncertainty_decomposition(
        blocks,
        output_dir / "global_from_glaciers_uncertainty.png",
    )

    print(f"\nDone → {output_dir}/")


if __name__ == "__main__":
    main()
