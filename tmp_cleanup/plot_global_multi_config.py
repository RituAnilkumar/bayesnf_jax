"""
src/plot_global_multi_config.py

Compare global Gt/yr mass balance across four model configurations plus OGGM,
all aggregated from per-glacier ensemble_glacier.csv files, against the
WGMS/Dussaillant regional observational sum.

Model configurations
--------------------
  te_r2_te    : outputs/egu_fin/ensemble_te_r2   / time_encoding
  te_r2_no_te : outputs/egu_fin/ensemble_te_r2   / no_time_encoding
  pt2000      : outputs/ensemble_pretrain2000     / no_time_encoding
  pt2000_no_te: outputs/ensemble_pretrain2000_no_te / no_time_encoding
  oggm        : data_for_model / oggm_targets_rXX.csv  (no uncertainty)

Uncertainty propagation (same as plot_global_from_glaciers.py)
  Annual global sum: all components in quadrature across glaciers
  Aleatoric:  independent across years → σ/√T for block means
  Epistemic + Structural: correlated across years → mean(σ) for block means

Conversion: Gt/yr = MWE/yr × area_km² × 1e-3
OGGM units: mm/yr  → ÷ 1000 → MWE/yr

Regions without a main_features file (r19) fall back to ensemble_regional_gt.csv.

Outputs (all saved to --output_dir)
  global_multi_annual.png        — annual Gt/yr, all configs + WGMS
  global_multi_blocks.png        — 20-year block means, all configs + WGMS

Usage:
    python src/plot_global_multi_config.py \\
        --te_r2_root   outputs/egu_fin/ensemble_te_r2 \\
        --pt2000_root  outputs/ensemble_pretrain2000 \\
        --pt2000_no_te_root outputs/ensemble_pretrain2000_no_te \\
        --data_root    data_for_model \\
        --wgms_dir     validation_data/regional_wgms_duss \\
        --output_dir   outputs/egu_fin/global_multi_config
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plot_global_from_glaciers import (
    load_glacier_gt,
    load_regional_gt_fallback,
    HIST_MIN,
    HIST_MAX,
    BLOCK_WIDTH,
    MWE_TO_GT,
)

_FS = 13

# ---------------------------------------------------------------------------
# Colour / style palette — one per config
# ---------------------------------------------------------------------------

CONFIGS = {
    "te_r2_te":     {"label": "TE (ensemble_te_r2)",        "color": "darkorange",   "ls": "-"},
    "te_r2_no_te":  {"label": "No TE (ensemble_te_r2)",     "color": "steelblue",    "ls": "-"},
    "pt2000":       {"label": "No TE (pretrain2000)",        "color": "mediumpurple", "ls": "-"},
    "pt2000_no_te": {"label": "No TE (pretrain2000_no_te)", "color": "seagreen",     "ls": "-"},
    "oggm":         {"label": "OGGM (training target)",     "color": "dimgray",      "ls": "--"},
}

WGMS_STYLE = {"label": "WGMS/Dussaillant (observed)", "color": "black", "marker": "o"}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def build_global_from_glacier_csv(
    ensemble_root: Path,
    data_root: Path,
    group: str,
) -> pd.DataFrame:
    """
    Aggregate per-glacier ensemble_glacier.csv across all regions.
    Falls back to ensemble_regional_gt.csv for regions with no features file (r19).
    Returns DataFrame indexed by year with columns:
      median_gt, var_alea, var_epist, var_struct, var_total,
      std_alea, std_epist, std_struct, std_total
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
            ann = load_regional_gt_fallback(ensemble_root, r, group)
            if ann is not None:
                print(f"    {r}: no features file — using regional GT fallback")
        if ann is None:
            continue
        global_acc = ann if global_acc is None else global_acc.add(ann, fill_value=0)

    if global_acc is None:
        raise RuntimeError(f"No data found: {ensemble_root} / {group}")

    global_acc["std_alea"]   = np.sqrt(global_acc["var_alea"])
    global_acc["std_epist"]  = np.sqrt(global_acc["var_epist"])
    global_acc["std_struct"] = np.sqrt(global_acc["var_struct"])
    global_acc["std_total"]  = np.sqrt(global_acc["var_total"])
    return global_acc


def build_oggm_global(data_root: Path) -> pd.Series:
    """
    Sum OGGM per-glacier mass balance (mm/yr → MWE/yr × area → Gt/yr)
    across all regions. Returns annual Series indexed by year.
    No uncertainty — OGGM is a single deterministic model output.
    """
    regions = sorted(
        d.name for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("r")
    )
    total_gt = pd.Series(dtype=float)
    for r in regions:
        og = data_root / r / f"oggm_targets_{r}.csv"
        ft = data_root / r / f"main_features_{r}.csv"
        if not og.exists() or not ft.exists():
            continue
        df   = pd.read_csv(og).dropna(subset=["mass_balance"])
        feat = pd.read_csv(ft, usecols=["rgi_id", "Area"]).drop_duplicates("rgi_id")
        m    = df.merge(feat, on="rgi_id")
        # mm/yr → MWE/yr (/1000) → Gt/yr (× area_km² × 1e-3)
        rgt  = (
            m.assign(gt=m["mass_balance"] / 1000 * m["Area"] * MWE_TO_GT)
            .groupby("year")["gt"].sum()
        )
        total_gt = total_gt.add(rgt, fill_value=0)

    return total_gt.sort_index()


def load_wgms_global(wgms_dir: Path) -> pd.DataFrame:
    """
    Sum WGMS/Dussaillant regional GT across all region files.
    SA1 + SA2 are both included (both sub-regions of r17).
    Returns DataFrame with columns: year, gt, gt_sigma.
    """
    csvs = sorted(wgms_dir.glob("*.csv"))
    dfs  = [pd.read_csv(p)[["year", "gt", "gt_sigma"]].dropna() for p in csvs]
    combined = pd.concat(dfs, ignore_index=True)
    return (
        combined
        .groupby("year")
        .agg(gt=("gt", "sum"),
             gt_sigma=("gt_sigma", lambda x: np.sqrt((x ** 2).sum())))
        .reset_index()
        .sort_values("year")
    )


# ---------------------------------------------------------------------------
# 20-year block means
# ---------------------------------------------------------------------------

def block_means(
    global_df: pd.DataFrame,
    width: int = BLOCK_WIDTH,
) -> list[dict]:
    """Same propagation as in plot_global_from_glaciers: aleatoric shrinks, rest does not."""
    blocks, y = [], HIST_MIN
    while y + width <= HIST_MAX + 1:
        sub = global_df[(global_df.index >= y) & (global_df.index < y + width)]
        if len(sub):
            T = len(sub)
            blocks.append({
                "label":       f"{y}–{y+width-1}",
                "y0": y, "y1": y + width,
                "mean":        sub["median_gt"].mean(),
                "sigma_alea":  np.sqrt(sub["var_alea"].mean() / T),
                "sigma_epist": sub["std_epist"].mean(),
                "sigma_struct":sub["std_struct"].mean(),
            })
            blocks[-1]["sigma_total"] = np.sqrt(
                blocks[-1]["sigma_alea"]   ** 2
                + blocks[-1]["sigma_epist"]  ** 2
                + blocks[-1]["sigma_struct"] ** 2
            )
        y += width
    return blocks


def block_means_series(series: pd.Series, width: int = BLOCK_WIDTH) -> list[dict]:
    """Block means for a plain Series (e.g. OGGM — no uncertainty)."""
    blocks, y = [], HIST_MIN
    while y + width <= HIST_MAX + 1:
        sub = series[(series.index >= y) & (series.index < y + width)]
        if len(sub):
            blocks.append({
                "label": f"{y}–{y+width-1}",
                "y0": y, "y1": y + width,
                "mean": sub.mean(),
                "sigma_total": 0.0,
            })
        y += width
    return blocks


def block_means_wgms(wgms_df: pd.DataFrame, width: int = BLOCK_WIDTH) -> list[dict]:
    """Block means for WGMS data. Sigma assumed correlated → mean(sigma)."""
    blocks, y = [], HIST_MIN
    while y + width <= HIST_MAX + 1:
        sub = wgms_df[(wgms_df["year"] >= y) & (wgms_df["year"] < y + width)]
        if len(sub):
            T = len(sub)
            blocks.append({
                "label": f"{y}–{y+width-1}",
                "y0": y, "y1": y + width,
                "mean": sub["gt"].mean(),
                # gt_sigma is a systematic per-region uncertainty → treat as correlated
                "sigma_total": sub["gt_sigma"].mean(),
            })
        y += width
    return blocks


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _savefig(fig, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_annual(
    model_data: dict[str, pd.DataFrame],
    oggm_gt:    pd.Series,
    wgms_df:    pd.DataFrame,
    output_path: Path,
    year_min: int = 1950,
    year_max: int = 2023,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))

    for cfg_key, df in model_data.items():
        style = CONFIGS[cfg_key]
        mask  = (df.index >= year_min) & (df.index <= year_max)
        yrs   = df.index[mask].values
        mu    = df.loc[mask, "median_gt"].values
        s     = df.loc[mask, "std_total"].values
        ax.fill_between(yrs, mu - 2*s, mu + 2*s, alpha=0.12, color=style["color"])
        ax.plot(yrs, mu, color=style["color"], lw=1.8,
                ls=style["ls"], label=style["label"])

    # OGGM — no uncertainty band
    og = oggm_gt[(oggm_gt.index >= year_min) & (oggm_gt.index <= year_max)]
    ax.plot(og.index, og.values, color=CONFIGS["oggm"]["color"],
            lw=1.5, ls=CONFIGS["oggm"]["ls"], label=CONFIGS["oggm"]["label"])

    # WGMS
    w = wgms_df[(wgms_df["year"] >= year_min) & (wgms_df["year"] <= year_max)]
    ax.errorbar(w["year"], w["gt"], yerr=2*w["gt_sigma"],
                fmt=WGMS_STYLE["marker"], color=WGMS_STYLE["color"],
                ms=4, lw=1.2, capsize=3, zorder=6, label=f"{WGMS_STYLE['label']} ±2σ")

    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlim(year_min, year_max)
    ax.set_xlabel("Year", fontsize=_FS)
    ax.set_ylabel("Global Mass Balance  (Gt yr⁻¹)", fontsize=_FS)
    ax.set_title("Global Glacier Mass Balance — multi-configuration comparison\n"
                 "Model bands: ±2σ total  |  WGMS error bars: ±2σ",
                 fontsize=_FS + 1, pad=8)
    ax.tick_params(labelsize=_FS - 1)
    ax.legend(fontsize=_FS - 2, loc="lower left")
    fig.tight_layout()
    _savefig(fig, output_path)


def plot_blocks(
    model_blocks: dict[str, list[dict]],
    oggm_blocks:  list[dict],
    wgms_blocks:  list[dict],
    output_path:  Path,
) -> None:
    """
    Grouped bar chart: one group per 20-year period, one bar per config.
    All configs + WGMS + OGGM side by side.
    Error bars show ±2σ.
    """
    # Use WGMS blocks as the reference period labels
    labels = [b["label"] for b in wgms_blocks]
    n_periods = len(labels)

    # Order: te_r2_te, te_r2_no_te, pt2000, pt2000_no_te, oggm, wgms
    bar_defs = [
        ("te_r2_te",     model_blocks, CONFIGS["te_r2_te"]["color"],     CONFIGS["te_r2_te"]["label"]),
        ("te_r2_no_te",  model_blocks, CONFIGS["te_r2_no_te"]["color"],  CONFIGS["te_r2_no_te"]["label"]),
        ("pt2000",       model_blocks, CONFIGS["pt2000"]["color"],        CONFIGS["pt2000"]["label"]),
        ("pt2000_no_te", model_blocks, CONFIGS["pt2000_no_te"]["color"],  CONFIGS["pt2000_no_te"]["label"]),
    ]
    n_bars = len(bar_defs) + 2  # +2 for OGGM and WGMS
    width  = 0.13
    x      = np.arange(n_periods)
    offsets = np.linspace(-(n_bars-1)/2 * width, (n_bars-1)/2 * width, n_bars)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, (key, blocks_dict, color, label) in enumerate(bar_defs):
        blks = blocks_dict.get(key, [])
        means  = [b["mean"]        for b in blks if b["label"] in labels]
        sigmas = [b["sigma_total"] for b in blks if b["label"] in labels]
        ax.bar(x + offsets[i], means, width, color=color, alpha=0.8,
               label=label, edgecolor="white", lw=0.4)
        ax.errorbar(x + offsets[i], means, yerr=[2*s for s in sigmas],
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=3)

    # OGGM
    og_means = [b["mean"] for b in oggm_blocks if b["label"] in labels]
    ax.bar(x + offsets[-2], og_means, width, color=CONFIGS["oggm"]["color"],
           alpha=0.7, label=CONFIGS["oggm"]["label"],
           edgecolor="white", lw=0.4, hatch="//")

    # WGMS
    w_means  = [b["mean"]        for b in wgms_blocks]
    w_sigmas = [b["sigma_total"] for b in wgms_blocks]
    ax.bar(x + offsets[-1], w_means, width, color=WGMS_STYLE["color"],
           alpha=0.85, label=WGMS_STYLE["label"], edgecolor="white", lw=0.4)
    ax.errorbar(x + offsets[-1], w_means, yerr=[2*s for s in w_sigmas],
                fmt="none", ecolor="dimgray", elinewidth=1.0, capsize=3)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=_FS - 1)
    ax.set_xlabel("20-year period", fontsize=_FS)
    ax.set_ylabel("Mean Mass Balance  (Gt yr⁻¹)", fontsize=_FS)
    ax.set_title("Global 20-year block means — multi-configuration comparison\n"
                 "Error bars: ±2σ", fontsize=_FS + 1, pad=8)
    ax.tick_params(labelsize=_FS - 1)
    ax.legend(fontsize=_FS - 3, loc="lower left", ncol=2)
    fig.tight_layout()
    _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--te_r2_root",         default="outputs/egu_fin/ensemble_te_r2")
    parser.add_argument("--pt2000_root",         default="outputs/ensemble_pretrain2000")
    parser.add_argument("--pt2000_no_te_root",   default="outputs/ensemble_pretrain2000_no_te")
    parser.add_argument("--data_root",           default="data_for_model")
    parser.add_argument("--wgms_dir",            default="validation_data/regional_wgms_duss")
    parser.add_argument("--output_dir",          default="outputs/egu_fin/global_multi_config")
    parser.add_argument("--year_min", type=int,  default=1950)
    parser.add_argument("--year_max", type=int,  default=2023)
    args = parser.parse_args()

    te_r2_root       = Path(args.te_r2_root)
    pt2000_root      = Path(args.pt2000_root)
    pt2000_no_te_root= Path(args.pt2000_no_te_root)
    data_root        = Path(args.data_root)
    wgms_dir         = Path(args.wgms_dir)
    output_dir       = Path(args.output_dir)

    print("\n=== Global multi-config comparison ===\n")

    print("Loading model configs from ensemble_glacier.csv...")
    configs_raw: dict[str, pd.DataFrame] = {}
    for cfg_key, root, group in [
        ("te_r2_te",     te_r2_root,        "time_encoding"),
        ("te_r2_no_te",  te_r2_root,        "no_time_encoding"),
        ("pt2000",       pt2000_root,       "no_time_encoding"),
        ("pt2000_no_te", pt2000_no_te_root, "no_time_encoding"),
    ]:
        print(f"  {cfg_key}...")
        configs_raw[cfg_key] = build_global_from_glacier_csv(root, data_root, group)

    print("\nLoading OGGM...")
    oggm_gt = build_oggm_global(data_root)
    print(f"  OGGM 2000-2020 mean: {oggm_gt.loc[2000:2020].mean():+.1f} Gt/yr")

    print("\nLoading WGMS/Dussaillant...")
    wgms_df = load_wgms_global(wgms_dir)
    print(f"  WGMS years: {wgms_df.year.min()}–{wgms_df.year.max()}")

    # Print summary table
    print("\n=== 2000-2019 mean Gt/yr ===")
    for key, df in configs_raw.items():
        m = df.loc[2000:2019, "median_gt"].mean()
        s = df.loc[2000:2019, "std_total"].mean()
        print(f"  {CONFIGS[key]['label']:<40}: {m:+.1f} ± {2*s:.1f} Gt/yr")
    og_mean = oggm_gt.loc[2000:2019].mean()
    print(f"  {CONFIGS['oggm']['label']:<40}: {og_mean:+.1f} Gt/yr")
    w_mean = wgms_df[(wgms_df.year>=2000)&(wgms_df.year<2020)]['gt'].mean()
    w_sig  = wgms_df[(wgms_df.year>=2000)&(wgms_df.year<2020)]['gt_sigma'].mean()
    print(f"  {'WGMS/Dussaillant':<40}: {w_mean:+.1f} ± {2*w_sig:.1f} Gt/yr")

    # Compute block means
    model_blocks = {k: block_means(v) for k, v in configs_raw.items()}
    oggm_blocks  = block_means_series(oggm_gt)
    wgms_blocks  = block_means_wgms(wgms_df)

    print("\n  Figure 1: annual time series")
    plot_annual(configs_raw, oggm_gt, wgms_df,
                output_dir / "global_multi_annual.png",
                year_min=args.year_min, year_max=args.year_max)

    print("  Figure 2: 20-year block comparison")
    plot_blocks(model_blocks, oggm_blocks, wgms_blocks,
                output_dir / "global_multi_blocks.png")

    print(f"\nDone → {output_dir}/")


if __name__ == "__main__":
    main()
