"""
src/validate_regional.py

Validates ensemble regional MWE predictions against WGMS / Dussaillant et al.
regional reference series.

For each matched region the script produces (files named by RGI number, e.g. r06):
  - timeseries_{rgi}.png     ensemble median ± N×σ bands vs. WGMS ±1σ error bars;
                              title includes RMSE, bias, r, and % years within bounds
  - scatter_{rgi}.png        per-region scatter: ensemble median vs. WGMS with
                              ±N×σ ensemble error bars and WGMS uncertainty band
  - residuals_all_regions.png residual bars per region with ±N×σ WGMS uncertainty band
  - summary_multipanel.png   all regions in one figure
  - validation_metrics.csv   RMSE, MAE, bias, correlation, coverage_pct, n_years

Only MWE is used; Gt columns are ignored because glacier area varies in WGMS.

Usage:
    python src/validate_regional.py
    python src/validate_regional.py --config conf/config_validate_regional.yaml
    python src/validate_regional.py --ensemble_base_dir /hpc/path/to/outputs \\
                                    --output_dir outputs/validation_regional
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
import yaml


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_ensemble_mwe(region_dir: Path, filename: str) -> pd.DataFrame | None:
    path = region_dir / filename
    if not path.exists():
        warnings.warn(f"Ensemble file not found: {path}")
        return None
    df = pd.read_csv(path)
    # Normalise column names: ensemble_ep_alea.py outputs total_std / epistemic_std /
    # structural_std; older ensemble_uncertainty.py used std_total / std_epistemic /
    # std_structural. Rename to the *_std convention so the rest of the script is uniform.
    rename = {
        "std_total":      "total_std",
        "std_epistemic":  "epistemic_std",
        "std_structural": "structural_std",
        "std_aleatoric":  "aleatoric_std",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"year", "median_mwe", "total_std"}
    missing = required - set(df.columns)
    if missing:
        warnings.warn(f"Ensemble file {path} missing columns: {missing}")
        return None
    return df.sort_values("year").reset_index(drop=True)


def _load_validation_mwe(val_dir: Path, wgms_abbrev: str) -> pd.DataFrame | None:
    path = val_dir / f"{wgms_abbrev}.csv"
    if not path.exists():
        warnings.warn(f"Validation file not found: {path}")
        return None
    df = pd.read_csv(path)
    required = {"year", "mwe", "mwe_sigma"}
    missing = required - set(df.columns)
    if missing:
        warnings.warn(f"Validation file {path} missing columns: {missing}")
        return None
    return df.sort_values("year").reset_index(drop=True)


def _merge_on_overlap(
    ens: pd.DataFrame,
    val: pd.DataFrame,
    min_years: int,
) -> pd.DataFrame | None:
    merged = pd.merge(ens, val[["year", "mwe", "mwe_sigma"]], on="year", how="inner")
    merged = merged.dropna(subset=["median_mwe", "mwe"]).reset_index(drop=True)
    if len(merged) < min_years:
        return None
    return merged


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(merged: pd.DataFrame, sigma_mult: float) -> dict:
    pred    = merged["median_mwe"].values
    obs     = merged["mwe"].values
    s_tot   = merged["total_std"].values
    diff    = pred - obs
    n       = len(diff)
    # Overlap: ensemble band [pred ± σ_mult·s_tot] intersects WGMS band [obs ± σ_mult·s_wgms].
    # With sigma_mult=1 (default) both bands are 1σ — a like-for-like comparison.
    # Two intervals [a,b] and [c,d] overlap iff a <= d and c <= b.
    s_wgms  = merged["mwe_sigma"].values
    within  = np.sum(
        (pred - sigma_mult * s_tot  <= obs + sigma_mult * s_wgms) &
        (obs  - sigma_mult * s_wgms <= pred + sigma_mult * s_tot)
    )
    return {
        "n_years":      n,
        "bias":         float(np.mean(diff)),
        "rmse":         float(np.sqrt(np.mean(diff ** 2))),
        "mae":          float(np.mean(np.abs(diff))),
        "corr":         float(np.corrcoef(pred, obs)[0, 1]) if n > 2 else np.nan,
        "coverage_pct": float(100.0 * within / n),
    }


# ---------------------------------------------------------------------------
# Individual time-series plot
# ---------------------------------------------------------------------------

def _plot_timeseries(
    merged: pd.DataFrame,
    region_name: str,
    sigma_mult: float,
    output_path: Path,
) -> None:
    years = merged["year"].values
    mu    = merged["median_mwe"].values
    s_tot = merged["total_std"].values

    fig, ax = plt.subplots(figsize=(11, 4))

    # Total uncertainty band
    ax.fill_between(
        years,
        mu - sigma_mult * s_tot,
        mu + sigma_mult * s_tot,
        alpha=0.20, color="steelblue",
        label=f"±{sigma_mult}σ total",
    )

    # Structural uncertainty band (if present)
    if "structural_std" in merged.columns:
        s_struct = merged["structural_std"].values
        ax.fill_between(
            years,
            mu - sigma_mult * s_struct,
            mu + sigma_mult * s_struct,
            alpha=0.25, color="mediumorchid",
            label=f"±{sigma_mult}σ structural",
        )

    # Epistemic uncertainty band (if present)
    if "epistemic_std" in merged.columns:
        s_epi = merged["epistemic_std"].values
        ax.fill_between(
            years,
            mu - sigma_mult * s_epi,
            mu + sigma_mult * s_epi,
            alpha=0.35, color="darkorange",
            label=f"±{sigma_mult}σ epistemic",
        )

    # Ensemble mean line
    ax.plot(years, mu, color="steelblue", lw=1.8, label="Ensemble median")

    # WGMS / Dussaillant: line + error bar dots (same colour throughout)
    ax.plot(merged["year"].values, merged["mwe"].values,
            color="black", lw=1.4, zorder=5)
    ax.errorbar(
        merged["year"].values,
        merged["mwe"].values,
        yerr=sigma_mult * merged["mwe_sigma"].values,
        fmt="o", color="black", ms=3.5, elinewidth=1.0, capsize=2.5, zorder=6,
        label=f"WGMS/Dussaillant ±{sigma_mult}σ",
    )

    metrics = _compute_metrics(merged, sigma_mult)
    ax.set_title(
        f"{region_name}  |  n={metrics['n_years']} yrs  "
        f"RMSE={metrics['rmse']:.3f}  bias={metrics['bias']:+.3f}  "
        f"r={metrics['corr']:.2f}  "
        f"coverage={metrics['coverage_pct']:.0f}% within ±{sigma_mult}σ",
        fontsize=9,
    )
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("MWE/yr (m w.e.)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-panel summary
# ---------------------------------------------------------------------------

def _plot_multipanel(
    region_data: dict[str, pd.DataFrame],
    region_names: dict[str, str],
    sigma_mult: float,
    output_path: Path,
) -> None:
    keys   = sorted(region_data.keys())
    n      = len(keys)
    ncols  = 3
    nrows  = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 3.2),
                             sharex=False, sharey=False)
    axes_flat = np.array(axes).ravel()

    for idx, abbrev in enumerate(keys):
        ax     = axes_flat[idx]
        merged = region_data[abbrev]
        name   = region_names.get(abbrev, abbrev)
        years  = merged["year"].values
        mu     = merged["median_mwe"].values
        s_tot  = merged["total_std"].values

        ax.fill_between(years,
                        mu - sigma_mult * s_tot, mu + sigma_mult * s_tot,
                        alpha=0.20, color="steelblue")
        if "structural_std" in merged.columns:
            ax.fill_between(years,
                            mu - sigma_mult * merged["structural_std"].values,
                            mu + sigma_mult * merged["structural_std"].values,
                            alpha=0.25, color="mediumorchid")
        ax.plot(years, mu, color="steelblue", lw=1.4)
        ax.plot(merged["year"].values, merged["mwe"].values,
                color="black", lw=1.2, zorder=5)
        ax.errorbar(merged["year"].values, merged["mwe"].values,
                    yerr=sigma_mult * merged["mwe_sigma"].values,
                    fmt="o", color="black", ms=2.0, elinewidth=0.8,
                    capsize=1.5, zorder=6)

        metrics = _compute_metrics(merged, sigma_mult)
        ax.set_title(
            f"{name}\nRMSE={metrics['rmse']:.3f}  r={metrics['corr']:.2f}  "
            f"cov={metrics['coverage_pct']:.0f}%",
            fontsize=7.5,
        )
        ax.axhline(0, color="black", lw=0.5, ls="--")
        ax.set_ylabel("MWE/yr", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(len(keys), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Ensemble vs. WGMS/Dussaillant — regional MWE/yr\n"
        f"Blue band: ±{sigma_mult}σ total  |  Purple: ±{sigma_mult}σ structural  |  "
        f"Black dots: WGMS ±{sigma_mult}σ",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-region scatter plot
# ---------------------------------------------------------------------------

def _plot_scatter_region(
    merged: pd.DataFrame,
    region_name: str,
    sigma_mult: float,
    output_path: Path,
) -> None:
    """
    Scatter plot for a single region: ensemble median vs. WGMS MWE.

    - Grey shaded band around the 1:1 line = ±sigma_mult × mwe_sigma (per-year
      WGMS observational uncertainty), showing where each observation "should" lie.
    - Vertical error bars = ±sigma_mult × ensemble total_std.
    - Points coloured by year to show temporal progression.
    """
    obs     = merged["mwe"].values
    pred    = merged["median_mwe"].values
    s_tot   = merged["total_std"].values
    s_wgms  = merged["mwe_sigma"].values
    years   = merged["year"].values

    lo  = min(np.nanmin(obs), np.nanmin(pred - sigma_mult * s_tot))
    hi  = max(np.nanmax(obs), np.nanmax(pred + sigma_mult * s_tot))
    pad = (hi - lo) * 0.06
    lo -= pad; hi += pad
    diag = np.linspace(lo, hi, 300)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Per-point WGMS uncertainty band: shade between diag ± each point's sigma.
    # Use fill_between with per-observation lower/upper sorted by obs value.
    sort_idx    = np.argsort(obs)
    obs_sorted  = obs[sort_idx]
    wgms_sorted = s_wgms[sort_idx]
    ax.fill_between(
        obs_sorted,
        obs_sorted - sigma_mult * wgms_sorted,
        obs_sorted + sigma_mult * wgms_sorted,
        color="lightgray", alpha=0.55, zorder=1,
        label=f"±{sigma_mult}σ WGMS obs. uncertainty",
    )
    ax.plot(diag, diag, "k--", lw=1.0, zorder=2, label="1:1 line")

    sc = ax.scatter(obs, pred, c=years, cmap="viridis", s=20, zorder=4,
                    alpha=0.85)
    ax.errorbar(obs, pred, yerr=sigma_mult * s_tot,
                fmt="none", ecolor="gray", elinewidth=0.7, capsize=2.0,
                alpha=0.6, zorder=3)
    plt.colorbar(sc, ax=ax, label="Year", shrink=0.8)

    metrics = _compute_metrics(merged, sigma_mult)
    ax.set_title(
        f"{region_name}\n"
        f"RMSE={metrics['rmse']:.3f}  bias={metrics['bias']:+.3f}  "
        f"r={metrics['corr']:.2f}  cov={metrics['coverage_pct']:.0f}%",
        fontsize=9,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("WGMS/Dussaillant MWE/yr (m w.e.)")
    ax.set_ylabel("Ensemble median MWE/yr (m w.e.)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Residual (bias) time series — one panel per region, with WGMS uncertainty band
# ---------------------------------------------------------------------------

def _plot_residuals(
    region_data: dict[str, pd.DataFrame],
    region_names: dict[str, str],
    sigma_mult: float,
    output_path: Path,
) -> None:
    keys  = sorted(region_data.keys())
    n     = len(keys)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 2.8),
                             sharex=False)
    axes_flat = np.array(axes).ravel()

    for idx, abbrev in enumerate(keys):
        ax     = axes_flat[idx]
        merged = region_data[abbrev]
        years  = merged["year"].values
        resid  = merged["median_mwe"].values - merged["mwe"].values
        s_wgms = merged["mwe_sigma"].values

        # WGMS observational uncertainty band around zero
        ax.fill_between(years,
                        -sigma_mult * s_wgms,
                         sigma_mult * s_wgms,
                        color="lightgray", alpha=0.55, zorder=1,
                        label=f"±{sigma_mult}σ WGMS obs. uncert.")

        ax.bar(years, resid, color=np.where(resid >= 0, "steelblue", "firebrick"),
               alpha=0.75, width=0.8, zorder=2)
        ax.axhline(0, color="black", lw=0.8, zorder=3)
        bias = float(np.mean(resid))
        ax.axhline(bias, color="orange", lw=1.2, ls="--", zorder=4,
                   label=f"mean bias={bias:+.3f}")
        ax.set_title(region_names.get(abbrev, abbrev), fontsize=8)
        ax.set_ylabel("residual (m w.e.)", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    for idx in range(len(keys), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Residuals: ensemble median − WGMS/Dussaillant (MWE/yr)\n"
        f"Grey band: ±{sigma_mult}σ WGMS observational uncertainty",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(cfg: dict) -> None:
    ensemble_base = Path(cfg["ensemble_base_dir"])
    val_dir       = Path(cfg["validation_dir"])
    output_dir    = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    region_map   = cfg["region_map"]        # {WGMS_abbrev: rgi_subdir, e.g. ISL: r06}
    region_names = cfg.get("region_names", {})
    ens_filename = cfg.get("ensemble_filename", "top_models_regional_mwe.csv")
    sigma_mult   = float(cfg.get("sigma_mult", 1))
    min_overlap  = int(cfg.get("min_overlap_years", 5))

    print(f"\n=== Regional MWE validation ===")
    print(f"  Ensemble base : {ensemble_base}")
    print(f"  Validation dir: {val_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Sigma mult    : {sigma_mult}  |  Min overlap years: {min_overlap}\n")

    region_data: dict[str, pd.DataFrame] = {}   # keyed by WGMS abbrev
    rgi_id: dict[str, str] = {}                 # abbrev → rgi subdir (e.g. r06)
    metrics_rows: list[dict] = []

    for abbrev, subdir in sorted(region_map.items()):
        region_dir = ensemble_base / subdir
        ens = _load_ensemble_mwe(region_dir, ens_filename)
        if ens is None:
            print(f"  [{abbrev}/{subdir}] SKIP — ensemble file missing")
            continue

        val = _load_validation_mwe(val_dir, abbrev)
        if val is None:
            print(f"  [{abbrev}/{subdir}] SKIP — validation file missing")
            continue

        merged = _merge_on_overlap(ens, val, min_overlap)
        if merged is None:
            print(f"  [{abbrev}/{subdir}] SKIP — fewer than {min_overlap} overlapping years")
            continue

        name    = region_names.get(abbrev, abbrev)
        yr_rng  = f"{int(merged['year'].min())}–{int(merged['year'].max())}"
        metrics = _compute_metrics(merged, sigma_mult)
        print(f"  [{abbrev}/{subdir}] {name:30s}  {yr_rng}  "
              f"n={metrics['n_years']:3d}  "
              f"RMSE={metrics['rmse']:.3f}  bias={metrics['bias']:+.3f}  "
              f"r={metrics['corr']:.2f}  cov={metrics['coverage_pct']:.0f}%")

        region_data[abbrev] = merged
        rgi_id[abbrev]      = subdir
        metrics_rows.append({
            "region": abbrev, "rgi": subdir, "name": name,
            **metrics,
            "year_min": int(merged["year"].min()),
            "year_max": int(merged["year"].max()),
        })

        # Files named by RGI number (e.g. timeseries_r06.png, scatter_r06.png)
        _plot_timeseries(
            merged, name, sigma_mult,
            output_dir / f"timeseries_{subdir}.png",
        )
        _plot_scatter_region(
            merged, name, sigma_mult,
            output_dir / f"scatter_{subdir}.png",
        )

    if not region_data:
        print("\nNo regions matched — check ensemble_base_dir and region_map in config.")
        return

    # Aggregated plots
    print(f"\n  Generating multi-panel summary ({len(region_data)} regions)...")
    _plot_multipanel(region_data, region_names, sigma_mult,
                     output_dir / "summary_multipanel.png")

    print("  Generating residual time series...")
    _plot_residuals(region_data, region_names, sigma_mult,
                    output_dir / "residuals_all_regions.png")

    # Metrics CSV
    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse").reset_index(drop=True)
    metrics_df.to_csv(output_dir / "validation_metrics.csv", index=False,
                      float_format="%.4f")
    print(f"\n  Saved validation_metrics.csv ({len(metrics_df)} regions)")

    print(f"\nDone. Outputs written to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ensemble regional MWE against WGMS/Dussaillant data."
    )
    parser.add_argument("--config", default="conf/config_validate_regional.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--ensemble_base_dir", default=None,
                        help="Override ensemble_base_dir from config.")
    parser.add_argument("--output_dir", default=None,
                        help="Override output_dir from config.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if args.ensemble_base_dir is not None:
        cfg["ensemble_base_dir"] = args.ensemble_base_dir
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    run_validation(cfg)


if __name__ == "__main__":
    main()
