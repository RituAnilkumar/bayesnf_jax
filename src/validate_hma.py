"""
src/validate_hma.py

Validates the High Mountain Asia (HMA) ensemble against CryoSat-2 and ICESat/ICESat-2
cumulative mass-change series from Qiuyu et al.

HMA is formed by combining r13 (Central Asia), r14 (S. Asia West), r15 (S. Asia East).
Ensemble regional Gt outputs (top_models_regional_gt.csv) are summed across the three
regions; uncertainties are propagated in quadrature.

Satellite data are cumulative Gt at sub-annual irregular intervals.  Each satellite
series is aligned to the ensemble cumulative at its first observation date so that
both are on the same zero reference.

Outputs written to {output_dir}/:
    hma_cumulative_gt.png    cumulative Gt: ensemble band + satellite overlay
    hma_annual_gt.png        annual Gt/yr: ensemble band + satellite annual averages
    hma_metrics.csv          RMSE, bias, r, coverage vs each satellite (annual)

Usage:
    python src/validate_hma.py
    python src/validate_hma.py --ensemble_base_dir /hpc/path/top5 \\
                               --output_dir outputs/validation_hma
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

# RGI subdirectories that make up HMA
HMA_REGIONS = {
    "r13": "Central Asia",
    "r14": "S. Asia West",
    "r15": "S. Asia East",
}

SATELLITE_STYLES = {
    "CryoSat-2":    {"color": "forestgreen", "marker": "s", "zorder": 5},
    "ICESat/ICESat-2": {"color": "purple",   "marker": "^", "zorder": 5},
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_regional_gt(region_dir: Path) -> pd.DataFrame | None:
    path = region_dir / "top_models_regional_gt.csv"
    if not path.exists():
        warnings.warn(f"top_models_regional_gt.csv not found in {region_dir}")
        return None
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)
    # Normalise column names (same as validate_regional.py)
    rename = {
        "std_total":      "total_std",
        "std_epistemic":  "epistemic_std",
        "std_structural": "structural_std",
        "std_aleatoric":  "aleatoric_std",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def _load_satellite(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip BOM and whitespace from column names
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["frac_year"] = df["date"].dt.year + (df["date"].dt.dayofyear - 1) / 365.25
    return df.sort_values("frac_year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Combine HMA regions
# ---------------------------------------------------------------------------

def combine_hma(ensemble_base: Path) -> pd.DataFrame | None:
    """Sum Gt across r13, r14, r15; propagate uncertainty in quadrature."""
    dfs = {}
    for rgi in HMA_REGIONS:
        df = _load_regional_gt(ensemble_base / rgi)
        if df is None:
            return None
        dfs[rgi] = df.set_index("year")

    # Align to common years
    common_years = sorted(
        set(dfs["r13"].index) & set(dfs["r14"].index) & set(dfs["r15"].index)
    )
    if not common_years:
        raise RuntimeError("No overlapping years across r13, r14, r15.")

    rows = []
    for yr in common_years:
        slices = [dfs[r].loc[yr] for r in HMA_REGIONS]
        median_gt = sum(s["median_gt"] for s in slices)

        def _quad(col: str) -> float:
            vals = [s[col] for s in slices if col in s.index]
            return float(np.sqrt(sum(v ** 2 for v in vals))) if vals else np.nan

        rows.append({
            "year":          yr,
            "median_gt":     float(median_gt),
            "epistemic_std": _quad("epistemic_std"),
            "structural_std": _quad("structural_std"),
            "aleatoric_std": _quad("aleatoric_std"),
            "total_std":     _quad("total_std"),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cumulative ensemble
# ---------------------------------------------------------------------------

def make_cumulative(
    ens: pd.DataFrame,
    ref_year: int,
    sigma_mult: float,
) -> pd.DataFrame:
    """
    Compute cumulative Gt from ref_year (set to 0 there).
    Uncertainty propagated in quadrature across years.
    """
    mask = ens["year"] >= ref_year
    sub  = ens[mask].copy().reset_index(drop=True)

    cum_median = np.cumsum(sub["median_gt"].values)
    cum_epi    = np.sqrt(np.cumsum(sub["epistemic_std"].values  ** 2))
    cum_struct = np.sqrt(np.cumsum(sub["structural_std"].values ** 2))
    cum_total  = np.sqrt(np.cumsum(sub["total_std"].values      ** 2))

    return pd.DataFrame({
        "year":          sub["year"].values,
        "cum_gt":        cum_median,
        "cum_epistemic": cum_epi,
        "cum_structural": cum_struct,
        "cum_total":     cum_total,
    })


def _interp_cum(cum_df: pd.DataFrame, frac_year: float) -> float:
    """Linearly interpolate ensemble cumulative Gt at a fractional year."""
    years = cum_df["year"].values.astype(float) + 0.5  # mid-year convention
    vals  = cum_df["cum_gt"].values
    return float(np.interp(frac_year, years, vals))


# ---------------------------------------------------------------------------
# Align satellite to ensemble cumulative
# ---------------------------------------------------------------------------

def align_satellite(sat: pd.DataFrame, cum_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift satellite cumulative series so its first observation aligns with
    the ensemble cumulative at that date.
    """
    sat = sat.copy()
    first_frac  = float(sat["frac_year"].iloc[0])
    ens_at_start = _interp_cum(cum_df, first_frac)
    sat_at_start = float(sat["mass_changes"].iloc[0])
    sat["mass_changes_aligned"] = sat["mass_changes"] - sat_at_start + ens_at_start
    return sat


# ---------------------------------------------------------------------------
# Annual averages from satellite (for annual rate plot and metrics)
# ---------------------------------------------------------------------------

def satellite_annual(sat: pd.DataFrame) -> pd.DataFrame:
    """Group satellite cumulative series into annual averages."""
    sat = sat.copy()
    sat["year"] = sat["date"].dt.year
    grp = sat.groupby("year").agg(
        mass_changes=("mass_changes", "mean"),
        mass_errors=("mass_errors", "mean"),
    ).reset_index()
    # Annual rate: difference of consecutive annual cumulative averages
    grp["annual_gt"]  = grp["mass_changes"].diff()
    grp["annual_err"] = grp["mass_errors"]   # conservative: keep original sigma
    return grp.dropna(subset=["annual_gt"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics (annual rate comparison)
# ---------------------------------------------------------------------------

def _compute_metrics(
    ens: pd.DataFrame,
    sat_annual: pd.DataFrame,
    sigma_mult: float,
    label: str,
) -> dict:
    merged = pd.merge(ens[["year", "median_gt", "total_std"]],
                      sat_annual[["year", "annual_gt", "annual_err"]],
                      on="year", how="inner").dropna()
    if len(merged) < 2:
        return {"source": label, "n_years": len(merged)}
    pred   = merged["median_gt"].values
    obs    = merged["annual_gt"].values
    s_ens  = merged["total_std"].values
    s_sat  = merged["annual_err"].values
    diff   = pred - obs
    within = int(np.sum(
        (pred - sigma_mult * s_ens <= obs + sigma_mult * s_sat) &
        (obs  - sigma_mult * s_sat <= pred + sigma_mult * s_ens)
    ))
    n = len(diff)
    return {
        "source":       label,
        "n_years":      n,
        "bias_gt":      float(np.mean(diff)),
        "rmse_gt":      float(np.sqrt(np.mean(diff ** 2))),
        "mae_gt":       float(np.mean(np.abs(diff))),
        "corr":         float(np.corrcoef(pred, obs)[0, 1]) if n > 2 else np.nan,
        "coverage_pct": float(100.0 * within / n),
    }


# ---------------------------------------------------------------------------
# Plot 1: cumulative Gt
# ---------------------------------------------------------------------------

def plot_cumulative(
    cum_df: pd.DataFrame,
    satellites: dict[str, pd.DataFrame],
    sigma_mult: float,
    output_path: Path,
) -> None:
    years  = cum_df["year"].values + 0.5   # plot at mid-year
    mu     = cum_df["cum_gt"].values
    s_tot  = cum_df["cum_total"].values
    s_str  = cum_df["cum_structural"].values
    s_epi  = cum_df["cum_epistemic"].values

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(years, mu - sigma_mult * s_tot, mu + sigma_mult * s_tot,
                    alpha=0.15, color="steelblue", label=f"±{sigma_mult}σ total")
    ax.fill_between(years, mu - sigma_mult * s_str, mu + sigma_mult * s_str,
                    alpha=0.22, color="mediumorchid", label=f"±{sigma_mult}σ structural")
    ax.fill_between(years, mu - sigma_mult * s_epi, mu + sigma_mult * s_epi,
                    alpha=0.30, color="darkorange", label=f"±{sigma_mult}σ epistemic")
    ax.plot(years, mu, color="steelblue", lw=1.8, label="Ensemble cumulative")

    for name, sat in satellites.items():
        sty = SATELLITE_STYLES.get(name, {"color": "gray", "marker": "o", "zorder": 4})
        ax.errorbar(
            sat["frac_year"].values,
            sat["mass_changes_aligned"].values,
            yerr=sigma_mult * sat["mass_errors"].values,
            fmt=sty["marker"], color=sty["color"], ms=3.5, lw=0.8,
            capsize=2.0, alpha=0.75, zorder=sty["zorder"],
            label=f"{name} ±{sigma_mult}σ",
        )

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative mass change (Gt)")
    ax.set_title(
        "HMA cumulative mass change — ensemble vs. CryoSat-2 & ICESat/ICESat-2\n"
        "r13 (Central Asia) + r14 (S. Asia West) + r15 (S. Asia East)\n"
        "Satellite series aligned to ensemble at each instrument's first observation"
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 2: annual Gt/yr
# ---------------------------------------------------------------------------

def plot_annual(
    ens: pd.DataFrame,
    satellites: dict[str, pd.DataFrame],
    sigma_mult: float,
    output_path: Path,
) -> None:
    years  = ens["year"].values
    mu     = ens["median_gt"].values
    s_tot  = ens["total_std"].values
    s_str  = ens["structural_std"].values
    s_epi  = ens["epistemic_std"].values

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(years, mu - sigma_mult * s_tot, mu + sigma_mult * s_tot,
                    alpha=0.15, color="steelblue", label=f"±{sigma_mult}σ total")
    ax.fill_between(years, mu - sigma_mult * s_str, mu + sigma_mult * s_str,
                    alpha=0.22, color="mediumorchid", label=f"±{sigma_mult}σ structural")
    ax.fill_between(years, mu - sigma_mult * s_epi, mu + sigma_mult * s_epi,
                    alpha=0.30, color="darkorange", label=f"±{sigma_mult}σ epistemic")
    ax.plot(years, mu, color="steelblue", lw=1.8, label="Ensemble median")

    for name, sat_annual in satellites.items():
        sty = SATELLITE_STYLES.get(name, {"color": "gray", "marker": "o", "zorder": 4})
        valid = sat_annual.dropna(subset=["annual_gt"])
        ax.errorbar(
            valid["year"].values,
            valid["annual_gt"].values,
            yerr=sigma_mult * valid["annual_err"].values,
            fmt=sty["marker"], color=sty["color"], ms=4, lw=0.8,
            capsize=2.5, alpha=0.8, zorder=sty["zorder"],
            label=f"{name} ±{sigma_mult}σ (annual avg)",
        )

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mass balance (Gt/yr)")
    ax.set_title(
        "HMA annual mass balance — ensemble vs. CryoSat-2 & ICESat/ICESat-2\n"
        "r13 (Central Asia) + r14 (S. Asia West) + r15 (S. Asia East)\n"
        f"Satellite: annual mean of cumulative series  |  Error bars ±{sigma_mult}σ"
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_hma_validation(cfg: dict) -> None:
    ensemble_base = Path(cfg["ensemble_base_dir"])
    val_dir       = Path(cfg.get("hma_val_dir",
                                  "validation_data/qiuyu_hma_full"))
    output_dir    = Path(cfg.get("hma_output_dir",
                                  "outputs/validation_hma"))
    sigma_mult    = float(cfg.get("sigma_mult", 2))
    ref_year      = int(cfg.get("hma_ref_year", 2003))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== HMA validation (r13 + r14 + r15) ===")
    print(f"  Ensemble base : {ensemble_base}")
    print(f"  Validation dir: {val_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Reference year: {ref_year}  |  Sigma mult: {sigma_mult}\n")

    # ------------------------------------------------------------------
    # 1. Build combined HMA ensemble
    # ------------------------------------------------------------------
    ens = combine_hma(ensemble_base)
    if ens is None:
        raise RuntimeError("Could not load top_models_regional_gt.csv for r13/r14/r15. "
                           "Check ensemble_base_dir.")
    print(f"  HMA ensemble: {len(ens)} years "
          f"({int(ens['year'].min())}–{int(ens['year'].max())})")
    print(f"  Peak annual Gt: {ens['median_gt'].min():.1f} to {ens['median_gt'].max():.1f}")

    cum_df = make_cumulative(ens, ref_year, sigma_mult)

    # ------------------------------------------------------------------
    # 2. Load and align satellite data
    # ------------------------------------------------------------------
    sat_files = {
        "CryoSat-2":        val_dir / "CryoSat2_self_mass_change.csv",
        "ICESat/ICESat-2":  val_dir / "ICESat11_2_mass_change.csv",
    }

    satellites_cum    = {}
    satellites_annual = {}
    for name, path in sat_files.items():
        if not path.exists():
            warnings.warn(f"Satellite file not found: {path} — skipped.")
            continue
        sat = _load_satellite(path)
        sat = align_satellite(sat, cum_df)
        satellites_cum[name]    = sat
        satellites_annual[name] = satellite_annual(sat)
        print(f"  {name}: {len(sat)} observations "
              f"({sat['date'].dt.year.min()}–{sat['date'].dt.year.max()})")

    # ------------------------------------------------------------------
    # 3. Plots
    # ------------------------------------------------------------------
    plot_cumulative(cum_df, satellites_cum, sigma_mult,
                    output_dir / "hma_cumulative_gt.png")
    plot_annual(ens, satellites_annual, sigma_mult,
                output_dir / "hma_annual_gt.png")

    # ------------------------------------------------------------------
    # 4. Metrics
    # ------------------------------------------------------------------
    metrics_rows = []
    for name, sat_annual in satellites_annual.items():
        metrics_rows.append(_compute_metrics(ens, sat_annual, sigma_mult, name))

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(output_dir / "hma_metrics.csv", index=False,
                          float_format="%.4f")
        print(f"\n  Saved hma_metrics.csv")
        print(metrics_df.to_string(index=False))

    print(f"\nDone. Outputs written to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate HMA ensemble (r13+r14+r15) against satellite altimetry."
    )
    parser.add_argument("--config", default="conf/config_validate_regional.yaml")
    parser.add_argument("--ensemble_base_dir", default=None)
    parser.add_argument("--output_dir",        default=None)
    parser.add_argument("--hma_val_dir",       default=None,
                        help="Directory containing CryoSat/ICESat CSV files.")
    parser.add_argument("--hma_ref_year",      type=int, default=None,
                        help="Reference year for cumulative (default 2003).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    cfg: dict = {}
    if config_path.exists():
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}

    if args.ensemble_base_dir: cfg["ensemble_base_dir"] = args.ensemble_base_dir
    if args.output_dir:        cfg["hma_output_dir"]    = args.output_dir
    if args.hma_val_dir:       cfg["hma_val_dir"]       = args.hma_val_dir
    if args.hma_ref_year:      cfg["hma_ref_year"]      = args.hma_ref_year

    run_hma_validation(cfg)


if __name__ == "__main__":
    main()
