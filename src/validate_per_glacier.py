"""
src/validate_per_glacier.py

Joins per-glacier ensemble predictions (top_models_glacier.csv) onto the
WGMS per-glacier reference time series (ref_mb_timeseries.csv), then
validates and plots model performance at the individual glacier level.

Steps
-----
1. Load ref_mb_timeseries.csv  (YEAR, rgi_id, ANNUAL_BALANCE in mm w.e.)
2. Convert ANNUAL_BALANCE mm w.e. → MWE/yr (÷ 1000)
3. For each glacier infer its RGI region (RGI60-01.* → r01) and load the
   corresponding top_models_glacier.csv from {ensemble_base_dir}/{rgi}/
4. Join on (rgi_id, YEAR); add model columns to the reference table
5. Save enriched CSV: {output_dir}/ref_mb_with_model.csv
6. Per-glacier and summary plots + metrics CSV

Metrics reported (all at 1σ, consistent with model output convention)
----------------------------------------------------------------------
  n_years       overlapping years
  bias          mean(model - obs)  in MWE/yr
  rmse          sqrt(mean((model-obs)²))
  mae           mean(|model-obs|)
  corr (r)      Pearson correlation
  coverage_pct  % years where obs falls within model ±2σ total band

Usage
-----
    python src/validate_per_glacier.py \\
        --ensemble_base_dir /hpc/path/top5_lim_glamb \\
        --output_dir outputs/validation_per_glacier
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

REF_CSV     = Path("validation_data/per_gla/ref_mb_timeseries.csv")
MM_TO_MWE   = 1e-3   # ANNUAL_BALANCE is mm w.e.; model is MWE/yr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgi_to_subdir(rgi_id: str) -> str:
    """'RGI60-01.09162'  →  'r01'"""
    try:
        region_num = int(rgi_id.split("-")[1].split(".")[0])
        return f"r{region_num:02d}"
    except Exception:
        raise ValueError(f"Cannot parse RGI region from '{rgi_id}'")


def _load_glacier_csv(ensemble_base: Path, subdir: str) -> pd.DataFrame | None:
    path = ensemble_base / subdir / "top_models_glacier.csv"
    if not path.exists():
        warnings.warn(f"top_models_glacier.csv not found: {path}")
        return None
    df = pd.read_csv(path)
    # Normalise column names for older runs that used std_* convention
    rename = {
        "std_total":      "total_std",
        "std_epistemic":  "epistemic_std",
        "std_structural": "structural_std",
        "std_aleatoric":  "aleatoric_std",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_joined(ref: pd.DataFrame, ensemble_base: Path) -> pd.DataFrame:
    """
    Join model predictions onto the reference table.

    Returns the reference table with added columns:
        model_mean_mwe, model_total_std, model_epistemic_std,
        model_structural_std, model_aleatoric_std
    """
    # Cache loaded regional glacier CSVs (one per region)
    region_cache: dict[str, pd.DataFrame] = {}
    subdirs = ref["rgi_id"].apply(_rgi_to_subdir)

    model_rows = []
    for _, row in ref.iterrows():
        subdir = _rgi_to_subdir(row["rgi_id"])
        if subdir not in region_cache:
            gl = _load_glacier_csv(ensemble_base, subdir)
            region_cache[subdir] = gl  # may be None

        gl = region_cache[subdir]
        if gl is None:
            model_rows.append({})
            continue

        match = gl[(gl["rgi_id"] == row["rgi_id"]) & (gl["year"] == row["YEAR"])]
        if match.empty:
            model_rows.append({})
            continue

        r = match.iloc[0]
        model_rows.append({
            "model_mean_mwe":      r.get("mean_mwe",      np.nan),
            "model_total_std":     r.get("total_std",     np.nan),
            "model_epistemic_std": r.get("epistemic_std", np.nan),
            "model_structural_std":r.get("structural_std",np.nan),
            "model_aleatoric_std": r.get("aleatoric_std", np.nan),
        })

    model_cols = ["model_mean_mwe", "model_total_std", "model_epistemic_std",
                  "model_structural_std", "model_aleatoric_std"]
    model_df = pd.DataFrame(model_rows, index=ref.index).reindex(columns=model_cols)
    joined   = pd.concat([ref, model_df], axis=1)
    return joined


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(sub: pd.DataFrame) -> dict:
    valid = sub.dropna(subset=["obs_mwe", "model_mean_mwe", "model_total_std"])
    if len(valid) < 2:
        return {"n_years": len(valid)}
    obs  = valid["obs_mwe"].values
    pred = valid["model_mean_mwe"].values
    s    = valid["model_total_std"].values
    diff = pred - obs
    n    = len(diff)
    within = int(np.sum((obs >= pred - 2 * s) & (obs <= pred + 2 * s)))
    return {
        "n_years":      n,
        "bias":         float(np.mean(diff)),
        "rmse":         float(np.sqrt(np.mean(diff ** 2))),
        "mae":          float(np.mean(np.abs(diff))),
        "corr":         float(np.corrcoef(pred, obs)[0, 1]),
        "coverage_pct": float(100.0 * within / n),
    }


# ---------------------------------------------------------------------------
# Plot 1: per-glacier time series (multi-panel)
# ---------------------------------------------------------------------------

def plot_timeseries(joined: pd.DataFrame, output_path: Path) -> None:
    glaciers = joined["rgi_id"].unique()
    ncols = 2
    nrows = int(np.ceil(len(glaciers) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 7, nrows * 3.5),
                             sharex=False)
    axes_flat = np.array(axes).ravel()

    for idx, rgi in enumerate(sorted(glaciers)):
        ax  = axes_flat[idx]
        sub = joined[joined["rgi_id"] == rgi].sort_values("YEAR")
        valid = sub.dropna(subset=["obs_mwe", "model_mean_mwe"])

        name = sub["NAME"].iloc[0]
        yr   = valid["YEAR"].values
        obs  = valid["obs_mwe"].values
        pred = valid["model_mean_mwe"].values
        s_tot  = valid["model_total_std"].values
        s_str  = valid["model_structural_std"].values if "model_structural_std" in valid else None
        s_epi  = valid["model_epistemic_std"].values  if "model_epistemic_std"  in valid else None

        ax.fill_between(yr, pred - s_tot, pred + s_tot,
                        alpha=0.20, color="steelblue", label="±1σ total")
        if s_str is not None:
            ax.fill_between(yr, pred - s_str, pred + s_str,
                            alpha=0.25, color="mediumorchid", label="±1σ structural")
        if s_epi is not None:
            ax.fill_between(yr, pred - s_epi, pred + s_epi,
                            alpha=0.35, color="darkorange", label="±1σ epistemic")
        ax.plot(yr, pred, color="steelblue", lw=1.6, label="Model median")
        ax.plot(yr, obs, color="black", lw=1.4, label="WGMS in-situ")

        m = _metrics(valid)
        ax.set_title(
            f"{name}  ({rgi})\n"
            f"n={m.get('n_years','?')}  "
            f"RMSE={m.get('rmse', float('nan')):.3f}  "
            f"bias={m.get('bias', float('nan')):+.3f}  "
            f"r={m.get('corr', float('nan')):.2f}  "
            f"cov2σ={m.get('coverage_pct', float('nan')):.0f}%",
            fontsize=8,
        )
        ax.axhline(0, color="black", lw=0.5, ls="--")
        ax.set_ylabel("MWE/yr (m w.e.)", fontsize=7)
        ax.legend(fontsize=6, ncol=2)
        ax.tick_params(labelsize=6)

    for idx in range(len(glaciers), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Per-glacier model vs. WGMS in-situ (MWE/yr)\n"
        "Blue: ±1σ total  |  Purple: ±1σ structural  |  Orange: ±1σ epistemic  |  Black: WGMS",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 2: per-glacier scatter (multi-panel)
# ---------------------------------------------------------------------------

def plot_scatter(joined: pd.DataFrame, output_path: Path) -> None:
    glaciers = sorted(joined["rgi_id"].unique())
    ncols = 3
    nrows = int(np.ceil(len(glaciers) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.5))
    axes_flat = np.array(axes).ravel()

    for idx, rgi in enumerate(glaciers):
        ax    = axes_flat[idx]
        sub   = joined[joined["rgi_id"] == rgi].dropna(
            subset=["obs_mwe", "model_mean_mwe", "model_total_std"])
        name  = joined[joined["rgi_id"] == rgi]["NAME"].iloc[0]
        if sub.empty:
            axes_flat[idx].set_title(f"{name}\nno matched data", fontsize=8)
            axes_flat[idx].set_visible(True)
            continue
        obs   = sub["obs_mwe"].values
        pred  = sub["model_mean_mwe"].values
        s_tot = sub["model_total_std"].values
        years = sub["YEAR"].values

        lo  = min(np.nanmin(obs), np.nanmin(pred - s_tot))
        hi  = max(np.nanmax(obs), np.nanmax(pred + s_tot))
        pad = (hi - lo) * 0.06
        lo -= pad; hi += pad
        diag = np.linspace(lo, hi, 200)

        # 1:1 line
        ax.plot(diag, diag, "k--", lw=1.0, zorder=1)

        # Scatter coloured by year with 1σ model error bars
        sc = ax.scatter(obs, pred, c=years, cmap="viridis", s=18,
                        zorder=4, alpha=0.85)
        ax.errorbar(obs, pred, yerr=s_tot,
                    fmt="none", ecolor="gray", elinewidth=0.7,
                    capsize=2.0, alpha=0.55, zorder=3)
        plt.colorbar(sc, ax=ax, label="Year", shrink=0.75)

        m = _metrics(sub)
        ax.set_title(
            f"{name}\n"
            f"r={m.get('corr', float('nan')):.2f}  "
            f"RMSE={m.get('rmse', float('nan')):.3f}  "
            f"cov2σ={m.get('coverage_pct', float('nan')):.0f}%",
            fontsize=8,
        )
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("WGMS in-situ (MWE/yr)", fontsize=7)
        ax.set_ylabel("Model median (MWE/yr)", fontsize=7)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6)

    for idx in range(len(glaciers), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Per-glacier scatter: model median vs. WGMS in-situ\n"
        "Error bars: ±1σ model total  |  Colour: year",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Plot 3: residuals per glacier
# ---------------------------------------------------------------------------

def plot_residuals(joined: pd.DataFrame, output_path: Path) -> None:
    glaciers = sorted(joined["rgi_id"].unique())
    ncols = 2
    nrows = int(np.ceil(len(glaciers) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 7, nrows * 3.0))
    axes_flat = np.array(axes).ravel()

    for idx, rgi in enumerate(glaciers):
        ax  = axes_flat[idx]
        sub = joined[joined["rgi_id"] == rgi].dropna(
            subset=["obs_mwe", "model_mean_mwe"]).sort_values("YEAR")
        name  = joined[joined["rgi_id"] == rgi]["NAME"].iloc[0]
        if sub.empty:
            axes_flat[idx].set_title(f"{name}\nno matched data", fontsize=8)
            continue
        years = sub["YEAR"].values
        resid = sub["model_mean_mwe"].values - sub["obs_mwe"].values
        s_tot = sub["model_total_std"].values

        # Model ±1σ band around zero
        ax.fill_between(years, -s_tot, s_tot,
                        color="steelblue", alpha=0.18, zorder=1,
                        label="±1σ model total")

        ax.bar(years, resid,
               color=np.where(resid >= 0, "steelblue", "firebrick"),
               alpha=0.75, width=0.8, zorder=2)
        ax.axhline(0, color="black", lw=0.8, zorder=3)
        bias = float(np.mean(resid))
        ax.axhline(bias, color="orange", lw=1.2, ls="--", zorder=4,
                   label=f"mean bias={bias:+.3f}")
        ax.set_title(f"{name}  ({rgi})", fontsize=8)
        ax.set_ylabel("residual (m w.e.)", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    for idx in range(len(glaciers), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Residuals: model median − WGMS in-situ (MWE/yr)\n"
        "Blue band: ±1σ model total uncertainty",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    ensemble_base = Path(cfg["ensemble_base_dir"])
    ref_csv       = Path(cfg.get("ref_csv", str(REF_CSV)))
    output_dir    = Path(cfg.get("per_glacier_output_dir",
                                  "outputs/validation_per_glacier"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Per-glacier validation ===")
    print(f"  Ensemble base : {ensemble_base}")
    print(f"  Reference CSV : {ref_csv}")
    print(f"  Output dir    : {output_dir}\n")

    # ------------------------------------------------------------------
    # 1. Load reference
    # ------------------------------------------------------------------
    ref = pd.read_csv(ref_csv)
    # Convert ANNUAL_BALANCE from mm w.e. to MWE/yr
    ref["obs_mwe"] = ref["ANNUAL_BALANCE"] * MM_TO_MWE

    print(f"  Reference: {len(ref)} rows | "
          f"{ref['rgi_id'].nunique()} glaciers | "
          f"{int(ref['YEAR'].min())}–{int(ref['YEAR'].max())}")

    # ------------------------------------------------------------------
    # 2. Join model predictions
    # ------------------------------------------------------------------
    joined = build_joined(ref, ensemble_base)
    n_matched = joined["model_mean_mwe"].notna().sum()
    print(f"  Matched {n_matched}/{len(joined)} rows to model predictions")

    # ------------------------------------------------------------------
    # 3. Save enriched CSV
    # ------------------------------------------------------------------
    out_csv = output_dir / "ref_mb_with_model.csv"
    joined.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"  Saved ref_mb_with_model.csv")

    # ------------------------------------------------------------------
    # 4. Metrics per glacier
    # ------------------------------------------------------------------
    metrics_rows = []
    for rgi in sorted(joined["rgi_id"].unique()):
        sub  = joined[joined["rgi_id"] == rgi]
        name = sub["NAME"].iloc[0]
        m    = _metrics(sub)
        yr_min = int(sub.dropna(subset=["model_mean_mwe"])["YEAR"].min()) \
            if sub["model_mean_mwe"].notna().any() else np.nan
        yr_max = int(sub.dropna(subset=["model_mean_mwe"])["YEAR"].max()) \
            if sub["model_mean_mwe"].notna().any() else np.nan
        metrics_rows.append({"rgi_id": rgi, "name": name,
                              "year_min": yr_min, "year_max": yr_max, **m})
        print(f"  [{rgi}] {name:20s}  "
              f"n={m.get('n_years','?'):3}  "
              f"RMSE={m.get('rmse', float('nan')):.3f}  "
              f"bias={m.get('bias', float('nan')):+.3f}  "
              f"r={m.get('corr', float('nan')):.2f}  "
              f"cov2σ={m.get('coverage_pct', float('nan')):.0f}%")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "per_glacier_metrics.csv",
                      index=False, float_format="%.4f")
    print(f"\n  Saved per_glacier_metrics.csv")

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    print("\n  Generating plots...")
    plot_timeseries(joined, output_dir / "timeseries_per_glacier.png")
    plot_scatter(joined,    output_dir / "scatter_per_glacier.png")
    plot_residuals(joined,  output_dir / "residuals_per_glacier.png")

    print(f"\nDone. Outputs written to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate per-glacier ensemble vs. WGMS in-situ reference."
    )
    p.add_argument("--config", default="conf/config_validate_regional.yaml")
    p.add_argument("--ensemble_base_dir", default=None,
                   help="Root directory containing r01/, r02/, ... sub-folders.")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--ref_csv", default=None,
                   help="Override path to ref_mb_timeseries.csv.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}

    if args.ensemble_base_dir: cfg["ensemble_base_dir"]       = args.ensemble_base_dir
    if args.output_dir:        cfg["per_glacier_output_dir"]  = args.output_dir
    if args.ref_csv:           cfg["ref_csv"]                 = args.ref_csv

    run(cfg)


if __name__ == "__main__":
    main()
