"""
src/ensemble_ep_alea.py

Selects the best single model from a Hydra multirun sweep and decomposes
its predictive uncertainty into epistemic (parameter) and aleatoric components.

No structural/model uncertainty is computed — this script is for a single best
model rather than an ensemble. Use ensemble_uncertainty.py when structural
uncertainty from model configuration spread is also required.

Uncertainty decomposition (single model, law of total variance):

    std_epistemic = MC std of mu samples
                    (spread of n_ensemble weight posterior draws in predict.py)
    std_aleatoric = mean predicted noise sigma across weight draws
                    (non-NaN only when heteroscedastic=true was used in training)
    std_total     = sqrt(std_epistemic² + std_aleatoric²)
                    (equals std_epistemic when aleatoric is unavailable)

The best model is selected by the lowest composite score from hyperparam_tuning
scoring (loyo_rmse + glambie_rmse, same weights as the ensemble config).

Inputs (all read from the selected run directory):
    preds_full.csv            — rgi_id, year, mean, std
                                [+ aleatoric_std, epistemic_std, total_std
                                   when run with heteroscedastic=true]
    regional_annual_mwe.csv   — year, total_area_km2, mean, std

Outputs written to {output_dir}/:
    best_model_info.csv          — which run was selected and its scores
    best_model_glacier.csv       — per (rgi_id, year): mean_mwe, epistemic_std,
                                   aleatoric_std, total_std
    best_model_regional_mwe.csv  — per year MWE/yr: same uncertainty columns
    best_model_regional_gt.csv   — per year Gt/yr:  same uncertainty columns
    best_model_regional_gt.png
    best_model_regional_mwe.png
    best_model_cumulative_gt.png

Usage:
    python src/ensemble_ep_alea.py
    python src/ensemble_ep_alea.py --config conf/config_ensemble_uncertainty.yaml
    python src/ensemble_ep_alea.py --multirun_root /path/to/r06_run \\
                                   --output_dir outputs/best_model/r06
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

from src.hyperparam_tuning import build_results_df, add_composite_score
from src.ensemble_uncertainty import (
    _read_run_model_cfg,
    _load_glambie_wide,
    _glambie_combined_gt,
    _glambie_sources_mwe,
    _load_oggm_regional,
    _aggregate_aleatoric_to_regional,
    _savefig,
)


# ---------------------------------------------------------------------------
# Best-model selection
# ---------------------------------------------------------------------------

def pick_best_run(
    multirun_root: Path,
    test_years: list[int],
    loyo_weight: float,
    glambie_weight: float,
    min_runs: int,
) -> pd.Series:
    """
    Return the single heteroscedastic run with the lowest composite score.

    Only runs with heteroscedastic=True in their Hydra overrides are considered.
    Raises if no such runs are found or none have a valid composite score.
    """
    results_df = build_results_df(multirun_root, test_years, min_runs_per_region=min_runs)
    results_df = add_composite_score(results_df, loyo_weight=loyo_weight, glambie_weight=glambie_weight)

    # Filter to heteroscedastic runs only
    if "heteroscedastic" in results_df.columns:
        hetero_df = results_df[results_df["heteroscedastic"] == True].reset_index(drop=True)
        n_total   = len(results_df)
        n_hetero  = len(hetero_df)
        if hetero_df.empty:
            raise RuntimeError(
                f"No runs with heteroscedastic=True found among {n_total} runs in "
                f"{multirun_root}. This script requires heteroscedastic=True training. "
                "Use ensemble_uncertainty.py for homoscedastic runs."
            )
        if n_hetero < n_total:
            print(f"  Filtered to {n_hetero}/{n_total} heteroscedastic runs.")
        results_df = hetero_df
    else:
        print("  WARNING: 'heteroscedastic' column not found in overrides — "
              "cannot pre-filter. Will validate after loading preds_full.csv.")

    valid = results_df.dropna(subset=["composite"]).reset_index(drop=True)
    if valid.empty:
        raise RuntimeError(
            "No heteroscedastic runs have a valid composite score. "
            "Check that pretrain_cv completed (metrics_oos.csv present) and "
            "that uncertainty floors are applied in finetune for this region."
        )

    best_idx  = valid["composite"].idxmin()
    best_row  = valid.loc[best_idx]
    print(f"  Best run: {best_row['run_id']}  composite={best_row['composite']:.4f}  "
          f"loyo_rmse={best_row['loyo_rmse']:.4f}  "
          f"glambie_rmse={best_row.get('glambie_rmse', float('nan')):.4f}")
    print(f"  Run dir: {best_row['run_dir']}")
    return best_row


# ---------------------------------------------------------------------------
# Per-glacier uncertainty extraction
# ---------------------------------------------------------------------------

def load_glacier_uncertainties(run_dir: Path) -> pd.DataFrame:
    """
    Load preds_full.csv and return per-glacier uncertainty decomposition.

    Columns always present:
        rgi_id, year, mean_mwe, epistemic_std

    Columns present when run with heteroscedastic=true:
        aleatoric_std, total_std

    When heteroscedastic columns are absent, aleatoric_std = NaN and
    total_std = epistemic_std.
    """
    path = run_dir / "preds_full.csv"
    if not path.exists():
        raise FileNotFoundError(f"preds_full.csv not found in {run_dir}")

    df = pd.read_csv(path)

    # epistemic_std is written explicitly when heteroscedastic; otherwise 'std' is epistemic
    if "epistemic_std" in df.columns:
        epistemic_std = df["epistemic_std"].values
    else:
        epistemic_std = df["std"].values

    if "aleatoric_std" not in df.columns:
        raise ValueError(
            f"preds_full.csv in {run_dir} does not contain 'aleatoric_std' column. "
            "This run was not trained with model.heteroscedastic=true. "
            "Re-run the sweep with heteroscedastic=true, or use ensemble_uncertainty.py "
            "for homoscedastic runs."
        )

    aleatoric_std = df["aleatoric_std"].values
    total_std     = df["total_std"].values if "total_std" in df.columns else (
        np.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)
    )
    print("  Heteroscedastic predictions detected — aleatoric uncertainty available.")

    return pd.DataFrame({
        "rgi_id":        df["rgi_id"].values,
        "year":          df["year"].values,
        "mean_mwe":      df["mean"].values,
        "epistemic_std": epistemic_std,
        "aleatoric_std": aleatoric_std,
        "total_std":     total_std,
    })


# ---------------------------------------------------------------------------
# Regional uncertainty aggregation
# ---------------------------------------------------------------------------

def compute_regional_uncertainties(
    glacier_df: pd.DataFrame,
    run_dir: Path,
    inp_dir: str,
    reg_subdir: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Aggregate per-glacier uncertainties to annual regional series.

    Epistemic:  read directly from regional_annual_mwe.csv (already computed
                from MC mu sample spread in predict.py, which is correct).
    Aleatoric:  propagated from per-glacier aleatoric via area-weighting:
                sigma_regional_t = sqrt( Σ_i (area_i/total_area_t)² * sigma_i² )
    Total:      sqrt(epistemic² + aleatoric²)

    Returns:
        regional_df     — DataFrame with columns: year, median_mwe, epistemic_std,
                          aleatoric_std, total_std
        total_area_km2  — numpy array of regional area per year (for Gt conversion)
    """
    mwe_path = run_dir / "regional_annual_mwe.csv"
    if not mwe_path.exists():
        raise FileNotFoundError(f"regional_annual_mwe.csv not found in {run_dir}")

    mwe_df = pd.read_csv(mwe_path).sort_values("year").reset_index(drop=True)
    years  = mwe_df["year"].values

    if "total_area_km2" in mwe_df.columns:
        total_area_km2 = mwe_df["total_area_km2"].values
    else:
        # Derive from regional_annual_gt.csv (older run format without total_area column)
        gt_path = run_dir / "regional_annual_gt.csv"
        if gt_path.exists():
            gt_df      = pd.read_csv(gt_path).sort_values("year").reset_index(drop=True)
            mwe_mean   = mwe_df["mean"].values
            total_area_km2 = np.where(
                np.abs(mwe_mean) > 1e-10,
                gt_df["mean"].values / (mwe_mean * 1e-3),
                0.0,
            )
            print("  Note: total_area_km2 derived from regional_annual_gt.csv.")
        else:
            total_area_km2 = np.zeros(len(years))
            warnings.warn("total_area_km2 unavailable — Gt conversion will be zero.")

    # Regional epistemic: std column in regional_annual_mwe.csv is from MC spread
    epistemic_std = mwe_df["std"].values

    # Regional aleatoric: propagate from per-glacier via area-weighted aggregation.
    # _aggregate_aleatoric_to_regional expects column 'std_aleatoric', so rename.
    glacier_for_agg = glacier_df.rename(columns={"aleatoric_std": "std_aleatoric"})
    reg_alea_df = _aggregate_aleatoric_to_regional(glacier_for_agg, inp_dir, reg_subdir)
    if reg_alea_df is not None:
        year_to_alea = dict(zip(reg_alea_df["year"], reg_alea_df["std_aleatoric"]))
        aleatoric_std = np.array([year_to_alea.get(y, np.nan) for y in years])
        valid = (~np.isnan(aleatoric_std)).sum()
        print(f"  Regional aleatoric propagated from per-glacier ({valid}/{len(years)} years).")
    else:
        aleatoric_std = np.full(len(years), np.nan)

    # Total: combine in quadrature, ignoring NaN aleatoric gracefully
    alea_sq = np.where(np.isnan(aleatoric_std), 0.0, aleatoric_std ** 2)
    total_std = np.sqrt(epistemic_std ** 2 + alea_sq)

    regional_df = pd.DataFrame({
        "year":          years,
        "median_mwe":    mwe_df["mean"].values,
        "epistemic_std": epistemic_std,
        "aleatoric_std": aleatoric_std,
        "total_std":     total_std,
    })
    return regional_df, total_area_km2


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _shade(ax, years, mu, s_epi, s_alea, s_tot):
    """
    Draw uncertainty bands:
      - ±2σ total    (steelblue, outer — includes aleatoric if available)
      - ±2σ epistemic (darkorange, inner — parameter uncertainty only)
    """
    ax.fill_between(years, mu - 2 * s_tot, mu + 2 * s_tot,
                    alpha=0.20, color="steelblue", label="±2σ total")
    ax.fill_between(years, mu - 2 * s_epi, mu + 2 * s_epi,
                    alpha=0.30, color="darkorange", label="±2σ epistemic")
    ax.plot(years, mu, color="steelblue", lw=1.8, label="best-model mean")


def plot_best_model_gt(
    regional_gt: pd.DataFrame,
    glambie_wide_df,
    oggm_df: pd.DataFrame,
    total_area_km2: np.ndarray,
    output_dir: Path,
) -> None:
    years           = regional_gt["year"].values
    mu              = regional_gt["median_gt"].values
    total_area_mean = float(total_area_km2.mean())

    gb_combined = _glambie_combined_gt(glambie_wide_df, total_area_mean)
    gb_sources  = _glambie_sources_mwe(glambie_wide_df)

    fig, ax = plt.subplots(figsize=(12, 5))
    _shade(ax, years, mu,
           regional_gt["epistemic_std"].values,
           regional_gt["aleatoric_std"].fillna(0).values,
           regional_gt["total_std"].values)

    if not gb_combined.empty:
        ax.errorbar(gb_combined["year"].values, gb_combined["gt"].values,
                    yerr=gb_combined["gt_err"].values,
                    fmt="o", color="black", ms=4, lw=1.2, capsize=3, label="GLaMBIE combined")

    for source, color, marker in [("altimetry", "forestgreen", "s"), ("gravimetry", "purple", "^")]:
        df = gb_sources[source]
        if not df.empty:
            gt_vals = df["mwe"].values * total_area_mean * 1e-3
            gt_errs = df["mwe_err"].values * total_area_mean * 1e-3
            ax.errorbar(df["year"].values, gt_vals, yerr=gt_errs,
                        fmt=marker, color=color, ms=4, lw=1.2, capsize=3,
                        label=f"GLaMBIE {source}")

    if not oggm_df.empty:
        ax.plot(oggm_df["year"].values, oggm_df["gt"].values,
                color="red", lw=1.2, ls="--", label="OGGM")

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year"); ax.set_ylabel("Gt/yr")
    ax.set_title("Regional mass balance — best model (Gt/yr)\nOrange: epistemic  Blue: total (epistemic + aleatoric)")
    ax.legend(fontsize=8); fig.tight_layout()
    _savefig(fig, output_dir / "best_model_regional_gt.png")


def plot_best_model_mwe(
    regional_mwe: pd.DataFrame,
    glambie_wide_df,
    oggm_df: pd.DataFrame,
    total_area_km2: np.ndarray,
    output_dir: Path,
) -> None:
    years           = regional_mwe["year"].values
    mu              = regional_mwe["median_mwe"].values
    total_area_mean = float(total_area_km2.mean())

    gb_sources     = _glambie_sources_mwe(glambie_wide_df)
    gb_combined_gt = _glambie_combined_gt(glambie_wide_df, total_area_mean)

    fig, ax = plt.subplots(figsize=(12, 5))
    _shade(ax, years, mu,
           regional_mwe["epistemic_std"].values,
           regional_mwe["aleatoric_std"].fillna(0).values,
           regional_mwe["total_std"].values)

    if not gb_combined_gt.empty and total_area_mean > 0:
        gb_mwe     = gb_combined_gt["gt"].values / (total_area_mean * 1e-3)
        gb_mwe_err = gb_combined_gt["gt_err"].values / (total_area_mean * 1e-3)
        ax.errorbar(gb_combined_gt["year"].values, gb_mwe, yerr=gb_mwe_err,
                    fmt="o", color="black", ms=4, lw=1.0, capsize=3, label="GLaMBIE combined")

    for source, color, marker in [("altimetry", "forestgreen", "s"), ("gravimetry", "purple", "^")]:
        df = gb_sources[source]
        if not df.empty:
            ax.errorbar(df["year"].values, df["mwe"].values, yerr=df["mwe_err"].values,
                        fmt=marker, color=color, ms=4, lw=1.0, capsize=3,
                        label=f"GLaMBIE {source}")

    if not oggm_df.empty:
        ax.plot(oggm_df["year"].values, oggm_df["mwe"].values,
                color="red", lw=1.2, ls="--", label="OGGM")

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year"); ax.set_ylabel("MWE/yr")
    ax.set_title("Regional mass balance — best model (MWE/yr)\nOrange: epistemic  Blue: total (epistemic + aleatoric)")
    ax.legend(fontsize=8); fig.tight_layout()
    _savefig(fig, output_dir / "best_model_regional_mwe.png")


def plot_best_model_cumulative_gt(
    regional_gt: pd.DataFrame,
    glambie_wide_df,
    oggm_df: pd.DataFrame,
    total_area_km2: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Cumulative Gt with epistemic and total uncertainty bands.

    Cumulative sigma is propagated in quadrature assuming year-to-year independence:
        cum_sigma_t = sqrt( Σ_{s≤t} sigma_s² )
    """
    total_area_mean = float(total_area_km2.mean())
    gb_combined = _glambie_combined_gt(glambie_wide_df, total_area_mean)

    start_year = int(gb_combined["year"].min()) if not gb_combined.empty \
        else int(regional_gt["year"].min())

    mask   = regional_gt["year"].values >= start_year
    years  = regional_gt["year"].values[mask]
    cum_mu = np.cumsum(regional_gt["median_gt"].values[mask])

    cum_epi = np.sqrt(np.cumsum(regional_gt["epistemic_std"].values[mask] ** 2))
    cum_tot = np.sqrt(np.cumsum(regional_gt["total_std"].values[mask] ** 2))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(years, cum_mu - 2 * cum_tot, cum_mu + 2 * cum_tot,
                    alpha=0.15, color="steelblue", label="±2σ total")
    ax.fill_between(years, cum_mu - 2 * cum_epi, cum_mu + 2 * cum_epi,
                    alpha=0.25, color="darkorange", label="±2σ epistemic")
    ax.plot(years, cum_mu, color="steelblue", lw=1.8, label="Best-model cumulative mean")

    if not oggm_df.empty:
        og = oggm_df[oggm_df["year"] >= start_year].copy()
        if not og.empty:
            ax.plot(og["year"].values, np.cumsum(og["gt"].values),
                    color="red", lw=1.3, ls="--", label="OGGM")

    if not gb_combined.empty:
        gb_from = gb_combined[gb_combined["year"] >= start_year].copy()
        if not gb_from.empty:
            gb_cum     = gb_from["gt"].cumsum().values
            gb_cum_err = np.sqrt(np.cumsum(gb_from["gt_err"].values ** 2))
            ax.fill_between(gb_from["year"].values,
                            gb_cum - 1.96 * gb_cum_err, gb_cum + 1.96 * gb_cum_err,
                            alpha=0.20, color="black")
            ax.plot(gb_from["year"].values, gb_cum, "k-", lw=1.5, label="GLaMBIE combined")

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year"); ax.set_ylabel("Cumulative mass balance (Gt)")
    ax.set_title(f"Cumulative regional mass balance from {start_year} — best model\n"
                 "Orange: epistemic  Blue: total (epistemic + aleatoric)")
    ax.legend(fontsize=8); fig.tight_layout()
    _savefig(fig, output_dir / "best_model_cumulative_gt.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ep_alea(cfg: dict) -> None:
    multirun_root = Path(cfg["multirun_root"])
    output_dir    = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    loyo_w     = float(cfg.get("loyo_weight", 0.0))
    glambie_w  = float(cfg.get("glambie_weight", 1.0))
    test_years = list(cfg.get("glambie_test_years", [2021, 2022, 2023]))
    min_runs   = int(cfg.get("min_runs_per_region", 1))

    print(f"\n=== Best-model epistemic + aleatoric: {multirun_root.name} ===")

    # ------------------------------------------------------------------
    # 1. Select best run
    # ------------------------------------------------------------------
    best_row = pick_best_run(multirun_root, test_years, loyo_w, glambie_w, min_runs)
    run_dir  = Path(best_row["run_dir"])

    pd.DataFrame([best_row]).to_csv(output_dir / "best_model_info.csv", index=False)
    print(f"  Saved best_model_info.csv")

    # ------------------------------------------------------------------
    # 2. Per-glacier uncertainty
    # ------------------------------------------------------------------
    print("\n--- Per-glacier uncertainty ---")
    glacier_df = load_glacier_uncertainties(run_dir)
    glacier_df.to_csv(output_dir / "best_model_glacier.csv", index=False)
    print(f"  Saved best_model_glacier.csv  ({len(glacier_df)} rows)")

    # ------------------------------------------------------------------
    # 3. Read model config paths (for aleatoric propagation + aux data)
    # ------------------------------------------------------------------
    run_model_cfg = _read_run_model_cfg([run_dir])
    inp_dir   = run_model_cfg.get("inp_dir", "")
    reg_subdir = run_model_cfg.get("reg_subdir", "")
    glambie_path = run_model_cfg.get("glambie_path", "")

    # ------------------------------------------------------------------
    # 4. Regional MWE uncertainty
    # ------------------------------------------------------------------
    print("\n--- Regional uncertainty ---")
    regional_mwe, total_area_km2 = compute_regional_uncertainties(
        glacier_df, run_dir, inp_dir, reg_subdir
    )
    regional_mwe.to_csv(output_dir / "best_model_regional_mwe.csv", index=False)
    print(f"  Saved best_model_regional_mwe.csv  ({len(regional_mwe)} years)")

    # ------------------------------------------------------------------
    # 5. Regional Gt (MWE × total_area × 1e-3)
    # ------------------------------------------------------------------
    scale = total_area_km2 * 1e-3

    regional_gt = pd.DataFrame({
        "year":          regional_mwe["year"].values,
        "median_gt":     regional_mwe["median_mwe"].values     * scale,
        "epistemic_std": regional_mwe["epistemic_std"].values  * scale,
        "aleatoric_std": np.where(
            regional_mwe["aleatoric_std"].isna(),
            np.nan,
            regional_mwe["aleatoric_std"].fillna(0).values * scale,
        ),
        "total_std":     regional_mwe["total_std"].values      * scale,
    })
    regional_gt.to_csv(output_dir / "best_model_regional_gt.csv", index=False)
    print(f"  Saved best_model_regional_gt.csv")

    # ------------------------------------------------------------------
    # 6. Auxiliary data for plots
    # ------------------------------------------------------------------
    glambie_wide_df = _load_glambie_wide(glambie_path)
    if glambie_wide_df is not None:
        print(f"  Loaded GLaMBIE data from {glambie_path}")
    else:
        print("  GLaMBIE path not set or missing — GLaMBIE series skipped in plots.")

    oggm_df = _load_oggm_regional(inp_dir, reg_subdir)
    if not oggm_df.empty:
        print(f"  Loaded OGGM regional series ({len(oggm_df)} years).")
    else:
        print("  OGGM data not loaded — skipped in plots.")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")
    plot_best_model_gt(regional_gt, glambie_wide_df, oggm_df, total_area_km2, output_dir)
    plot_best_model_mwe(regional_mwe, glambie_wide_df, oggm_df, total_area_km2, output_dir)
    plot_best_model_cumulative_gt(regional_gt, glambie_wide_df, oggm_df, total_area_km2, output_dir)

    print(f"\nDone. Outputs written to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Best-model epistemic + aleatoric uncertainty from a Hydra multirun."
    )
    parser.add_argument("--config",        default="conf/config_ensemble_uncertainty.yaml",
                        help="Path to YAML config file (same format as ensemble_uncertainty).")
    parser.add_argument("--multirun_root", default=None,
                        help="Override multirun_root from config.")
    parser.add_argument("--output_dir",    default=None,
                        help="Override output_dir from config.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if args.multirun_root is not None:
        cfg["multirun_root"] = args.multirun_root
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    run_ep_alea(cfg)


if __name__ == "__main__":
    main()
