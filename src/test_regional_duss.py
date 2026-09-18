"""
src/test_regional_duss.py

Regional TESTING (not validation — Dussaillant regional products were never
used for model/ensemble selection) of the combined ensemble against the
Dussaillant et al. regional annual mass-balance product, one CSV per RGI
region (WGMS-style abbreviation, e.g. ISL.csv = region 06).

Promoted from an earlier ad hoc script (outputs/detailed_val/validate_regional_wgms.py,
gitignored, not previously in src/) that produced the same "model vs.
Dussaillant regional product, with number-of-glaciers-measured overlay"
plots framed as validation. Reclassified here as testing and repointed at
the new data/ensemble locations. Logic is otherwise the same.

Assumptions (see conversation trail — flag if any of these are wrong):
  - Measured data now lives at validation_data/regional_wgms_duss/{CODE}.csv
    (columns: year, area_km2, mwe, mwe_sigma, gt, gt_sigma) — this directory
    does not exist in this checkout yet; it needs to be populated with the
    same ~19 files as the old outputs/detailed_val/measured_consol_gmb/regional_wgms_duss/
    location (byte-identical sample confirmed for ISL.csv) before this script
    can run for real, wherever it's run from.
  - Region 17 (Southern Andes) is still split across SA1.csv/SA2.csv in this
    dataset; combined here via the same area-weighted formula as the
    original script, since the model only produces one r17 estimate.
  - Region 20 (a Dussaillant-only "Antarctic mainland" split) is not
    modelled and is skipped, same as the original script.
  - Model source: {ensemble_base}/r{NN}/combined/ensemble_regional_mwe.csv
    (the pooled ensemble across all pretrain-year groups).
  - "Number of glaciers measured" overlay: re-derived from
    validation_data/per_gla/{mass_balance,glacier}.csv (the canonical copies
    already used elsewhere in this repo) via the gtng_region column, rather
    than the separate fog_number_insitu/ copy the old script kept — same
    counting logic (unique glacier_id per (region, year) with a non-null
    annual_balance), just pointed at the canonical files instead of a
    duplicate copy.
  - Metrics: added MAE alongside the original script's RMSE/r/coverage(2sigma),
    for consistency with the other testing scripts in this batch (the
    original script did not report MAE).

Outputs (to output_dir):
  regional_duss_joined.csv        stacked per-region joined table (all years)
  regional_duss_metrics.csv       per-region RMSE, MAE, r, coverage_2sig_pct, n
  timeseries_r{NN}_{CODE}.png     model vs. measured, + glacier-count bar overlay
  scatter_r{NN}_{CODE}.png        model vs. measured scatter, coloured by year

Usage:
    python src/test_regional_duss.py \\
        --measured_dir validation_data/regional_wgms_duss \\
        --ensemble_base outputs/ensemble_pretrain_year \\
        --output_dir outputs/testing_regional_duss
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd

MEASURED_LABEL = "Dussaillant et al."

# RGI region key ('01'..'19') -> (rgi_subdir, WGMS abbreviation, name)
REGION_META = {
    "01": ("r01", "ALA", "Alaska"),
    "02": ("r02", "WNA", "W. Canada & US"),
    "03": ("r03", "ACN", "Arctic Canada N"),
    "04": ("r04", "ACS", "Arctic Canada S"),
    "05": ("r05", "GRL", "Greenland periph."),
    "06": ("r06", "ISL", "Iceland"),
    "07": ("r07", "SJM", "Svalbard & Jan Mayen"),
    "08": ("r08", "SCA", "Scandinavia"),
    "09": ("r09", "RUA", "Russian Arctic"),
    "10": ("r10", "ASN", "North Asia"),
    "11": ("r11", "CEU", "Central Europe"),
    "12": ("r12", "CAU", "Caucasus"),
    "13": ("r13", "ASC", "Central Asia"),
    "14": ("r14", "ASW", "S. Asia West"),
    "15": ("r15", "ASE", "S. Asia East"),
    "16": ("r16", "TRP", "Low Latitudes"),
    "17": ("r17", "SAN", "Southern Andes"),
    "18": ("r18", "NZL", "New Zealand"),
    "19": ("r19", "ANT", "Antarctic & Subantarctic"),
    "20": ("r20", "ANT2", "Antarctic Mainland"),   # Dussaillant-only split, not modelled
}


# ---------------------------------------------------------------------------
# Glacier-count overlay (re-derived from the canonical WGMS FoG files)
# ---------------------------------------------------------------------------

def _region_key(gtng: str) -> str:
    """'11_central_europe' -> '11'"""
    return str(gtng).split("_")[0].zfill(2)


def load_glacier_counts(per_gla_dir: Path) -> pd.DataFrame:
    """region_key ('01'..'19'), year -> n_glaciers with in-situ measurements."""
    mb = pd.read_csv(per_gla_dir / "mass_balance.csv", low_memory=False)
    gl = pd.read_csv(per_gla_dir / "glacier.csv", low_memory=False)
    mb = mb.dropna(subset=["annual_balance"])
    merged = mb.merge(gl[["id", "gtng_region"]], left_on="glacier_id", right_on="id", how="left")
    merged = merged.dropna(subset=["gtng_region"])
    merged["region_key"] = merged["gtng_region"].apply(_region_key)
    return (merged.groupby(["region_key", "year"])["glacier_id"]
            .nunique().reset_index(name="n_glaciers"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_measured_by_region(measured_dir: Path) -> dict[int, pd.DataFrame]:
    result = {}
    for key, (_, code, _name) in REGION_META.items():
        region_num = int(key)
        if region_num in (17, 20):
            continue
        path = measured_dir / f"{code}.csv"
        if not path.exists():
            print(f"  WARNING: measured file not found, skipping: {path}")
            continue
        result[region_num] = pd.read_csv(path)[["year", "area_km2", "mwe", "mwe_sigma"]]

    sa1_path, sa2_path = measured_dir / "SA1.csv", measured_dir / "SA2.csv"
    if sa1_path.exists() and sa2_path.exists():
        sa1, sa2 = pd.read_csv(sa1_path), pd.read_csv(sa2_path)
        assert (sa1["year"].values == sa2["year"].values).all(), "SA1/SA2 year mismatch"
        combined_area = sa1["area_km2"] + sa2["area_km2"]
        result[17] = pd.DataFrame({
            "year": sa1["year"],
            "area_km2": combined_area,
            "mwe": (sa1["area_km2"] * sa1["mwe"] + sa2["area_km2"] * sa2["mwe"]) / combined_area,
            "mwe_sigma": np.sqrt((sa1["area_km2"] * sa1["mwe_sigma"]) ** 2 +
                                  (sa2["area_km2"] * sa2["mwe_sigma"]) ** 2) / combined_area,
        })
    else:
        print(f"  WARNING: SA1.csv/SA2.csv not found in {measured_dir} — region 17 skipped")
    return result


def load_model_regional(ensemble_base: Path, region_num: int) -> pd.DataFrame | None:
    path = ensemble_base / f"r{region_num:02d}" / "combined" / "ensemble_regional_mwe.csv"
    if not path.exists():
        print(f"  WARNING: model file not found, skipping: {path}")
        return None
    return pd.read_csv(path, usecols=["year", "median_mwe", "std_total"])


def build_joined(measured_by_region: dict[int, pd.DataFrame], ensemble_base: Path) -> pd.DataFrame:
    frames = []
    for region_num, measured in sorted(measured_by_region.items()):
        model = load_model_regional(ensemble_base, region_num)
        if model is None:
            continue
        model = model.rename(columns={"median_mwe": "model_median_mwe", "std_total": "model_std_total"})
        joined = measured.merge(model, on="year", how="left")
        assert len(joined) == len(measured)
        code = REGION_META[f"{region_num:02d}"][1]
        joined.insert(0, "region_code", code)
        joined.insert(0, "region_num", region_num)
        frames.append(joined)
    if not frames:
        raise RuntimeError("No regions matched — check measured_dir and ensemble_base.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(sub: pd.DataFrame) -> dict:
    val = sub.dropna(subset=["model_median_mwe", "mwe"])
    n_points = len(val)
    if n_points >= 2:
        obs, pred = val["mwe"].values, val["model_median_mwe"].values
        resid = pred - obs
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae  = float(np.mean(np.abs(resid)))
        r = float(np.corrcoef(pred, obs)[0, 1]) if obs.std() > 0 and pred.std() > 0 else np.nan
    else:
        rmse, mae, r = np.nan, np.nan, np.nan

    unc = sub.dropna(subset=["model_median_mwe", "model_std_total", "mwe", "mwe_sigma"])
    n_overlap_pairs = len(unc)
    if n_overlap_pairs > 0:
        pred, pstd = unc["model_median_mwe"].values, unc["model_std_total"].values
        obs, ounc  = unc["mwe"].values, unc["mwe_sigma"].values
        m_lo, m_hi = pred - 2 * pstd, pred + 2 * pstd
        o_lo, o_hi = obs - 2 * ounc, obs + 2 * ounc
        overlap = (m_lo <= o_hi) & (o_lo <= m_hi)
        cov2 = 100.0 * overlap.sum() / n_overlap_pairs
    else:
        cov2 = np.nan

    return {"n_points": n_points, "r": r, "rmse": rmse, "mae": mae,
            "n_overlap_pairs": n_overlap_pairs, "coverage_2sig_pct": cov2}


def _fmt(val: float, spec: str) -> str:
    return "N/A" if val is None or np.isnan(val) else format(val, spec)


def _fmt_pct(val: float) -> str:
    return "N/A" if val is None or np.isnan(val) else f"{val:.0f}%"


def _annotate_metrics(ax, metrics: dict, loc: str = "bottom") -> None:
    text = (f"RMSE={_fmt(metrics['rmse'], '.2f')}  MAE={_fmt(metrics['mae'], '.2f')}  |  "
            f"r={_fmt(metrics['r'], '.2f')}  |  Coverage (2σ)={_fmt_pct(metrics['coverage_2sig_pct'])}")
    y, va = (0.04, "bottom") if loc == "bottom" else (0.96, "top")
    ax.text(0.02, y, text, transform=ax.transAxes, fontsize=12, fontweight="bold", va=va, ha="left")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_timeseries(ax, sub: pd.DataFrame, counts: pd.DataFrame, region_key: str,
                      title: str, metrics: dict) -> None:
    sub = sub.sort_values("year")
    yr = sub["year"].values
    model_pred = sub["model_median_mwe"].values
    model_std  = sub["model_std_total"].values
    obs        = sub["mwe"].values
    obs_unc    = sub["mwe_sigma"].values
    has_obs_unc = np.isfinite(obs_unc).any()

    if has_obs_unc:
        ax.fill_between(yr, obs - 2 * obs_unc, obs + 2 * obs_unc, color="firebrick", alpha=0.20, zorder=1)
    ax.plot(yr, obs, "-", color="firebrick", lw=1.4, zorder=2)

    valid = ~np.isnan(model_pred)
    ax.fill_between(yr[valid], (model_pred - 2 * model_std)[valid], (model_pred + 2 * model_std)[valid],
                    color="steelblue", alpha=0.20, zorder=3)
    ax.plot(yr[valid], model_pred[valid], "-", color="steelblue", lw=1.6, zorder=4)
    ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual mass balance (MWE/yr)")
    _annotate_metrics(ax, metrics, loc="bottom")

    cnt = counts[(counts["region_key"] == region_key) &
                 (counts["year"] >= yr.min()) & (counts["year"] <= yr.max())].sort_values("year")
    ax2 = ax.twinx()
    ax2.bar(cnt["year"], cnt["n_glaciers"], color="gray", alpha=0.25, width=0.8, zorder=0)
    ax2.set_ylabel("Number of glaciers measured (in-situ)", color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax2.set_ylim(0, cnt["n_glaciers"].max() * 3 if not cnt.empty else 1)
    ax.set_xlim(yr.min() - 1, yr.max() + 1)

    model_handle = (mpatches.Patch(facecolor="steelblue", alpha=0.20), Line2D([0], [0], color="steelblue", lw=1.6))
    handles = [model_handle]
    labels = ["Model (±2σ)"]
    if has_obs_unc:
        obs_handle = (mpatches.Patch(facecolor="firebrick", alpha=0.20), Line2D([0], [0], color="firebrick", lw=1.4))
        labels.append(f"{MEASURED_LABEL} (±2σ)")
    else:
        obs_handle = Line2D([0], [0], color="firebrick", lw=1.4)
        labels.append(MEASURED_LABEL)
    handles.append(obs_handle)
    handles.append(mpatches.Patch(facecolor="gray", alpha=0.25))
    labels.append("# glaciers measured")
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc="upper right", fontsize=9)

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)


def _draw_scatter(ax, sub: pd.DataFrame, title: str, metrics: dict) -> None:
    obs, obs_unc = sub["mwe"].values, sub["mwe_sigma"].values
    pred, pred_std = sub["model_median_mwe"].values, sub["model_std_total"].values
    years = sub["year"].values

    valid = ~(np.isnan(obs) | np.isnan(pred))
    lo = min(np.nanmin(obs[valid] - 2 * obs_unc[valid]), np.nanmin(pred[valid] - 2 * pred_std[valid]))
    hi = max(np.nanmax(obs[valid] + 2 * obs_unc[valid]), np.nanmax(pred[valid] + 2 * pred_std[valid]))
    pad = (hi - lo) * 0.08
    lo -= pad; hi += pad
    diag = np.linspace(lo, hi, 200)
    ax.plot(diag, diag, "k--", lw=1.0, zorder=1, label="1:1")

    ax.errorbar(obs, pred, xerr=2 * obs_unc, yerr=2 * pred_std, fmt="none", ecolor="gray",
                elinewidth=0.8, capsize=2.5, alpha=0.6, zorder=2, label="±2σ")
    sc = ax.scatter(obs, pred, c=years, cmap="viridis", s=28, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Year")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured regional MB (MWE/yr)")
    ax.set_ylabel("Model regional MB (MWE/yr)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left")
    _annotate_metrics(ax, metrics, loc="bottom")


def plot_regions(joined: pd.DataFrame, counts: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    for region_num in sorted(joined["region_num"].unique()):
        sub = joined[joined["region_num"] == region_num].sort_values("year")
        code = sub["region_code"].iloc[0]
        region_key = f"{region_num:02d}"
        name = REGION_META[region_key][2]
        title = f"r{region_num:02d} {code} — {name}"

        metrics = _compute_metrics(sub)
        metrics_rows.append({"region_num": region_num, "region_code": code, "region_name": name, **metrics})

        fig, ax = plt.subplots(figsize=(12, 5.5))
        _draw_timeseries(ax, sub, counts, region_key, title, metrics)
        fig.tight_layout()
        fig.savefig(output_dir / f"timeseries_r{region_num:02d}_{code}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6.5))
        _draw_scatter(ax, sub, title, metrics)
        fig.tight_layout()
        fig.savefig(output_dir / f"scatter_r{region_num:02d}_{code}.png", dpi=150)
        plt.close(fig)

        print(f"  [{title}] n={metrics['n_points']} r={_fmt(metrics['r'], '.2f')} "
              f"RMSE={_fmt(metrics['rmse'], '.2f')} MAE={_fmt(metrics['mae'], '.2f')} "
              f"Cov2σ={_fmt_pct(metrics['coverage_2sig_pct'])}")

    return pd.DataFrame(metrics_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(measured_dir: Path, ensemble_base: Path, per_gla_dir: Path, output_dir: Path) -> None:
    print("=== Regional Dussaillant testing (regional_wgms_duss) ===")
    measured_by_region = load_measured_by_region(measured_dir)
    print(f"  Loaded {len(measured_by_region)} regions (17=SA1+SA2 combined)")

    joined = build_joined(measured_by_region, ensemble_base)
    n_matched = joined["model_median_mwe"].notna().sum()
    print(f"  Matched {n_matched}/{len(joined)} rows to model predictions")

    output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_dir / "regional_duss_joined.csv", index=False, float_format="%.6f")
    print(f"  Saved regional_duss_joined.csv")

    print("  Loading glacier in-situ counts...")
    counts = load_glacier_counts(per_gla_dir)

    metrics_df = plot_regions(joined, counts, output_dir)
    metrics_df.to_csv(output_dir / "regional_duss_metrics.csv", index=False, float_format="%.4f")
    print(f"  Saved regional_duss_metrics.csv")
    print(f"\nDone. Outputs written to {output_dir}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test the combined ensemble against Dussaillant regional MWE products."
    )
    p.add_argument("--measured_dir", default="validation_data/regional_wgms_duss")
    p.add_argument("--ensemble_base", default="outputs/ensemble_pretrain_year")
    p.add_argument("--per_gla_dir", default="validation_data/per_gla",
                   help="Directory containing mass_balance.csv/glacier.csv (for the glacier-count overlay).")
    p.add_argument("--output_dir", default="outputs/testing_regional_duss")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(Path(args.measured_dir), Path(args.ensemble_base), Path(args.per_gla_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
