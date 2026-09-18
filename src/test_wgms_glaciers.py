"""
src/test_wgms_glaciers.py

Per-glacier TESTING against the WGMS reference/benchmark glaciers held out as
the "testing" split (see build_reference_glaciers.py / build_reference_timeseries.py
/ conversation trail — these were never used for ensemble selection, only the
"validation" split was).

Two different year windows are used deliberately, per instruction:
  - PLOTS show each test glacier's entire available WGMS record (not just the
    pre-2000/post-2020 subset), for visual context, against the model's full
    prediction trajectory. The 2000-2020 training window is shaded so it's
    visually clear which portion did NOT feed into the metrics below.
  - METRICS (r, RMSE, MAE, coverage) use ONLY the pre-2000/post-2020 subset —
    i.e. exactly the rows already in reference_benchmark_mb_timeseries.csv
    with split=="testing" (that file already restricts to years <2000 or
    >2020 for every region — see build_reference_timeseries.py; no separate
    year-filtering logic is needed here, it's inherited from that file).

Coverage (2 sigma) is defined as: the percentage of WGMS testing
mass-balance values that fall within [model_mean - 2*std_total,
model_mean + 2*std_total].

Assumptions (see conversation trail — flag if any of these are wrong):
  - Model source: outputs/ensemble_pretrain_year/r{NN}/combined/ensemble_glacier.csv
    (the pooled ensemble across all pretrain-year groups), keyed by (rgi_id, year).
  - The FULL WGMS record for plotting is re-read fresh from
    validation_data/per_gla/mass_balance.csv (annual_balance is already m w.e., no conversion)
    for exactly the glacier_ids in the "testing" split — this file is NOT the
    same one used for metrics (reference_benchmark_mb_timeseries.csv only
    contains the tier-filtered, metrics-eligible years). This means
    mass_balance.csv must be present alongside the other two reference_*
    files wherever this script runs — an extra data dependency beyond what
    was needed for ensembling alone.
  - Metrics are reported both per-glacier (one row each) and pooled (one
    overall row combining every test glacier's testing-window points).
  - Regions with zero test-split glaciers (most regions with <4 eligible
    glaciers went entirely to validation — see reference_benchmark_region_summary.csv)
    simply contribute nothing here; that's expected, not an error.

Outputs (to output_dir):
  wgms_test_joined.csv          full per-glacier time series (all years) with
                                 model predictions attached where available
  wgms_test_metrics_per_glacier.csv   r, RMSE, MAE, coverage_2sig_pct per glacier
                                       (testing-window years only)
  wgms_test_metrics_overall.csv       one row: pooled metrics across all test glaciers
  timeseries_all_test_glaciers.png    multi-panel: every test glacier's full record
  individual/timeseries_{slug}.png    one file per test glacier

Usage:
    python src/test_wgms_glaciers.py \\
        --ensemble_base outputs/ensemble_pretrain_year \\
        --output_dir outputs/testing_wgms_glaciers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GLACIERS_CSV     = Path("validation_data/per_gla/reference_benchmark_glaciers.csv")
TIMESERIES_CSV   = Path("validation_data/per_gla/reference_benchmark_mb_timeseries.csv")
MASS_BALANCE_CSV = Path("validation_data/per_gla/mass_balance.csv")
TRAIN_YEAR_MIN, TRAIN_YEAR_MAX = 2000, 2020


def load_test_glaciers() -> pd.DataFrame:
    """One row per test-split glacier: glacier_id, NAME, RGI_ID, rgi_region."""
    ts = pd.read_csv(TIMESERIES_CSV)
    test_ids = ts.loc[ts["split"] == "testing", "glacier_id"].unique()
    if len(test_ids) == 0:
        raise RuntimeError(
            f"No glaciers with split=='testing' in {TIMESERIES_CSV} — "
            "every region may have had <4 eligible glaciers (all went to validation)."
        )
    meta = pd.read_csv(GLACIERS_CSV)
    meta = meta[(meta["status"] == "kept") & (meta["glacier_id"].isin(test_ids))]
    return meta[["glacier_id", "input_name", "matched_wgms_name", "rgi_id", "rgi_region"]].drop_duplicates()


def load_full_wgms_series(glacier_ids: list[int]) -> pd.DataFrame:
    """Full (unfiltered by year) WGMS annual_balance series for the given glacier_ids.

    mass_balance.csv's annual_balance is already in m w.e. — no conversion
    needed (confirmed by cross-checking against ref_mb_timeseries.csv, an
    official WGMS export whose ANNUAL_BALANCE is mm w.e. and exactly 1000x
    this file's annual_balance for the same glacier/year — see the same note
    in build_reference_timeseries.py::load_obs)."""
    mb = pd.read_csv(MASS_BALANCE_CSV, low_memory=False)
    sub = mb[mb["glacier_id"].isin(glacier_ids)].dropna(subset=["annual_balance"])
    sub = sub[["glacier_id", "year", "annual_balance"]].rename(columns={"year": "YEAR"})
    sub["obs_mwe"] = sub["annual_balance"]
    return sub[["glacier_id", "YEAR", "obs_mwe"]]


def load_testing_window_series() -> pd.DataFrame:
    """The metrics-eligible subset: split=='testing' rows only (already <2000 or >2020)."""
    ts = pd.read_csv(TIMESERIES_CSV)
    return ts[ts["split"] == "testing"][["glacier_id", "YEAR", "obs_mwe"]]


def load_model_glacier_preds(ensemble_base: Path, region_num: int) -> pd.DataFrame | None:
    path = ensemble_base / f"r{region_num:02d}" / "combined" / "ensemble_glacier.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"rgi_id", "year", "median_mwe", "std_total"}
    if not required.issubset(df.columns):
        return None
    return df[["rgi_id", "year", "median_mwe", "std_total"]].rename(columns={"year": "YEAR"})


def _metrics(obs: np.ndarray, pred: np.ndarray, std: np.ndarray) -> dict:
    n = len(obs)
    if n < 2:
        return {"n_years": n, "corr": np.nan, "rmse": np.nan, "mae": np.nan, "coverage_2sig_pct": np.nan}
    diff = pred - obs
    within = np.sum((obs >= pred - 2 * std) & (obs <= pred + 2 * std))
    return {
        "n_years": n,
        "corr": float(np.corrcoef(pred, obs)[0, 1]) if np.std(obs) > 0 and np.std(pred) > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae":  float(np.mean(np.abs(diff))),
        "coverage_2sig_pct": float(100.0 * within / n),
    }


def build(ensemble_base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (full_joined, testing_window_joined) — both with model_mean_mwe/model_std_total attached."""
    glaciers = load_test_glaciers()
    full_obs = load_full_wgms_series(glaciers["glacier_id"].tolist())
    window_obs = load_testing_window_series()

    full_obs = full_obs.merge(glaciers, on="glacier_id", how="left")
    window_obs = window_obs.merge(glaciers, on="glacier_id", how="left")

    model_cache: dict[int, pd.DataFrame | None] = {}

    def _attach_model(df: pd.DataFrame) -> pd.DataFrame:
        pieces = []
        for region_num, grp in df.groupby("rgi_region"):
            region_num = int(region_num)
            if region_num not in model_cache:
                model_cache[region_num] = load_model_glacier_preds(ensemble_base, region_num)
            model = model_cache[region_num]
            if model is None:
                print(f"  [r{region_num:02d}] WARNING — combined/ensemble_glacier.csv not found; "
                      f"{grp['glacier_id'].nunique()} test glacier(s) in this region will have no model predictions")
                grp = grp.copy()
                grp["model_mean_mwe"] = np.nan
                grp["model_std_total"] = np.nan
            else:
                grp = grp.merge(
                    model.rename(columns={"rgi_id": "rgi_id", "median_mwe": "model_mean_mwe",
                                           "std_total": "model_std_total"}),
                    left_on=["rgi_id", "YEAR"], right_on=["rgi_id", "YEAR"], how="left",
                )
            pieces.append(grp)
        return pd.concat(pieces, ignore_index=True) if pieces else df

    return _attach_model(full_obs), _attach_model(window_obs)


def compute_metrics(window_joined: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for glacier_id, grp in window_joined.groupby("glacier_id"):
        valid = grp.dropna(subset=["obs_mwe", "model_mean_mwe", "model_std_total"])
        m = _metrics(valid["obs_mwe"].values, valid["model_mean_mwe"].values, valid["model_std_total"].values)
        rows.append({
            "glacier_id": glacier_id,
            "name": grp["input_name"].iloc[0],
            "rgi_id": grp["rgi_id"].iloc[0],
            "rgi_region": int(grp["rgi_region"].iloc[0]),
            **m,
        })
    per_glacier = pd.DataFrame(rows)

    pooled = window_joined.dropna(subset=["obs_mwe", "model_mean_mwe", "model_std_total"])
    overall = _metrics(pooled["obs_mwe"].values, pooled["model_mean_mwe"].values, pooled["model_std_total"].values)
    overall["n_glaciers"] = pooled["glacier_id"].nunique()
    overall["n_glaciers_no_model_data"] = (
        window_joined["glacier_id"].nunique() - pooled["glacier_id"].nunique()
    )
    return per_glacier, overall


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(".", "").strip("_")


def _draw_glacier(ax, sub: pd.DataFrame, name: str, rgi_id: str, metrics_row: dict | None) -> None:
    sub = sub.sort_values("YEAR")
    yr = sub["YEAR"].values
    obs = sub["obs_mwe"].values
    pred = sub["model_mean_mwe"].values
    std  = sub["model_std_total"].values

    ax.axvspan(TRAIN_YEAR_MIN, TRAIN_YEAR_MAX, color="gray", alpha=0.12, zorder=0,
               label="Training window (excluded from metrics)")

    has_pred = ~np.isnan(pred)
    if has_pred.any():
        ax.fill_between(yr[has_pred], (pred - 2 * std)[has_pred], (pred + 2 * std)[has_pred],
                        alpha=0.20, color="steelblue", label="Model ±2σ")
        ax.plot(yr[has_pred], pred[has_pred], color="steelblue", lw=1.6, label="Model median")
    ax.plot(yr, obs, color="black", lw=1.3, marker="o", ms=3, label="WGMS observed")
    ax.axhline(0, color="black", lw=0.5, ls="--", zorder=0)

    title = f"{name}  ({rgi_id})"
    if metrics_row is not None and not np.isnan(metrics_row.get("rmse", np.nan)):
        title += (f"\n[testing years only] n={metrics_row['n_years']:.0f}  "
                  f"r={metrics_row['corr']:.2f}  RMSE={metrics_row['rmse']:.3f}  "
                  f"MAE={metrics_row['mae']:.3f}  Cov(2σ)={metrics_row['coverage_2sig_pct']:.0f}%")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("MWE/yr")


def plot_all(full_joined: pd.DataFrame, per_glacier_metrics: pd.DataFrame, output_dir: Path) -> None:
    glacier_ids = sorted(full_joined["glacier_id"].unique())
    metrics_by_id = per_glacier_metrics.set_index("glacier_id").to_dict("index")

    ncols = 2
    nrows = int(np.ceil(len(glacier_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 3.2))
    axes_flat = np.array(axes).ravel()
    for idx, gid in enumerate(glacier_ids):
        sub = full_joined[full_joined["glacier_id"] == gid]
        name = sub["input_name"].iloc[0]
        rgi_id = sub["rgi_id"].iloc[0]
        _draw_glacier(axes_flat[idx], sub, name, rgi_id, metrics_by_id.get(gid))
        if idx == 0:
            axes_flat[idx].legend(fontsize=6, ncol=2)
    for idx in range(len(glacier_ids), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.suptitle("WGMS testing glaciers — full record vs. model (grey band = excluded from metrics)",
                fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "timeseries_all_test_glaciers.png", dpi=150)
    plt.close(fig)
    print(f"  Saved timeseries_all_test_glaciers.png")

    ind_dir = output_dir / "individual"
    ind_dir.mkdir(exist_ok=True)
    for gid in glacier_ids:
        sub = full_joined[full_joined["glacier_id"] == gid]
        name = sub["input_name"].iloc[0]
        rgi_id = sub["rgi_id"].iloc[0]
        fig, ax = plt.subplots(figsize=(10, 4))
        _draw_glacier(ax, sub, name, rgi_id, metrics_by_id.get(gid))
        ax.legend(fontsize=7, ncol=2)
        ax.set_xlabel("Year")
        fig.tight_layout()
        fig.savefig(ind_dir / f"timeseries_{_slug(name)}.png", dpi=150)
        plt.close(fig)
    print(f"  Saved {len(glacier_ids)} individual plots -> {ind_dir}/")


def run(ensemble_base: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== WGMS per-glacier testing (testing-split reference/benchmark glaciers) ===")
    print(f"  Ensemble base: {ensemble_base}\n")

    full_joined, window_joined = build(ensemble_base)
    full_joined.to_csv(output_dir / "wgms_test_joined.csv", index=False, float_format="%.5f")
    print(f"  Saved wgms_test_joined.csv ({full_joined['glacier_id'].nunique()} test glaciers, "
          f"{len(full_joined)} total obs rows)")

    per_glacier, overall = compute_metrics(window_joined)
    per_glacier.to_csv(output_dir / "wgms_test_metrics_per_glacier.csv", index=False, float_format="%.4f")
    pd.DataFrame([overall]).to_csv(output_dir / "wgms_test_metrics_overall.csv", index=False, float_format="%.4f")
    print(f"  Saved wgms_test_metrics_per_glacier.csv and wgms_test_metrics_overall.csv")
    print(f"  Overall (pooled, testing years only): n_glaciers={overall['n_glaciers']} "
          f"(+{overall['n_glaciers_no_model_data']} with no model data)  "
          f"n_points={overall['n_years']}  r={overall['corr']:.3f}  RMSE={overall['rmse']:.3f}  "
          f"MAE={overall['mae']:.3f}  Coverage(2σ)={overall['coverage_2sig_pct']:.0f}%")

    plot_all(full_joined, per_glacier, output_dir)
    print(f"\nDone. Outputs written to {output_dir}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test the combined ensemble against WGMS testing-split glaciers.")
    p.add_argument("--ensemble_base", default="outputs/ensemble_pretrain_year")
    p.add_argument("--output_dir", default="outputs/testing_wgms_glaciers")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(Path(args.ensemble_base), Path(args.output_dir))


if __name__ == "__main__":
    main()
