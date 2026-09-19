"""
src/test_glambie_regional.py

Regional-scale TESTING (not validation) of the combined ensemble against
GLaMBIE for years 2020-2024. This is testing, not validation, because GLaMBIE
post-2020 data was never used for ensemble/model selection (only pre-2020
GLaMBIE was used as a Stage-2 finetuning target; selection itself is now
WGMS-correlation-based — see src/wgms_validation.py).

Deliberately uses GLaMBIE's "combined" product only, regardless of
altimetry/gravimetry availability — unlike src/data_utils.py::load_glambie
(used at finetune time), which prefers altimetry/gravimetry and only falls
back to combined when a region has no primary source at all. Here every
region is compared on the same homogeneous "combined" product, per explicit
instruction, rather than a per-region mix of source types.

Assumptions (see conversation trail — flag if any of these are wrong):
  - Model source: {ensemble_base}/r{NN}/combined/ensemble_regional_mwe.csv
    (the pooled ensemble across all pretrain-year groups — the ensembling
    script's "combined" output, not any single pt{year} group).
  - GLaMBIE source: {data_root}/r{NN}/glambie_targets_r{NN}.csv, combined_mwe
    column only; year = floor(end_date), matching the convention used
    elsewhere in this codebase (data_utils.py::load_glambie).
  - Test years: 2020-2024 inclusive (glambie_test_years, matching the default
    used throughout the ensembling scripts).
  - Region-years with NaN combined_mwe, or a missing model/GLaMBIE file, are
    skipped with a printed note, not treated as an error — GLaMBIE combined
    is not available for every region/year (e.g. no 2023/2024 rows for some
    regions in the current data).
  - "Coverage (2 sigma)" = whether the model's ±2σ band
    [median_mwe - 2*std_total, median_mwe + 2*std_total] overlaps GLaMBIE's own
    ±2σ band [glambie_mwe - 2*glambie_mwe_err, glambie_mwe + 2*glambie_mwe_err]
    — a band-overlap check (same definition as src/test_regional_duss.py),
    deliberately different from the one-sided containment check used for the
    per-glacier WGMS testing in src/test_wgms_glaciers.py (WGMS obs there have
    no per-point uncertainty column to build a second band from). Assumes
    glambie_mwe_err is 1σ, consistent with every other error column in this
    codebase.
  - r/RMSE/MAE/coverage are reported once, pooled across all region-years
    (not asked for explicitly here, but computed and annotated the same way
    every other comparison plot in this codebase reports them — flag if you
    wanted the scatter plots bare).

Outputs (to output_dir):
  glambie_test_pairs.csv         region, year, model_mwe, model_std_total,
                                  glambie_mwe, glambie_mwe_err
  glambie_test_metrics.csv       one row: pooled r, RMSE, MAE, coverage_2sig_pct, n
  glambie_scatter_by_region.png  model vs. GLaMBIE combined, coloured by region
  glambie_scatter_by_year.png    model vs. GLaMBIE combined, coloured by year

Usage:
    python src/test_glambie_regional.py \\
        --ensemble_base outputs/ensemble_pretrain_year \\
        --output_dir outputs/testing_glambie_regional
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEST_YEARS = [2020, 2021, 2022, 2023, 2024]
N_REGIONS = 19


def _load_glambie_combined(path: Path) -> pd.DataFrame:
    """year, glambie_mwe, glambie_mwe_err — combined product only, NaN rows dropped."""
    if not path.exists():
        return pd.DataFrame(columns=["year", "glambie_mwe", "glambie_mwe_err"])
    df = pd.read_csv(path)
    df["year"] = np.floor(df["end_date"]).astype(int)
    df = df.dropna(subset=["combined_mwe", "combined_mwe_errors"])
    return df[["year", "combined_mwe", "combined_mwe_errors"]].rename(
        columns={"combined_mwe": "glambie_mwe", "combined_mwe_errors": "glambie_mwe_err"}
    )


def _load_model_regional_mwe(ensemble_base: Path, region_num: int) -> pd.DataFrame | None:
    path = ensemble_base / f"r{region_num:02d}" / "combined" / "ensemble_regional_mwe.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "median_mwe" not in df.columns or "std_total" not in df.columns:
        return None
    return df[["year", "median_mwe", "std_total"]]


def build_pairs(ensemble_base: Path, data_root: Path) -> pd.DataFrame:
    rows = []
    for region_num in range(1, N_REGIONS + 1):
        rlabel = f"r{region_num:02d}"
        model = _load_model_regional_mwe(ensemble_base, region_num)
        if model is None:
            print(f"  [{rlabel}] SKIP — combined/ensemble_regional_mwe.csv not found")
            continue

        glambie_path = data_root / rlabel / f"glambie_targets_{rlabel}.csv"
        gb = _load_glambie_combined(glambie_path)
        if gb.empty:
            print(f"  [{rlabel}] SKIP — no valid GLaMBIE combined data")
            continue

        merged = model.merge(gb, on="year", how="inner")
        merged = merged[merged["year"].isin(TEST_YEARS)].reset_index(drop=True)
        if merged.empty:
            print(f"  [{rlabel}] SKIP — no overlap in test years {TEST_YEARS}")
            continue

        merged.insert(0, "region", rlabel)
        rows.append(merged)
        print(f"  [{rlabel}] {len(merged)} test-year point(s)")

    if not rows:
        raise RuntimeError("No regions produced any GLaMBIE test pairs — "
                            "check ensemble_base/data_root paths.")
    return pd.concat(rows, ignore_index=True)


def compute_metrics(pairs: pd.DataFrame) -> dict:
    pred = pairs["median_mwe"].values
    obs  = pairs["glambie_mwe"].values
    std  = pairs["std_total"].values
    err  = pairs["glambie_mwe_err"].values
    diff = pred - obs
    n = len(diff)
    # Band-overlap coverage: model's own ±2σ band vs. GLaMBIE's own ±2σ band.
    m_lo, m_hi = pred - 2 * std, pred + 2 * std
    o_lo, o_hi = obs - 2 * err, obs + 2 * err
    within = np.sum((m_lo <= o_hi) & (o_lo <= m_hi))
    return {
        "n":              n,
        "corr":           float(np.corrcoef(pred, obs)[0, 1]) if n > 2 else float("nan"),
        "rmse":           float(np.sqrt(np.mean(diff ** 2))),
        "mae":            float(np.mean(np.abs(diff))),
        "coverage_2sig_pct": float(100.0 * within / n),
    }


def _scatter_common(ax, pairs: pd.DataFrame, metrics: dict) -> tuple[float, float]:
    obs, pred = pairs["glambie_mwe"].values, pairs["median_mwe"].values
    lo = min(obs.min(), pred.min())
    hi = max(obs.max(), pred.max())
    pad = (hi - lo) * 0.08
    lo -= pad; hi += pad
    diag = np.linspace(lo, hi, 200)
    ax.plot(diag, diag, "k--", lw=1.0, zorder=1, label="1:1")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("GLaMBIE combined MWE/yr (m w.e.)")
    ax.set_ylabel("Model median MWE/yr (m w.e.)")
    ax.set_title(
        f"Model vs. GLaMBIE combined ({TEST_YEARS[0]}–{TEST_YEARS[-1]})\n"
        f"n={metrics['n']}  r={metrics['corr']:.2f}  RMSE={metrics['rmse']:.3f}  "
        f"MAE={metrics['mae']:.3f}  Coverage(2σ)={metrics['coverage_2sig_pct']:.0f}%",
        fontsize=10,
    )
    return lo, hi


def plot_by_region(pairs: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    _scatter_common(ax, pairs, metrics)

    regions = sorted(pairs["region"].unique())
    cmap = matplotlib.colormaps["tab20"].resampled(max(len(regions), 1))
    for i, r in enumerate(regions):
        sub = pairs[pairs["region"] == r]
        ax.errorbar(sub["glambie_mwe"], sub["median_mwe"],
                    xerr=2 * sub["glambie_mwe_err"], yerr=2 * sub["std_total"],
                    fmt="o", color=cmap(i), ms=6, elinewidth=0.8, capsize=2.0,
                    alpha=0.85, label=r, zorder=3)
    ax.legend(fontsize=6, ncol=2, loc="upper left", title="Region")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_by_year(pairs: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    _scatter_common(ax, pairs, metrics)

    ax.errorbar(pairs["glambie_mwe"], pairs["median_mwe"],
                xerr=2 * pairs["glambie_mwe_err"], yerr=2 * pairs["std_total"],
                fmt="none", ecolor="gray", elinewidth=0.7, capsize=2.0, alpha=0.5, zorder=2)
    sc = ax.scatter(pairs["glambie_mwe"], pairs["median_mwe"], c=pairs["year"],
                    cmap="viridis", s=40, zorder=3, edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label="Year", shrink=0.8, ticks=TEST_YEARS)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def run(ensemble_base: Path, data_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== GLaMBIE combined regional testing ({TEST_YEARS[0]}-{TEST_YEARS[-1]}) ===")
    print(f"  Ensemble base: {ensemble_base}")
    print(f"  Data root    : {data_root}\n")

    pairs = build_pairs(ensemble_base, data_root)
    pairs.to_csv(output_dir / "glambie_test_pairs.csv", index=False, float_format="%.5f")
    print(f"\n  Saved glambie_test_pairs.csv ({len(pairs)} region-year points, "
          f"{pairs['region'].nunique()} regions)")

    metrics = compute_metrics(pairs)
    pd.DataFrame([metrics]).to_csv(output_dir / "glambie_test_metrics.csv", index=False,
                                    float_format="%.4f")
    print(f"  Pooled metrics: n={metrics['n']}  r={metrics['corr']:.3f}  "
          f"RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  "
          f"Coverage(2σ)={metrics['coverage_2sig_pct']:.0f}%")

    plot_by_region(pairs, metrics, output_dir / "glambie_scatter_by_region.png")
    plot_by_year(pairs, metrics, output_dir / "glambie_scatter_by_year.png")
    print(f"\nDone. Outputs written to {output_dir}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test the combined ensemble's regional MWE against GLaMBIE combined (2020-2024)."
    )
    p.add_argument("--ensemble_base", default="outputs/ensemble_pretrain_year",
                   help="Parent dir containing r{NN}/combined/ensemble_regional_mwe.csv")
    p.add_argument("--data_root", default="data_for_model",
                   help="Parent dir containing r{NN}/glambie_targets_r{NN}.csv")
    p.add_argument("--output_dir", default="outputs/testing_glambie_regional")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(Path(args.ensemble_base), Path(args.data_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
