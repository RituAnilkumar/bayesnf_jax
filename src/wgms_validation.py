"""
src/wgms_validation.py

Per-run WGMS reference/benchmark-glacier validation & testing metrics, used by
the revised ensemble-selection scheme in src/ensemble_common.py.

Consumes the files built by build_reference_glaciers.py / build_reference_timeseries.py:
    validation_data/per_gla/reference_benchmark_mb_timeseries.csv
    validation_data/per_gla/reference_benchmark_region_summary.csv

For a given RGI region, three tiers exist (see reference_benchmark_region_summary.csv):
    tier 1 — WGMS obs restricted to years <2000 or >2020 (no training-window overlap)
    tier 2 — fallback: full WGMS record including 2000-2020 (only used when tier 1
             had zero eligible glaciers for that region)
    tier 3 — no usable WGMS data at all for that region; callers must fall back to
             a LOYO/LOGO-only selection criterion (this module reports
             available=False, it does not compute a LOYO/LOGO score itself).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

TIMESERIES_CSV      = Path("validation_data/per_gla/reference_benchmark_mb_timeseries.csv")
REGION_SUMMARY_CSV  = Path("validation_data/per_gla/reference_benchmark_region_summary.csv")


@lru_cache(maxsize=1)
def _load_timeseries() -> pd.DataFrame:
    if not TIMESERIES_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(TIMESERIES_CSV)


@lru_cache(maxsize=1)
def _load_region_summary() -> pd.DataFrame:
    if not REGION_SUMMARY_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(REGION_SUMMARY_CSV)


def region_tier(region_num: int) -> int:
    """Tier used for this region: 1, 2, or 3 (3 = no WGMS data -> LOYO/LOGO fallback)."""
    summary = _load_region_summary()
    if summary.empty:
        return 3
    row = summary[summary["rgi_region"] == region_num]
    if row.empty:
        return 3
    return int(row.iloc[0]["tier_used"])


def _metrics_from_residuals(obs: np.ndarray, pred: np.ndarray) -> dict:
    obs, pred = np.asarray(obs, dtype=float), np.asarray(pred, dtype=float)
    n = len(obs)
    if n < 2:
        return {"rmse": float("nan"), "corr": float("nan"), "medae": float("nan"), "n_points": n}
    diff = pred - obs
    rmse  = float(np.sqrt(np.mean(diff ** 2)))
    medae = float(np.median(np.abs(diff)))
    corr  = float(np.corrcoef(pred, obs)[0, 1]) if np.std(obs) > 0 and np.std(pred) > 0 else float("nan")
    return {"rmse": rmse, "corr": corr, "medae": medae, "n_points": n}


def _empty_result(tier: int, n_glaciers: int = 0) -> dict:
    return {
        "available": False, "rmse": float("nan"), "corr": float("nan"),
        "medae": float("nan"), "n_points": 0, "n_glaciers": n_glaciers,
        "tier_used": tier,
    }


def compute_wgms_metrics(
    run_dir: Path,
    region_num: int | None,
    split: str = "validation",
    period_mean: bool = False,
) -> dict:
    """
    Compute WGMS {split} metrics for one candidate run's preds_full.csv.

    Returns a dict with keys: available, rmse, corr, medae, n_points,
    n_glaciers, tier_used. `available=False` means this region/run has no
    usable WGMS data at this tier — callers must use a non-WGMS fallback
    (e.g. LOYO/LOGO R²) for selection.

    period_mean=True aggregates each glacier's residuals to a single
    (glacier-mean obs, glacier-mean pred) point before computing metrics —
    the fallback used when too few runs have usable point-wise data.
    """
    if region_num is None:
        return _empty_result(tier=3)

    tier = region_tier(region_num)
    if tier == 3:
        return _empty_result(tier=3)

    ts = _load_timeseries()
    if ts.empty:
        return _empty_result(tier=tier)

    sub = ts[(ts["rgi_region"] == region_num) & (ts["split"] == split)]
    if sub.empty:
        return _empty_result(tier=tier)

    preds_path = Path(run_dir) / "preds_full.csv"
    if not preds_path.exists():
        return _empty_result(tier=tier, n_glaciers=sub["glacier_id"].nunique())

    preds = pd.read_csv(preds_path, usecols=["rgi_id", "year", "mean"]).rename(
        columns={"rgi_id": "RGI_ID", "year": "YEAR", "mean": "pred_mwe"}
    )
    joined = sub.merge(preds, on=["RGI_ID", "YEAR"], how="inner")
    if joined.empty:
        return _empty_result(tier=tier, n_glaciers=sub["glacier_id"].nunique())

    if period_mean:
        agg = (joined.groupby("glacier_id")
               .agg(obs_mwe=("obs_mwe", "mean"), pred_mwe=("pred_mwe", "mean"))
               .reset_index())
        m = _metrics_from_residuals(agg["obs_mwe"].values, agg["pred_mwe"].values)
        m["n_glaciers"] = len(agg)
    else:
        m = _metrics_from_residuals(joined["obs_mwe"].values, joined["pred_mwe"].values)
        m["n_glaciers"] = joined["glacier_id"].nunique()

    m["available"] = True
    m["tier_used"] = tier
    return m


# ---------------------------------------------------------------------------
# Shared run-selection logic (used by ensemble_common.py, ensemble_ep_alea.py,
# and ensemble_uncertainty.py). Lives here rather than in ensemble_common.py
# to avoid a circular import: ensemble_common.py already imports helpers from
# ensemble_uncertainty.py, so ensemble_uncertainty.py cannot import the
# selection logic back from ensemble_common.py.
# ---------------------------------------------------------------------------

def _minmax_norm_series(s: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]. Returns zeros if all values are equal."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def _region_num_from_label(region_label) -> int | None:
    """'r06' -> 6. Returns None if unparseable."""
    import re
    m = re.match(r"r(\d+)$", str(region_label))
    return int(m.group(1)) if m else None


def _recompute_wgms_period_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute wgms_val_* columns using per-glacier period-means instead of
    point-wise (glacier, year) residuals — the fallback used when too few
    runs have usable point-wise validation data."""
    df = df.copy()
    cols = {"wgms_val_rmse": [], "wgms_val_corr": [], "wgms_val_medae": [],
            "wgms_val_n_points": [], "wgms_val_available": []}
    for _, row in df.iterrows():
        region_num = _region_num_from_label(row.get("region"))
        m = compute_wgms_metrics(Path(row["run_dir"]), region_num,
                                  split="validation", period_mean=True)
        cols["wgms_val_rmse"].append(m["rmse"])
        cols["wgms_val_corr"].append(m["corr"])
        cols["wgms_val_medae"].append(m["medae"])
        cols["wgms_val_n_points"].append(m["n_points"])
        cols["wgms_val_available"].append(m["available"])
    for k, v in cols.items():
        df[k] = v
    return df


def _wgms_corr_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank by WGMS validation correlation (r), descending (higher = better).

    Correlation is used instead of an RMSE/MedAE-based composite because WGMS
    glaciological series are often themselves bias-corrected against geodetic
    (satellite) measurements — the same family of signal GLaMBIE/Hugonnet
    contribute during finetuning. An absolute-error metric like RMSE or MedAE
    can therefore reward a model for matching a bias term it may have partly
    learned from training data, whereas correlation only credits matching the
    interannual pattern and is far less sensitive to that shared bias.

    Runs with an undefined correlation (e.g. zero variance in obs or pred)
    are treated as worst rather than dropped.
    """
    df = df.copy()
    corr_filled = df["wgms_val_corr"].fillna(df["wgms_val_corr"].min()
                                              if df["wgms_val_corr"].notna().any() else 0.0)
    df["_composite"] = 1.0 - _minmax_norm_series(corr_filled)
    return df.sort_values("_composite", ascending=True).reset_index(drop=True)


def _loyo_logo_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback ranking for regions with no usable WGMS data (tier 3):
    average of normalised loyo_r2 and logo_r2 (higher=better), inverted so
    lower=better for consistency with the WGMS composite."""
    df = df.copy()
    norm_loyo = _minmax_norm_series(df["loyo_r2"])
    norm_logo = _minmax_norm_series(df["logo_r2"])
    df["_composite"] = 1.0 - 0.5 * (norm_loyo + norm_logo)
    return df.sort_values("_composite", ascending=True).reset_index(drop=True)


def select_top_n_runs(
    group_df: pd.DataFrame,
    top_n: int,
    min_runs: int,
    loyo_r2_min: float | None = 0.0,
    logo_r2_min: float | None = 0.0,
    wgms_rmse_max: float = 10.0,
) -> tuple[pd.DataFrame | None, str]:
    """
    Apply the shared selection criteria used by every ensemble-building script
    in this repo (ensemble_common.py::_run_top_n_group, ensemble_ep_alea.py,
    and ensemble_uncertainty.py — see CLAUDE.md / conversation trail for the
    full reasoning; this replaces the earlier glambie_rmse-ranked selection).

      Hard gates (always applied):
        - loyo_r2 > loyo_r2_min   (pretrain leave-one-year-out R², strictly positive by default)
        - logo_r2 > logo_r2_min   (pretrain leave-one-glacier-out R², strictly positive by default)
        - wgms_val_rmse <= wgms_rmse_max (excludes runs with WGMS validation
          RMSE above 10 m w.e./yr by default — a loose sanity backstop against
          catastrophic misfit, not a ranking criterion; see below for why) —
          only applied when the region has usable WGMS data (see tiers below).

      Ranking:
        - Regions with usable WGMS reference/benchmark data (tier 1 or 2,
          per reference_benchmark_region_summary.csv): rank by WGMS validation
          correlation (r), descending. RMSE/MedAE are deliberately NOT used
          for ranking — WGMS glaciological series are often bias-corrected
          against geodetic measurements, the same family of signal GLaMBIE/
          Hugonnet contribute during finetuning, so an absolute-error metric
          risks rewarding a model for matching a bias term it may have partly
          learned from training data. Correlation is far less sensitive to
          that shared bias. (wgms_val_rmse/medae are still computed and saved
          for reporting — see the caller's saved info CSV.)
        - Regions with no usable WGMS data (tier 3, e.g. r04/r09): rank by a
          composite of normalised loyo_r2/logo_r2 instead — GLaMBIE is never
          part of selection at any tier (it is testing-only; see wgms_test_*
          / glambie_rmse columns in the caller's saved info CSV for that).

      Year-by-year -> period-mean fallback:
        - Ranking is computed first from point-wise (glacier, year) WGMS
          residuals. If fewer than top_n runs remain after gating, the WGMS
          metrics are recomputed using per-glacier period-means (reduces
          noise at the cost of fewer effective points) and gating/ranking
          is redone before falling through to a hard skip.

    Args:
        group_df:          Candidate runs (one row per run) with the columns
                          produced by hyperparam_tuning.build_results_df.
        top_n:             Number of top runs to return (pass len(group_df)
                          to get everything that passed gating, ranked).
        min_runs:          Minimum runs required after gating to proceed.
        loyo_r2_min:       Hard gate threshold (strict >). None disables.
        logo_r2_min:       Hard gate threshold (strict >). None disables.
        wgms_rmse_max:     Hard exclusion threshold on WGMS validation RMSE
                          (m w.e./yr). Only applied where WGMS data is used.

    Returns:
        (top_df, rank_label) — top_df has a `_composite` column (lower =
        better) and may have fewer than top_n rows. (None, reason) if fewer
        than min_runs runs survive even after the period-mean fallback.
    """
    gated = group_df.copy().reset_index(drop=True)

    if loyo_r2_min is not None and "loyo_r2" in gated.columns:
        n_before = len(gated)
        gated = gated[gated["loyo_r2"] > loyo_r2_min].reset_index(drop=True)
        n_rejected = n_before - len(gated)
        if n_rejected:
            print(f"  [loyo_r2 gate] Rejected {n_rejected}/{n_before} runs with "
                  f"loyo_r2 <= {loyo_r2_min}")

    if logo_r2_min is not None and "logo_r2" in gated.columns:
        n_before = len(gated)
        gated = gated[gated["logo_r2"] > logo_r2_min].reset_index(drop=True)
        n_rejected = n_before - len(gated)
        if n_rejected:
            print(f"  [logo_r2 gate] Rejected {n_rejected}/{n_before} runs with "
                  f"logo_r2 <= {logo_r2_min}")

    if len(gated) < min_runs:
        return None, f"only {len(gated)} run(s) pass LOYO/LOGO R² gates (need {min_runs})"

    tier = int(gated["wgms_tier"].iloc[0]) if "wgms_tier" in gated.columns else 3
    has_wgms = tier != 3 and "wgms_val_available" in gated.columns and gated["wgms_val_available"].any()

    def _score(period_mean: bool) -> tuple[pd.DataFrame, str]:
        if not has_wgms:
            return _loyo_logo_rank(gated), "loyo/logo composite (no usable WGMS data for this region — tier 3)"

        scored = _recompute_wgms_period_mean(gated) if period_mean else gated.copy()
        n_before = len(scored)
        scored = scored[scored["wgms_val_rmse"].isna() | (scored["wgms_val_rmse"] <= wgms_rmse_max)]
        n_rejected = n_before - len(scored)
        if n_rejected:
            print(f"  [wgms_rmse gate{'  (period-mean)' if period_mean else ''}] "
                  f"Rejected {n_rejected}/{n_before} runs with wgms_val_rmse > {wgms_rmse_max} m w.e./yr")
        scored = scored.dropna(subset=["wgms_val_rmse"])
        label = ("WGMS validation correlation (r)"
                 + ("  [period-mean fallback]" if period_mean else "  [point-wise]"))
        return _wgms_corr_rank(scored), label

    ranked, rank_label = _score(period_mean=False)
    if has_wgms and len(ranked) < top_n:
        print(f"  Only {len(ranked)} run(s) pass point-wise WGMS gating (need {top_n} for a full "
              f"ensemble) — falling back to per-glacier period-mean WGMS metrics.")
        ranked, rank_label = _score(period_mean=True)

    if len(ranked) < min_runs:
        return None, f"only {len(ranked)} run(s) pass all gates under {rank_label} (need {min_runs})"

    return ranked.head(top_n).reset_index(drop=True), rank_label
