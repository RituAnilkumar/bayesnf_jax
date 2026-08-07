"""
src/hyperparam_tuning.py

Hyperparameter sweep analysis for bayesnf_jax multirun outputs.

Reads per-run metric files from a Hydra multirun directory tree and produces
plots + ranked tables to guide model selection.

Expected multirun layout (one SLURM job per region):
  {multirun_root}/{region}_{jobid}/{run_id}/.hydra/overrides.yaml
  {multirun_root}/{region}_{jobid}/{run_id}/metrics_glambie_test.csv
  {multirun_root}/{region}_{jobid}/{run_id}/._cv/metrics_oos.csv

Metrics used:
  - LOYO RMSE       : pretrain out-of-sample (leave-one-year-out) — from ._cv/metrics_oos.csv
  - GLaMBIE RMSE    : finetune vs held-out GLaMBIE years (2021-2023) — from metrics_glambie_test.csv
  - Composite score : weighted average of per-region normalised LOYO + GLaMBIE RMSE

Usage:
  python src/hyperparam_tuning.py
  python src/hyperparam_tuning.py --config conf/config_hyperparam_tuning.yaml
  python src/hyperparam_tuning.py --multirun_root /path/to/multirun --output_dir outputs/analysis

Batch usage:
for i in $(seq -w 1 19); do
  python src/hyperparam_tuning.py \
    --multirun_root /scratch/b5at/ranil.b5at/bayesnf_jax/multirun/r${i}_397*/
done
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from itertools import combinations
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWEEP_PARAMS = [
    "model.model_nlayers",
    "model.model_nhidden",
    "model.heteroscedastic",
    "model.pretrain_year_min",
]

# Short names used in plot labels / column headers
PARAM_SHORT = {
    "model.model_nlayers":    "nlayers",
    "model.model_nhidden":    "nhidden",
    "model.heteroscedastic":  "heteroscedastic",
    "model.pretrain_year_min": "pretrain_year_min",
}

# Human-readable axis labels for plots
PARAM_LABEL = {
    "nlayers":           "N layers",
    "nhidden":           "Hidden width",
    "heteroscedastic":   "Heteroscedastic",
    "pretrain_year_min": "Pretrain year min",
}

# Key 2-D pairs to show in heatmaps (short names)
HEATMAP_PAIRS = [
    ("nlayers",           "nhidden"),
    ("heteroscedastic",   "nlayers"),
    ("heteroscedastic",   "nhidden"),
    ("pretrain_year_min", "nlayers"),
    ("pretrain_year_min", "nhidden"),
]

# Non-sweep overrides tracked as metadata columns (not used in scoring)
EXTRA_OVERRIDES = {
    "model.glambie_weight":     "glambie_weight",
    "model.beta_anneal_epochs": "beta_anneal_epochs",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_overrides(overrides_path: Path) -> dict:
    """Parse a Hydra overrides.yaml (list of 'key=value' strings) into a dict.

    Returns a dict with short param names for sweep params plus 'region'.
    Returns None if the file cannot be parsed.
    """
    try:
        with open(overrides_path) as fh:
            entries = yaml.safe_load(fh)  # list of strings like "model.model_nlayers=2"
    except Exception as exc:
        warnings.warn(f"Cannot parse {overrides_path}: {exc}")
        return None

    raw = {}
    for entry in entries:
        if "=" in entry:
            k, v = entry.split("=", 1)
            raw[k.lstrip("+-~ ")] = v

    # Extract region from "model=bnf_regional_seasonal/r06"
    region = None
    model_val = raw.get("model", "")
    m = re.search(r"/r(\d+)$", model_val)
    if m:
        region = f"r{m.group(1):>02}"

    out = {"region": region}
    for full_key, short in PARAM_SHORT.items():
        val_str = raw.get(full_key)
        if val_str is not None:
            if val_str.lower() in ("true", "false"):
                out[short] = val_str.lower() == "true"
            else:
                try:
                    out[short] = float(val_str)
                except ValueError:
                    out[short] = val_str
        else:
            out[short] = float("nan")

    for full_key, short in EXTRA_OVERRIDES.items():
        val_str = raw.get(full_key)
        if val_str is not None:
            # Coerce booleans; try numeric; fall back to string
            if val_str.lower() in ("true", "false"):
                out[short] = val_str.lower() == "true"
            else:
                try:
                    out[short] = float(val_str)
                except ValueError:
                    out[short] = val_str
        else:
            out[short] = None

    return out


def _load_glambie_test_rmse(run_dir: Path, test_years: list[int]) -> float:
    """Return RMSE of finetune-stage predictions vs held-out GLaMBIE test years.

    Filters metrics_glambie_test.csv to stage=='finetune' and year in test_years.

    Returns NaN when:
      - file absent (run did not complete or region has no GLaMBIE data)
      - all residuals are NaN (finetune diverged — NaN params → NaN predictions)
      - no rows match test_years

    NaN residuals typically indicate finetune loss diverged due to zero uncertainty
    values in the Hugonnet temporal_avg data (affects r13-r15, r17). Fix: re-run
    with the uncertainty_floor in prepare_finetune_arrays.
    """
    path = run_dir / "metrics_glambie_test.csv"
    if not path.exists():
        return float("nan")
    try:
        df = pd.read_csv(path)
    except Exception:
        return float("nan")

    df = df[(df["stage"] == "finetune") & (df["year"].isin(test_years))]
    if df.empty:
        return float("nan")

    residuals = df["residual"].dropna()
    if residuals.empty:
        return float("nan")  # all NaN — finetune diverged
    return float(np.sqrt((residuals ** 2).mean()))


def _load_loyo_rmse(run_dir: Path) -> float:
    """Return LOYO RMSE from the pretrain_cv run.

    Looks in {run_dir}/._cv/metrics_oos.csv (the _cv suffix appended by
    main_pipeline.py to the output_dir '.' at runtime).
    Returns NaN if the file is absent or contains no loyo row.
    """
    path = run_dir / "._cv" / "metrics_oos.csv"
    if not path.exists():
        return float("nan")
    try:
        df = pd.read_csv(path)
    except Exception:
        return float("nan")

    row = df[df["split"] == "loyo"]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["rmse"])


def _load_loyo_r2(run_dir: Path) -> float:
    """Return LOYO R² from the pretrain_cv run (same file as _load_loyo_rmse)."""
    path = run_dir / "._cv" / "metrics_oos.csv"
    if not path.exists():
        return float("nan")
    try:
        df = pd.read_csv(path)
    except Exception:
        return float("nan")
    row = df[df["split"] == "loyo"]
    if row.empty or "r2" not in df.columns:
        return float("nan")
    return float(row.iloc[0]["r2"])


def discover_runs(multirun_root: Path) -> list[Path]:
    """Return a sorted list of run directories inside multirun_root.

    Expected layout: {multirun_root}/{run_id}/.hydra/overrides.yaml
    where multirun_root is the region-job directory (e.g. r06_3645680/).
    """
    if not multirun_root.exists():
        raise FileNotFoundError(f"multirun_root not found: {multirun_root}")

    runs = []
    for run_dir in sorted(multirun_root.iterdir(), key=lambda p: p.name):
        if not run_dir.is_dir():
            continue
        if (run_dir / ".hydra" / "overrides.yaml").exists():
            runs.append(run_dir)

    print(f"Discovered {len(runs)} runs in {multirun_root.name}.")
    return runs


def _region_from_dir_name(dir_name: str) -> str:
    """Extract region label from a directory name like 'r06_3645680' → 'r06'."""
    m = re.match(r"(r\d+)_", dir_name)
    return m.group(1) if m else dir_name


def build_results_df(
    multirun_root: Path,
    test_years: list[int],
    min_runs_per_region: int = 5,
) -> pd.DataFrame:
    """Load metrics for every discovered run and return a combined DataFrame.

    Columns: region, run_id, run_dir, nlayers, nhidden, heteroscedastic,
             glambie_weight, beta_anneal_epochs, loyo_rmse, loyo_r2, glambie_rmse
    """
    run_dirs = discover_runs(multirun_root)
    # Region is derived from the multirun_root dir name (e.g. r06_3645680 → r06)
    region = _region_from_dir_name(multirun_root.name)

    rows = []
    missing_glambie = 0
    missing_loyo = 0

    for run_dir in run_dirs:
        params = _parse_overrides(run_dir / ".hydra" / "overrides.yaml")
        if params is None:
            continue

        glambie_rmse = _load_glambie_test_rmse(run_dir, test_years)
        loyo_rmse    = _load_loyo_rmse(run_dir)
        loyo_r2      = _load_loyo_r2(run_dir)

        if np.isnan(glambie_rmse):
            missing_glambie += 1
        if np.isnan(loyo_rmse):
            missing_loyo += 1

        rows.append({
            "region":  params.get("region") or region,
            "run_id":  run_dir.name,
            "run_dir": str(run_dir),
            **{k: params[k] for k in PARAM_SHORT.values()},
            **{k: params.get(k) for k in EXTRA_OVERRIDES.values()},
            "loyo_rmse":    loyo_rmse,
            "loyo_r2":      loyo_r2,
            "glambie_rmse": glambie_rmse,
        })

    df = pd.DataFrame(rows)

    if missing_glambie:
        print(f"  WARNING: {missing_glambie} runs missing glambie_test metrics")
    if missing_loyo:
        print(f"  WARNING: {missing_loyo} runs missing LOYO metrics")

    n_complete = df.dropna(subset=["loyo_rmse", "glambie_rmse"]).shape[0]
    if n_complete < min_runs_per_region:
        print(f"  WARNING: only {n_complete} complete runs (threshold={min_runs_per_region})")
    print(f"  {len(df)} runs loaded, {n_complete} complete (both metrics present).")
    return df


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def add_composite_score(
    df: pd.DataFrame,
    loyo_weight: float = 0.5,
    glambie_weight: float = 0.5,
) -> pd.DataFrame:
    """Add a composite score column (lower = better).

    Each metric is normalised per-region to [0, 1] (min-max) before combining,
    so regions with different absolute error scales contribute equally.

    Fallback behaviour when glambie_rmse is NaN for all runs in a region
    (e.g. r13-r15, r17 where finetune loss diverged due to zero uncertainties in
    the Hugonnet data, or regions with no GLaMBIE test coverage):
      - composite is computed from loyo_rmse only (weight effectively = 1.0)
      - a warning is printed so the user knows glambie is excluded

    Rows with NaN loyo_rmse always get NaN composite (pretrain failed).
    """
    df = df.copy()

    def _minmax_norm(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(0.0, index=series.index)
        return (series - lo) / (hi - lo)

    df["norm_loyo"]    = float("nan")
    df["norm_glambie"] = float("nan")

    for region, grp in df.groupby("region"):
        idx = grp.index
        df.loc[idx, "norm_loyo"]    = _minmax_norm(grp["loyo_rmse"]).values
        df.loc[idx, "norm_glambie"] = _minmax_norm(grp["glambie_rmse"]).values

    # Build composite, falling back to loyo-only per row when glambie is NaN.
    # This handles regions where glambie is unavailable or finetune diverged.
    has_glambie_mask = df["norm_glambie"].notna()
    all_loyo_only = (~has_glambie_mask).all()

    if all_loyo_only:
        print("  WARNING: glambie_rmse is NaN for all runs. "
              "Scoring by loyo_rmse only. "
              "If finetune produced NaN loss, re-run after the uncertainty_floor fix.")
        df["composite"] = df["norm_loyo"]
    else:
        if (~has_glambie_mask).any():
            n_missing = (~has_glambie_mask).sum()
            print(f"  NOTE: {n_missing} runs have NaN glambie_rmse — those use loyo-only composite.")
        # Per row: use full formula when glambie is available, loyo-only otherwise
        full_score = loyo_weight * df["norm_loyo"] + glambie_weight * df["norm_glambie"]
        df["composite"] = np.where(has_glambie_mask, full_score, df["norm_loyo"])

    return df


def add_within_region_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank_in_region (1 = best composite score) and mean_rank across regions."""
    df = df.copy()
    df["rank_in_region"] = df.groupby("region")["composite"].rank(method="min")

    # Mean rank across regions for each unique hyperparameter combination
    param_cols = list(PARAM_SHORT.values())
    mean_rank = (
        df.groupby(param_cols)["rank_in_region"]
        .mean()
        .reset_index()
        .rename(columns={"rank_in_region": "mean_rank"})
    )
    df = df.merge(mean_rank, on=param_cols, how="left")
    return df


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_param_importance(df: pd.DataFrame, metric: str = "composite") -> pd.Series:
    """Estimate the marginal importance of each sweep parameter via eta-squared.

    eta² = SS_between / SS_total — fraction of metric variance explained by
    the parameter alone (ignoring interactions). Higher = more important.
    """
    valid = df.dropna(subset=[metric])
    total_ss = ((valid[metric] - valid[metric].mean()) ** 2).sum()
    if total_ss == 0:
        return pd.Series(0.0, index=list(PARAM_SHORT.values()))

    importances = {}
    for short in PARAM_SHORT.values():
        group_means = valid.groupby(short)[metric].mean()
        group_counts = valid.groupby(short)[metric].count()
        between_ss = (group_counts * (group_means - valid[metric].mean()) ** 2).sum()
        importances[short] = between_ss / total_ss

    return pd.Series(importances).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_marginals(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plots of LOYO RMSE and GLaMBIE RMSE for each hyperparameter value,
    marginalised over all other parameters and all regions.
    """
    metrics = [
        ("loyo_rmse",    "LOYO RMSE (MWE/yr)"),
        ("glambie_rmse", "GLaMBIE test RMSE (MWE/yr)"),
        ("composite",    "Composite score (normalised)"),
    ]
    param_cols = list(PARAM_SHORT.values())

    for metric_col, metric_label in metrics:
        valid = df.dropna(subset=[metric_col])
        fig, axes = plt.subplots(1, len(param_cols), figsize=(4 * len(param_cols), 5))
        fig.suptitle(f"Marginal effect on {metric_label}", fontsize=13)

        for ax, short in zip(axes, param_cols):
            unique_vals = sorted(valid[short].dropna().unique())
            pairs = [
                (str(val), valid[valid[short] == val][metric_col].values)
                for val in unique_vals
            ]
            # Drop any value whose group has no data (float comparison edge cases)
            pairs = [(lbl, grp) for lbl, grp in pairs if len(grp) > 0]
            if not pairs:
                ax.set_visible(False)
                continue
            labels, groups = zip(*pairs)
            bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, notch=False)
            for patch in bp["boxes"]:
                patch.set_facecolor("steelblue")
                patch.set_alpha(0.6)
            ax.set_title(PARAM_LABEL.get(short, short), fontsize=10)
            ax.set_xlabel(short)
            if ax is axes[0]:
                ax.set_ylabel(metric_label, fontsize=9)
            ax.tick_params(axis="x", labelsize=8, rotation=30)
            ax.grid(axis="y", alpha=0.4)

        fig.tight_layout()
        fname = f"marginals_{metric_col}.png"
        _savefig(fig, output_dir / "plots" / fname)


def plot_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    """2-D mean composite score heatmaps for key parameter pairs,
    with a separate panel per region (plus an 'all regions' mean panel).
    """
    valid = df.dropna(subset=["composite"])
    regions = sorted(valid["region"].unique())
    n_panels = len(regions) + 1  # +1 for aggregate

    for p1, p2 in HEATMAP_PAIRS:
        vals1 = sorted(valid[p1].dropna().unique())
        vals2 = sorted(valid[p2].dropna().unique())
        ncols = min(4, n_panels)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes_flat = np.array(axes).flatten()
        fig.suptitle(
            f"Mean composite score: {PARAM_LABEL.get(p1, p1)} × {PARAM_LABEL.get(p2, p2)}\n"
            "(lower is better)",
            fontsize=12,
        )

        def _make_grid(subset: pd.DataFrame) -> np.ndarray:
            grid = np.full((len(vals2), len(vals1)), np.nan)
            for i, v2 in enumerate(vals2):
                for j, v1 in enumerate(vals1):
                    cell = subset[(subset[p1] == v1) & (subset[p2] == v2)]["composite"]
                    if len(cell):
                        grid[i, j] = cell.mean()
            return grid

        all_grids = [_make_grid(valid[valid["region"] == r]) for r in regions]
        agg_grid  = _make_grid(valid)

        all_data = np.concatenate([g[~np.isnan(g)] for g in all_grids] + [agg_grid[~np.isnan(agg_grid)]])
        if all_data.size == 0:
            plt.close(fig)
            print(f"  Skipping heatmap {p1} vs {p2} — no complete runs for this pair")
            continue
        vmin, vmax = all_data.min(), all_data.max()

        for ax_idx, (label, grid) in enumerate(
            list(zip(regions, all_grids)) + [("All regions", agg_grid)]
        ):
            ax = axes_flat[ax_idx]
            im = ax.imshow(grid, aspect="auto", origin="lower",
                           cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(vals1))); ax.set_xticklabels([str(v) for v in vals1], fontsize=7, rotation=30)
            ax.set_yticks(range(len(vals2))); ax.set_yticklabels([str(v) for v in vals2], fontsize=7)
            ax.set_xlabel(PARAM_LABEL.get(p1, p1), fontsize=8)
            ax.set_ylabel(PARAM_LABEL.get(p2, p2), fontsize=8)
            ax.set_title(label, fontsize=9)
            # Annotate cells
            for i in range(len(vals2)):
                for j in range(len(vals1)):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                                fontsize=6, color="black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused axes
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        fig.tight_layout()
        fname = f"heatmap_{p1}_vs_{p2}.png"
        _savefig(fig, output_dir / "plots" / fname)


def plot_parallel_coords(df: pd.DataFrame, output_dir: Path) -> None:
    """Parallel coordinates plot: each line is a run, coloured by composite score.

    Only complete runs (no NaN in composite) are shown.
    """
    valid = df.dropna(subset=["composite"]).copy()
    if valid.empty:
        print("  Skipping parallel coords — no complete runs")
        return
    param_cols = list(PARAM_SHORT.values())

    def _to_num(v) -> float:
        """Convert any scalar (including numpy/Python bool) to Python float."""
        if pd.isna(v) if not isinstance(v, (list, np.ndarray)) else False:
            return float("nan")
        if isinstance(v, (bool, np.bool_)):
            return float(int(v))
        return float(v)

    # Normalise each param axis to [0, 1] for visual alignment
    # Booleans (Python and numpy) are cast to 0/1 before normalisation.
    norm_df = valid[param_cols].copy()
    for col in param_cols:
        norm_df[col] = norm_df[col].map(_to_num)
        lo, hi = norm_df[col].min(), norm_df[col].max()
        norm_df[col] = (norm_df[col] - lo) / (hi - lo) if hi > lo else 0.0

    composite = valid["composite"].values
    cmap = plt.get_cmap("RdYlGn_r")
    c_norm = mcolors.Normalize(vmin=composite.min(), vmax=composite.max())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-0.5, len(param_cols) - 0.5)
    ax.set_ylim(-0.05, 1.05)

    for idx in range(len(valid)):
        y_vals = norm_df.iloc[idx].values
        color = cmap(c_norm(composite[idx]))
        alpha = 0.15 + 0.5 * (1 - c_norm(composite[idx]))  # top runs more opaque
        ax.plot(range(len(param_cols)), y_vals, color=color, alpha=float(alpha), linewidth=0.8)

    ax.set_xticks(range(len(param_cols)))
    ax.set_xticklabels(
        [PARAM_LABEL.get(p, p) for p in param_cols],
        fontsize=10, rotation=15,
    )

    # Draw vertical guide lines with actual param values annotated
    for j, col in enumerate(param_cols):
        ax.axvline(j, color="grey", linewidth=0.6, alpha=0.5)
        vals = sorted(valid[col].dropna().unique(), key=_to_num)
        num_vals = [_to_num(v) for v in vals]
        lo = float(min(num_vals)); hi = float(max(num_vals))
        for v, nv in zip(vals, num_vals):
            y_pos = (nv - lo) / (hi - lo) if hi > lo else 0.5
            ax.annotate(
                str(v), xy=(j, y_pos), xytext=(-4, 0), textcoords="offset points",
                fontsize=6, ha="right", va="center", color="dimgray",
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=c_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Composite score (lower = better)", fraction=0.03, pad=0.02)
    ax.set_ylabel("Normalised parameter value", fontsize=9)
    ax.set_title(
        f"Parallel coordinates — all runs ({len(valid)} complete)\n"
        "Green lines = low composite score (good), Red = high (bad)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.0)

    _savefig(fig, output_dir / "plots" / "parallel_coords.png", dpi=150)


def plot_nhidden_vs_rmse(df: pd.DataFrame, output_dir: Path) -> None:
    """nhidden on x-axis, RMSE on y-axis, one line per (nlayers, heteroscedastic) combo.

    Each point is the mean RMSE across all regions for that combination.
    Error bars show ±1 std. Two subplots: LOYO RMSE and GLaMBIE test RMSE.
    When heteroscedastic is not a sweep param, lines are split by nlayers only.
    """
    markers = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
    colors  = {1: "steelblue", 2: "darkorange", 3: "forestgreen", 4: "purple", 5: "crimson"}
    linestyles = {True: "-", False: "--"}

    metrics = [
        ("loyo_rmse",    "LOYO RMSE (MWE/yr)"),
        ("glambie_rmse", "GLaMBIE test RMSE (MWE/yr)"),
    ]

    has_hetero = "heteroscedastic" in df.columns and df["heteroscedastic"].notna().any()
    hetero_vals = sorted(df["heteroscedastic"].dropna().unique(), key=lambda x: int(x) if isinstance(x, bool) else x) if has_hetero else [None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, (metric_col, metric_label) in zip(axes, metrics):
        valid = df.dropna(subset=[metric_col, "nhidden", "nlayers"])
        if valid.empty:
            ax.set_visible(False)
            continue

        nhidden_vals = sorted(valid["nhidden"].unique())
        nlayers_vals = sorted(valid["nlayers"].unique())

        for nl in nlayers_vals:
            for hv in hetero_vals:
                if hv is None:
                    sub = valid[valid["nlayers"] == nl]
                    ls = "-"
                    lbl = f"{int(nl)} layer{'s' if nl > 1 else ''}"
                else:
                    sub = valid[(valid["nlayers"] == nl) & (valid["heteroscedastic"] == hv)]
                    ls = linestyles.get(hv, "-")
                    het_tag = "aleatoric" if hv else "ep+struct"
                    lbl = f"{int(nl)}L {het_tag}"

                means, stds, xs = [], [], []
                for nh in nhidden_vals:
                    cell = sub[sub["nhidden"] == nh][metric_col]
                    if len(cell) == 0:
                        continue
                    xs.append(nh)
                    means.append(cell.mean())
                    stds.append(cell.std() if len(cell) > 1 else 0.0)

                if not xs:
                    continue

                marker = markers.get(int(nl), "o")
                color  = colors.get(int(nl), "grey")
                ax.errorbar(
                    xs, means, yerr=stds,
                    marker=marker, color=color, lw=1.5, ms=6, capsize=4,
                    linestyle=ls, label=lbl,
                )

        ax.set_xlabel("Hidden width (neurons)", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11)
        ax.set_xticks(nhidden_vals)
        legend_title = "N layers × heteroscedastic" if has_hetero else "N layers"
        ax.legend(fontsize=8, title=legend_title)
        ax.grid(alpha=0.35)

    subtitle = "(mean ± std across regions; solid=aleatoric, dashed=ep+struct)" if has_hetero else "(mean ± std across regions)"
    fig.suptitle(f"RMSE vs hidden width by number of layers\n{subtitle}", fontsize=12)
    fig.tight_layout()
    _savefig(fig, output_dir / "plots" / "nhidden_vs_rmse.png")


def plot_heteroscedastic_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plots comparing aleatoric (heteroscedastic=True) vs epistemic+structural
    (heteroscedastic=False) groups for each metric, broken down by region.

    Skipped silently when heteroscedastic is not a sweep parameter.
    """
    if "heteroscedastic" not in df.columns or df["heteroscedastic"].isna().all():
        return
    if df["heteroscedastic"].nunique() < 2:
        print("  Skipping heteroscedastic comparison — only one group found")
        return

    metrics = [
        ("loyo_rmse",    "LOYO RMSE (MWE/yr)"),
        ("glambie_rmse", "GLaMBIE test RMSE (MWE/yr)"),
        ("composite",    "Composite score (normalised)"),
    ]
    regions = sorted(df["region"].dropna().unique())
    group_labels = ["ep+struct (False)", "aleatoric (True)"]

    for metric_col, metric_label in metrics:
        valid = df.dropna(subset=[metric_col, "heteroscedastic"])
        if valid.empty:
            continue

        ncols = min(4, len(regions))
        nrows = int(np.ceil(len(regions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        fig.suptitle(
            f"{metric_label}: aleatoric vs epistemic+structural\n(per region)",
            fontsize=12,
        )

        for ax_idx, region in enumerate(regions):
            ax = axes_flat[ax_idx]
            sub = valid[valid["region"] == region]
            grp_false = sub[sub["heteroscedastic"] == False][metric_col].values
            grp_true  = sub[sub["heteroscedastic"] == True][metric_col].values
            groups = [g for g in [grp_false, grp_true] if len(g) > 0]
            lbls   = [gl for g, gl in [(grp_false, group_labels[0]), (grp_true, group_labels[1])] if len(g) > 0]
            if not groups:
                ax.set_visible(False)
                continue
            bp = ax.boxplot(groups, tick_labels=lbls, patch_artist=True)
            colors_box = ["steelblue", "darkorange"]
            for patch, c in zip(bp["boxes"], colors_box):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax.set_title(region, fontsize=9)
            ax.tick_params(axis="x", labelsize=7, rotation=15)
            ax.grid(axis="y", alpha=0.4)
            if ax_idx % ncols == 0:
                ax.set_ylabel(metric_label, fontsize=8)

        for ax in axes_flat[len(regions):]:
            ax.set_visible(False)

        fig.tight_layout()
        fname = f"heteroscedastic_comparison_{metric_col}.png"
        _savefig(fig, output_dir / "plots" / fname)


def plot_pretrain_vs_finetune(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter of LOYO RMSE (pretrain) vs GLaMBIE RMSE (finetune),
    one subplot per region, coloured by composite score.
    """
    valid = df.dropna(subset=["loyo_rmse", "glambie_rmse"])
    if valid.empty:
        print("  Skipping pretrain vs finetune scatter — no complete runs")
        return
    regions = sorted(valid["region"].unique())
    ncols = min(4, len(regions))
    nrows = int(np.ceil(len(regions) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    cmap = plt.get_cmap("RdYlGn_r")

    for ax_idx, region in enumerate(regions):
        ax = axes_flat[ax_idx]
        sub = valid[valid["region"] == region]
        c_norm = mcolors.Normalize(vmin=sub["composite"].min(), vmax=sub["composite"].max())
        sc = ax.scatter(
            sub["loyo_rmse"], sub["glambie_rmse"],
            c=sub["composite"], cmap=cmap, norm=c_norm,
            s=30, alpha=0.7, edgecolors="none",
        )
        ax.set_xlabel("LOYO RMSE (MWE/yr)", fontsize=8)
        ax.set_ylabel("GLaMBIE test RMSE (MWE/yr)", fontsize=8)
        ax.set_title(region, fontsize=9)
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="composite")

    for ax in axes_flat[len(regions):]:
        ax.set_visible(False)

    fig.suptitle(
        "Pretrain LOYO RMSE vs Finetune GLaMBIE test RMSE\n(colour = composite score)",
        fontsize=12,
    )
    fig.tight_layout()
    _savefig(fig, output_dir / "plots" / "pretrain_vs_finetune.png")


def plot_param_importance(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of marginal eta-squared for each sweep parameter.

    Computed globally (all regions pooled, composite score as the target).
    Also computes per-metric importance for LOYO and GLaMBIE separately.
    """
    metrics = [
        ("composite",    "Composite score"),
        ("loyo_rmse",    "LOYO RMSE"),
        ("glambie_rmse", "GLaMBIE test RMSE"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))

    for ax, (metric_col, metric_label) in zip(axes, metrics):
        imp = compute_param_importance(df, metric=metric_col)
        if imp.empty or imp.max() == 0:
            ax.set_visible(False)
            continue
        colors = ["steelblue" if v >= imp.max() * 0.5 else "lightsteelblue" for v in imp.values]
        ax.barh(
            [PARAM_LABEL.get(k, k) for k in imp.index],
            imp.values,
            color=colors,
        )
        ax.set_xlabel("η² (fraction of variance explained)", fontsize=9)
        ax.set_title(metric_label, fontsize=10)
        ax.set_xlim(0, min(1.0, imp.max() * 1.3 + 0.05))
        ax.grid(axis="x", alpha=0.4)
        for i, v in enumerate(imp.values):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    fig.suptitle("Parameter importance (marginal η²)", fontsize=12)
    fig.tight_layout()
    _savefig(fig, output_dir / "plots" / "param_importance.png")


def plot_regional_ranking(df: pd.DataFrame, output_dir: Path, top_n: int = 10) -> None:
    """Heatmap: rows = top-N configs by mean rank, columns = regions.
    Cell colour = within-region rank (green = low rank = good).

    Helps assess whether the best configs generalise across regions.
    """
    valid = df.dropna(subset=["composite"])
    if valid.empty:
        print("  Skipping regional ranking — no complete runs")
        return
    param_cols = list(PARAM_SHORT.values())

    # Top-N configs by mean_rank
    top_configs = (
        valid.groupby(param_cols)["rank_in_region"]
        .mean()
        .reset_index()
        .rename(columns={"rank_in_region": "mean_rank"})
        .sort_values("mean_rank")
        .head(top_n)
    )

    regions = sorted(valid["region"].unique())
    n_runs_per_region = valid.groupby("region").size().max()

    # Build label for each config row
    def _config_label(row) -> str:
        parts = [f"L={int(row.nlayers)}", f"H={int(row.nhidden)}"]
        if "heteroscedastic" in row.index and not pd.isna(row.heteroscedastic):
            parts.append("het=" + ("Y" if row.heteroscedastic else "N"))
        if "pretrain_year_min" in row.index and not pd.isna(row.pretrain_year_min):
            parts.append(f"pt={int(row.pretrain_year_min)}")
        return " ".join(parts)

    config_labels = [_config_label(r) for _, r in top_configs.iterrows()]

    # Build rank matrix (config × region)
    rank_matrix = np.full((len(top_configs), len(regions)), np.nan)
    for i, (_, cfg_row) in enumerate(top_configs.iterrows()):
        mask = pd.Series([True] * len(valid), index=valid.index)
        for col in param_cols:
            mask &= valid[col] == cfg_row[col]
        subset = valid[mask]
        for j, region in enumerate(regions):
            reg_row = subset[subset["region"] == region]
            if not reg_row.empty:
                rank_matrix[i, j] = reg_row.iloc[0]["rank_in_region"]

    fig, ax = plt.subplots(figsize=(max(8, len(regions) * 0.9), max(5, top_n * 0.55)))
    im = ax.imshow(
        rank_matrix, aspect="auto", cmap="RdYlGn_r",
        vmin=1, vmax=n_runs_per_region,
    )
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(top_configs)))
    ax.set_yticklabels(config_labels, fontsize=8)
    ax.set_xlabel("Region", fontsize=10)
    ax.set_title(
        f"Top-{top_n} configs by mean rank across regions\n"
        "(cell = within-region rank; green = rank 1 = best)",
        fontsize=11,
    )

    # Annotate cells
    for i in range(len(top_configs)):
        for j in range(len(regions)):
            if not np.isnan(rank_matrix[i, j]):
                ax.text(j, i, str(int(rank_matrix[i, j])), ha="center", va="center",
                        fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="Rank (1 = best)", fraction=0.03, pad=0.02)
    fig.tight_layout()
    _savefig(fig, output_dir / "plots" / "regional_ranking.png")


# ---------------------------------------------------------------------------
# Summary outputs
# ---------------------------------------------------------------------------

def save_top_runs(df: pd.DataFrame, output_dir: Path, top_n: int = 10) -> None:
    """Save CSV of top-N configs ranked by mean composite score across regions."""
    param_cols = list(PARAM_SHORT.values())
    valid = df.dropna(subset=["composite"])

    extra_cols = [c for c in EXTRA_OVERRIDES.values() if c in valid.columns]

    top = (
        valid.groupby(param_cols)
        .agg(
            run_id=           ("run_id",       "first"),
            mean_loyo_rmse=   ("loyo_rmse",    "mean"),
            std_loyo_rmse=    ("loyo_rmse",    "std"),
            mean_glambie_rmse=("glambie_rmse", "mean"),
            std_glambie_rmse= ("glambie_rmse", "std"),
            mean_composite=   ("composite",    "mean"),
            mean_rank=        ("rank_in_region","mean"),
            n_regions=        ("region",        "nunique"),
            **{c: (c, "first") for c in extra_cols},
        )
        .reset_index()
        .sort_values("mean_composite")
        .head(top_n)
    )

    col_order = ["run_id"] + param_cols + extra_cols + [
        "mean_loyo_rmse", "std_loyo_rmse",
        "mean_glambie_rmse", "std_glambie_rmse",
        "mean_composite", "mean_rank", "n_regions",
    ]
    top = top[[c for c in col_order if c in top.columns]]

    path = output_dir / "top_runs.csv"
    top.to_csv(path, index=False, float_format="%.5f")
    print(f"  Saved {path}")
    print("\nTop configurations (run_id = folder number to inspect):")
    print(top.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse bayesnf_jax hyperparameter sweep.")
    parser.add_argument(
        "--config",
        default="conf/config_hyperparam_tuning.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument("--multirun_root", default=None, help="Override multirun_root in config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.multirun_root:
        cfg["multirun_root"] = args.multirun_root

    multirun_root = Path(cfg["multirun_root"])
    # Output lives under hyperparam_analysis/{region_jobid}/  e.g. hyperparam_analysis/r06_3645680/
    output_dir = Path("hyperparam_analysis") / multirun_root.name
    test_years    = list(cfg.get("glambie_test_years", [2021, 2022, 2023]))
    loyo_weight   = float(cfg.get("loyo_weight",   0.5))
    glambie_weight= float(cfg.get("glambie_weight", 0.5))
    min_runs      = int(cfg.get("min_runs_per_region", 5))
    top_n         = int(cfg.get("top_n", 10))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    print(f"\n=== bayesnf_jax hyperparameter analysis ===")
    print(f"Multirun root : {multirun_root}")
    print(f"Output dir    : {output_dir}")
    print(f"Test years    : {test_years}")
    print(f"Weights       : LOYO={loyo_weight}  GLaMBIE={glambie_weight}\n")

    # ---- Load data ----
    df = build_results_df(multirun_root, test_years, min_runs_per_region=min_runs)
    df = add_composite_score(df, loyo_weight=loyo_weight, glambie_weight=glambie_weight)
    df = add_within_region_rank(df)

    results_path = output_dir / "sweep_results.csv"
    df.to_csv(results_path, index=False, float_format="%.5f")
    print(f"  Saved {results_path}")

    # ---- Plots ----
    print("\nGenerating plots...")

    print("  [1/8] Marginal box plots")
    plot_marginals(df, output_dir)

    print("  [2/8] 2-D heatmaps")
    plot_heatmaps(df, output_dir)

    print("  [3/8] Parallel coordinates")
    plot_parallel_coords(df, output_dir)

    print("  [4/8] Pretrain vs finetune scatter")
    plot_pretrain_vs_finetune(df, output_dir)

    print("  [5/8] Parameter importance")
    plot_param_importance(df, output_dir)

    print("  [6/8] Regional ranking heatmap")
    plot_regional_ranking(df, output_dir, top_n=top_n)

    print("  [7/8] Hidden width vs RMSE by n_layers")
    plot_nhidden_vs_rmse(df, output_dir)

    print("  [8/8] Heteroscedastic group comparison")
    plot_heteroscedastic_comparison(df, output_dir)

    # ---- Summary table ----
    print("\nTop runs:")
    save_top_runs(df, output_dir, top_n=top_n)

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
