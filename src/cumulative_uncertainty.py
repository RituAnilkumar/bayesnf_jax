"""
src/cumulative_uncertainty.py

Correlation-aware cumulative-Gt uncertainty propagation, shared by every script
that produces a cumulative Gt plot: ensemble_common.py (used by
ensemble_uncertainty_pretrain_year.py), ensemble_uncertainty.py, ensemble_ep_alea.py,
validate_hma.py, plot_model_animations.py, and plot_acceleration_analysis.py.

Deliberately dependency-free (no import of ensemble_uncertainty.py or similar)
so every consumer — including ensemble_uncertainty.py itself — can import from
here without a circular import.

Background
----------
Naively cumulating annual std_total via sqrt(cumsum(std_total**2)) assumes every
year's uncertainty is independent of every other year's. That's true for aleatoric
noise (roughly independent point noise) but false for structural and epistemic
uncertainty: both arise from a *fixed* trained model (or a fixed small set of
models) evaluated at different years, so a model's bias persists across nearby
years rather than averaging out. Treating them as independent understates the
true cumulative uncertainty, sometimes by several-fold over a multi-decade record
(empirically ~6-7x for one 86-year regional test case — see CONTEXT.md).

compute_cumulative_gt_variants() computes five scenarios so the difference is
visible and auditable rather than silently assumed:

  cum_std_total                   (audited default, shown in the sensitivity
                                   comparison) structural=exact persistent,
                                   epistemic=persistent (approx.), aleatoric=independent
  cum_std_structural_independent  structural=independent (naive), epistemic=persistent,
                                   aleatoric=independent
  cum_std_all_correlated          structural=exact persistent, epistemic=persistent,
                                   aleatoric=persistent  (upper bound)
  cum_std_all_independent         structural=independent, epistemic=independent,
                                   aleatoric=independent (equivalent to the old,
                                   pre-fix formula — kept for direct comparison)
  cum_std_structural_indep_epi_alea_persist
                                   (chosen treatment for the operational
                                   ensemble_cumulative_gt_*.png outputs — a
                                   deliberate, different choice from the
                                   audited default above) structural=independent,
                                   epistemic=persistent, aleatoric=persistent.
                                   See compute_preferred_cumulative_gt() below,
                                   which recomputes this same formula over an
                                   arbitrary re-zeroed year window (e.g. from
                                   the GLaMBIE start year) rather than only the
                                   single full-record window this function uses.

"structural=exact persistent" is computed from the K ensemble members' own
regional Gt trajectories (each read from run_dir/regional_annual_gt.csv), not
approximated: cumsum each model's own trajectory, then take the weighted variance
across the K cumulative trajectories at each year. This requires the models'
source run directories to still exist (e.g. on /scratch) — if they don't
(cleaned up, moved), the exact variant falls back to the same persistent
*approximation* already used for epistemic (linear sum of annual std_structural,
no shrink), NOT to the independent (naive) treatment — falling all the way back
to independent would silently reproduce the pre-fix bug whenever source run
directories are unavailable, which defeats the purpose of the fix. A warning is
printed either way so the fallback is visible.

"epistemic=persistent" is an approximation (linear sum of per-year epistemic std,
not exact — exact would need raw per-draw MC samples, which are not currently
saved anywhere in this pipeline; see CONTEXT.md, "Per-glacier temporal
aggregation"). This is a minor caveat in practice since epistemic is consistently
the smallest of the three components.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _exact_structural_cumulative(
    run_dirs: list[Path],
    weights: np.ndarray,
    years: np.ndarray,
    file_name: str,
    value_col: str,
) -> np.ndarray | None:
    """
    Exact cumulative structural std: read each member's own per-year trajectory,
    cumsum it, then take the weighted std across members' cumulative trajectories
    at each year. Returns None if any member's file is unreadable or misaligned,
    or if run_dirs is empty (e.g. the source member list could not be determined).
    """
    if not run_dirs:
        return None
    trajectories = []
    for run_dir in run_dirs:
        path = Path(run_dir) / file_name
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path).sort_values("year").reset_index(drop=True)
        except Exception:
            return None
        df = df[df["year"].isin(years)].reset_index(drop=True)
        if len(df) != len(years) or not np.array_equal(df["year"].values, years):
            return None
        trajectories.append(df[value_col].values)

    traj_mat = np.stack(trajectories)          # (K, T)
    cum_mat  = np.cumsum(traj_mat, axis=1)     # (K, T) — each member's own cumulative trajectory

    w = weights[:, np.newaxis]                 # (K, 1)
    cum_mean = (w * cum_mat).sum(axis=0)       # (T,)
    cum_var  = (w * (cum_mat - cum_mean[np.newaxis]) ** 2).sum(axis=0)
    return np.sqrt(cum_var)


def compute_cumulative_gt_variants(
    run_dirs: list[Path],
    weights: np.ndarray,
    years: np.ndarray,
    median_gt: np.ndarray,
    std_structural: np.ndarray,
    std_epistemic: np.ndarray,
    std_aleatoric: np.ndarray,
    file_name: str = "regional_annual_gt.csv",
    value_col: str = "mean",
) -> pd.DataFrame:
    """
    Build the four-scenario cumulative Gt uncertainty table for one region/group.

    Args:
        run_dirs:       Source run directories for the K ensemble members (used to
                         read each member's own per-year trajectory for the exact
                         structural term). May be stale (e.g. scratch cleaned up);
                         handled gracefully with a fallback.
        weights:        Normalised weights for the K members (equal or performance).
        years:          Sorted year array, shape (T,).
        median_gt:      Ensemble weighted-mean Gt/yr per year, shape (T,) — already
                         computed by the caller (this is what gets cumsum'd for the
                         point estimate).
        std_structural, std_epistemic, std_aleatoric:
                         Annual per-component std (Gt/yr), shape (T,) — already
                         computed by the caller via the standard per-year ensemble
                         decomposition. NaNs are not accepted — callers should
                         fill missing aleatoric values with 0.0 before calling.
        file_name:      Per-run file to read for each member's own trajectory
                         ("regional_annual_gt.csv" or "regional_annual_mwe.csv").
        value_col:      Column to read from that file ("mean").

    Returns:
        DataFrame with columns:
            year, cum_median_gt,
            cum_std_total, cum_std_structural_independent,
            cum_std_all_correlated, cum_std_all_independent
    """
    cum_median = np.cumsum(median_gt)
    comp = compute_cumulative_components(
        run_dirs, weights, years, std_structural, std_epistemic, std_aleatoric,
        file_name, value_col,
    )
    scenarios = combine_cumulative_scenarios(comp)
    return pd.DataFrame({
        "year":          years,
        "cum_median_gt": cum_median,
        **scenarios,
    })


def compute_cumulative_components(
    run_dirs: list[Path],
    weights: np.ndarray,
    years: np.ndarray,
    std_structural: np.ndarray,
    std_epistemic: np.ndarray,
    std_aleatoric: np.ndarray,
    file_name: str = "regional_annual_gt.csv",
    value_col: str = "mean",
) -> dict:
    """
    Compute the six intermediate cumulative-per-component arrays that the four
    scenarios in compute_cumulative_gt_variants() are built from. Exposed
    separately so callers that need to combine multiple independent ensembles
    first (e.g. validate_hma.py summing r13+r14+r15 in quadrature) can do so at
    the per-component level before re-deriving the final scenario columns via
    combine_cumulative_scenarios().

    Returns a dict with keys:
        cum_struct_exact, cum_struct_indep,
        cum_epi_persist,  cum_epi_indep,
        cum_alea_persist, cum_alea_indep
    each an array of shape (T,) matching `years`.
    """
    # --- "independent" (naive, shrinking) cumulative std for each component ---
    cum_struct_indep = np.sqrt(np.cumsum(std_structural ** 2))
    cum_epi_indep    = np.sqrt(np.cumsum(std_epistemic ** 2))
    cum_alea_indep   = np.sqrt(np.cumsum(std_aleatoric ** 2))

    # --- "persistent" (no-shrink) cumulative std for epistemic and aleatoric ---
    # Epistemic persistence is an approximation (see module docstring); aleatoric
    # persistence is only used for the all-correlated upper-bound scenario.
    cum_epi_persist  = np.cumsum(std_epistemic)
    cum_alea_persist = np.cumsum(std_aleatoric)

    # --- exact structural persistence from each member's own trajectory ---
    cum_struct_exact = _exact_structural_cumulative(
        run_dirs, weights, years, file_name, value_col
    )
    if cum_struct_exact is None:
        warnings.warn(
            "  compute_cumulative_components: could not read source run "
            f"trajectories ({file_name}) for one or more of {len(run_dirs)} "
            "members — falling back to the persistent-approximation structural "
            "treatment (linear sum of annual std_structural, same approximation "
            "already used for epistemic) for cum_std_total and "
            "cum_std_all_correlated, rather than the exact per-model computation. "
            "This is still a large improvement over the independent (naive) "
            "treatment — it is NOT the same as falling back to the old pre-fix "
            "formula. Source run directories may have been cleaned up."
        )
        cum_struct_exact = np.cumsum(std_structural)

    return {
        "cum_struct_exact": cum_struct_exact,
        "cum_struct_indep": cum_struct_indep,
        "cum_epi_persist":  cum_epi_persist,
        "cum_epi_indep":    cum_epi_indep,
        "cum_alea_persist": cum_alea_persist,
        "cum_alea_indep":   cum_alea_indep,
    }


def combine_cumulative_scenarios(comp: dict) -> dict:
    """
    Combine the six per-component cumulative arrays (from
    compute_cumulative_components, or a quadrature-sum of several regions'
    components — see validate_hma.py) into the four final scenario std arrays.
    """
    cum_std_total = np.sqrt(
        comp["cum_struct_exact"] ** 2 + comp["cum_epi_persist"] ** 2 + comp["cum_alea_indep"] ** 2
    )
    cum_std_structural_independent = np.sqrt(
        comp["cum_struct_indep"] ** 2 + comp["cum_epi_persist"] ** 2 + comp["cum_alea_indep"] ** 2
    )
    cum_std_all_correlated = np.sqrt(
        comp["cum_struct_exact"] ** 2 + comp["cum_epi_persist"] ** 2 + comp["cum_alea_persist"] ** 2
    )
    cum_std_all_independent = np.sqrt(
        comp["cum_struct_indep"] ** 2 + comp["cum_epi_indep"] ** 2 + comp["cum_alea_indep"] ** 2
    )
    # Chosen treatment for the operational ensemble_cumulative_gt_*.png plots
    # (see module docstring) — structural independent, epistemic+aleatoric persistent.
    cum_std_structural_indep_epi_alea_persist = np.sqrt(
        comp["cum_struct_indep"] ** 2 + comp["cum_epi_persist"] ** 2 + comp["cum_alea_persist"] ** 2
    )
    return {
        "cum_std_total":                  cum_std_total,
        "cum_std_structural_independent": cum_std_structural_independent,
        "cum_std_all_correlated":         cum_std_all_correlated,
        "cum_std_all_independent":        cum_std_all_independent,
        "cum_std_structural_indep_epi_alea_persist": cum_std_structural_indep_epi_alea_persist,
    }


def compute_preferred_cumulative_gt(
    years: np.ndarray,
    median_gt: np.ndarray,
    std_structural: np.ndarray,
    std_epistemic: np.ndarray,
    std_aleatoric: np.ndarray,
    start_year: int | None = None,
) -> pd.DataFrame:
    """
    Cumulative Gt using the chosen treatment for the operational
    ensemble_cumulative_gt_*.png outputs: structural=independent,
    epistemic=persistent, aleatoric=persistent (see module docstring — this
    is a deliberate choice, different from cum_std_total in
    compute_cumulative_gt_variants(), which is kept as the audited default
    shown in the sensitivity comparison).

    Cheap and self-contained: unlike compute_cumulative_gt_variants(), this
    never needs each ensemble member's own run_dir/regional_annual_gt.csv,
    because structural is treated as independent here, not exact-persistent.

    If start_year is given, all arrays are first restricted to years >=
    start_year and the cumulative sum restarts (re-zeros) from there — e.g.
    for a plot aligned to the first year GLaMBIE data is available, rather
    than the full model record.

    Returns a DataFrame with columns:
        year, cum_median_gt, cum_std_gt (total),
        cum_std_structural (independent), cum_std_epistemic (persistent)
    — the last two are exposed for plots that show per-component bands
    (e.g. ensemble_ep_alea.py's epistemic/structural/total overlay).
    """
    years          = np.asarray(years)
    median_gt      = np.asarray(median_gt, dtype=float)
    std_structural = np.nan_to_num(np.asarray(std_structural, dtype=float))
    std_epistemic  = np.nan_to_num(np.asarray(std_epistemic, dtype=float))
    std_aleatoric  = np.nan_to_num(np.asarray(std_aleatoric, dtype=float))

    if start_year is not None:
        mask = years >= start_year
        years, median_gt = years[mask], median_gt[mask]
        std_structural, std_epistemic, std_aleatoric = (
            std_structural[mask], std_epistemic[mask], std_aleatoric[mask]
        )

    cum_median       = np.cumsum(median_gt)
    cum_struct_indep = np.sqrt(np.cumsum(std_structural ** 2))
    cum_epi_persist  = np.cumsum(std_epistemic)
    cum_alea_persist = np.cumsum(std_aleatoric)
    cum_std = np.sqrt(cum_struct_indep ** 2 + cum_epi_persist ** 2 + cum_alea_persist ** 2)

    return pd.DataFrame({
        "year": years,
        "cum_median_gt": cum_median,
        "cum_std_gt": cum_std,
        "cum_std_structural": cum_struct_indep,
        "cum_std_epistemic": cum_epi_persist,
    })


def plot_cumulative_sensitivity(
    cum_df: pd.DataFrame,
    output_path: Path,
    title: str = "Cumulative Gt uncertainty — sensitivity to correlation assumptions",
) -> None:
    """
    4-subplot comparison of the four cumulative-uncertainty scenarios, shared
    y-axis scale across panels so relative band widths are directly comparable.
    """
    years = cum_df["year"].values
    med   = cum_df["cum_median_gt"].values

    scenarios = [
        ("cum_std_total",                  "Recommended\n(structural+epistemic persistent, aleatoric independent)"),
        ("cum_std_structural_independent", "Structural treated as independent\n(epistemic persistent, aleatoric independent)"),
        ("cum_std_all_correlated",         "All correlated\n(upper bound)"),
        ("cum_std_all_independent",        "All independent\n(≡ pre-fix formula)"),
    ]

    all_std = np.concatenate([cum_df[col].values for col, _ in scenarios])
    ylim = (
        float(np.min(med - 2 * all_std.max())),
        float(np.max(med + 2 * all_std.max())),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, (col, label) in zip(axes.flat, scenarios):
        std = cum_df[col].values
        ax.fill_between(years, med - 2 * std, med + 2 * std, alpha=0.25, color="steelblue")
        ax.plot(years, med, color="steelblue", lw=1.6)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(*ylim)

    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative Gt")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, output_path)
