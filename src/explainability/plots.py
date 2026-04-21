"""
Seaborn/matplotlib explainability plots.

All functions accept:
  mean_attrs:     (N, n_features) float array — mean attribution per point per feature
  feature_names:  list of n_features strings — must include "year" at index 0
  feature_values: (N, n_features) float array — raw (or standardised) feature values,
                  used only for colouring / x-axes

The expected feature ordering throughout matches compute_attributions():
  [year, cov_0, cov_1, ..., cov_{n_cov-1}]
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)

_POS_COLOR = "#4575b4"   # blue  — positive attribution
_NEG_COLOR = "#d73027"   # red   — negative attribution
_BEESWARM_CMAP = "RdBu_r"


# ---------------------------------------------------------------------------
# 1. Global importance bar chart
# ---------------------------------------------------------------------------

def plot_importance_bar(
    mean_attrs: np.ndarray,
    feature_names: list[str],
    output_path: str,
    std_attrs: np.ndarray | None = None,
    top_k: int | None = None,
    std_epistemic_attrs: np.ndarray | None = None,
    std_structural_attrs: np.ndarray | None = None,
) -> None:
    """
    Horizontal bar chart of mean |attribution| per feature, sorted descending.

    Args:
        mean_attrs:           (N, n_features) mean attributions.
        feature_names:        n_features feature name strings.
        output_path:          File path to save the figure.
        std_attrs:            (N, n_features) total std; shown as grey error bars.
        top_k:                Only show top_k features (default: all).
        std_epistemic_attrs:  (N, n_features) epistemic (VI) std; orange whiskers.
        std_structural_attrs: (N, n_features) structural (between-model) std; purple whiskers.
    """
    importance = np.abs(mean_attrs).mean(axis=0)            # (n_features,)

    order = np.argsort(importance)[::-1]
    if top_k is not None:
        order = order[:top_k]

    imp_sorted   = importance[order]
    names_sorted = [feature_names[i] for i in order]

    # Glacier heterogeneity std: how variable the attribution magnitude is across
    # glacier-years (always computed; independent of model uncertainty).
    glacier_std = np.abs(mean_attrs).std(axis=0)[order]

    show_total      = std_attrs is not None
    show_components = std_epistemic_attrs is not None and std_structural_attrs is not None

    fig, ax = plt.subplots(figsize=(8, max(3, len(order) * 0.35 + 1)))
    y_pos = np.arange(len(order))

    ax.barh(y_pos, imp_sorted, color=_POS_COLOR, alpha=0.85, height=0.65)

    def _xerr_clipped(err):
        """Clip lower whisker so it never extends past zero."""
        return np.array([np.minimum(err, imp_sorted), err])

    # Four whiskers at fixed offsets (drawn only when data is available).
    # Positions (from top to bottom of each bar):
    #   +0.24  black  — glacier heterogeneity (always)
    #   +0.08  grey   — total model uncertainty
    #   -0.08  orange — epistemic (VI weight uncertainty)
    #   -0.24  purple — structural (between-model)
    ax.errorbar(imp_sorted, y_pos + 0.24,
                xerr=_xerr_clipped(glacier_std),
                fmt="none", ecolor="black", capsize=2, lw=1.0,
                label="±1σ across glaciers")

    if show_total:
        total_err = np.abs(std_attrs).mean(axis=0)[order]
        ax.errorbar(imp_sorted, y_pos + 0.08,
                    xerr=_xerr_clipped(total_err),
                    fmt="none", ecolor="grey", capsize=2, lw=1.0,
                    label="±1σ total (model)")

    if show_components:
        epi_err    = np.abs(std_epistemic_attrs).mean(axis=0)[order]
        struct_err = np.abs(std_structural_attrs).mean(axis=0)[order]
        ax.errorbar(imp_sorted, y_pos - 0.08,
                    xerr=_xerr_clipped(epi_err),
                    fmt="none", ecolor="darkorange", capsize=2, lw=1.0,
                    label="±1σ epistemic (VI)")
        ax.errorbar(imp_sorted, y_pos - 0.24,
                    xerr=_xerr_clipped(struct_err),
                    fmt="none", ecolor="mediumorchid", capsize=2, lw=1.0,
                    label="±1σ structural (ensemble)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |Attribution| (MWE/yr per unit input change)")
    ax.set_title("Global feature importance")
    ax.axvline(0, color="black", lw=0.7)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# 2. Beeswarm (SHAP-style)
# ---------------------------------------------------------------------------

def plot_beeswarm(
    mean_attrs: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    output_path: str,
    n_max: int = 1000,
    seed: int = 0,
) -> None:
    """
    Beeswarm plot: one dot per data point per feature.
    x = attribution value, y = feature (jittered), colour = normalised feature value.

    Args:
        mean_attrs:     (N, n_features) mean attributions.
        feature_values: (N, n_features) feature values (for colour encoding).
        feature_names:  n_features feature name strings.
        output_path:    File path to save the figure.
        n_max:          Max points to plot (subsampled if N > n_max).
        seed:           RNG seed for subsampling.
    """
    N = len(mean_attrs)
    rng_np = np.random.default_rng(seed)
    idx = rng_np.choice(N, min(N, n_max), replace=False) if N > n_max else np.arange(N)

    attrs = mean_attrs[idx]             # (M, n_features)
    fvals = feature_values[idx]         # (M, n_features)
    n_features = len(feature_names)

    fig, ax = plt.subplots(figsize=(10, max(4, n_features * 0.45 + 1)))
    cmap = cm.get_cmap(_BEESWARM_CMAP)

    rng_jitter = np.random.default_rng(seed + 1)
    for fi in range(n_features):
        a = attrs[:, fi]
        fv = fvals[:, fi]

        # Normalise feature values to [0, 1] per feature for colour encoding
        fv_min, fv_max = fv.min(), fv.max()
        fv_norm = (fv - fv_min) / (fv_max - fv_min) if fv_max > fv_min else np.full_like(fv, 0.5)

        y_jitter = fi + rng_jitter.uniform(-0.25, 0.25, size=len(a))
        colors = cmap(fv_norm)

        ax.scatter(a, y_jitter, c=colors, alpha=0.55, s=8, linewidths=0)

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(feature_names)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Attribution (MWE/yr per unit input change)")
    ax.set_title(f"Feature attribution beeswarm  (n={len(idx)} points)")

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("Feature value\n(low → high)", fontsize=9)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["low", "mid", "high"])

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# 3. Dependence scatter
# ---------------------------------------------------------------------------

def plot_dependence(
    mean_attrs: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    feature: str | int,
    output_path: str,
    color_feature: str | int | None = "year",
    alpha: float = 0.4,
    s: float = 8,
    color_values: np.ndarray | None = None,
    color_label: str = "year",
) -> None:
    """
    Dependence scatter: attribution for one feature vs that feature's value.

    x-axis: the raw feature value (e.g. summer temperature in °C).
    y-axis: the IG attribution for that feature — how much it shifted the
            predicted mass balance away from the baseline.
    colour: by default, year (viridis sequential colormap) so you can see
            whether the relationship between a feature and its effect on mass
            balance has changed over time.  Pass color_feature=None to
            auto-select the most correlated other feature instead (RdBu_r).

    Args:
        mean_attrs:     (N, n_features) mean attributions.
        feature_values: (N, n_features) feature values.
        feature_names:  n_features strings.
        feature:        Feature to plot — name string or column index.
        output_path:    File path to save the figure.
        color_feature:  Feature to use for colour encoding (name, index, or None).
                        Default "year" — colours by year using viridis so temporal
                        drift is immediately visible.  Set to None to fall back to
                        auto-selecting the most correlated other feature.
                        Ignored when color_values is provided.
        alpha / s:      Scatter transparency and marker size.
        color_values:   (N,) array to use directly for colouring, bypassing any
                        feature lookup.  Use this to force year colouring on masked
                        plots where "year" has been removed from feature_names.
        color_label:    Colorbar label when color_values is provided (default "year").
    """
    n_features = len(feature_names)
    fi = feature_names.index(feature) if isinstance(feature, str) else int(feature)

    x = feature_values[:, fi]
    y = mean_attrs[:, fi]

    if color_values is not None:
        # Caller supplied colour array directly (e.g. year from unmasked data)
        c          = color_values
        color_name = color_label
        color_by_year = True
    else:
        # Resolve colour feature index — fall back to auto-select if the requested
        # feature is not present (e.g. "year" requested but masked out).
        def _auto_ci():
            corrs = [
                abs(np.corrcoef(feature_values[:, j], mean_attrs[:, fi])[0, 1])
                for j in range(n_features) if j != fi
            ]
            return [j for j in range(n_features) if j != fi][int(np.argmax(corrs))]

        if color_feature is None:
            ci = _auto_ci()
        elif isinstance(color_feature, str):
            ci = feature_names.index(color_feature) if color_feature in feature_names \
                 else _auto_ci()
        else:
            ci = int(color_feature)

        color_name    = feature_names[ci]
        color_by_year = (color_name == "year")
        c             = feature_values[:, ci]

    if color_by_year:
        cmap_use   = "viridis"
        scatter_kw = dict(c=c, cmap=cmap_use, alpha=alpha, s=s, linewidths=0)
        cbar_label = color_name
        cbar_ticks = None   # matplotlib auto ticks show real year values
    else:
        c_norm = (c - c.min()) / (c.max() - c.min() + 1e-12)
        cmap_use   = _BEESWARM_CMAP
        scatter_kw = dict(c=c_norm, cmap=cmap_use, vmin=0, vmax=1,
                          alpha=alpha, s=s, linewidths=0)
        cbar_label = color_name
        cbar_ticks = ([0.0, 0.5, 1.0], ["low", "mid", "high"])

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x, y, **scatter_kw)
    ax.axhline(0, color="black", lw=0.7, ls="--")

    # Light LOWESS trend line
    try:
        import statsmodels  # noqa: F401
        sns.regplot(x=x, y=y, ax=ax, scatter=False,
                    lowess=True, line_kws={"color": "black", "lw": 1.2, "ls": "-"})
    except ImportError:
        pass

    ax.set_xlabel(f"{feature_names[fi]} (feature value)")
    ax.set_ylabel(f"Attribution for {feature_names[fi]}")
    ax.set_title(f"Dependence: {feature_names[fi]}  (colour = {color_name})")

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks[0])
        cbar.set_ticklabels(cbar_ticks[1])

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# 4. Local waterfall
# ---------------------------------------------------------------------------

def plot_waterfall(
    mean_attr_1d: np.ndarray,
    feature_names: list[str],
    output_path: str,
    std_attr_1d: np.ndarray | None = None,
    baseline_pred: float | None = None,
    title: str = "Local attribution waterfall",
) -> None:
    """
    Horizontal waterfall bar chart for a single prediction.
    Positive = pushes prediction up (blue), negative = down (red).
    Sorted by absolute attribution magnitude.

    Args:
        mean_attr_1d:  (n_features,) attribution vector for one data point.
        feature_names: n_features strings.
        output_path:   File path to save the figure.
        std_attr_1d:   (n_features,) MC std; shown as error bars if provided.
        baseline_pred: Value of the prediction at the baseline input.
                       Shown as an annotation if provided.
        title:         Figure title.
    """
    attrs = np.array(mean_attr_1d)
    errs  = np.array(std_attr_1d) if std_attr_1d is not None else None

    order = np.argsort(np.abs(attrs))[::-1]
    attrs_sorted = attrs[order]
    errs_sorted  = errs[order] if errs is not None else None
    names_sorted = [feature_names[i] for i in order]

    colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in attrs_sorted]

    fig, ax = plt.subplots(figsize=(8, max(3, len(order) * 0.35 + 1.2)))
    y_pos = np.arange(len(order))

    ax.barh(
        y_pos, attrs_sorted,
        xerr=errs_sorted,
        color=colors, alpha=0.85,
        ecolor="grey", capsize=3,
        height=0.65,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Attribution (MWE/yr per unit input change)")
    ax.set_title(title)

    if baseline_pred is not None:
        ax.text(
            0.98, 0.02,
            f"Baseline prediction: {baseline_pred:.4f} MWE/yr",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="grey",
        )

    # Add value labels on bars
    for yi, (val, name) in enumerate(zip(attrs_sorted, names_sorted)):
        ha = "left" if val >= 0 else "right"
        offset = 0.001 * (np.abs(attrs_sorted).max() or 1)
        ax.text(val + (offset if val >= 0 else -offset), yi, f"{val:+.4f}",
                va="center", ha=ha, fontsize=7.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# 5. Summary: top-k dependence grid
# ---------------------------------------------------------------------------

def plot_dependence_grid(
    mean_attrs: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    top_k: int = 6,
    color_feature: str | int | None = "year",
    color_values: np.ndarray | None = None,
    color_label: str = "year",
) -> None:
    """
    Convenience wrapper: save a dependence plot for each of the top_k most
    important features (ranked by mean |attribution|).

    color_feature:  passed to each plot_dependence call (default "year").
                    Pass None to auto-select the most correlated other feature.
    color_values:   if provided, overrides color_feature entirely (used to
                    force year colouring on masked plots).
    """
    os.makedirs(output_dir, exist_ok=True)
    importance = np.abs(mean_attrs).mean(axis=0)
    top_features = np.argsort(importance)[::-1][:top_k]

    for fi in top_features:
        fname = os.path.join(output_dir, f"dependence_{feature_names[fi]}.png")
        plot_dependence(mean_attrs, feature_values, feature_names, fi, fname,
                        color_feature=color_feature,
                        color_values=color_values, color_label=color_label)


# ---------------------------------------------------------------------------
# 6. Year slice — importance bar for a single year
# ---------------------------------------------------------------------------

def plot_year_slice(
    mean_attrs: np.ndarray,
    years: np.ndarray,
    feature_names: list[str],
    target_year: int,
    output_path: str,
    std_epistemic_attrs: np.ndarray | None = None,
    std_structural_attrs: np.ndarray | None = None,
    top_k: int | None = None,
) -> None:
    """
    Importance bar chart for a single year: mean |attribution| across all
    glaciers in target_year, with cross-glacier std (black) and optional
    model uncertainty whiskers (grey/orange/purple).

    Args:
        mean_attrs:           (N, n_features) attributions for all points.
        years:                (N,) year for each data point.
        feature_names:        n_features strings.
        target_year:          The year to slice.
        output_path:          File path to save the figure.
        std_epistemic_attrs:  (N, n_features) epistemic std (ensemble mode).
        std_structural_attrs: (N, n_features) structural std (ensemble mode).
        top_k:                Only show top_k features (default: all).
    """
    mask = years == target_year
    if mask.sum() == 0:
        print(f"WARNING: No data points found for year {target_year} — skipping year slice.")
        return

    attrs_yr = mean_attrs[mask]      # (n_glaciers_yr, F)
    epi_yr   = std_epistemic_attrs[mask]  if std_epistemic_attrs  is not None else None
    struct_yr= std_structural_attrs[mask] if std_structural_attrs is not None else None

    importance = np.abs(attrs_yr).mean(axis=0)
    glacier_std = np.abs(attrs_yr).std(axis=0)

    order = np.argsort(importance)[::-1]
    if top_k is not None:
        order = order[:top_k]

    imp_sorted   = importance[order]
    gs_sorted    = glacier_std[order]
    names_sorted = [feature_names[i] for i in order]

    show_components = epi_yr is not None and struct_yr is not None

    fig, ax = plt.subplots(figsize=(8, max(3, len(order) * 0.35 + 1)))
    y_pos = np.arange(len(order))

    ax.barh(y_pos, imp_sorted, color=_POS_COLOR, alpha=0.85, height=0.65)

    def _clip(err):
        return np.array([np.minimum(err, imp_sorted), err])

    ax.errorbar(imp_sorted, y_pos + (0.24 if show_components else 0),
                xerr=_clip(gs_sorted),
                fmt="none", ecolor="black", capsize=2, lw=1.0,
                label="±1σ across glaciers")

    if show_components:
        ax.errorbar(imp_sorted, y_pos + 0.08,
                    xerr=_clip(np.abs(epi_yr).mean(axis=0)[order]),
                    fmt="none", ecolor="darkorange", capsize=2, lw=1.0,
                    label="±1σ epistemic (VI)")
        ax.errorbar(imp_sorted, y_pos - 0.08,
                    xerr=_clip(np.abs(struct_yr).mean(axis=0)[order]),
                    fmt="none", ecolor="mediumorchid", capsize=2, lw=1.0,
                    label="±1σ structural (ensemble)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |Attribution| (MWE/yr per unit input change)")
    ax.set_title(f"Feature importance — year {target_year}  (n={mask.sum()} glaciers)")
    ax.axvline(0, color="black", lw=0.7)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# 7. Temporal importance — stacked area plots across years
# ---------------------------------------------------------------------------

def plot_temporal_importance(
    mean_attrs: np.ndarray,
    years: np.ndarray,
    feature_names: list[str],
    output_path: str,
    top_k: int = 6,
) -> None:
    """
    Vertically stacked subplots — one panel per top-k feature, each with its
    own y-scale.  Shows signed mean attribution ± 1σ across glaciers per year.

    Each panel:
      - Shaded band = ±1σ of attribution across glaciers in that year
      - Line = mean signed attribution
      - Dashed zero line

    Args:
        mean_attrs:    (N, n_features) attributions (signed).
        years:         (N,) year for each data point.
        feature_names: n_features strings.
        output_path:   File path to save the figure.
        top_k:         Number of top features (ranked by global mean |attr|).
    """
    global_importance = np.abs(mean_attrs).mean(axis=0)
    top_idx   = np.argsort(global_importance)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_idx]

    unique_years = np.sort(np.unique(years))
    yr_mean = np.full((len(unique_years), len(top_idx)), np.nan)
    yr_std  = np.full((len(unique_years), len(top_idx)), np.nan)
    for yi, yr in enumerate(unique_years):
        mask = years == yr
        if mask.sum() == 0:
            continue
        sub = mean_attrs[mask][:, top_idx]
        yr_mean[yi] = sub.mean(axis=0)
        yr_std[yi]  = sub.std(axis=0)

    valid = ~np.isnan(yr_mean[:, 0])
    yrs   = unique_years[valid]

    cmap   = cm.get_cmap("tab10")
    n      = len(top_idx)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.2 * n + 0.8),
                             sharex=True)
    if n == 1:
        axes = [axes]

    for j, (ax, fi, name) in enumerate(zip(axes, top_idx, top_names)):
        color = cmap(j % 10)
        mu = yr_mean[valid, j]
        sd = yr_std[valid, j]

        shade = ax.fill_between(yrs, mu - sd, mu + sd, color=color, alpha=0.25,
                                label="±1σ across glaciers")
        ax.plot(yrs, mu, color=color, lw=1.6, label="mean attribution")
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.set_ylabel(name, fontsize=9)
        ax.yaxis.set_major_formatter(
            plt.matplotlib.ticker.FormatStrFormatter("%.3f")
        )
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
        # Only add legend to first panel to avoid repetition
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Year")
    fig.suptitle(
        f"Feature attribution over time  (top {top_k})\n"
        "line = mean signed attribution across glaciers per year   "
        "shading = ±1σ across glaciers",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")
