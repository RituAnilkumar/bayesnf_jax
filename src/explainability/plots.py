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

    show_total      = std_attrs is not None
    show_components = std_epistemic_attrs is not None and std_structural_attrs is not None

    fig, ax = plt.subplots(figsize=(8, max(3, len(order) * 0.35 + 1)))
    y_pos = np.arange(len(order))

    ax.barh(y_pos, imp_sorted, color=_POS_COLOR, alpha=0.85, height=0.65)

    def _xerr_clipped(std_2d):
        err = np.abs(std_2d).mean(axis=0)[order]
        return np.array([np.minimum(err, imp_sorted), err])

    if show_total:
        ax.errorbar(imp_sorted, y_pos,
                    xerr=_xerr_clipped(std_attrs),
                    fmt="none", ecolor="grey", capsize=3, lw=1.2,
                    label="±1σ total")

    if show_components:
        ax.errorbar(imp_sorted, y_pos + 0.18,
                    xerr=_xerr_clipped(std_epistemic_attrs),
                    fmt="none", ecolor="darkorange", capsize=2, lw=1.0,
                    label="±1σ epistemic (VI)")
        ax.errorbar(imp_sorted, y_pos - 0.18,
                    xerr=_xerr_clipped(std_structural_attrs),
                    fmt="none", ecolor="mediumorchid", capsize=2, lw=1.0,
                    label="±1σ structural (ensemble)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |Attribution| (MWE/yr per unit input change)")
    ax.set_title("Global feature importance")
    ax.axvline(0, color="black", lw=0.7)
    if show_total or show_components:
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
    color_feature: str | int | None = None,
    alpha: float = 0.4,
    s: float = 8,
) -> None:
    """
    Dependence scatter: attribution for one feature vs that feature's value.
    Optionally colour points by a second feature.

    Args:
        mean_attrs:     (N, n_features) mean attributions.
        feature_values: (N, n_features) feature values.
        feature_names:  n_features strings.
        feature:        Feature to plot — name string or column index.
        output_path:    File path to save the figure.
        color_feature:  Feature to use for colour encoding (name or index).
                        Defaults to the feature with highest mean |attr|
                        correlation with the target feature's attribution.
        alpha / s:      Scatter transparency and marker size.
    """
    n_features = len(feature_names)

    fi = feature_names.index(feature) if isinstance(feature, str) else int(feature)

    # Auto-select colour feature if not given: feature with highest abs-corr
    # between its values and the target attribution
    if color_feature is None:
        corrs = [
            abs(np.corrcoef(feature_values[:, j], mean_attrs[:, fi])[0, 1])
            for j in range(n_features) if j != fi
        ]
        ci = [j for j in range(n_features) if j != fi][int(np.argmax(corrs))]
    else:
        ci = feature_names.index(color_feature) if isinstance(color_feature, str) else int(color_feature)

    x = feature_values[:, fi]
    y = mean_attrs[:, fi]
    c = feature_values[:, ci]

    # Normalise colour feature
    c_norm = (c - c.min()) / (c.max() - c.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x, y, c=c_norm, cmap=_BEESWARM_CMAP,
                         alpha=alpha, s=s, linewidths=0, vmin=0, vmax=1)
    ax.axhline(0, color="black", lw=0.7, ls="--")

    # Light LOWESS trend line via seaborn
    try:
        import statsmodels  # noqa: F401
        sns.regplot(x=x, y=y, ax=ax, scatter=False,
                    lowess=True, line_kws={"color": "black", "lw": 1.2, "ls": "-"})
    except ImportError:
        pass

    ax.set_xlabel(f"{feature_names[fi]} (feature value)")
    ax.set_ylabel(f"Attribution for {feature_names[fi]}")
    ax.set_title(f"Dependence: {feature_names[fi]}")

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label(feature_names[ci], fontsize=9)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["low", "mid", "high"])

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
) -> None:
    """
    Convenience wrapper: save a dependence plot for each of the top_k most
    important features (ranked by mean |attribution|).
    """
    os.makedirs(output_dir, exist_ok=True)
    importance = np.abs(mean_attrs).mean(axis=0)
    top_features = np.argsort(importance)[::-1][:top_k]

    for fi in top_features:
        fname = os.path.join(output_dir, f"dependence_{feature_names[fi]}.png")
        plot_dependence(mean_attrs, feature_values, feature_names, fi, fname)
