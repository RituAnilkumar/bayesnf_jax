"""
Inference utilities — MC forward passes and output formatting.

Loads finetuned_params.pkl and generates predictions over the full
(glacier × year) grid specified in for_preds.csv.

Output files:
  preds_full.csv           — per (glacier, year): rgi_id, year, p2_5, p50, p97_5, mean, std
  regional_annual_mwe.csv  — area-weighted regional mean MWE/yr per year (MC quantiles)
  regional_annual_gt.csv   — regional Gt/yr per year (MC + density uncertainty)
  regional_mass_loss.png   — Gt/yr time series + GLaMBIE combined overlay
  hugonnet_scatter_*.png   — per-period scatter: model vs Hugonnet dmdtda per glacier
"""

import os
import jax
import jax.numpy as jnp
import cloudpickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.model.bnf_module import BayesianNeuralField, T_MIN
from src.data_utils import (
    load_features,
    load_dmdtda,
    build_model_inputs,
    apply_scaler,
    FEATURE_COLS,
)

# Ice density assumption (Huss 2013) for Gt uncertainty propagation
_ICE_DENSITY     = 850.0   # kg/m³
_ICE_DENSITY_SIG = 60.0    # kg/m³ 1-sigma uncertainty


# ---------------------------------------------------------------------------
# Param loading and reconstruction
# ---------------------------------------------------------------------------

def load_finetuned_params(finetuned_params_path: str) -> tuple[dict, dict]:
    """
    Load (mu_dict, log_sigma_dict) from finetuned_params.pkl.

    Returns:
        (mu_dict, log_sigma_dict) with same pytree structure as model params['params']
    """
    with open(finetuned_params_path, "rb") as f:
        result = cloudpickle.load(f)
    assert isinstance(result, tuple) and len(result) == 2, \
        f"Expected (mu_dict, log_sigma_dict), got {type(result)}"
    return result


def build_params_from_posterior(mu_dict: dict, log_sigma_dict: dict) -> dict:
    """
    Reconstruct a full Flax params dict from (mu_dict, log_sigma_dict).

    Inverse of extract_vi_params() — merges the two split dicts back into
    the structure expected by model.apply().

    Returns:
        {'params': {merged leaves}}
    """
    def _merge(d_mu, d_ls):
        result = {**d_mu}
        for k, v in d_ls.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = _merge(result[k], v)
            else:
                result[k] = v
        return result

    return {"params": _merge(mu_dict, log_sigma_dict)}


# ---------------------------------------------------------------------------
# Prediction grid loading
# ---------------------------------------------------------------------------

def load_pred_grid(cfg: DictConfig, ft_cols: list[str]) -> pd.DataFrame:
    """
    Load the full prediction grid for the configured region.

    Uses main_features_{region}.csv directly — for_preds corresponds to
    all rows in the features file (full glacier × year grid).
    time_index = year - T_MIN is already added by load_features().

    Returns DataFrame with rgi_id, year, time_index, + feature columns.
    """
    region = cfg.model.reg_subdir
    base   = os.path.join(cfg.model.inp_dir, region)
    path   = os.path.join(base, f"main_features_{region}.csv")
    return load_features(path, feature_cols=ft_cols)


# ---------------------------------------------------------------------------
# MC prediction
# ---------------------------------------------------------------------------

def mc_predict_region(
    model: BayesianNeuralField,
    params: dict,
    time_index: jax.Array,
    covariates: jax.Array,
    rng: jax.Array,
    n_samples: int,
) -> jax.Array:
    """
    Draw n_samples MC predictions over the full prediction grid.

    Returns:
        Array of shape (n_samples, N)  [MWE/yr]
    """
    return model.apply(
        params,
        time_index,
        covariates,
        rng,
        n_samples=n_samples,
        method=model.mc_predict,
    )


# ---------------------------------------------------------------------------
# Quantile extraction and output formatting
# ---------------------------------------------------------------------------

def extract_quantiles(mc_preds: jax.Array) -> dict:
    """
    Compute summary statistics over MC samples (axis=0).

    Args:
        mc_preds: shape (n_samples, N)

    Returns:
        Dict with keys: p2_5, p50, p97_5, mean, std — each shape (N,) as numpy arrays.
    """
    mc_np = np.array(mc_preds)
    return {
        "p2_5":  np.percentile(mc_np, 2.5,  axis=0),
        "p50":   np.percentile(mc_np, 50.0,  axis=0),
        "p97_5": np.percentile(mc_np, 97.5,  axis=0),
        "mean":  mc_np.mean(axis=0),
        "std":   mc_np.std(axis=0),
    }


def predictions_to_df(pred_grid: pd.DataFrame, quantiles: dict) -> pd.DataFrame:
    """
    Assemble the full predictions DataFrame.

    Output columns:
        rgi_id, year, p2_5, p50, p97_5, mean, std  — all mass balance in MWE/yr

    Args:
        pred_grid:  DataFrame with rgi_id and year columns (N rows)
        quantiles:  Output of extract_quantiles()

    Returns:
        DataFrame with N rows.
    """
    return pd.DataFrame({
        "rgi_id": pred_grid["rgi_id"].values,
        "year":   pred_grid["year"].values,
        "p2_5":   quantiles["p2_5"],
        "p50":    quantiles["p50"],
        "p97_5":  quantiles["p97_5"],
        "mean":   quantiles["mean"],
        "std":    quantiles["std"],
    })


# ---------------------------------------------------------------------------
# Regional aggregate outputs
# ---------------------------------------------------------------------------

def compute_regional_series(
    mc_preds_np: np.ndarray,   # (n_samples, N) MWE/yr, already unscaled
    pred_grid: pd.DataFrame,    # N rows, must have year and Area (km²)
) -> dict:
    """
    Compute annual area-weighted regional mean MWE/yr and Gt/yr across MC samples.

    Area column is assumed to be in km².
    Gt/yr = MWE/yr × total_area_km² × 1e-3  (via ρ_water = 1000 kg/m³).
    Density uncertainty (±60 kg/m³ on 850) is propagated separately and stored
    for use in plots and CSV output.

    Returns dict with keys:
        years            — sorted list of int years
        mwe_samples      — (n_samples, n_years) area-weighted mean MWE/yr
        gt_samples       — (n_samples, n_years) Gt/yr
        total_area_km2   — (n_years,) total region area per year
    """
    years = sorted(pred_grid["year"].unique().tolist())
    year_to_idx = {y: i for i, y in enumerate(years)}
    n_years  = len(years)
    areas    = pred_grid["Area"].values.astype(np.float64)
    year_idx = pred_grid["year"].map(year_to_idx).values

    total_area = np.array([areas[year_idx == i].sum() for i in range(n_years)])

    # Area-weighted mean per sample per year: one dot-product per year
    n_samples = mc_preds_np.shape[0]
    mwe_samples = np.zeros((n_samples, n_years), dtype=np.float64)
    for yi in range(n_years):
        mask = year_idx == yi
        if total_area[yi] > 0:
            w = areas[mask] / total_area[yi]
            mwe_samples[:, yi] = mc_preds_np[:, mask].astype(np.float64) @ w

    # Gt: MWE × area_km² × 1e6 m²/km² × 1000 kg/m³ / 1e12 kg/Gt = MWE × area × 1e-3
    gt_samples = mwe_samples * total_area[np.newaxis, :] * 1e-3

    return {
        "years":          years,
        "mwe_samples":    mwe_samples,
        "gt_samples":     gt_samples,
        "total_area_km2": total_area,
    }


def _summarise(samples: np.ndarray) -> dict:
    """Return p2_5, p50, p97_5, mean, std over axis=0."""
    return {
        "p2_5":  np.percentile(samples, 2.5,  axis=0),
        "p50":   np.percentile(samples, 50.0, axis=0),
        "p97_5": np.percentile(samples, 97.5, axis=0),
        "mean":  samples.mean(axis=0),
        "std":   samples.std(axis=0),
    }


def save_regional_csvs(regional: dict, output_dir: str) -> None:
    """
    Save regional_annual_mwe.csv and regional_annual_gt.csv.

    For Gt, also saves p2_5_total / p97_5_total which fold in the density
    uncertainty (±60 kg/m³) in quadrature with the MC interval half-widths.
    """
    years = regional["years"]

    # --- MWE ---
    mwe = _summarise(regional["mwe_samples"])
    pd.DataFrame({"year": years, **{k: mwe[k] for k in ("p2_5", "p50", "p97_5", "mean", "std")}}).to_csv(
        os.path.join(output_dir, "regional_annual_mwe.csv"), index=False
    )

    # --- Gt with density uncertainty ---
    gt = _summarise(regional["gt_samples"])
    density_sig = np.abs(gt["p50"]) * (_ICE_DENSITY_SIG / _ICE_DENSITY)
    half_lo = gt["p50"] - gt["p2_5"]
    half_hi = gt["p97_5"] - gt["p50"]
    p2_5_total  = gt["p50"] - np.sqrt(half_lo ** 2 + density_sig ** 2)
    p97_5_total = gt["p50"] + np.sqrt(half_hi ** 2 + density_sig ** 2)

    pd.DataFrame({
        "year":        years,
        "p2_5":        gt["p2_5"],
        "p50":         gt["p50"],
        "p97_5":       gt["p97_5"],
        "mean":        gt["mean"],
        "std":         gt["std"],
        "p2_5_total":  p2_5_total,
        "p97_5_total": p97_5_total,
    }).to_csv(os.path.join(output_dir, "regional_annual_gt.csv"), index=False)

    print(f"Saved regional_annual_mwe.csv and regional_annual_gt.csv → {output_dir}")


def plot_regional_mass_loss(
    regional: dict,
    glambie_wide_df,   # pd.DataFrame from glambie_targets CSV, or None
    output_dir: str,
) -> None:
    """
    Plot annual regional Gt/yr time series with uncertainty shading.

    MC uncertainty shown as darker shade; total (MC + density) as lighter shade.
    GLaMBIE combined_mwe overlaid as a line with error bars where available.
    """
    years = np.array(regional["years"])
    gt    = _summarise(regional["gt_samples"])
    total_area_mean = regional["total_area_km2"].mean()  # scalar for GLaMBIE conversion

    density_sig = np.abs(gt["p50"]) * (_ICE_DENSITY_SIG / _ICE_DENSITY)
    half_lo = gt["p50"] - gt["p2_5"]
    half_hi = gt["p97_5"] - gt["p50"]
    p2_5_total  = gt["p50"] - np.sqrt(half_lo ** 2 + density_sig ** 2)
    p97_5_total = gt["p50"] + np.sqrt(half_hi ** 2 + density_sig ** 2)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Total uncertainty (lighter)
    ax.fill_between(years, p2_5_total, p97_5_total, alpha=0.20, color="steelblue",
                    label="95% CI (MC + density unc.)")
    # MC uncertainty (darker)
    ax.fill_between(years, gt["p2_5"], gt["p97_5"], alpha=0.40, color="steelblue",
                    label="95% CI (MC)")
    # Median
    ax.plot(years, gt["p50"], color="steelblue", lw=1.5, label="Model median")

    # GLaMBIE combined overlay
    if glambie_wide_df is not None and "combined_mwe" in glambie_wide_df.columns:
        gb = glambie_wide_df.dropna(subset=["combined_mwe", "combined_mwe_errors"]).copy()
        gb["year"] = np.floor(gb["end_date"]).astype(int)
        gb_gt    = gb["combined_mwe"].values * total_area_mean * 1e-3
        gb_gt_err = gb["combined_mwe_errors"].values * total_area_mean * 1e-3
        ax.errorbar(
            gb["year"].values, gb_gt,
            yerr=gb_gt_err,
            fmt="o", color="darkorange", ms=4, lw=1.2, capsize=3,
            label="GLaMBIE combined",
        )

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Regional mass balance (Gt/yr)")
    ax.set_title("Annual regional mass balance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "regional_mass_loss.png"), dpi=150)
    plt.close(fig)
    print(f"Saved regional_mass_loss.png → {output_dir}")


def plot_hugonnet_scatters(
    preds_df: pd.DataFrame,   # per-glacier per-year: rgi_id, year, mean, std
    pred_grid: pd.DataFrame,  # N rows: rgi_id, year, Area
    hugo_df: pd.DataFrame,    # from load_dmdtda: rgi_id, start_date, end_date, avg_mb_mwe, uncertainty_mwe
    output_dir: str,
) -> None:
    """
    One scatter plot per Hugonnet period: model predicted period mean vs Hugonnet dmdtda.

    For each (start_date, end_date) period in hugo_df, filters preds_df to
    years in [start_date, end_date) and computes per-glacier mean prediction.
    Error bars show ±1 std of per-glacier MC predictions over the period.
    Separate plot files are named hugonnet_scatter_{start}_{end}.png.
    """
    for (pstart, pend), period_hugo in hugo_df.groupby(["start_date", "end_date"]):
        period_preds = preds_df[(preds_df["year"] >= pstart) & (preds_df["year"] < pend)]
        if period_preds.empty:
            continue

        pred_means = (
            period_preds.groupby("rgi_id")[["mean", "std"]]
            .agg({"mean": "mean", "std": "mean"})
            .reset_index()
        )
        merged = pred_means.merge(
            period_hugo[["rgi_id", "avg_mb_mwe", "uncertainty_mwe"]],
            on="rgi_id", how="inner",
        )
        if merged.empty:
            continue

        x = merged["avg_mb_mwe"].values
        y = merged["mean"].values
        y_err = merged["std"].values

        vmin = min(x.min(), y.min()) - 0.1
        vmax = max(x.max(), y.max()) + 0.1

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(x, y, yerr=y_err, fmt="o", ms=3, alpha=0.5, lw=0.8,
                    color="steelblue", ecolor="steelblue", capsize=0, label="Glaciers")
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=0.8, label="1:1")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel("Hugonnet dmdtda (MWE/yr)")
        ax.set_ylabel("Model predicted mean (MWE/yr)")
        ax.set_title(f"Hugonnet scatter {pstart}–{pend}")

        # Diagonal R² annotation
        ss_res = ((y - x) ** 2).sum()
        ss_tot = ((x - x.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(((y - x) ** 2).mean()))
        ax.text(0.05, 0.95, f"R²={r2:.3f}  RMSE={rmse:.3f}",
                transform=ax.transAxes, fontsize=9, va="top")
        ax.legend(fontsize=8)
        fig.tight_layout()

        fname = f"hugonnet_scatter_{pstart}_{pend}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"Saved {fname} → {output_dir}")


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def run_predict(cfg: DictConfig) -> None:
    """
    Full inference run for one region.

    Loads finetuned params, runs MC predictions over the full feature grid,
    writes per-glacier and regional outputs to cfg.model.output_dir.
    """
    os.makedirs(cfg.model.output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(cfg.model.seed)

    # --- Resolve relative paths against original cwd (Hydra chdir moves us to output dir) ---
    orig = get_original_cwd()
    cfg_mutable = OmegaConf.to_container(cfg, resolve=True)
    for key in ("finetuned_params_path", "inp_dir", "temporal_avg_path", "glambie_path"):
        if key in cfg_mutable["model"] and cfg_mutable["model"][key]:
            cfg_mutable["model"][key] = os.path.join(orig, cfg_mutable["model"][key])
    cfg = OmegaConf.create(cfg_mutable)

    # --- Feature columns ---
    ft_cols = list(cfg.model.model_ftcols) if cfg.model.model_ftcols else FEATURE_COLS
    rm_fts  = list(cfg.model.rm_fts) if cfg.model.get("rm_fts") else []
    ft_cols = [c for c in ft_cols if c not in rm_fts]

    # --- Load params ---
    mu_dict, log_sigma_dict = load_finetuned_params(cfg.model.finetuned_params_path)
    params = build_params_from_posterior(mu_dict, log_sigma_dict)

    # --- Load scalers (written by finetune alongside finetuned_params.pkl) ---
    finetune_dir = os.path.dirname(cfg.model.finetuned_params_path)

    scaler_path = os.path.join(finetune_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = cloudpickle.load(f)
        print(f"Loaded feature scaler from {scaler_path}")
    else:
        scaler = None
        print(f"WARNING: scaler.pkl not found at {scaler_path} — covariates will be unscaled")

    target_scaler_path = os.path.join(finetune_dir, "target_scaler.pkl")
    if os.path.exists(target_scaler_path):
        with open(target_scaler_path, "rb") as f:
            target_scaler = cloudpickle.load(f)  # (mean, std)
        print(f"Loaded target scaler from {target_scaler_path}")
    else:
        target_scaler = None
        print(f"WARNING: target_scaler.pkl not found at {target_scaler_path} — predictions will not be unscaled")

    # --- Load prediction grid ---
    pred_grid = load_pred_grid(cfg, ft_cols)
    time_index, covariates, _, _ = build_model_inputs(pred_grid, ft_cols)
    if scaler is not None:
        covariates = apply_scaler(covariates, scaler)

    # --- Model ---
    model = BayesianNeuralField(
        hidden_sizes=tuple([cfg.model.model_nhidden] * cfg.model.model_nlayers),
        n_fourier=cfg.model.n_fourier,
    )

    # --- MC predictions ---
    print(f"Running {cfg.model.model_nensemble} MC samples over {len(pred_grid)} rows...")
    rng, rng_pred = jax.random.split(rng)
    mc_preds = mc_predict_region(
        model, params,
        jnp.array(time_index),
        jnp.array(covariates),
        rng_pred,
        n_samples=cfg.model.model_nensemble,
    )  # (n_samples, N)

    # --- Unscale predictions to physical MWE/yr, convert to numpy once ---
    if target_scaler is not None:
        t_mean, t_std = target_scaler
        mc_preds = mc_preds * t_std + t_mean
    mc_preds_np = np.array(mc_preds)  # (n_samples, N) — reused for per-glacier and regional

    # --- Per-glacier predictions CSV ---
    quantiles = extract_quantiles(mc_preds_np)
    preds_df  = predictions_to_df(pred_grid, quantiles)
    full_path = os.path.join(cfg.model.output_dir, "preds_full.csv")
    preds_df.to_csv(full_path, index=False)
    print(f"Saved preds_full.csv → {full_path}")

    # --- Regional time series CSVs (MWE + Gt) ---
    regional = compute_regional_series(mc_preds_np, pred_grid)
    save_regional_csvs(regional, cfg.model.output_dir)

    # --- GLaMBIE wide-format (for plot overlay) ---
    glambie_wide_df = None
    glambie_path_cfg = cfg.model.get("glambie_path", None)
    if glambie_path_cfg and os.path.exists(glambie_path_cfg):
        glambie_wide_df = pd.read_csv(glambie_path_cfg)

    # --- Regional mass loss plot ---
    plot_regional_mass_loss(regional, glambie_wide_df, cfg.model.output_dir)

    # --- Hugonnet scatter plots ---
    hugo_path = cfg.model.get("temporal_avg_path", None)
    if hugo_path and os.path.exists(hugo_path):
        hugo_df = load_dmdtda(hugo_path)
        plot_hugonnet_scatters(preds_df, pred_grid, hugo_df, cfg.model.output_dir)
