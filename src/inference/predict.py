"""
Inference utilities — MC forward passes and output formatting.

Loads finetuned_params.pkl (and optionally pretrained_params.pkl) and generates
predictions over the full (glacier × year) feature grid.

Output files:
  preds_full.csv                  — per (glacier, year): rgi_id, year, p2_5, p50, p97_5, mean, std  [finetune]
  preds_full_pretrain.csv         — same, from pretrained model
  regional_annual_mwe.csv         — area-weighted regional mean MWE/yr per year  [finetune]
  regional_annual_mwe_pretrain.csv
  regional_annual_gt.csv          — regional Gt/yr per year (MC uncertainty)  [finetune]
  regional_annual_gt_pretrain.csv
  regional_mass_loss.png          — finetune Gt/yr + GLaMBIE combined
  regional_mwe_comparison.png     — pretrain vs finetune MWE on same axes
  glambie_scatter_finetune.png    — GLaMBIE combined annual regional mean vs model
  glambie_scatter_pretrain.png
  cumulative_gt.png               — cumulative Gt: pretrain, finetune, OGGM vs GLaMBIE
  hugonnet_scatter_{finetune,pretrain}.png    — per-glacier Hugonnet scatter (longest period)
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

from src.model.bnf_module import BayesianNeuralField, T_MIN, mc_predict_chunked
from src.data_utils import (
    load_features,
    load_oggm,
    load_dmdtda,
    merge_features_targets,
    build_model_inputs,
    apply_scaler,
    extract_hugonnet_region,
    FEATURE_COLS,
)


# ---------------------------------------------------------------------------
# Param loading and reconstruction
# ---------------------------------------------------------------------------

def _load_params(path: str) -> dict:
    """Load (mu_dict, log_sigma_dict) from a .pkl file and reconstruct Flax params."""
    with open(path, "rb") as f:
        result = cloudpickle.load(f)
    assert isinstance(result, tuple) and len(result) == 2, \
        f"Expected (mu_dict, log_sigma_dict), got {type(result)}"
    mu_dict, log_sigma_dict = result

    def _merge(d_mu, d_ls):
        out = {**d_mu}
        for k, v in d_ls.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out

    return {"params": _merge(mu_dict, log_sigma_dict)}


def _load_scalers(params_dir: str) -> tuple:
    """Load (scaler, target_scaler) from a params directory. Either may be None."""
    scaler = None
    scaler_path = os.path.join(params_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = cloudpickle.load(f)
        print(f"Loaded feature scaler from {scaler_path}")
    else:
        print(f"WARNING: scaler.pkl not found at {scaler_path}")

    target_scaler = None
    ts_path = os.path.join(params_dir, "target_scaler.pkl")
    if os.path.exists(ts_path):
        with open(ts_path, "rb") as f:
            target_scaler = cloudpickle.load(f)
        t_mean, t_std = target_scaler
        print(f"Loaded target scaler from {ts_path}  (mean={t_mean:.4f}, std={t_std:.4f})")
    else:
        print(f"WARNING: target_scaler.pkl not found at {ts_path}")

    return scaler, target_scaler


# ---------------------------------------------------------------------------
# Prediction grid loading
# ---------------------------------------------------------------------------

def load_pred_grid(cfg: DictConfig, ft_cols: list[str]) -> pd.DataFrame:
    """Load the full prediction grid (main_features_{region}.csv)."""
    region = cfg.model.reg_subdir
    base   = os.path.join(cfg.model.inp_dir, region)
    return load_features(os.path.join(base, f"main_features_{region}.csv"), feature_cols=ft_cols)


# ---------------------------------------------------------------------------
# MC prediction
# ---------------------------------------------------------------------------

def _run_mc(
    model: BayesianNeuralField,
    params: dict,
    time_index: np.ndarray,
    covariates: np.ndarray,
    rng: jax.Array,
    n_samples: int,
    target_scaler,
):
    """
    Run MC inference and return unscaled predictions.

    Returns:
        heteroscedastic=False: numpy array (n_samples, N)
        heteroscedastic=True:  tuple (mu_np, sigma_np), each (n_samples, N)
                               mu unscaled with mean+std, sigma unscaled with std only.
    """
    result = mc_predict_chunked(
        model, params, jnp.array(time_index), jnp.array(covariates), rng, n_samples
    )
    if isinstance(result, tuple):
        mu_np, sigma_np = result
        if target_scaler is not None:
            t_mean, t_std = target_scaler
            mu_np    = mu_np    * t_std + t_mean
            sigma_np = sigma_np * t_std           # scale only — sigma is not shifted
        return mu_np, sigma_np
    else:
        mc_np = result
        if target_scaler is not None:
            t_mean, t_std = target_scaler
            mc_np = mc_np * t_std + t_mean
        return mc_np


# ---------------------------------------------------------------------------
# Quantile extraction and output formatting
# ---------------------------------------------------------------------------

def _summarise(samples: np.ndarray) -> dict:
    """Return p2_5, p50, p97_5, mean, std over axis=0."""
    return {
        "p2_5":  np.percentile(samples, 2.5,  axis=0),
        "p50":   np.percentile(samples, 50.0, axis=0),
        "p97_5": np.percentile(samples, 97.5, axis=0),
        "mean":  samples.mean(axis=0),
        "std":   samples.std(axis=0),
    }


def _predictions_to_df(
    pred_grid: pd.DataFrame,
    mc_preds_np: np.ndarray,
    mc_sigma_np: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build per-glacier prediction DataFrame.

    If mc_sigma_np is provided (heteroscedastic mode):
      - 'std' remains the epistemic std (spread of mu samples)
      - 'aleatoric_std' = mean predicted noise sigma across samples
      - 'total_std'     = sqrt(epistemic² + aleatoric²)
    Quantiles (p2_5, p50, p97_5) are from mu samples only.
    """
    q = _summarise(mc_preds_np)
    row = {
        "rgi_id": pred_grid["rgi_id"].values,
        "year":   pred_grid["year"].values,
        "p2_5":   q["p2_5"],
        "p50":    q["p50"],
        "p97_5":  q["p97_5"],
        "mean":   q["mean"],
        "std":    q["std"],
    }
    if mc_sigma_np is not None:
        aleatoric_std = mc_sigma_np.mean(axis=0)                             # (N,)
        row["aleatoric_std"] = aleatoric_std
        row["epistemic_std"] = q["std"]
        row["total_std"]     = np.sqrt(q["std"] ** 2 + aleatoric_std ** 2)
    return pd.DataFrame(row)


# ---------------------------------------------------------------------------
# Regional aggregate computation
# ---------------------------------------------------------------------------

def compute_regional_series(mc_preds_np: np.ndarray, pred_grid: pd.DataFrame) -> dict:
    """
    Compute annual area-weighted regional mean MWE/yr and Gt/yr across MC samples.

    Area column assumed to be in km².
    Gt/yr = MWE/yr × total_area_km² × 1e-3.

    Returns dict:
        years, mwe_samples (n_samples, n_years), gt_samples (n_samples, n_years),
        total_area_km2 (n_years,)
    """
    years = sorted(pred_grid["year"].unique().tolist())
    year_to_idx = {y: i for i, y in enumerate(years)}
    n_years  = len(years)
    areas    = pred_grid["Area"].values.astype(np.float64)
    year_idx = pred_grid["year"].map(year_to_idx).values
    total_area = np.array([areas[year_idx == i].sum() for i in range(n_years)])

    n_samples = mc_preds_np.shape[0]
    mwe_samples = np.zeros((n_samples, n_years), dtype=np.float64)
    for yi in range(n_years):
        mask = year_idx == yi
        if total_area[yi] > 0:
            w = areas[mask] / total_area[yi]
            mwe_samples[:, yi] = mc_preds_np[:, mask].astype(np.float64) @ w

    gt_samples = mwe_samples * total_area[np.newaxis, :] * 1e-3

    return {
        "years":          years,
        "mwe_samples":    mwe_samples,
        "gt_samples":     gt_samples,
        "total_area_km2": total_area,
    }




def _oggm_regional_gt(cfg: DictConfig, pred_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Compute OGGM area-weighted regional Gt/yr per year.

    Returns DataFrame with columns: year, gt (scalar, no MC uncertainty).
    """
    region = cfg.model.reg_subdir
    base   = os.path.join(cfg.model.inp_dir, region)
    targets_df  = load_oggm(os.path.join(base, f"oggm_targets_{region}.csv"))
    features_df = pred_grid[["rgi_id", "year", "Area"]].drop_duplicates(["rgi_id", "year"])
    merged = targets_df.merge(features_df, on=["rgi_id", "year"], how="inner")

    rows = []
    for yr, grp in merged.groupby("year"):
        total_area = grp["Area"].sum()
        if total_area > 0:
            w_mean_mwe = (grp["mass_balance_mwe"] * grp["Area"]).sum() / total_area
            rows.append({"year": yr, "gt": w_mean_mwe * total_area * 1e-3})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _glambie_combined_gt(glambie_wide_df: pd.DataFrame, total_area_km2: float) -> pd.DataFrame:
    """
    Extract GLaMBIE combined as Gt/yr per year.

    total_area_km2: representative scalar area for MWE→Gt conversion.
    Returns DataFrame: year, gt, gt_err.
    """
    if glambie_wide_df is None or "combined_mwe" not in glambie_wide_df.columns:
        return pd.DataFrame(columns=["year", "gt", "gt_err"])
    gb = glambie_wide_df.dropna(subset=["combined_mwe", "combined_mwe_errors"]).copy()
    gb["year"]   = np.floor(gb["end_date"]).astype(int)
    gb["gt"]     = gb["combined_mwe"].values     * total_area_km2 * 1e-3
    gb["gt_err"] = gb["combined_mwe_errors"].values * total_area_km2 * 1e-3
    return gb[["year", "gt", "gt_err"]].sort_values("year").reset_index(drop=True)


def _glambie_sources_mwe(glambie_wide_df) -> dict:
    """
    Extract altimetry and gravimetry from GLaMBIE wide format as MWE/yr per year.

    Returns dict with keys 'altimetry' and 'gravimetry', each a DataFrame with
    columns: year, mwe, mwe_err. Missing sources return empty DataFrames.
    """
    empty = pd.DataFrame(columns=["year", "mwe", "mwe_err"])
    if glambie_wide_df is None:
        return {"altimetry": empty, "gravimetry": empty}

    result = {}
    for source in ("altimetry", "gravimetry"):
        val_col = f"{source}_mwe"
        err_col = f"{source}_mwe_errors"
        if val_col not in glambie_wide_df.columns or err_col not in glambie_wide_df.columns:
            result[source] = empty
            continue
        df = glambie_wide_df.dropna(subset=[val_col, err_col]).copy()
        if df.empty:
            result[source] = empty
            continue
        df["year"]    = np.floor(df["end_date"]).astype(int)
        df["mwe"]     = df[val_col].values
        df["mwe_err"] = df[err_col].values
        result[source] = df[["year", "mwe", "mwe_err"]].sort_values("year").reset_index(drop=True)
    return result


def _oggm_regional_mwe(oggm_gt_df: pd.DataFrame, total_area_km2: float) -> pd.DataFrame:
    """
    Convert OGGM regional Gt/yr to MWE/yr using total_area_km2.

    Returns DataFrame: year, mwe.
    """
    if oggm_gt_df.empty or total_area_km2 <= 0:
        return pd.DataFrame(columns=["year", "mwe"])
    df = oggm_gt_df.copy()
    df["mwe"] = df["gt"] / (total_area_km2 * 1e-3)
    return df[["year", "mwe"]]


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _save_regional_csvs(regional: dict, output_dir: str, suffix: str = "") -> None:
    years = regional["years"]
    mwe   = _summarise(regional["mwe_samples"])
    gt    = _summarise(regional["gt_samples"])

    pd.DataFrame({
        "year": years,
        "total_area_km2": regional["total_area_km2"],
        **{k: mwe[k] for k in ("p2_5", "p50", "p97_5", "mean", "std")},
    }).to_csv(os.path.join(output_dir, f"regional_annual_mwe{suffix}.csv"), index=False)
    pd.DataFrame({
        "year": years, "p2_5": gt["p2_5"], "p50": gt["p50"], "p97_5": gt["p97_5"],
        "mean": gt["mean"], "std": gt["std"],
    }).to_csv(os.path.join(output_dir, f"regional_annual_gt{suffix}.csv"), index=False)
    print(f"Saved regional_annual_mwe{suffix}.csv and regional_annual_gt{suffix}.csv")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _shade_mc(ax, years, gt_summary, color, alpha_mc=0.40, label_prefix=""):
    """Fill 95% MC credible interval band and plot median on ax."""
    ax.fill_between(years, gt_summary["p2_5"], gt_summary["p97_5"], alpha=alpha_mc, color=color,
                    label=f"{label_prefix}95% CI")
    ax.plot(years, gt_summary["p50"], color=color, lw=1.5, label=f"{label_prefix}median")


# --- Plot 1: finetune Gt/yr + GLaMBIE sources + OGGM ---
def plot_regional_mass_loss(
    regional_fin: dict, glambie_wide_df, oggm_gt_df: pd.DataFrame, output_dir: str
) -> None:
    years = np.array(regional_fin["years"])
    gt    = _summarise(regional_fin["gt_samples"])
    total_area_mean = float(regional_fin["total_area_km2"].mean())

    gb_combined  = _glambie_combined_gt(glambie_wide_df, total_area_mean)
    gb_sources   = _glambie_sources_mwe(glambie_wide_df)

    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_mc(ax, years, gt, "steelblue")

    if not gb_combined.empty:
        ax.errorbar(gb_combined["year"].values, gb_combined["gt"].values,
                    yerr=gb_combined["gt_err"].values,
                    fmt="o", color="darkorange", ms=4, lw=1.2, capsize=3, label="GLaMBIE combined")

    for source, color, marker in [("altimetry", "forestgreen", "s"), ("gravimetry", "purple", "^")]:
        df = gb_sources[source]
        if not df.empty:
            gt_vals = df["mwe"].values * total_area_mean * 1e-3
            gt_errs = df["mwe_err"].values * total_area_mean * 1e-3
            ax.errorbar(df["year"].values, gt_vals, yerr=gt_errs,
                        fmt=marker, color=color, ms=4, lw=1.2, capsize=3,
                        label=f"GLaMBIE {source}")

    if not oggm_gt_df.empty:
        ax.plot(oggm_gt_df["year"].values, oggm_gt_df["gt"].values,
                color="seagreen", lw=1.2, ls="--", label="OGGM")

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year"); ax.set_ylabel("Gt/yr"); ax.set_title("Regional mass balance (finetuned)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "regional_mass_loss.png"), dpi=150)
    plt.close(fig)
    print("Saved regional_mass_loss.png")


# --- Plot 2: pretrain vs finetune MWE time series ---
def plot_regional_mwe_comparison(
    regional_pre: dict, regional_fin: dict, glambie_wide_df, oggm_gt_df: pd.DataFrame, output_dir: str
) -> None:
    years_pre = np.array(regional_pre["years"])
    years_fin = np.array(regional_fin["years"])
    mwe_pre   = _summarise(regional_pre["mwe_samples"])
    mwe_fin   = _summarise(regional_fin["mwe_samples"])

    total_area_mean = float(regional_fin["total_area_km2"].mean())
    gb_combined = _glambie_combined_gt(glambie_wide_df, total_area_mean)
    gb_sources  = _glambie_sources_mwe(glambie_wide_df)
    oggm_mwe    = _oggm_regional_mwe(oggm_gt_df, total_area_mean)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(years_pre, mwe_pre["p2_5"], mwe_pre["p97_5"], alpha=0.25, color="darkorange")
    ax.plot(years_pre, mwe_pre["p50"], color="darkorange", lw=1.3, label="Pretrain median")
    ax.fill_between(years_fin, mwe_fin["p2_5"], mwe_fin["p97_5"], alpha=0.25, color="steelblue")
    ax.plot(years_fin, mwe_fin["p50"], color="steelblue", lw=1.3, label="Finetune median")

    if not gb_combined.empty:
        gb_mwe     = gb_combined["gt"].values / (total_area_mean * 1e-3)
        gb_mwe_err = gb_combined["gt_err"].values / (total_area_mean * 1e-3)
        ax.errorbar(gb_combined["year"].values, gb_mwe, yerr=gb_mwe_err,
                    fmt="o", color="black", ms=3, lw=1.0, capsize=3, label="GLaMBIE combined")

    for source, color, marker in [("altimetry", "forestgreen", "s"), ("gravimetry", "purple", "^")]:
        df = gb_sources[source]
        if not df.empty:
            ax.errorbar(df["year"].values, df["mwe"].values, yerr=df["mwe_err"].values,
                        fmt=marker, color=color, ms=4, lw=1.0, capsize=3,
                        label=f"GLaMBIE {source}")

    if not oggm_mwe.empty:
        ax.plot(oggm_mwe["year"].values, oggm_mwe["mwe"].values,
                color="seagreen", lw=1.2, ls="--", label="OGGM")

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year"); ax.set_ylabel("MWE/yr")
    ax.set_title("Regional annual mean mass balance: pretrain vs finetune")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "regional_mwe_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved regional_mwe_comparison.png")


# --- Plot 3: GLaMBIE combined vs predicted regional mean scatter (one per stage) ---
def plot_glambie_scatter(
    regional: dict,
    glambie_wide_df,
    stage: str,   # 'pretrain' or 'finetune'
    output_dir: str,
) -> None:
    if glambie_wide_df is None or "combined_mwe" not in glambie_wide_df.columns:
        return
    gb = glambie_wide_df.dropna(subset=["combined_mwe", "combined_mwe_errors"]).copy()
    if gb.empty:
        return
    gb["year"] = np.floor(gb["end_date"]).astype(int)

    years       = regional["years"]
    year_to_idx = {y: i for i, y in enumerate(years)}
    mwe         = _summarise(regional["mwe_samples"])

    rows = []
    for _, row in gb.iterrows():
        yr = int(row["year"])
        if yr in year_to_idx:
            yi = year_to_idx[yr]
            rows.append({
                "year":         yr,
                "glambie_mwe":  row["combined_mwe"],
                "glambie_err":  row["combined_mwe_errors"],
                "pred_mwe":     mwe["p50"][yi],
                "pred_err":     (mwe["p97_5"][yi] - mwe["p2_5"][yi]) / 2.0,
            })
    if not rows:
        return
    df = pd.DataFrame(rows)

    x, y = df["glambie_mwe"].values, df["pred_mwe"].values
    x_err, y_err = df["glambie_err"].values, df["pred_err"].values
    vmin = min(x.min(), y.min()) - 0.1
    vmax = max(x.max(), y.max()) + 0.1
    ss_res = ((y - x) ** 2).sum()
    ss_tot = ((x - x.mean()) ** 2).sum()
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(((y - x) ** 2).mean()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", ms=5, alpha=0.7,
                color="steelblue", ecolor="steelblue", capsize=3)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=0.8, label="1:1")
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_xlabel("GLaMBIE combined (MWE/yr)")
    ax.set_ylabel(f"Model {stage} regional mean (MWE/yr)")
    ax.set_title(f"GLaMBIE vs {stage} — annual regional mean")
    ax.text(0.05, 0.95, f"R²={r2:.3f}  RMSE={rmse:.4f}", transform=ax.transAxes, fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"glambie_scatter_{stage}.png"), dpi=150)
    plt.close(fig)
    print(f"Saved glambie_scatter_{stage}.png")


# --- Plot 4: cumulative Gt — pretrain, finetune, OGGM vs GLaMBIE ---
def plot_cumulative_gt(
    regional_pre: dict,
    regional_fin: dict,
    oggm_gt_df: pd.DataFrame,    # year, gt
    glambie_wide_df,
    output_dir: str,
) -> None:
    total_area_mean = float(regional_fin["total_area_km2"].mean())
    gb = _glambie_combined_gt(glambie_wide_df, total_area_mean)

    # Determine common start year: first year where GLaMBIE combined has data
    # (so all series cumulate from the same baseline)
    if not gb.empty:
        start_year = int(gb["year"].min())
    else:
        start_year = min(regional_pre["years"][0], regional_fin["years"][0])

    def _cumulate_samples(regional: dict) -> tuple[np.ndarray, np.ndarray]:
        """Cumulate gt_samples from start_year; return (years_arr, cumgt_samples)."""
        years = np.array(regional["years"])
        mask  = years >= start_year
        yrs   = years[mask]
        samples = regional["gt_samples"][:, mask]   # (n_samples, n_valid)
        return yrs, np.cumsum(samples, axis=1)

    yrs_pre, cum_pre = _cumulate_samples(regional_pre)
    yrs_fin, cum_fin = _cumulate_samples(regional_fin)

    pre_s = _summarise(cum_pre)
    fin_s = _summarise(cum_fin)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Pretrain
    ax.fill_between(yrs_pre, pre_s["p2_5"], pre_s["p97_5"], alpha=0.20, color="darkorange")
    ax.plot(yrs_pre, pre_s["p50"], color="darkorange", lw=1.3, label="Pretrain (median, 95% CI)")

    # Finetune
    ax.fill_between(yrs_fin, fin_s["p2_5"], fin_s["p97_5"], alpha=0.20, color="steelblue")
    ax.plot(yrs_fin, fin_s["p50"], color="steelblue", lw=1.3, label="Finetune (median, 95% CI)")

    # OGGM — single deterministic line (already in MWE/yr, no density assumption needed)
    if not oggm_gt_df.empty:
        og = oggm_gt_df[oggm_gt_df["year"] >= start_year].copy()
        if not og.empty:
            og_cum = og["gt"].cumsum().values
            ax.plot(og["year"].values, og_cum, color="seagreen", lw=1.3, ls="--", label="OGGM")

    # GLaMBIE combined cumulative
    if not gb.empty:
        gb_from = gb[gb["year"] >= start_year].copy()
        if not gb_from.empty:
            gb_cum     = gb_from["gt"].cumsum().values
            gb_cum_err = np.sqrt(np.cumsum(gb_from["gt_err"].values ** 2))
            ax.fill_between(gb_from["year"].values,
                            gb_cum - 1.96 * gb_cum_err, gb_cum + 1.96 * gb_cum_err,
                            alpha=0.25, color="black")
            ax.plot(gb_from["year"].values, gb_cum, "k-", lw=1.5, label="GLaMBIE combined")

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year"); ax.set_ylabel("Cumulative mass balance (Gt)")
    ax.set_title(f"Cumulative regional mass balance from {start_year}")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cumulative_gt.png"), dpi=150)
    plt.close(fig)
    print("Saved cumulative_gt.png")


# --- Plot 5: Hugonnet scatter per period (one per stage) ---
def plot_hugonnet_scatters(
    preds_df: pd.DataFrame,
    hugo_df: pd.DataFrame,
    stage: str,
    output_dir: str,
) -> None:
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
            period_hugo[["rgi_id", "avg_mb_mwe", "uncertainty_mwe"]], on="rgi_id", how="inner"
        )
        if merged.empty:
            continue

        x, y, y_err = merged["avg_mb_mwe"].values, merged["mean"].values, merged["std"].values
        vmin = min(x.min(), y.min()) - 0.1
        vmax = max(x.max(), y.max()) + 0.1
        ss_res = ((y - x) ** 2).sum()
        ss_tot = ((x - x.mean()) ** 2).sum()
        r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(((y - x) ** 2).mean()))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(x, y, yerr=y_err, fmt="o", ms=3, alpha=0.5, lw=0.8,
                    color="steelblue", ecolor="steelblue", capsize=0)
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=0.8, label="1:1")
        ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
        ax.set_xlabel("Hugonnet dmdtda (MWE/yr)")
        ax.set_ylabel(f"Model {stage} mean (MWE/yr)")
        ax.set_title(f"Hugonnet scatter {pstart}–{pend}  [{stage}]")
        ax.text(0.05, 0.95, f"R²={r2:.3f}  RMSE={rmse:.3f}", transform=ax.transAxes, fontsize=9, va="top")
        fig.tight_layout()
        fname = f"hugonnet_scatter_{pstart}_{pend}_{stage}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def run_predict(cfg: DictConfig) -> None:
    """
    Full inference run for one region.

    Runs MC inference with both finetuned and pretrained params (if available),
    writes per-glacier CSVs, regional aggregate CSVs, and all diagnostic plots.
    """
    os.makedirs(cfg.model.output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(cfg.model.seed)

    # --- Resolve relative paths against original cwd (Hydra chdir moves us to output dir) ---
    orig = get_original_cwd()
    cfg_mutable = OmegaConf.to_container(cfg, resolve=True)
    for key in ("finetuned_params_path", "pretrained_params_path", "inp_dir",
                "temporal_avg_path", "glambie_path"):
        if key in cfg_mutable["model"] and cfg_mutable["model"][key]:
            cfg_mutable["model"][key] = os.path.join(orig, cfg_mutable["model"][key])
    cfg = OmegaConf.create(cfg_mutable)

    # --- Feature columns ---
    ft_cols = list(cfg.model.model_ftcols) if cfg.model.model_ftcols else FEATURE_COLS
    rm_fts  = list(cfg.model.rm_fts) if cfg.model.get("rm_fts") else []
    ft_cols = [c for c in ft_cols if c not in rm_fts]

    # --- Prediction grid + scaled covariates (scalers from finetune dir, same for both stages) ---
    pred_grid = load_pred_grid(cfg, ft_cols)
    time_index, covariates, _, _ = build_model_inputs(pred_grid, ft_cols)

    finetune_dir = os.path.dirname(cfg.model.finetuned_params_path)
    scaler, target_scaler = _load_scalers(finetune_dir)
    if scaler is not None:
        covariates = apply_scaler(covariates, scaler)

    # --- Model architecture ---
    heteroscedastic = bool(cfg.model.get("heteroscedastic", False))
    model = BayesianNeuralField(
        hidden_sizes=tuple([cfg.model.model_nhidden] * cfg.model.model_nlayers),
        n_fourier=cfg.model.n_fourier,
        heteroscedastic=heteroscedastic,
        sigma_floor=float(cfg.model.get("sigma_floor", 0.05)),
    )
    n_samples = cfg.model.model_nensemble
    print(f"Running {n_samples} MC samples over {len(pred_grid)} rows "
          f"({'heteroscedastic' if heteroscedastic else 'homoscedastic'})...")

    def _unpack_mc(result):
        """Return (mc_mu, mc_sigma_or_None) from _run_mc output."""
        if isinstance(result, tuple):
            return result  # (mc_mu, mc_sigma)
        return result, None

    # --- Finetuned inference ---
    params_fin = _load_params(cfg.model.finetuned_params_path)
    rng, rng_fin = jax.random.split(rng)
    mc_fin_raw = _run_mc(model, params_fin, time_index, covariates, rng_fin, n_samples, target_scaler)
    mc_fin, mc_fin_sigma = _unpack_mc(mc_fin_raw)

    preds_fin = _predictions_to_df(pred_grid, mc_fin, mc_fin_sigma)
    preds_fin.to_csv(os.path.join(cfg.model.output_dir, "preds_full.csv"), index=False)
    print("Saved preds_full.csv")

    regional_fin = compute_regional_series(mc_fin, pred_grid)
    _save_regional_csvs(regional_fin, cfg.model.output_dir, suffix="")

    # --- Pretrained inference (if params exist) ---
    pretrain_path = cfg.model.get("pretrained_params_path", None)
    has_pretrain  = pretrain_path and os.path.exists(pretrain_path)
    regional_pre  = None
    preds_pre     = None

    if has_pretrain:
        params_pre = _load_params(pretrain_path)
        rng, rng_pre = jax.random.split(rng)
        mc_pre_raw = _run_mc(model, params_pre, time_index, covariates, rng_pre, n_samples, target_scaler)
        mc_pre, mc_pre_sigma = _unpack_mc(mc_pre_raw)

        preds_pre = _predictions_to_df(pred_grid, mc_pre, mc_pre_sigma)
        preds_pre.to_csv(os.path.join(cfg.model.output_dir, "preds_full_pretrain.csv"), index=False)
        print("Saved preds_full_pretrain.csv")

        regional_pre = compute_regional_series(mc_pre, pred_grid)
        _save_regional_csvs(regional_pre, cfg.model.output_dir, suffix="_pretrain")
    else:
        print("WARNING: pretrained_params_path not found — pretrain plots skipped")

    # --- Load auxiliary data ---
    glambie_wide_df = None
    glambie_path_cfg = cfg.model.get("glambie_path", None)
    if glambie_path_cfg and os.path.exists(glambie_path_cfg):
        glambie_wide_df = pd.read_csv(glambie_path_cfg)

    hugo_df = None
    hugo_path = cfg.model.get("temporal_avg_path", None)
    if hugo_path:
        if not os.path.exists(hugo_path):
            reg_int = int(cfg.model.reg_subdir.lstrip("r"))
            all_csv = os.path.join(orig, cfg.model.inp_dir, "dmdtda_all.csv")
            print(f"  temporal_avg_path not found — generating from {all_csv} for reg={reg_int}")
            extract_hugonnet_region(all_csv, reg_int, hugo_path, period=None)
        if os.path.exists(hugo_path):
            hugo_df = load_dmdtda(hugo_path)

    oggm_gt_df = _oggm_regional_gt(cfg, pred_grid)

    # --- Plots ---

    # 1. Finetune Gt/yr vs GLaMBIE sources + OGGM
    plot_regional_mass_loss(regional_fin, glambie_wide_df, oggm_gt_df, cfg.model.output_dir)

    # 2. Pretrain vs finetune MWE time series + GLaMBIE sources + OGGM
    if regional_pre is not None:
        plot_regional_mwe_comparison(regional_pre, regional_fin, glambie_wide_df, oggm_gt_df, cfg.model.output_dir)

    # 3. GLaMBIE scatter — finetune and pretrain
    plot_glambie_scatter(regional_fin, glambie_wide_df, "finetune", cfg.model.output_dir)
    if regional_pre is not None:
        plot_glambie_scatter(regional_pre, glambie_wide_df, "pretrain", cfg.model.output_dir)

    # 4. Cumulative Gt
    plot_cumulative_gt(
        regional_pre if regional_pre is not None else regional_fin,
        regional_fin, oggm_gt_df, glambie_wide_df, cfg.model.output_dir,
    )

    # 5. Hugonnet scatters — finetune and pretrain
    if hugo_df is not None:
        plot_hugonnet_scatters(preds_fin, hugo_df, "finetune", cfg.model.output_dir)
        if preds_pre is not None:
            plot_hugonnet_scatters(preds_pre, hugo_df, "pretrain", cfg.model.output_dir)
