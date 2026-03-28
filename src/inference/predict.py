"""
Inference utilities — MC forward passes and output formatting.

Loads finetuned_params.pkl and generates predictions over the full
(glacier × year) grid specified in for_preds.csv.

Output CSV columns:
  preds_full.csv     — one row per (glacier, year): rgi_id, year, p2_5, p50, p97_5, mean, std
  preds_quantiles.csv — same as preds_full but wide-format quantile summary
"""

import os
import jax
import jax.numpy as jnp
import cloudpickle
import pandas as pd
import numpy as np
from omegaconf import DictConfig

from src.model.bnf_module import BayesianNeuralField, T_MIN
from src.data_utils import (
    load_features,
    build_model_inputs,
    FEATURE_COLS,
)


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
# Main prediction function
# ---------------------------------------------------------------------------

def run_predict(cfg: DictConfig) -> None:
    """
    Full inference run for one region.

    Loads finetuned params, runs MC predictions over the full feature grid,
    writes preds_full.csv and preds_quantiles.csv to cfg.model.output_dir.
    """
    os.makedirs(cfg.model.output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(cfg.model.seed)

    # --- Feature columns ---
    ft_cols = list(cfg.model.model_ftcols) if cfg.model.model_ftcols else FEATURE_COLS
    rm_fts  = list(cfg.model.rm_fts) if cfg.model.get("rm_fts") else []
    ft_cols = [c for c in ft_cols if c not in rm_fts]

    # --- Load params ---
    mu_dict, log_sigma_dict = load_finetuned_params(cfg.model.finetuned_params_path)
    params = build_params_from_posterior(mu_dict, log_sigma_dict)

    # --- Load prediction grid ---
    pred_grid = load_pred_grid(cfg, ft_cols)
    time_index, covariates, _, _ = build_model_inputs(pred_grid, ft_cols)

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

    # --- Quantiles and output ---
    quantiles = extract_quantiles(mc_preds)
    preds_df  = predictions_to_df(pred_grid, quantiles)

    full_path = os.path.join(cfg.model.output_dir, "preds_full.csv")
    preds_df.to_csv(full_path, index=False)
    print(f"Saved predictions → {full_path}")

    # preds_quantiles: p2_5 / p50 / p97_5 only (compact summary)
    quantiles_df = preds_df[["rgi_id", "year", "p2_5", "p50", "p97_5"]].copy()
    q_path = os.path.join(cfg.model.output_dir, "preds_quantiles.csv")
    quantiles_df.to_csv(q_path, index=False)
    print(f"Saved quantile summary → {q_path}")
