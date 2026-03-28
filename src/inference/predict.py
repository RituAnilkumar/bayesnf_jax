"""
Inference utilities — MC forward passes and output formatting.

Loads finetuned_params.pkl and generates predictions over the full
(glacier × year) grid specified in for_preds.csv.

Output CSV columns: glacier_id, year, mass_balance_mwe, uncertainty_mwe
  - mass_balance_mwe: median (p50) of MC predictive distribution [MWE/yr]
  - uncertainty_mwe:  std dev of MC predictive distribution [MWE/yr]

The p2.5 and p97.5 percentiles are also computed internally and can be
written to an extended output if needed.
"""

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from omegaconf import DictConfig

from src.model.bnf_module import BayesianNeuralField, T_MIN
from src.loss.aggregation import MM_TO_MWE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_finetuned_params(finetuned_params_path: str) -> tuple[dict, dict]:
    """
    Load (mu_dict, log_sigma_dict) from finetuned_params.pkl.

    Returns:
        (mu_dict, log_sigma_dict) with same pytree structure as model params
    """
    # TODO: cloudpickle.load, assert tuple of length 2
    raise NotImplementedError


def load_pred_grid(cfg) -> pd.DataFrame:
    """
    Load the full prediction grid (for_preds.csv) for the configured region.

    Returns DataFrame with columns: rgi_id, year, + all feature columns.
    time_index = year - T_MIN is added here.
    """
    # TODO: implement
    raise NotImplementedError


def build_params_from_posterior(mu_dict: dict, log_sigma_dict: dict) -> dict:
    """
    Reconstruct a Flax params dict from (mu_dict, log_sigma_dict).

    Inverse of extract_vi_params() — merges the split dicts back into
    the structure expected by model.apply().
    """
    # TODO: implement by merging mu and log_sigma leaves
    raise NotImplementedError


# ---------------------------------------------------------------------------
# MC prediction
# ---------------------------------------------------------------------------

def mc_predict_region(
    model: BayesianNeuralField,
    params: dict,
    time_index: jax.Array,   # shape (N,)
    covariates: jax.Array,   # shape (N, n_features)
    rng: jax.Array,
    n_samples: int,
) -> jax.Array:
    """
    Draw n_samples MC predictions over the full prediction grid.

    Uses model.mc_predict via vmap for efficiency.

    Returns:
        Array of shape (n_samples, N)  [MWE/yr]
    """
    # TODO: model.apply(params, time_index, covariates, rng, n_samples=n_samples,
    #                   method=model.mc_predict)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Quantile extraction and output formatting
# ---------------------------------------------------------------------------

def extract_quantiles(mc_preds: jax.Array) -> dict:
    """
    Compute summary statistics over MC samples.

    Args:
        mc_preds: shape (n_samples, N)

    Returns:
        Dict with keys: p2_5, p50, p97_5, mean, std — each shape (N,)
    """
    # TODO: jnp.percentile over axis=0
    raise NotImplementedError


def predictions_to_df(
    pred_grid: pd.DataFrame,
    quantiles: dict,
) -> pd.DataFrame:
    """
    Assemble the final predictions DataFrame.

    Output columns:
        glacier_id        — rgi_id from pred_grid
        year              — year from pred_grid
        mass_balance_mwe  — p50 (median) of MC distribution [MWE/yr]
        uncertainty_mwe   — std dev of MC distribution [MWE/yr]

    Args:
        pred_grid:  DataFrame with rgi_id and year columns (N rows)
        quantiles:  Output of extract_quantiles()

    Returns:
        DataFrame with 4 columns, N rows.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def run_predict(cfg: DictConfig) -> None:
    """
    Full inference run for one region.

    Loads finetuned params, runs MC predictions over for_preds.csv grid,
    writes predictions.csv to cfg.model.output_dir.
    """
    import os
    os.makedirs(cfg.model.output_dir, exist_ok=True)

    # TODO: load_finetuned_params
    # TODO: load_pred_grid
    # TODO: build_params_from_posterior
    # TODO: instantiate BayesianNeuralField from cfg (same arch as training)
    # TODO: mc_predict_region (n_samples = cfg.model.model_nensemble)
    # TODO: extract_quantiles
    # TODO: predictions_to_df → write predictions.csv

    raise NotImplementedError
