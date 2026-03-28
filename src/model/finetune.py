"""
Stage 2 finetuning loop — temporal-avg + GLaMBIE observational supervision.

Loads pretrained_params.pkl as the Bayesian prior (Bayesian continual learning).
All available OGGM data (full.csv) is used — no train/loyo/logo/loygo splits here.

Observation data:
  - Temporal-average per-glacier period mean (typically 2001-2020)
  - GLaMBIE regional annual means (gravimetry + altimetry, where available)

GLaMBIE test evaluation:
  - A held-out set of GLaMBIE years (specified in cfg.model.glambie_test_years)
    is withheld from training and used to assess finetuned model fit.
  - This replaces the loyo/logo cross-validation used in Stage 1.

Outputs written to cfg.model.output_dir:
  - finetuned_params.pkl          (tuple of mu_dict, log_sigma_dict)
  - training_loss.png
  - metrics_glambie_test.csv      (held-out GLaMBIE years)
"""

import os
import jax
import jax.numpy as jnp
import optax
import cloudpickle
import pandas as pd
import numpy as np
from omegaconf import DictConfig

from src.model.bnf_module import (
    BayesianNeuralField,
    compute_total_kl,
    extract_vi_params,
    T_MIN,
)
from src.loss.elbo import finetune_elbo, make_beta_schedule
from src.loss.likelihood import temporal_avg_loss, glambie_loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pretrained_prior(pretrained_params_path: str) -> tuple[dict, dict]:
    """
    Load (mu_dict, log_sigma_dict) from pretrained_params.pkl.

    These become the prior distribution for Stage 2 KL.

    Returns:
        (prior_mu, prior_log_sigma) — same pytree structure as model params
    """
    # TODO: cloudpickle.load, assert tuple of length 2
    raise NotImplementedError


def load_oggm_full(cfg) -> pd.DataFrame:
    """Load full.csv for the configured region."""
    # TODO: same as pretrain.load_oggm_split but always 'full'
    raise NotImplementedError


def load_temporal_avg(cfg) -> pd.DataFrame:
    """
    Load temporal-average CSV. Values are already in MWE/yr — no conversion.

    Required columns: rgi_id, start_date, end_date, avg_mb_mwe, uncertainty_mwe

    IMPORTANT: rgi_id ordering must be asserted to match the factorize
    codes from the OGGM training data (same pattern as oggm_combined_loss.py).
    """
    # TODO: implement; assert rgi_id alignment with OGGM factorize codes
    raise NotImplementedError


def load_glambie(cfg, oggm_years: set) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load GLaMBIE CSV for the configured region.

    Splits the data into:
      - train portion (years to be used in finetuning loss)
      - test portion  (cfg.model.glambie_test_years, withheld for evaluation)

    Handles missing file or empty source gracefully (returns None).

    Args:
        cfg:        Hydra config
        oggm_years: Set of years present in the OGGM training data
                    (GLaMBIE years must be a subset — assert this)

    Returns:
        (glambie_train_df, glambie_test_df) — either may be None
    """
    # TODO: implement; handle missing file gracefully
    # TODO: assert glambie_years ⊆ oggm_years
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Segment code preparation (outside JIT)
# ---------------------------------------------------------------------------

def prepare_finetune_arrays(
    oggm_df: pd.DataFrame,
    temporal_avg_df: pd.DataFrame,
    glambie_train_df: pd.DataFrame | None,
    ft_cols: list[str],
) -> dict:
    """
    Build all JAX arrays needed for the finetuning train_step.

    Factorize calls happen here — once, outside the training loop.
    The returned dict of arrays is passed to make_train_step and stays static.

    Returns dict with keys:
        time_index, covariates                        — full OGGM grid arrays
        temporal_avg_glacier_ids, temporal_avg_targets,
        temporal_avg_errs, n_glaciers                 — period-window arrays
        glambie_year_ids, glambie_targets,             — GLaMBIE arrays (or None)
        glambie_errs, n_glambie_obs
        period_window_mask                             — boolean mask for start_date–end_date rows
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Training step (JIT-compiled)
# ---------------------------------------------------------------------------

def make_train_step(
    model: BayesianNeuralField,
    optimizer,
    prior_mu: dict,
    prior_log_sigma: dict,
    static_arrays: dict,
):
    """
    Factory: returns a JIT-compiled finetune train_step.

    The prior (prior_mu, prior_log_sigma) is closed over — it is static
    and never updated during Stage 2 training.

    Signature of returned function:
        train_step(params, opt_state, rng, beta)
        -> (params, opt_state, loss_scalar)
    """
    # TODO: implement
    # Loss = finetune_elbo(
    #     temporal_avg_loss(...),
    #     glambie_loss(...),  # 0.0 if no GLaMBIE
    #     compute_total_kl(params, prior_mu, prior_log_sigma),
    #     beta
    # )
    raise NotImplementedError


# ---------------------------------------------------------------------------
# GLaMBIE test evaluation
# ---------------------------------------------------------------------------

def evaluate_glambie_test(
    model: BayesianNeuralField,
    params: dict,
    oggm_df: pd.DataFrame,
    glambie_test_df: pd.DataFrame,
    ft_cols: list[str],
    rng: jax.Array,
    n_samples: int = 100,
) -> dict:
    """
    Evaluate finetuned model against held-out GLaMBIE test years.

    Predicts regional annual means and compares against glambie_test_df.

    Returns dict with keys: rmse, bias, n_obs (per source)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main finetuning function
# ---------------------------------------------------------------------------

def run_finetune(cfg: DictConfig) -> None:
    """
    Full Stage 2 finetuning run.

    Loads pretrained prior, trains on Hugonnet + GLaMBIE,
    evaluates on held-out GLaMBIE test years, saves outputs.
    """
    os.makedirs(cfg.model.output_dir, exist_ok=True)

    # TODO: load_pretrained_prior
    # TODO: load_oggm_full, load_temporal_avg, load_glambie
    # TODO: prepare_finetune_arrays (factorize outside JIT)
    # TODO: init model + optimizer (same architecture as Stage 1)
    # TODO: initialise params from prior mu (warm start from pretrained posterior mean)
    # TODO: make_beta_schedule
    # TODO: make_train_step (closed over prior)
    # TODO: training loop
    # TODO: extract_vi_params → save finetuned_params.pkl
    # TODO: save training_loss.png
    # TODO: evaluate_glambie_test → save metrics_glambie_test.csv

    raise NotImplementedError
