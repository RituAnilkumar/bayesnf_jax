"""
Stage 1 pretraining loop — OGGM point-level supervision.

Two run modes controlled by cfg.model.train_split:
  - 'train': trains on train.csv, evaluates on train/loyo/logo/loygo splits.
             Produces out-of-sample metrics for performance assessment.
             loyo is the primary evaluation metric (leave-one-year-out).
  - 'full':  trains on full.csv (all available data), skips OOS evaluation.
             Produces pretrained_params.pkl used as the Stage 2 prior.

Outputs written to cfg.model.output_dir:
  - pretrained_params.pkl         (always — tuple of mu_dict, log_sigma_dict)
  - training_loss.png             (always)
  - metrics_train.csv             (train_split='train' only)
  - metrics_loyo.csv              (train_split='train' only — PRIMARY metric)
  - metrics_logo.csv              (train_split='train' only)
  - metrics_loygo.csv             (train_split='train' only)
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
    make_standard_normal_prior,
    T_MIN,
)
from src.loss.elbo import pretrain_elbo, make_beta_schedule
from src.loss.likelihood import oggm_loss
from src.loss.aggregation import MM_TO_MWE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_oggm_split(cfg: DictConfig, split: str) -> pd.DataFrame:
    """
    Load one OGGM data split CSV for the configured region.

    Args:
        cfg:   Hydra config (model sub-config)
        split: One of 'train', 'loyo', 'logo', 'loygo', 'full'

    Returns:
        DataFrame with at least columns: rgi_id, year, mass_balance_mm_yr
    """
    # TODO: implement — path = cfg.model.inp_dir / cfg.model.reg_subdir / f"{split}.csv"
    # TODO: handle year column as int or date string (inherit jungle3 pattern)
    # TODO: compute time_index = year - T_MIN
    # TODO: apply MM_TO_MWE conversion to target column
    raise NotImplementedError


def prepare_arrays(df: pd.DataFrame, ft_cols: list[str]) -> dict:
    """
    Convert a loaded OGGM DataFrame into JAX arrays ready for training.

    Returns dict with keys:
        time_index: jnp.array, shape (N,)
        covariates: jnp.array, shape (N, n_features)
        targets:    jnp.array, shape (N,)  [MWE/yr]
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Training step (JIT-compiled)
# ---------------------------------------------------------------------------

def make_train_step(model: BayesianNeuralField, optimizer):
    """
    Factory: returns a JIT-compiled train_step function closed over model and optimizer.

    The returned function signature:
        train_step(params, opt_state, rng, time_index, covariates, targets, beta)
        -> (params, opt_state, loss_scalar)
    """
    # TODO: implement using jax.jit + jax.value_and_grad
    # Loss = pretrain_elbo(oggm_loss(...), compute_total_kl(...), beta)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Evaluation (no grad)
# ---------------------------------------------------------------------------

def evaluate_split(
    model: BayesianNeuralField,
    params: dict,
    arrays: dict,
    rng: jax.Array,
    n_samples: int = 50,
) -> dict:
    """
    Compute evaluation metrics for one data split.

    Uses MC predictions (n_samples forward passes) to get the predictive mean,
    then computes RMSE and bias against the OGGM targets.

    Returns dict with keys: rmse, bias, n_points
    """
    # TODO: implement using model.apply(..., method=model.mc_predict)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_pretrain(cfg: DictConfig) -> None:
    """
    Full Stage 1 pretraining run.

    Reads cfg.model.train_split to determine data split.
    Saves outputs to cfg.model.output_dir.
    """
    os.makedirs(cfg.model.output_dir, exist_ok=True)

    # --- Data loading ---
    # TODO: load train split (or full split)
    # TODO: pd.factorize on rgi_id for any future use — consistent ordering
    # TODO: prepare_arrays

    # --- Model + optimizer init ---
    # TODO: instantiate BayesianNeuralField from cfg
    # TODO: init params
    # TODO: make Adam optimizer (lr from cfg)

    # --- Beta schedule ---
    # TODO: beta_schedule = make_beta_schedule(cfg.model.model_nepochs, cfg.model.beta_anneal_epochs)

    # --- Training loop ---
    # TODO: loop over epochs, call train_step, log loss

    # --- Save pretrained params ---
    # TODO: mu_dict, log_sigma_dict = extract_vi_params(params['params'])
    # TODO: cloudpickle.dump((mu_dict, log_sigma_dict), output_dir/pretrained_params.pkl)

    # --- Save training curve ---
    # TODO: matplotlib plot of loss vs epoch → training_loss.png

    # --- Evaluate splits (only when train_split='train') ---
    # TODO: for each split in [train, loyo, logo, loygo]:
    #         arrays = prepare_arrays(load_oggm_split(cfg, split), ...)
    #         metrics = evaluate_split(model, params, arrays, rng)
    #         write metrics_{split}.csv
    # NOTE: loyo is the primary metric — log it prominently

    raise NotImplementedError
