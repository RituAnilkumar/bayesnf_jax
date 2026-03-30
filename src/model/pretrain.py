"""
Stage 1 pretraining loop — OGGM point-level supervision.

Two run modes controlled by cfg.model.train_split:
  - 'train': trains on train split, evaluates on loyo/logo/loygo splits.
             Produces out-of-sample metrics for performance assessment.
             loyo is the primary evaluation metric (leave-one-year-out).
  - 'full':  trains on full split (all available data), skips OOS evaluation.
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
import matplotlib.pyplot as plt
from hydra.utils import get_original_cwd
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
from src.data_utils import (
    load_features,
    load_oggm,
    merge_features_targets,
    select_held_years,
    make_cv_splits,
    build_model_inputs,
    fit_scaler,
    apply_scaler,
    FEATURE_COLS,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_oggm_split(cfg: DictConfig, split: str) -> tuple[pd.DataFrame, list[int]]:
    """
    Load and return one CV split of the merged (features + OGGM targets) DataFrame.

    Args:
        cfg:   Hydra config
        split: One of 'train', 'loyo', 'logo', 'loygo', 'full'

    Returns:
        (split_df, held_years) — the split DataFrame and the globally held years
        (needed so evaluate_split can use the same held_years).
    """
    region = cfg.model.reg_subdir
    base   = os.path.join(get_original_cwd(), cfg.model.inp_dir, region)

    features_df = load_features(os.path.join(base, f"main_features_{region}.csv"))
    targets_df  = load_oggm(os.path.join(base, f"oggm_targets_{region}.csv"))

    merged_df   = merge_features_targets(features_df, targets_df)

    held_years = select_held_years(merged_df["year"].unique().tolist())
    splits     = make_cv_splits(merged_df, held_years)

    return splits[split], held_years


def _get_ft_cols(cfg: DictConfig) -> list[str]:
    """Resolve feature columns from config, applying any rm_fts exclusions."""
    ft_cols = list(cfg.model.model_ftcols) if cfg.model.model_ftcols else FEATURE_COLS
    rm_fts  = list(cfg.model.rm_fts) if cfg.model.get("rm_fts") else []
    return [c for c in ft_cols if c not in rm_fts]


def prepare_arrays(df: pd.DataFrame, ft_cols: list[str], scaler=None) -> dict:
    """
    Convert a loaded OGGM DataFrame into JAX arrays ready for training.

    Args:
        df:      Merged OGGM DataFrame.
        ft_cols: Feature column names.
        scaler:  Fitted StandardScaler to apply to covariates, or None for raw values.
                 Must be the scaler fitted on the training split only.

    Returns dict with keys:
        time_index: jnp.array, shape (N,)
        covariates: jnp.array, shape (N, n_features)
        targets:    jnp.array, shape (N,)  [MWE/yr]
    """
    time_index, covariates, _, _ = build_model_inputs(df, ft_cols)
    if scaler is not None:
        covariates = apply_scaler(covariates, scaler)
    targets = df["mass_balance_mwe"].to_numpy(dtype=np.float32)
    return {
        "time_index": jnp.array(time_index),
        "covariates": jnp.array(covariates),
        "targets":    jnp.array(targets),
    }


# ---------------------------------------------------------------------------
# Training step (JIT-compiled)
# ---------------------------------------------------------------------------

def make_train_step(model: BayesianNeuralField, optimizer):
    """
    Factory: returns a JIT-compiled train_step closed over model and optimizer.

    Returned function signature:
        train_step(params, opt_state, rng, time_index, covariates, targets, beta)
        -> (params, opt_state, loss_scalar)
    """
    @jax.jit
    def train_step(params, opt_state, rng, time_index, covariates, targets, beta):
        def loss_fn(params):
            rng_fwd, rng_kl = jax.random.split(rng)
            preds = model.apply(params, time_index, covariates, rng_fwd)
            l_oggm = oggm_loss(preds, targets)

            mu_dict, log_sigma_dict   = extract_vi_params(params["params"])
            prior_mu, prior_log_sigma = make_standard_normal_prior(mu_dict, log_sigma_dict)
            kl = compute_total_kl(mu_dict, log_sigma_dict, prior_mu, prior_log_sigma)

            return pretrain_elbo(l_oggm, kl, beta)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    return train_step


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
    Compute evaluation metrics for one data split using MC predictive mean.

    Returns dict with keys: rmse, bias, n_points
    """
    mc_preds = model.apply(
        params,
        arrays["time_index"],
        arrays["covariates"],
        rng,
        n_samples=n_samples,
        method=model.mc_predict,
    )  # (n_samples, N)
    pred_mean = jnp.mean(mc_preds, axis=0)  # (N,)
    targets   = arrays["targets"]

    residuals = pred_mean - targets
    rmse = float(jnp.sqrt(jnp.mean(residuals ** 2)))
    bias = float(jnp.mean(residuals))
    ss_res = float(jnp.sum(residuals ** 2))
    ss_tot = float(jnp.sum((targets - jnp.mean(targets)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "bias": bias, "r2": r2, "n_points": len(targets)}


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
    rng = jax.random.PRNGKey(cfg.model.seed)

    # --- Feature columns ---
    ft_cols = _get_ft_cols(cfg)

    # --- Data loading ---
    train_split = cfg.model.train_split  # 'train' or 'full'
    train_df, held_years = load_oggm_split(cfg, train_split)

    # Fit scaler on training covariates only (no leakage from eval splits)
    _, raw_covariates, _, _ = build_model_inputs(train_df, ft_cols)
    scaler = fit_scaler(raw_covariates)

    # Save scaler alongside pretrained_params so finetune/predict apply identical scaling
    scaler_path = os.path.join(cfg.model.output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        cloudpickle.dump(scaler, f)
    print(f"Saved feature scaler → {scaler_path}")

    train_arrays = prepare_arrays(train_df, ft_cols, scaler=scaler)

    # --- Model init ---
    model = BayesianNeuralField(
        hidden_sizes=tuple([cfg.model.model_nhidden] * cfg.model.model_nlayers),
        n_fourier=cfg.model.n_fourier,
    )
    rng, rng_init, rng_fwd = jax.random.split(rng, 3)
    params = model.init(
        rng_init,
        train_arrays["time_index"][:1],
        train_arrays["covariates"][:1],
        rng_fwd,
    )

    # --- Optimizer ---
    optimizer  = optax.adam(cfg.model.lr)
    opt_state  = optimizer.init(params)

    # --- Beta schedule ---
    beta_schedule = make_beta_schedule(
        cfg.model.model_nepochs,
        cfg.model.beta_anneal_epochs,
    )

    # --- Training loop ---
    train_step = make_train_step(model, optimizer)
    losses = []

    for epoch in range(cfg.model.model_nepochs):
        rng, rng_step = jax.random.split(rng)
        beta = beta_schedule[epoch]
        params, opt_state, loss = train_step(
            params, opt_state, rng_step,
            train_arrays["time_index"],
            train_arrays["covariates"],
            train_arrays["targets"],
            beta,
        )
        losses.append(float(loss))
        if (epoch + 1) % max(1, cfg.model.model_nepochs // 10) == 0:
            print(f"  epoch {epoch+1}/{cfg.model.model_nepochs}  loss={loss:.6f}  beta={beta:.3f}")

    # --- Save pretrained params ---
    mu_dict, log_sigma_dict = extract_vi_params(params["params"])
    params_path = os.path.join(cfg.model.output_dir, "pretrained_params.pkl")
    with open(params_path, "wb") as f:
        cloudpickle.dump((mu_dict, log_sigma_dict), f)
    print(f"Saved pretrained params → {params_path}")

    # --- Training loss curve ---
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO loss")
    ax.set_title("Stage 1 pretraining loss")
    fig.savefig(os.path.join(cfg.model.output_dir, "training_loss.png"), dpi=150)
    plt.close(fig)

    # --- OOS evaluation (train_split='train' only) ---
    if train_split != "full":
        # Re-load all splits using same held_years
        region = cfg.model.reg_subdir
        base   = os.path.join(get_original_cwd(), cfg.model.inp_dir, region)
        features_df = load_features(os.path.join(base, f"main_features_{region}.csv"))
        targets_df  = load_oggm(os.path.join(base, f"oggm_targets_{region}.csv"))
        merged_df   = merge_features_targets(features_df, targets_df)
        all_splits  = make_cv_splits(merged_df, held_years)

        rows = []
        for split_name in ["train", "loyo", "logo", "loygo"]:
            split_df = all_splits[split_name]
            if len(split_df) == 0:
                continue
            arrays = prepare_arrays(split_df, ft_cols, scaler=scaler)
            rng, rng_eval = jax.random.split(rng)
            metrics = evaluate_split(model, params, arrays, rng_eval, n_samples=cfg.model.model_nensemble)
            metrics["region"] = region
            metrics["split"]  = split_name
            rows.append(metrics)
            tag = " ← PRIMARY" if split_name == "loyo" else ""
            print(f"  {split_name}: rmse={metrics['rmse']:.4f}  bias={metrics['bias']:.4f}  r2={metrics['r2']:.4f}  n={metrics['n_points']}{tag}")

        pd.DataFrame(rows)[["region", "split", "rmse", "bias", "r2", "n_points"]].to_csv(
            os.path.join(cfg.model.output_dir, "metrics_oos.csv"), index=False
        )
