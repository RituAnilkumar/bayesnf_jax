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
import functools
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
    mc_predict_chunked,
    T_MIN,
)
from src.loss.elbo import pretrain_elbo, make_beta_schedule
from src.loss.likelihood import oggm_loss, oggm_nll
from src.data_utils import (
    load_features,
    load_oggm,
    merge_features_targets,
    select_held_years,
    make_cv_splits,
    build_model_inputs,
    fit_scaler,
    apply_scaler,
    fit_target_scaler,
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


def prepare_arrays(
    df: pd.DataFrame,
    ft_cols: list[str],
    scaler=None,
    target_scaler: tuple[float, float] | None = None,
) -> dict:
    """
    Convert a loaded OGGM DataFrame into JAX arrays ready for training.

    Args:
        df:             Merged OGGM DataFrame.
        ft_cols:        Feature column names.
        scaler:         Fitted StandardScaler for covariates, or None.
        target_scaler:  (mean, std) tuple from fit_target_scaler(), or None.
                        When provided, targets are standardised to ~N(0,1).

    Returns dict with keys:
        time_index: jnp.array, shape (N,)
        covariates: jnp.array, shape (N, n_features)
        targets:    jnp.array, shape (N,)  [scaled if target_scaler provided]
    """
    time_index, covariates, _, _ = build_model_inputs(df, ft_cols)
    if scaler is not None:
        covariates = apply_scaler(covariates, scaler)
    targets = df["mass_balance_mwe"].to_numpy(dtype=np.float32)
    if target_scaler is not None:
        t_mean, t_std = target_scaler
        targets = ((targets - t_mean) / t_std).astype(np.float32)
    return {
        "time_index": jnp.array(time_index),
        "covariates": jnp.array(covariates),
        "targets":    jnp.array(targets),
    }


# ---------------------------------------------------------------------------
# Training step (JIT-compiled)
# ---------------------------------------------------------------------------

def make_train_step(
    model: BayesianNeuralField,
    optimizer,
    n_data: int,
    loss_fn_name: str = "huber",
    huber_delta: float = 0.5,
):
    """
    Factory: returns a pmap-compiled train_step for multi-GPU data parallelism.

    All arguments must have a leading device axis of size n_devices:
        params, opt_state   — replicated via jax.device_put_replicated
        rng                 — (n_devices, 2) from jax.random.split(key, n_devices)
        time_index          — (n_devices, N//n_devices)
        covariates          — (n_devices, N//n_devices, n_features)
        targets             — (n_devices, N//n_devices)
        beta                — (n_devices,) broadcast scalar

    n_data must be the TOTAL dataset size (before sharding) so the KL
    normalisation in pretrain_elbo is correct after pmean.
    """
    heteroscedastic = model.heteroscedastic

    @functools.partial(jax.pmap, axis_name='devices')
    def train_step(params, opt_state, rng, time_index, covariates, targets, beta):
        def loss_fn(params):
            rng_fwd, _ = jax.random.split(rng)
            if heteroscedastic:
                mu, sigma = model.apply(params, time_index, covariates, rng_fwd)
                l_oggm = oggm_nll(mu, sigma, targets)
            else:
                preds = model.apply(params, time_index, covariates, rng_fwd)
                l_oggm = oggm_loss(preds, targets, loss_fn=loss_fn_name, delta=huber_delta)

            mu_dict, log_sigma_dict   = extract_vi_params(params["params"])
            prior_mu, prior_log_sigma = make_standard_normal_prior(mu_dict, log_sigma_dict)
            kl = compute_total_kl(mu_dict, log_sigma_dict, prior_mu, prior_log_sigma)

            # n_data is the TOTAL dataset size (not the per-device shard size) so
            # KL scaling is correct after pmean averages the likelihood gradients.
            return pretrain_elbo(l_oggm, kl, beta, n_data)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        # Average gradients and loss across all devices.  KL gradients are
        # identical on every device (params are replicated), so pmean leaves
        # them unchanged.  Likelihood gradients from each shard are averaged
        # to recover the full-batch gradient.
        grads = jax.lax.pmean(grads, axis_name='devices')
        loss  = jax.lax.pmean(loss,  axis_name='devices')
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
    target_scaler: tuple[float, float] | None = None,
) -> dict:
    """
    Compute evaluation metrics for one data split using MC predictive mean.

    Metrics are always returned in physical units (MWE/yr). If target_scaler
    is provided, both predictions and targets are unscaled before computing
    residuals so that rmse/bias/r2 are interpretable.

    Returns dict with keys: rmse, bias, r2, n_points
    """
    # mc_predict uses jax.vmap over n_samples, creating an (n_samples, N, H)
    # intermediate per layer.  For large N (hundreds of thousands of rows) and
    # large H this exceeds GPU memory.  mc_predict_chunked runs one sample at a
    # time (chunk_size=1), keeping peak memory at a plain (N, H) forward pass.
    mc_result = mc_predict_chunked(
        model, params,
        arrays["time_index"],
        arrays["covariates"],
        rng,
        n_samples=n_samples,
        chunk_size=1,
    )
    if isinstance(mc_result, tuple):
        mc_preds = mc_result[0]  # mu samples only for evaluation
    else:
        mc_preds = mc_result    # (n_samples, N) numpy array
    pred_mean = jnp.mean(mc_preds, axis=0)  # (N,)
    targets   = arrays["targets"]

    if target_scaler is not None:
        t_mean, t_std = target_scaler
        pred_mean = pred_mean * t_std + t_mean
        targets   = targets   * t_std + t_mean

    residuals = pred_mean - targets
    rmse = float(jnp.sqrt(jnp.mean(residuals ** 2)))
    bias = float(jnp.mean(residuals))
    ss_res = float(jnp.sum(residuals ** 2))
    ss_tot = float(jnp.sum((targets - jnp.mean(targets)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "bias": bias, "r2": r2, "n_points": len(targets)}


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def _shard_arrays(arrays: dict, n_devices: int) -> dict:
    """
    Pad N to the nearest multiple of n_devices, then reshape each array from
    (N, ...) to (n_devices, N//n_devices, ...) so pmap can distribute shards.

    Padding duplicates the first few rows — for large N (tens of thousands)
    the effect on the loss is negligible.
    """
    n = next(iter(arrays.values())).shape[0]
    pad = (-n) % n_devices
    if pad > 0:
        arrays = {k: jnp.concatenate([v, v[:pad]], axis=0) for k, v in arrays.items()}
        n = n + pad
    n_per = n // n_devices
    return {k: v.reshape(n_devices, n_per, *v.shape[1:]) for k, v in arrays.items()}


def _unreplicate(tree):
    """Extract the single-device copy of a pmap-replicated pytree (takes device 0)."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_pretrain(cfg: DictConfig) -> None:
    """
    Full Stage 1 pretraining run.

    Reads cfg.model.train_split to determine data split.
    Saves outputs to cfg.model.output_dir.
    """
    print("default_backend:", jax.default_backend())
    print("devices:", jax.devices())
    print("local_device_count:", jax.local_device_count())
    print("device_count:", jax.device_count())

    os.makedirs(cfg.model.output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(cfg.model.seed)

    # --- Feature columns ---
    ft_cols = _get_ft_cols(cfg)

    # --- Data loading ---
    train_split = cfg.model.train_split  # 'train' or 'full'
    train_df, held_years = load_oggm_split(cfg, train_split)

    # Restrict to the observationally-constrained period to align the pretrained
    # prior with the Hugonnet/GLaMBIE finetuning window.
    pretrain_year_min = int(cfg.model.get("pretrain_year_min", 0))
    pretrain_year_max = int(cfg.model.get("pretrain_year_max", 9999))
    if pretrain_year_min > 0 or pretrain_year_max < 9999:
        n_before = len(train_df)
        train_df = train_df[
            (train_df["year"] >= pretrain_year_min) &
            (train_df["year"] <= pretrain_year_max)
        ].reset_index(drop=True)
        print(f"  Year filter [{pretrain_year_min}, {pretrain_year_max}]: "
              f"{n_before} → {len(train_df)} rows")

    # Fit feature and target scalers on training data only (no leakage from eval splits)
    _, raw_covariates, _, _ = build_model_inputs(train_df, ft_cols)
    scaler = fit_scaler(raw_covariates)

    raw_targets = train_df["mass_balance_mwe"].to_numpy(dtype=np.float32)
    target_scaler = fit_target_scaler(raw_targets)  # (mean, std)

    scaler_path = os.path.join(cfg.model.output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        cloudpickle.dump(scaler, f)
    print(f"Saved feature scaler → {scaler_path}")

    target_scaler_path = os.path.join(cfg.model.output_dir, "target_scaler.pkl")
    with open(target_scaler_path, "wb") as f:
        cloudpickle.dump(target_scaler, f)
    print(f"Saved target scaler  → {target_scaler_path}  (mean={target_scaler[0]:.4f}, std={target_scaler[1]:.4f})")

    train_arrays = prepare_arrays(train_df, ft_cols, scaler=scaler, target_scaler=target_scaler)

    # --- Model init ---
    model = BayesianNeuralField(
        hidden_sizes=tuple([cfg.model.model_nhidden] * cfg.model.model_nlayers),
        n_fourier=cfg.model.n_fourier,
        heteroscedastic=bool(cfg.model.get("heteroscedastic", False)),
        sigma_floor=float(cfg.model.get("sigma_floor", 0.05)),
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

    # --- Early stopping setup (only when train_split != 'full' and patience > 0) ---
    patience       = int(cfg.model.get("early_stopping_patience", 20))
    eval_interval  = int(cfg.model.get("early_stopping_interval", 100))
    use_early_stop = (train_split != "full") and (patience > 0)

    # Pre-load loyo arrays for val if early stopping is active
    if use_early_stop:
        region = cfg.model.reg_subdir
        base   = os.path.join(get_original_cwd(), cfg.model.inp_dir, region)
        features_df_es = load_features(os.path.join(base, f"main_features_{region}.csv"))
        targets_df_es  = load_oggm(os.path.join(base, f"oggm_targets_{region}.csv"))
        merged_df_es   = merge_features_targets(features_df_es, targets_df_es)
        loyo_df        = make_cv_splits(merged_df_es, held_years)["loyo"]
        loyo_arrays    = prepare_arrays(loyo_df, ft_cols, scaler=scaler, target_scaler=target_scaler)
        best_val_rmse  = float("inf")
        best_params    = params
        no_improve     = 0

    # --- Multi-GPU setup ---
    # n_data must be captured from the FULL (unsharded) dataset so the KL
    # normalisation in pretrain_elbo is correct after pmean.
    n_data    = len(train_arrays["targets"])
    n_devices = jax.local_device_count()
    devices   = jax.devices()
    print(f"  Using {n_devices} device(s): {devices}")

    sharded_arrays = _shard_arrays(train_arrays, n_devices)

    # Replicate params and optimizer state across all devices.
    params    = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    # --- Training loop ---
    train_step = make_train_step(
        model, optimizer, n_data,
        loss_fn_name=cfg.model.get("oggm_loss_fn", "huber"),
        huber_delta=cfg.model.get("huber_delta", 0.5),
    )
    losses = []
    stopped_epoch = cfg.model.model_nepochs

    for epoch in range(cfg.model.model_nepochs):
        rng, rng_step = jax.random.split(rng)
        beta = beta_schedule[epoch]
        # Give each device a distinct RNG key so weight samples are independent.
        rng_keys = jax.random.split(rng_step, n_devices)       # (n_devices, 2)
        beta_arr = jnp.full((n_devices,), beta)                # (n_devices,)
        params, opt_state, loss_arr = train_step(
            params, opt_state, rng_keys,
            sharded_arrays["time_index"],
            sharded_arrays["covariates"],
            sharded_arrays["targets"],
            beta_arr,
        )
        # loss_arr has shape (n_devices,); all values identical after pmean.
        loss = float(loss_arr[0])
        losses.append(loss)
        if (epoch + 1) % max(1, cfg.model.model_nepochs // 10) == 0:
            print(f"  epoch {epoch+1}/{cfg.model.model_nepochs}  loss={loss:.6f}  beta={beta:.3f}")

        # --- Early stopping check ---
        if use_early_stop and (epoch + 1) % eval_interval == 0:
            rng, rng_es = jax.random.split(rng)
            val_metrics = evaluate_split(model, _unreplicate(params), loyo_arrays, rng_es,
                                         n_samples=10, target_scaler=target_scaler)
            val_rmse = val_metrics["rmse"]
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_params   = params  # store replicated; unreplicate at save time
                no_improve    = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} — best loyo RMSE={best_val_rmse:.4f}")
                stopped_epoch = epoch + 1
                break

    if use_early_stop:
        params = best_params
        print(f"  Restored best params (loyo RMSE={best_val_rmse:.4f}, stopped at epoch {stopped_epoch})")

    # --- Save pretrained params ---
    # Un-replicate before extracting VI params — all devices hold identical values.
    single_params = _unreplicate(params)
    mu_dict, log_sigma_dict = extract_vi_params(single_params["params"])
    params_path = os.path.join(cfg.model.output_dir, "pretrained_params.pkl")
    with open(params_path, "wb") as f:
        cloudpickle.dump((mu_dict, log_sigma_dict), f)
    print(f"Saved pretrained params → {params_path}")

    # --- Training loss curve ---
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO loss")
    ax.set_title(f"Stage 1 pretraining loss (stopped epoch {stopped_epoch})")
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
            arrays = prepare_arrays(split_df, ft_cols, scaler=scaler, target_scaler=target_scaler)
            rng, rng_eval = jax.random.split(rng)
            metrics = evaluate_split(model, single_params, arrays, rng_eval, n_samples=cfg.model.model_nensemble, target_scaler=target_scaler)
            metrics["region"] = region
            metrics["split"]  = split_name
            rows.append(metrics)
            tag = " ← PRIMARY" if split_name == "loyo" else ""
            print(f"  {split_name}: rmse={metrics['rmse']:.4f}  bias={metrics['bias']:.4f}  r2={metrics['r2']:.4f}  n={metrics['n_points']}{tag}")

        pd.DataFrame(rows)[["region", "split", "rmse", "bias", "r2", "n_points"]].to_csv(
            os.path.join(cfg.model.output_dir, "metrics_oos.csv"), index=False
        )
