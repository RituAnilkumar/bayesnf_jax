"""
Stage 2 finetuning loop — temporal-avg + GLaMBIE observational supervision.

Loads pretrained_params.pkl as the Bayesian prior (Bayesian continual learning).
All available OGGM data (full split) is used — no train/loyo/logo/loygo splits here.

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
  - metrics_glambie_test.csv      (held-out GLaMBIE years, if glambie_test_years set)
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
    T_MIN,
)
from src.loss.elbo import finetune_elbo, make_beta_schedule
from src.loss.likelihood import temporal_avg_loss, glambie_loss
from src.loss.aggregation import regional_annual_mean
from src.data_utils import (
    load_features,
    load_oggm,
    merge_features_targets,
    select_held_years,
    make_cv_splits,
    load_dmdtda as _load_temporal_avg,
    load_glambie as _load_glambie_raw,
    build_model_inputs,
    apply_scaler,
    FEATURE_COLS,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pretrained_prior(pretrained_params_path: str) -> tuple[dict, dict]:
    """
    Load (mu_dict, log_sigma_dict) from pretrained_params.pkl.

    These become the prior distribution for Stage 2 KL.

    Returns:
        (prior_mu, prior_log_sigma) — same pytree structure as model params['params']
    """
    with open(pretrained_params_path, "rb") as f:
        prior = cloudpickle.load(f)
    assert isinstance(prior, tuple) and len(prior) == 2, \
        f"Expected (mu_dict, log_sigma_dict), got {type(prior)}"
    return prior  # (mu_dict, log_sigma_dict)


def load_oggm_full(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the full merged OGGM dataset and the raw features for the configured region.

    Returns:
        (oggm_df, features_df) — oggm_df is the inner-join of features + OGGM targets
        (years limited by OGGM target coverage); features_df is the full features file
        including years beyond OGGM coverage (e.g. 2023/2024 for GLaMBIE evaluation).
    """
    region = cfg.model.reg_subdir
    base   = os.path.join(get_original_cwd(), cfg.model.inp_dir, region)

    features_df = load_features(os.path.join(base, f"main_features_{region}.csv"))
    targets_df  = load_oggm(os.path.join(base, f"oggm_targets_{region}.csv"))
    merged_df   = merge_features_targets(features_df, targets_df)

    held_years = select_held_years(merged_df["year"].unique().tolist())
    splits     = make_cv_splits(merged_df, held_years)
    return splits["full"], features_df


def load_temporal_avg(cfg: DictConfig, rgi_ids_ordered: np.ndarray) -> pd.DataFrame:
    """
    Load temporal-average CSV and assert rgi_id ordering matches the OGGM factorize codes.

    rgi_ids_ordered must be the unique rgi_ids in the same order as produced by
    pd.factorize on the OGGM full dataset — so temporal_avg targets index [i]
    corresponds to glacier code i.

    Returns DataFrame sorted to match rgi_ids_ordered, columns:
        rgi_id, start_date, end_date, avg_mb_mwe, uncertainty_mwe
    """
    df = _load_temporal_avg(cfg.model.temporal_avg_path)

    # Assert exact rgi_id set match between temporal_avg and OGGM
    ta_ids = set(df["rgi_id"].unique())
    og_ids = set(rgi_ids_ordered)
    extra   = ta_ids - og_ids
    missing = og_ids - ta_ids
    assert not extra,   f"temporal_avg has rgi_ids not in OGGM data: {extra}"
    assert not missing, f"OGGM glaciers missing from temporal_avg: {missing}"

    # Sort to match factorize ordering
    id_to_idx = {rid: i for i, rid in enumerate(rgi_ids_ordered)}
    df = df[df["rgi_id"].isin(og_ids)].copy()
    df["_order"] = df["rgi_id"].map(id_to_idx)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


def load_glambie(
    cfg: DictConfig,
    feature_years: set,
    temporal_avg_end_year: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load and three-way split GLaMBIE data for the configured region.

    Splits:
      train      — years <= temporal_avg_end_year  (used in finetuning loss)
      val        — first post-temporal_avg year (typically 2021)  (early stopping)
      final_test — remaining post-temporal_avg years (2022+)  (reported metrics)

    GLaMBIE years not present in feature_years (the full features file, not just
    OGGM-merged rows) are dropped with a warning. Using feature_years instead of
    oggm_years means 2023/2024 are retained when features exist but OGGM targets
    don't extend that far.

    Args:
        cfg:                   Hydra config (needs cfg.model.glambie_path)
        feature_years:         Set of years present in the full features file.
        temporal_avg_end_year: Last year of temporal_avg period (e.g. 2020).

    Returns:
        (glambie_train_df, glambie_val_df, glambie_final_test_df) — any may be None.
    """
    glambie_path = cfg.model.get("glambie_path", None)
    if not glambie_path or not os.path.exists(glambie_path):
        return None, None, None

    df = _load_glambie_raw(glambie_path)
    if df.empty:
        return None, None, None

    glambie_years = set(df["year"].unique())
    out_of_range  = glambie_years - feature_years
    if out_of_range:
        print(f"  GLaMBIE years dropped (no feature data): {sorted(out_of_range)}")
        df = df[df["year"].isin(feature_years)].reset_index(drop=True)
    if df.empty:
        return None, None, None

    train_df = df[df["year"] <= temporal_avg_end_year].reset_index(drop=True)
    post_df  = df[df["year"] >  temporal_avg_end_year].reset_index(drop=True)

    post_years = sorted(post_df["year"].unique().tolist())
    if post_years:
        val_year   = post_years[0]         # e.g. 2021
        test_years = post_years[1:]        # e.g. [2022, 2023, 2024]
        val_df       = post_df[post_df["year"] == val_year].reset_index(drop=True)
        final_test_df = post_df[post_df["year"].isin(test_years)].reset_index(drop=True) if test_years else pd.DataFrame()
        print(f"  GLaMBIE val year (early stopping): {val_year}")
        if test_years:
            print(f"  GLaMBIE final test years: {test_years}")
    else:
        val_df        = pd.DataFrame()
        final_test_df = pd.DataFrame()

    return (
        train_df      if not train_df.empty      else None,
        val_df        if not val_df.empty        else None,
        final_test_df if not final_test_df.empty else None,
    )


# ---------------------------------------------------------------------------
# Array preparation (outside JIT)
# ---------------------------------------------------------------------------

def prepare_finetune_arrays(
    oggm_df: pd.DataFrame,
    temporal_avg_df: pd.DataFrame,
    glambie_train_df: pd.DataFrame | None,
    ft_cols: list[str],
    scaler=None,
) -> dict:
    """
    Build all JAX arrays needed for the finetuning train_step.

    Factorize calls happen here — once, outside the training loop.

    Returns dict with keys:
        time_index, covariates              — full OGGM grid
        period_mask                         — bool mask selecting period-window rows
        temporal_avg_glacier_ids            — int glacier codes for period rows
        temporal_avg_targets, temporal_avg_errs, n_glaciers
        has_glambie                         — bool
        glambie_time_index, glambie_covariates  — rows matching GLaMBIE years
        glambie_year_ids, n_glambie_years   — per-row year code in [0, n_glambie_years)
        obs_year_idx                        — per-GLaMBIE-obs index into glambie years
        glambie_means, glambie_errs         — GLaMBIE targets
    """
    # Full OGGM grid arrays
    time_index, covariates, rgi_ids_all, _ = build_model_inputs(oggm_df, ft_cols)
    if scaler is not None:
        covariates = apply_scaler(covariates, scaler)

    # Factorize glacier IDs — codes[i] gives glacier index for row i
    glacier_codes, unique_glaciers = pd.factorize(oggm_df["rgi_id"], sort=True)
    n_glaciers = len(unique_glaciers)

    # Period window mask (temporal_avg period, same for all glaciers in our data)
    period_start = int(temporal_avg_df["start_date"].min())
    period_end   = int(temporal_avg_df["end_date"].max())
    period_mask  = (oggm_df["year"] >= period_start) & (oggm_df["year"] <= period_end)

    # Temporal-avg arrays (sorted to match factorize ordering above)
    # temporal_avg_df is already sorted to match unique_glaciers (asserted upstream)
    ta_targets = temporal_avg_df["avg_mb_mwe"].to_numpy(dtype=np.float32)
    ta_errs    = temporal_avg_df["uncertainty_mwe"].to_numpy(dtype=np.float32)

    out = {
        "time_index":               jnp.array(time_index),
        "covariates":               jnp.array(covariates),
        "period_mask":              jnp.array(period_mask.to_numpy()),
        "temporal_avg_glacier_ids": jnp.array(glacier_codes[period_mask.to_numpy()], dtype=jnp.int32),
        "temporal_avg_targets":     jnp.array(ta_targets),
        "temporal_avg_errs":        jnp.array(ta_errs),
        "n_glaciers":               n_glaciers,
        "has_glambie":              glambie_train_df is not None,
    }

    # GLaMBIE arrays
    if glambie_train_df is not None:
        glambie_years = sorted(glambie_train_df["year"].unique().tolist())
        year_to_idx   = {y: i for i, y in enumerate(glambie_years)}
        n_glambie_years = len(glambie_years)

        # OGGM rows whose year appears in GLaMBIE train years
        glambie_row_mask = oggm_df["year"].isin(glambie_years).to_numpy()
        g_time_idx   = time_index[glambie_row_mask]
        g_covariates = covariates[glambie_row_mask]
        g_year_ids   = oggm_df.loc[glambie_row_mask, "year"].map(year_to_idx).to_numpy(dtype=np.int32)
        # Area weights for area-weighted regional mean (MWE/yr requires area weighting)
        g_areas      = oggm_df.loc[glambie_row_mask, "Area"].to_numpy(dtype=np.float32)

        # Per-observation year index (one per GLaMBIE row)
        obs_year_idx = glambie_train_df["year"].map(year_to_idx).to_numpy(dtype=np.int32)

        out.update({
            "glambie_time_index":  jnp.array(g_time_idx),
            "glambie_covariates":  jnp.array(g_covariates),
            "glambie_year_ids":    jnp.array(g_year_ids),
            "glambie_areas":       jnp.array(g_areas),
            "n_glambie_years":     n_glambie_years,
            "obs_year_idx":        jnp.array(obs_year_idx),
            "glambie_means":       jnp.array(glambie_train_df["value_mwe"].to_numpy(dtype=np.float32)),
            "glambie_errs":        jnp.array(glambie_train_df["error_mwe"].to_numpy(dtype=np.float32)),
        })

    return out


# ---------------------------------------------------------------------------
# Training step (JIT-compiled)
# ---------------------------------------------------------------------------

def make_train_step(
    model: BayesianNeuralField,
    optimizer,
    prior_mu: dict,
    prior_log_sigma: dict,
    sa: dict,  # static_arrays
    n_data: int,
    target_mean: float = 0.0,
    target_std: float = 1.0,
):
    """
    Factory: returns a JIT-compiled finetune train_step.

    The prior, data arrays, n_data, and target scaling constants are all closed
    over — static during Stage 2 training.

    target_mean / target_std: from pretrain target_scaler.pkl. Model outputs
    are in scaled units; these are used to unscale to physical MWE/yr before
    the temporal_avg and glambie losses (which expect physical units).

    n_data normalises KL to the per-observation scale of the likelihood terms.

    Returned function signature:
        train_step(params, opt_state, rng, beta)
        -> (params, opt_state, loss_scalar)
    """
    has_glambie = sa["has_glambie"]

    @jax.jit
    def train_step(params, opt_state, rng, beta):
        def loss_fn(params):
            rng_period, rng_glambie = jax.random.split(rng)

            # --- Temporal-avg loss (unscale model output → physical MWE/yr) ---
            period_preds_scaled = model.apply(
                params,
                sa["time_index"][sa["period_mask"]],
                sa["covariates"][sa["period_mask"]],
                rng_period,
            )
            period_preds = period_preds_scaled * target_std + target_mean
            l_ta = temporal_avg_loss(
                period_preds,
                sa["temporal_avg_glacier_ids"],
                sa["n_glaciers"],
                sa["temporal_avg_targets"],
                sa["temporal_avg_errs"],
            )

            # --- GLaMBIE loss (unscale model output → physical MWE/yr) ---
            if has_glambie:
                glambie_preds_scaled = model.apply(
                    params,
                    sa["glambie_time_index"],
                    sa["glambie_covariates"],
                    rng_glambie,
                )
                glambie_preds = glambie_preds_scaled * target_std + target_mean
                l_glambie = glambie_loss(
                    glambie_preds,
                    sa["glambie_year_ids"],
                    sa["n_glambie_years"],
                    sa["obs_year_idx"],
                    sa["glambie_means"],
                    sa["glambie_errs"],
                    sa["glambie_areas"],
                )
            else:
                l_glambie = jnp.array(0.0)

            # --- KL against pretrained prior ---
            mu_dict, log_sigma_dict = extract_vi_params(params["params"])
            kl = compute_total_kl(mu_dict, log_sigma_dict, prior_mu, prior_log_sigma)

            return finetune_elbo(l_ta, l_glambie, kl, beta, n_data)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    return train_step


# ---------------------------------------------------------------------------
# GLaMBIE test evaluation
# ---------------------------------------------------------------------------

def evaluate_glambie_test(
    model: BayesianNeuralField,
    params: dict,
    *,
    features_df: pd.DataFrame,
    glambie_test_df: pd.DataFrame,
    ft_cols: list[str],
    rng: jax.Array,
    n_samples: int = 100,
    scaler=None,
    target_scaler: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate model against held-out GLaMBIE test years.

    Uses features_df (the full features file, not the OGGM-merged df) so that
    years with feature coverage but no OGGM targets (e.g. 2023/2024) are not
    silently skipped.

    For each test year, computes the MC area-weighted regional mean and compares
    against the GLaMBIE observation. Returns a DataFrame of per-observation metrics.
    """
    test_years = sorted(glambie_test_df["year"].unique().tolist())
    year_to_idx = {y: i for i, y in enumerate(test_years)}

    # Use full features file — includes years beyond OGGM target coverage
    test_df = features_df[features_df["year"].isin(test_years)].reset_index(drop=True)

    time_idx, covariates, _, _ = build_model_inputs(test_df, ft_cols)
    if scaler is not None:
        covariates = apply_scaler(covariates, scaler)
    year_ids  = test_df["year"].map(year_to_idx).to_numpy(dtype=np.int32)
    areas     = test_df["Area"].to_numpy(dtype=np.float32)

    # MC predictions → (n_samples, N)
    mc_preds = model.apply(
        params,
        jnp.array(time_idx),
        jnp.array(covariates),
        rng,
        n_samples=n_samples,
        method=model.mc_predict,
    )

    # Area-weighted regional annual mean per sample → (n_samples, n_years)
    jnp_year_ids = jnp.array(year_ids)
    jnp_areas    = jnp.array(areas)
    def sample_regional_mean(preds_i):
        return regional_annual_mean(preds_i, jnp_year_ids, len(test_years), jnp_areas)

    regional_means = jax.vmap(sample_regional_mean)(mc_preds)  # (n_samples, n_years)
    pred_mean = jnp.mean(regional_means, axis=0)  # (n_years,)

    # Unscale regional mean predictions to physical MWE/yr
    pred_mean_np = np.array(pred_mean)
    if target_scaler is not None:
        t_mean_v, t_std_v = target_scaler
        pred_mean_np = pred_mean_np * t_std_v + t_mean_v

    rows = []
    for _, obs in glambie_test_df.iterrows():
        y_idx    = year_to_idx[obs["year"]]
        pred_val = float(pred_mean_np[y_idx])
        rows.append({
            "year":       obs["year"],
            "source":     obs["source"],
            "glambie_mwe": obs["value_mwe"],
            "pred_mwe":   pred_val,
            "residual":   pred_val - obs["value_mwe"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main finetuning function
# ---------------------------------------------------------------------------

def run_finetune(cfg: DictConfig) -> None:
    """
    Full Stage 2 finetuning run.

    Loads pretrained prior, trains on temporal-avg + GLaMBIE,
    evaluates on held-out GLaMBIE test years, saves outputs.
    """
    os.makedirs(cfg.model.output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(cfg.model.seed)

    ft_cols = list(cfg.model.model_ftcols) if cfg.model.model_ftcols else FEATURE_COLS
    rm_fts  = list(cfg.model.rm_fts) if cfg.model.get("rm_fts") else []
    ft_cols = [c for c in ft_cols if c not in rm_fts]

    # --- Resolve relative paths against original cwd (Hydra chdir moves us to output dir) ---
    orig = get_original_cwd()
    from omegaconf import OmegaConf
    cfg_mutable = OmegaConf.to_container(cfg, resolve=True)
    for key in ("pretrained_params_path", "temporal_avg_path", "glambie_path"):
        if key in cfg_mutable["model"] and cfg_mutable["model"][key]:
            cfg_mutable["model"][key] = os.path.join(orig, cfg_mutable["model"][key])
    cfg = OmegaConf.create(cfg_mutable)

    # --- Load prior and scaler ---
    prior_mu, prior_log_sigma = load_pretrained_prior(cfg.model.pretrained_params_path)

    pretrain_dir = os.path.dirname(cfg.model.pretrained_params_path)

    scaler_path = os.path.join(pretrain_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = cloudpickle.load(f)
        print(f"Loaded feature scaler from {scaler_path}")
        with open(os.path.join(cfg.model.output_dir, "scaler.pkl"), "wb") as f:
            cloudpickle.dump(scaler, f)
    else:
        scaler = None
        print(f"WARNING: scaler.pkl not found at {scaler_path} — covariates will be unscaled")

    target_scaler_path = os.path.join(pretrain_dir, "target_scaler.pkl")
    if os.path.exists(target_scaler_path):
        with open(target_scaler_path, "rb") as f:
            target_scaler = cloudpickle.load(f)  # (mean, std)
        print(f"Loaded target scaler from {target_scaler_path}  (mean={target_scaler[0]:.4f}, std={target_scaler[1]:.4f})")
        with open(os.path.join(cfg.model.output_dir, "target_scaler.pkl"), "wb") as f:
            cloudpickle.dump(target_scaler, f)
    else:
        target_scaler = None
        print(f"WARNING: target_scaler.pkl not found at {target_scaler_path} — predictions will be unscaled")

    # --- Load data ---
    oggm_df, features_df = load_oggm_full(cfg)
    oggm_years  = set(oggm_df["year"].unique().tolist())
    # features_df covers years beyond OGGM target coverage (e.g. 2023/2024)
    # and is used for GLaMBIE evaluation of those years
    all_feature_years = set(features_df["year"].unique().tolist())

    # Get unique glaciers in factorize order (sort=True matches prepare_finetune_arrays)
    _, unique_glaciers = pd.factorize(oggm_df["rgi_id"], sort=True)

    temporal_avg_df              = load_temporal_avg(cfg, unique_glaciers)
    temporal_avg_end_year = int(temporal_avg_df["end_date"].max())
    # Use all_feature_years so GLaMBIE years 2023/2024 aren't silently dropped
    # when OGGM targets don't extend that far but features do.
    glambie_train_df, glambie_val_df, glambie_final_test_df = load_glambie(cfg, all_feature_years, temporal_avg_end_year)

    # --- Prepare static arrays (factorize outside JIT) ---
    static_arrays = prepare_finetune_arrays(
        oggm_df, temporal_avg_df, glambie_train_df, ft_cols, scaler=scaler
    )

    # --- Model init (warm-start from pretrained posterior mean) ---
    model = BayesianNeuralField(
        hidden_sizes=tuple([cfg.model.model_nhidden] * cfg.model.model_nlayers),
        n_fourier=cfg.model.n_fourier,
    )
    rng, rng_init, rng_fwd = jax.random.split(rng, 3)
    # Initialise with dummy input to get params structure, then overwrite with prior
    dummy_params = model.init(
        rng_init,
        static_arrays["time_index"][:1],
        static_arrays["covariates"][:1],
        rng_fwd,
    )
    # Reconstruct full params dict from pretrained (mu, log_sigma)
    def _merge(d_mu, d_ls):
        result = {**d_mu}
        for k, v in d_ls.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = _merge(result[k], v)
            else:
                result[k] = v
        return result
    params = {"params": _merge(prior_mu, prior_log_sigma)}

    # Keep a copy of the pretrained params for baseline evaluation after finetuning
    prior_params = params

    # --- Optimizer ---
    optimizer = optax.adam(cfg.model.lr)
    opt_state = optimizer.init(params)

    # --- Beta schedule + train step ---
    n_finetune_obs = static_arrays["n_glaciers"]
    if static_arrays["has_glambie"]:
        n_finetune_obs += int(static_arrays["obs_year_idx"].shape[0])

    t_mean, t_std = target_scaler if target_scaler is not None else (0.0, 1.0)

    beta_schedule = make_beta_schedule(cfg.model.model_nepochs, cfg.model.beta_anneal_epochs)
    train_step    = make_train_step(
        model, optimizer, prior_mu, prior_log_sigma, static_arrays,
        n_data=n_finetune_obs, target_mean=t_mean, target_std=t_std,
    )

    # --- Early stopping setup (val = GLaMBIE 2021, if available) ---
    patience      = int(cfg.model.get("early_stopping_patience", 20))
    eval_interval = int(cfg.model.get("early_stopping_interval", 100))
    use_early_stop = (patience > 0) and (glambie_val_df is not None)
    if use_early_stop:
        val_eval_kwargs = dict(
            features_df=features_df, ft_cols=ft_cols,
            n_samples=10, scaler=scaler, target_scaler=target_scaler,
        )
        best_val_rmse = float("inf")
        best_params   = params
        no_improve    = 0

    # --- Training loop ---
    losses = []
    stopped_epoch = cfg.model.model_nepochs

    for epoch in range(cfg.model.model_nepochs):
        rng, rng_step = jax.random.split(rng)
        beta = beta_schedule[epoch]
        params, opt_state, loss = train_step(params, opt_state, rng_step, beta)
        losses.append(float(loss))
        if (epoch + 1) % max(1, cfg.model.model_nepochs // 10) == 0:
            print(f"  epoch {epoch+1}/{cfg.model.model_nepochs}  loss={loss:.6f}  beta={beta:.3f}")

        if use_early_stop and (epoch + 1) % eval_interval == 0:
            rng, rng_es = jax.random.split(rng)
            val_df_es = evaluate_glambie_test(model, params,
                                              glambie_test_df=glambie_val_df,
                                              rng=rng_es, **val_eval_kwargs)
            val_rmse = float(np.sqrt((val_df_es["residual"] ** 2).mean()))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_params   = params
                no_improve    = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} — best val RMSE (2021)={best_val_rmse:.4f}")
                stopped_epoch = epoch + 1
                break

    if use_early_stop:
        params = best_params
        print(f"  Restored best params (val RMSE={best_val_rmse:.4f}, stopped epoch {stopped_epoch})")

    # --- Save finetuned params ---
    mu_dict, log_sigma_dict = extract_vi_params(params["params"])
    params_path = os.path.join(cfg.model.output_dir, "finetuned_params.pkl")
    with open(params_path, "wb") as f:
        cloudpickle.dump((mu_dict, log_sigma_dict), f)
    print(f"Saved finetuned params → {params_path}")

    # --- Training loss curve ---
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO loss")
    ax.set_title(f"Stage 2 finetuning loss (stopped epoch {stopped_epoch})")
    fig.savefig(os.path.join(cfg.model.output_dir, "training_loss.png"), dpi=150)
    plt.close(fig)

    # --- GLaMBIE final test evaluation (2022+) ---
    if glambie_final_test_df is not None:
        eval_kwargs = dict(
            features_df=features_df, ft_cols=ft_cols,
            n_samples=cfg.model.model_nensemble, scaler=scaler, target_scaler=target_scaler,
        )

        # Pretrain baseline (prior params — no finetuning applied)
        rng, rng_pre = jax.random.split(rng)
        pre_df = evaluate_glambie_test(model, prior_params, glambie_test_df=glambie_final_test_df, rng=rng_pre, **eval_kwargs)
        pre_df["stage"] = "pretrain"

        # Finetuned model
        rng, rng_fin = jax.random.split(rng)
        fin_df = evaluate_glambie_test(model, params, glambie_test_df=glambie_final_test_df, rng=rng_fin, **eval_kwargs)
        fin_df["stage"] = "finetune"

        metrics_df = pd.concat([pre_df, fin_df], ignore_index=True)
        metrics_path = os.path.join(cfg.model.output_dir, "metrics_glambie_test.csv")
        metrics_df.to_csv(metrics_path, index=False)

        for stage, sdf in metrics_df.groupby("stage"):
            rmse = float(np.sqrt((sdf["residual"] ** 2).mean()))
            bias = float(sdf["residual"].mean())
            ss_res = (sdf["residual"] ** 2).sum()
            ss_tot = ((sdf["glambie_mwe"] - sdf["glambie_mwe"].mean()) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"GLaMBIE test [{stage}]: rmse={rmse:.4f}  bias={bias:.4f}  r2={r2:.4f}  n={len(sdf)}")
