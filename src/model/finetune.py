"""
Stage 2 finetuning loop — temporal-avg + GLaMBIE observational supervision.

Loads pretrained_params.pkl as the Bayesian prior (Bayesian continual learning).
All available OGGM data (full split) is used — no train/loyo/logo/loygo splits here.

Observation data:
  - Temporal-average per-glacier period mean (typically 2001-2020)
  - GLaMBIE regional annual means (gravimetry + altimetry, where available)

GLaMBIE test evaluation:
  - All post-temporal_avg GLaMBIE years (2021+) are withheld from training
    and used to assess finetuned model fit.
  - Early stopping monitors training loss, not a validation set.

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
    assert not extra, f"temporal_avg has rgi_ids not in OGGM data: {extra}"
    if missing:
        print(f"  Warning: {len(missing)} OGGM glaciers not in temporal_avg, excluded from temporal_avg loss: {missing}")

    # Sort to match factorize ordering (missing glaciers simply absent from df)
    id_to_idx = {rid: i for i, rid in enumerate(rgi_ids_ordered)}
    df = df[df["rgi_id"].isin(og_ids)].copy()
    df["_order"] = df["rgi_id"].map(id_to_idx)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


def load_glambie(
    cfg: DictConfig,
    feature_years: set,
    temporal_avg_end_year: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load and two-way split GLaMBIE data for the configured region.

    Splits:
      train — years <= temporal_avg_end_year  (used in finetuning loss)
      test  — all post-temporal_avg years (2021+)  (reported metrics only)

    GLaMBIE years not present in feature_years (the full features file, not just
    OGGM-merged rows) are dropped with a warning. Using feature_years instead of
    oggm_years means 2023/2024 are retained when features exist but OGGM targets
    don't extend that far.

    Args:
        cfg:                   Hydra config (needs cfg.model.glambie_path)
        feature_years:         Set of years present in the full features file.
        temporal_avg_end_year: Last year of temporal_avg period (e.g. 2020).

    Returns:
        (glambie_train_df, glambie_test_df) — either may be None.
    """
    glambie_path = cfg.model.get("glambie_path", None)
    if not glambie_path or not os.path.exists(glambie_path):
        return None, None

    df = _load_glambie_raw(glambie_path)
    if df.empty:
        return None, None

    glambie_years = set(df["year"].unique())
    out_of_range  = glambie_years - feature_years
    if out_of_range:
        print(f"  GLaMBIE years dropped (no feature data): {sorted(out_of_range)}")
        df = df[df["year"].isin(feature_years)].reset_index(drop=True)
    if df.empty:
        return None, None

    train_df = df[df["year"] <= temporal_avg_end_year].reset_index(drop=True)
    test_df  = df[df["year"] >  temporal_avg_end_year].reset_index(drop=True)

    test_years = sorted(test_df["year"].unique().tolist())
    if test_years:
        print(f"  GLaMBIE test years: {test_years}")

    return (
        train_df if not train_df.empty else None,
        test_df  if not test_df.empty  else None,
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
    time_index, covariates, _, _ = build_model_inputs(oggm_df, ft_cols)
    if scaler is not None:
        covariates = apply_scaler(covariates, scaler)

    # Factorize glacier IDs — codes[i] gives glacier index for row i
    _, unique_glaciers = pd.factorize(oggm_df["rgi_id"], sort=True)
    n_glaciers = len(unique_glaciers)

    # Hugonnet temporal-avg arrays — currently always a single period (longest span).
    # The loop handles the general case; at present temporal_avg_df contains one period.
    # Period mask uses exclusive end (year < end_year) — consistent convention.
    hugonnet_periods = []
    for (pstart, pend), period_df in temporal_avg_df.groupby(["start_date", "end_date"]):
        period_rgi_set = set(period_df["rgi_id"].unique())
        # Only rows within the year window AND whose glacier has temporal_avg data
        pmask = ((oggm_df["year"] >= pstart) & (oggm_df["year"] < pend)
                 & oggm_df["rgi_id"].isin(period_rgi_set)).to_numpy()
        # Re-factorize glacier IDs for this period to contiguous [0, n_period_glaciers)
        period_rgi_series = oggm_df.loc[pmask, "rgi_id"]
        p_gids_local, period_unique = pd.factorize(period_rgi_series, sort=True)
        n_p_glaciers = len(period_unique)
        # Reorder period_df rows to match sorted period_unique order
        period_df_sorted = period_df.set_index("rgi_id").loc[period_unique].reset_index()
        p_tgt   = period_df_sorted["avg_mb_mwe"].to_numpy(dtype=np.float32)
        p_errs  = period_df_sorted["uncertainty_mwe"].to_numpy(dtype=np.float32)
        hugonnet_periods.append({
            "period_mask": jnp.array(pmask),
            "glacier_ids": jnp.array(p_gids_local, dtype=jnp.int32),
            "n_glaciers":  n_p_glaciers,
            "ta_targets":  jnp.array(p_tgt),
            "ta_errs":     jnp.array(p_errs),
        })

    out = {
        "time_index":         jnp.array(time_index),
        "covariates":         jnp.array(covariates),
        "hugonnet_periods":   hugonnet_periods,
        "n_hugonnet_periods": len(hugonnet_periods),
        "n_glaciers":         n_glaciers,
        "has_glambie":        glambie_train_df is not None,
    }

    # GLaMBIE arrays
    if glambie_train_df is not None:
        # Restrict GLaMBIE train years to those present in oggm_df.
        # Years like 2020 can appear in glambie_train_df (year <= temporal_avg_end_year)
        # but be absent from oggm_df (OGGM targets end at 2019). Including them
        # would produce zero OGGM rows for that year, causing 0/0 = NaN in the
        # area-weighted segment mean and NaN gradients throughout.
        oggm_years_set = set(oggm_df["year"].unique().tolist())
        glambie_train_df = glambie_train_df[glambie_train_df["year"].isin(oggm_years_set)].reset_index(drop=True)
        if glambie_train_df.empty:
            out["has_glambie"] = False
            return out

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

            # --- Temporal-avg loss (single Hugonnet period) ---
            # Loop kept for generality; n_hugonnet_periods == 1 in current usage.
            # Dividing by n_periods is a no-op (÷1) but keeps scale correct if
            # multiple periods are ever re-enabled.
            l_ta = jnp.array(0.0)
            for period in sa["hugonnet_periods"]:
                period_preds_scaled = model.apply(
                    params,
                    sa["time_index"][period["period_mask"]],
                    sa["covariates"][period["period_mask"]],
                    rng_period,
                )
                period_preds = period_preds_scaled * target_std + target_mean
                l_ta += temporal_avg_loss(
                    period_preds,
                    period["glacier_ids"],
                    period["n_glaciers"],
                    period["ta_targets"],
                    period["ta_errs"],
                )
            l_ta = l_ta / sa["n_hugonnet_periods"]

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

def evaluate_oggm_glambie(
    oggm_df: pd.DataFrame,
    glambie_test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute OGGM-as-baseline regional means against held-out GLaMBIE test years.

    For each GLaMBIE test observation whose year appears in oggm_df, computes
    the area-weighted regional mean of OGGM mass_balance_mwe and compares it
    to the GLaMBIE observation.  Years outside OGGM coverage are silently skipped
    (they will simply be absent from the returned DataFrame).

    Args:
        oggm_df:         Full OGGM dataframe — columns include year, mass_balance_mwe, Area.
        glambie_test_df: GLaMBIE held-out observations — columns year, source, value_mwe.

    Returns:
        DataFrame with columns: year, source, glambie_mwe, pred_mwe, residual.
    """
    rows = []
    for _, obs in glambie_test_df.iterrows():
        yr_df = oggm_df[oggm_df["year"] == obs["year"]]
        if yr_df.empty:
            continue
        total_area = yr_df["Area"].sum()
        if total_area == 0:
            continue
        oggm_regional_mean = float((yr_df["mass_balance_mwe"] * yr_df["Area"]).sum() / total_area)
        rows.append({
            "year":        obs["year"],
            "source":      obs["source"],
            "glambie_mwe": obs["value_mwe"],
            "pred_mwe":    oggm_regional_mean,
            "residual":    oggm_regional_mean - obs["value_mwe"],
        })
    return pd.DataFrame(rows)


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
    # features_df covers years beyond OGGM target coverage (e.g. 2023/2024)
    # and is used for GLaMBIE evaluation of those years
    all_feature_years = set(features_df["year"].unique().tolist())

    # Get unique glaciers in factorize order (sort=True matches prepare_finetune_arrays)
    _, unique_glaciers = pd.factorize(oggm_df["rgi_id"], sort=True)

    temporal_avg_df       = load_temporal_avg(cfg, unique_glaciers)
    # Use only the longest Hugonnet period (2000-2020). The decadal periods
    # (2000-2010, 2010-2020) have higher uncertainties and empirically hurt performance.
    _spans = temporal_avg_df.groupby(["start_date", "end_date"]).size().reset_index(name="_n")
    _spans["_span"] = _spans["end_date"] - _spans["start_date"]
    _best = _spans.sort_values("_span", ascending=False).iloc[0]
    temporal_avg_df = temporal_avg_df[
        (temporal_avg_df["start_date"] == _best["start_date"]) &
        (temporal_avg_df["end_date"]   == _best["end_date"])
    ].reset_index(drop=True)
    print(f"  Using Hugonnet period {int(_best['start_date'])}-{int(_best['end_date'])} only "
          f"({len(temporal_avg_df)} glaciers)")
    temporal_avg_end_year = int(temporal_avg_df["end_date"].max())
    # Use all_feature_years so GLaMBIE years 2023/2024 aren't silently dropped
    # when OGGM targets don't extend that far but features do.
    glambie_train_df, glambie_test_df = load_glambie(cfg, all_feature_years, temporal_avg_end_year)

    # --- Prepare static arrays (factorize outside JIT) ---
    static_arrays = prepare_finetune_arrays(
        oggm_df, temporal_avg_df, glambie_train_df, ft_cols, scaler=scaler
    )

    # --- Model init (warm-start from pretrained posterior mean) ---
    model = BayesianNeuralField(
        hidden_sizes=tuple([cfg.model.model_nhidden] * cfg.model.model_nlayers),
        n_fourier=cfg.model.n_fourier,
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

    # --- Optimizer: finetune_lr + cosine decay starting after beta annealing + grad clipping ---
    # Cosine decay runs from beta_anneal_epochs → model_nepochs (the post-annealing window).
    # During beta annealing the LR is held constant at finetune_lr to preserve the flexibility
    # needed while KL weight is still ramping up.
    finetune_lr     = float(cfg.model.get("finetune_lr", cfg.model.lr))
    grad_clip_norm  = float(cfg.model.get("grad_clip_norm", 0.0))
    anneal_epochs   = int(cfg.model.beta_anneal_epochs)
    decay_steps     = max(1, cfg.model.model_nepochs - anneal_epochs)

    lr_schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(finetune_lr),
            optax.cosine_decay_schedule(finetune_lr, decay_steps=decay_steps, alpha=0.05),
        ],
        boundaries=[anneal_epochs],
    )
    optimizer_chain = [optax.adam(lr_schedule)]
    if grad_clip_norm > 0.0:
        optimizer_chain.insert(0, optax.clip_by_global_norm(grad_clip_norm))
    optimizer  = optax.chain(*optimizer_chain)
    opt_state  = optimizer.init(params)

    # --- Beta schedule + train step ---
    # n_data for KL normalisation: single Hugonnet period, so n_glaciers for that period.
    # GLaMBIE observation count is added as before.
    _period_counts = [p["n_glaciers"] for p in static_arrays["hugonnet_periods"]]
    n_finetune_obs = round(sum(_period_counts) / len(_period_counts)) if _period_counts else 0
    if static_arrays["has_glambie"]:
        n_finetune_obs += int(static_arrays["obs_year_idx"].shape[0])

    t_mean, t_std = target_scaler if target_scaler is not None else (0.0, 1.0)

    beta_schedule = make_beta_schedule(cfg.model.model_nepochs, cfg.model.beta_anneal_epochs)
    train_step    = make_train_step(
        model, optimizer, prior_mu, prior_log_sigma, static_arrays,
        n_data=n_finetune_obs, target_mean=t_mean, target_std=t_std,
    )

    # --- Early stopping setup (monitors training loss, not validation) ---
    # IMPORTANT: early stopping only activates AFTER beta annealing completes.
    # During annealing the ELBO total increases even as the model improves (KL
    # weight is ramping up), so the minimum total loss often falls at an early
    # low-beta checkpoint.  Tracking best_params during annealing would restore
    # an undertrained model.  We start tracking only once beta = 1.0.
    patience      = int(cfg.model.get("early_stopping_patience", 20))
    eval_interval = int(cfg.model.get("early_stopping_interval", 100))
    use_early_stop = patience > 0
    if use_early_stop:
        best_loss   = float("inf")
        best_params = params
        no_improve  = 0

    # --- Training loop ---
    losses = []
    stopped_epoch = cfg.model.model_nepochs

    for epoch in range(cfg.model.model_nepochs):
        rng, rng_step = jax.random.split(rng)
        beta = beta_schedule[epoch]
        params, opt_state, loss = train_step(params, opt_state, rng_step, beta)
        loss_val = float(loss)
        losses.append(loss_val)
        if (epoch + 1) % max(1, cfg.model.model_nepochs // 10) == 0:
            print(f"  epoch {epoch+1}/{cfg.model.model_nepochs}  loss={loss_val:.6f}  beta={beta:.3f}")

        annealing_done = (epoch + 1) >= cfg.model.beta_anneal_epochs
        if use_early_stop and annealing_done and (epoch + 1) % eval_interval == 0:
            if loss_val < best_loss:
                best_loss   = loss_val
                best_params = params
                no_improve  = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} — best train loss={best_loss:.6f}")
                stopped_epoch = epoch + 1
                break

    if use_early_stop and best_loss < float("inf"):
        params = best_params
        print(f"  Restored best params (train loss={best_loss:.6f}, stopped epoch {stopped_epoch})")

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

    # --- GLaMBIE test evaluation (all post-temporal_avg years: 2021+) ---
    if glambie_test_df is not None:
        eval_kwargs = dict(
            features_df=features_df, ft_cols=ft_cols,
            n_samples=cfg.model.model_nensemble, scaler=scaler, target_scaler=target_scaler,
        )

        # OGGM baseline (area-weighted regional mean from raw OGGM targets)
        oggm_df_eval = evaluate_oggm_glambie(oggm_df, glambie_test_df)
        oggm_df_eval["stage"] = "oggm"

        # Pretrain baseline (prior params — no finetuning applied)
        rng, rng_pre = jax.random.split(rng)
        pre_df = evaluate_glambie_test(model, prior_params, glambie_test_df=glambie_test_df, rng=rng_pre, **eval_kwargs)
        pre_df["stage"] = "pretrain"

        # Finetuned model
        rng, rng_fin = jax.random.split(rng)
        fin_df = evaluate_glambie_test(model, params, glambie_test_df=glambie_test_df, rng=rng_fin, **eval_kwargs)
        fin_df["stage"] = "finetune"

        metrics_df = pd.concat([oggm_df_eval, pre_df, fin_df], ignore_index=True)
        metrics_path = os.path.join(cfg.model.output_dir, "metrics_glambie_test.csv")
        metrics_df.to_csv(metrics_path, index=False)

        for stage, sdf in metrics_df.groupby("stage"):
            rmse = float(np.sqrt((sdf["residual"] ** 2).mean()))
            bias = float(sdf["residual"].mean())
            ss_res = (sdf["residual"] ** 2).sum()
            ss_tot = ((sdf["glambie_mwe"] - sdf["glambie_mwe"].mean()) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"GLaMBIE test [{stage}]: rmse={rmse:.4f}  bias={bias:.4f}  r2={r2:.4f}  n={len(sdf)}")
