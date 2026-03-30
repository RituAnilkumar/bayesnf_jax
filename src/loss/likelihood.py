"""
Individual likelihood terms for the two-stage BNF pipeline.

All functions accept JAX arrays only — no pandas inside these functions.
Unit conversion and factorize alignment happen upstream in pretrain.py / finetune.py
before the JIT boundary.

Units throughout: MWE/yr (metres water equivalent per year).
"""

import jax
import jax.numpy as jnp
from .aggregation import glacier_annual_mean, regional_annual_mean


# ---------------------------------------------------------------------------
# Stage 1 — OGGM point-level likelihood
# ---------------------------------------------------------------------------

def oggm_loss(
    preds: jax.Array,   # shape (N,)  model predictions [MWE/yr]
    targets: jax.Array, # shape (N,)  OGGM targets [MWE/yr], pre-divided by 1000
    delta: float = 0.5, # Huber transition point [MWE/yr] — MSE below, MAE above
) -> jax.Array:
    """
    Huber loss over all (glacier × year) point predictions.

    Behaves as MSE for |residual| < delta and MAE (with a constant offset) for
    larger residuals, limiting the influence of divergent OGGM outliers while
    preserving well-conditioned gradients near zero.

    delta=0.5 MWE/yr is a reasonable default: typical inter-annual variability
    is ~0.1-0.3 MWE/yr, so residuals above 0.5 are genuine outliers.

    Args:
        preds:   Model predictions, shape (N,)   [MWE/yr]
        targets: OGGM mass balance targets, shape (N,)  [MWE/yr]
        delta:   Huber transition point in MWE/yr.

    Returns:
        Scalar Huber loss.
    """
    r = preds - targets
    return jnp.mean(jnp.where(jnp.abs(r) <= delta,
                               0.5 * r ** 2,
                               delta * (jnp.abs(r) - 0.5 * delta)))


# ---------------------------------------------------------------------------
# Stage 2 — temporal-average per-glacier mean likelihood
# ---------------------------------------------------------------------------

def temporal_avg_loss(
    preds: jax.Array,           # shape (N,)  predictions for period rows [MWE/yr]
    glacier_ids: jax.Array,     # shape (N,)  integer glacier codes
    n_glaciers: int,             # number of distinct glaciers (static)
    avg_mb: jax.Array,          # shape (n_glaciers,)  temporal-avg targets [MWE/yr]
    uncertainty: jax.Array,     # shape (n_glaciers,)  temporal-avg uncertainties [MWE/yr]
) -> jax.Array:
    """
    Inverse-variance weighted MSE between predicted per-glacier period means
    and temporal-average observations (typically 2001-2020).

    L_temporal_avg = (1/N_glaciers) * sum_i [
        (pred_period_mean_i - avg_mb_i)^2 / uncertainty_i^2
    ]

    where pred_period_mean_i = mean of preds for glacier i over start_date–end_date rows.

    Inputs are already in MWE/yr — no unit conversion needed.
    rgi_id ordering must match the factorize codes in glacier_ids (assert upstream).

    Args:
        preds:        Predictions for the period window rows, shape (N,)
        glacier_ids:  Integer glacier codes from pd.factorize, shape (N,)
        n_glaciers:   Number of distinct glaciers
        avg_mb:       Temporal-avg targets, shape (n_glaciers,)       [MWE/yr]
        uncertainty:  Temporal-avg uncertainties, shape (n_glaciers,) [MWE/yr]

    Returns:
        Scalar inverse-variance weighted MSE.
    """
    pred_means = glacier_annual_mean(preds, glacier_ids, n_glaciers)  # (n_glaciers,)
    return jnp.mean(((pred_means - avg_mb) / uncertainty) ** 2)


# ---------------------------------------------------------------------------
# Stage 2 — GLaMBIE regional annual mean likelihood
# ---------------------------------------------------------------------------

def glambie_loss(
    preds: jax.Array,         # shape (N,)    predictions for GLaMBIE year rows [MWE/yr]
    year_ids: jax.Array,      # shape (N,)    integer year codes in [0, n_years)
    n_years: int,              # number of distinct years covered (static)
    obs_year_idx: jax.Array,  # shape (N_obs,) index into [0, n_years) per observation
    glambie_means: jax.Array, # shape (N_obs,) GLaMBIE regional mean targets [MWE/yr]
    glambie_errs: jax.Array,  # shape (N_obs,) GLaMBIE uncertainties [MWE/yr]
) -> jax.Array:
    """
    Inverse-variance weighted MSE between predicted regional annual means
    and GLaMBIE observations.

    Gravimetry and altimetry are treated as independent residuals — each is
    a separate row in obs_year_idx / glambie_means / glambie_errs.
    N_obs = total (year, source) pairs; years with both sources count as 2.

    L_glambie = (1/N_obs) * sum_k [
        (pred_regional_mean_t(k) - glambie_mean_t(k))^2 / glambie_err_t(k)^2
    ]

    Args:
        preds:         Predictions for rows whose year appears in GLaMBIE, shape (N,)
        year_ids:      Integer year codes for each prediction row, shape (N,)
                       maps each row to a year index in [0, n_years)
        n_years:       Number of distinct years (static)
        obs_year_idx:  Year index for each GLaMBIE observation, shape (N_obs,)
                       aligns glambie_means[k] with pred_regional_means[obs_year_idx[k]]
        glambie_means: GLaMBIE targets, shape (N_obs,)        [MWE/yr]
        glambie_errs:  GLaMBIE uncertainties, shape (N_obs,)  [MWE/yr]

    Returns:
        Scalar inverse-variance weighted MSE, or 0.0 if N_obs == 0.
    """
    pred_regional_means = regional_annual_mean(preds, year_ids, n_years)  # (n_years,)
    pred_at_obs = pred_regional_means[obs_year_idx]                        # (N_obs,)
    return jnp.mean(((pred_at_obs - glambie_means) / glambie_errs) ** 2)
