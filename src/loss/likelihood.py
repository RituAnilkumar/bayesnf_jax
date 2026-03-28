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
) -> jax.Array:
    """
    MSE loss over all (glacier × year) point predictions.

    L_oggm = (1 / N) * sum_ij (pred_ij - oggm_ij)^2

    Args:
        preds:   Model predictions, shape (N,)   [MWE/yr]
        targets: OGGM mass balance targets, shape (N,)  [MWE/yr]

    Returns:
        Scalar MSE loss.
    """
    # TODO: implement
    raise NotImplementedError


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
    # TODO: implement using glacier_annual_mean from aggregation.py
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2 — GLaMBIE regional annual mean likelihood
# ---------------------------------------------------------------------------

def glambie_loss(
    preds: jax.Array,           # shape (N,)  predictions for GLaMBIE year rows [MWE/yr]
    year_ids: jax.Array,        # shape (N,)  integer year codes (per GLaMBIE obs)
    n_years_per_source: list,   # [n_years_gravimetry, n_years_altimetry] (static)
    glambie_means: jax.Array,   # shape (N_obs,)  GLaMBIE regional mean targets [MWE/yr]
    glambie_errs: jax.Array,    # shape (N_obs,)  GLaMBIE uncertainties [MWE/yr]
) -> jax.Array:
    """
    Inverse-variance weighted MSE between predicted regional annual means
    and GLaMBIE observations.

    Gravimetry and altimetry are treated as independent residuals.
    N_obs = total (year, source) pairs — years with both sources count as 2.

    L_glambie = (1/N_obs) * sum_k [
        (pred_ann_mean_t(k) - glambie_mean_t(k))^2 / err_glambie_t(k)^2
    ]

    GLaMBIE inputs must be pre-converted from regional Gt/yr sum to
    regional mean MWE/yr (÷ N_glaciers in region). Missing source for a
    region returns 0.0 gracefully.

    Args:
        preds:              Predictions for rows matching GLaMBIE years, shape (N,)
        year_ids:           Integer year codes (aligned per source), shape (N,)
        n_years_per_source: Static list of year counts per source
        glambie_means:      GLaMBIE targets, shape (N_obs,)   [MWE/yr]
        glambie_errs:       GLaMBIE uncertainties, shape (N_obs,)  [MWE/yr]

    Returns:
        Scalar inverse-variance weighted MSE, or 0.0 if no GLaMBIE data.
    """
    # TODO: implement; handle missing sources gracefully
    raise NotImplementedError
