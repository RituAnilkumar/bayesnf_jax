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
# Stage 2 — Hugonnet per-glacier 20-year mean likelihood
# ---------------------------------------------------------------------------

def hugonnet_loss(
    preds: jax.Array,           # shape (N,)  predictions for 2000-2019 rows [MWE/yr]
    glacier_ids: jax.Array,     # shape (N,)  integer glacier codes
    n_glaciers: int,             # number of distinct glaciers (static)
    dmdtda: jax.Array,          # shape (n_glaciers,)  Hugonnet targets [MWE/yr]
    err_dmdtda: jax.Array,      # shape (n_glaciers,)  Hugonnet uncertainties [MWE/yr]
) -> jax.Array:
    """
    Inverse-variance weighted MSE between predicted 20yr glacier means
    and Hugonnet dmdtda observations.

    L_hugonnet = (1/N_glaciers) * sum_i [
        (pred_20yr_i - dmdtda_i)^2 / err_dmdtda_i^2
    ]

    where pred_20yr_i = mean of preds for glacier i over rows in 2000-2019.

    Hugonnet inputs must be pre-converted from kg/m²/yr to MWE/yr (÷1000).
    rgi_id ordering must match the factorize codes in glacier_ids (assert upstream).

    Args:
        preds:        Predictions for the Hugonnet window rows, shape (N,)
        glacier_ids:  Integer glacier codes from pd.factorize, shape (N,)
        n_glaciers:   Number of distinct glaciers
        dmdtda:       Hugonnet 20yr mean targets, shape (n_glaciers,)  [MWE/yr]
        err_dmdtda:   Hugonnet uncertainties, shape (n_glaciers,)      [MWE/yr]

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
