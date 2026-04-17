"""
ELBO assembly and beta annealing schedule.

These functions return a *loss to be minimised* (not the ELBO to be maximised).
The relationship is:  loss = -ELBO / n_data  (up to constants).

Stage 1 loss:  L_oggm  + (beta / n_data) * KL(q || N(0,1))
Stage 2 loss:  L_temporal_avg + glambie_weight * L_glambie + (beta / n_data) * KL(q || q_pretrained)

Beta is annealed linearly from 0 → 1 over the first `beta_anneal_epochs`
epochs (default 20% of total epochs). This prevents posterior collapse
early in training when likelihood gradients dominate.

Beta reaches 1.0 at epoch `beta_anneal_epochs` and stays there.
"""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Beta annealing
# ---------------------------------------------------------------------------

def make_beta_schedule(total_epochs: int, anneal_epochs: int) -> list[float]:
    """
    Construct the full beta schedule as a Python list (evaluated once, outside JIT).

    Beta = 0 at epoch 0, ramps linearly to 1.0 at epoch `anneal_epochs`,
    then stays at 1.0 for the remainder.

    Args:
        total_epochs:  Total number of training epochs.
        anneal_epochs: Number of epochs over which beta ramps 0 → 1.
                       Should be ~20% of total_epochs per CLAUDE.md.

    Returns:
        List of float beta values, length = total_epochs.
    """
    betas = jnp.linspace(0.0, 1.0, anneal_epochs).tolist()
    betas += [1.0] * (total_epochs - anneal_epochs)
    return betas


# ---------------------------------------------------------------------------
# Stage 1 ELBO
# ---------------------------------------------------------------------------

def pretrain_elbo(
    oggm_loss_val,   # scalar
    kl_val,          # scalar
    beta: float,
    n_data: int,
) -> jnp.ndarray:
    """
    ELBO for Stage 1 pretraining.

    ELBO = L_oggm + (beta / n_data) * KL(q || N(0,1))

    KL is divided by n_data to match the per-data-point scale of L_oggm
    (which is a mean over N points). Without this, KL — a sum over all
    variational parameters — dominates once beta annealing completes.

    Args:
        oggm_loss_val: Scalar OGGM loss (mean over data points; Huber or MSE)
        kl_val:        Scalar KL divergence against standard normal prior (sum over params)
        beta:          Current annealing coefficient in [0, 1]
        n_data:        Number of training data points (normalises KL to per-point scale)

    Returns:
        Scalar ELBO (to be minimised)
    """
    return oggm_loss_val + beta * (kl_val / n_data)


# ---------------------------------------------------------------------------
# Stage 2 ELBO
# ---------------------------------------------------------------------------

def finetune_elbo(
    temporal_avg_loss_val,       # scalar
    glambie_loss_val,             # scalar (0.0 if no GLaMBIE data)
    kl_val,                       # scalar
    beta: float,
    n_data: int,
    glambie_weight: float = 1.0,
) -> jnp.ndarray:
    """
    ELBO = L_temporal_avg + glambie_weight * L_glambie + (beta / n_data) * KL(q || q_pretrained)

    KL is divided by n_data (n_glaciers + n_glambie_obs) for the same reason
    as in pretrain_elbo: KL is a sum over parameters, losses are means over obs.

    Both L_temporal_avg and L_glambie are already normalised per-observation
    (jnp.mean), so glambie_weight=1.0 gives equal per-observation weight to both
    datasets. Reduce glambie_weight below 1.0 if GLaMBIE dominates and per-glacier
    Hugonnet performance suffers; increase above 1.0 to prioritise regional fit.

    Args:
        temporal_avg_loss_val: Scalar temporal-avg inverse-variance MSE
        glambie_loss_val:      Scalar GLaMBIE inverse-variance MSE (0.0 if absent)
        kl_val:                Scalar KL divergence against pretrained posterior
        beta:                  Current annealing coefficient in [0, 1]
        n_data:                Number of observations (n_glaciers + n_glambie_obs)
        glambie_weight:        Relative weight of GLaMBIE loss vs Hugonnet (default 1.0)

    Returns:
        Scalar ELBO (to be minimised)
    """
    return temporal_avg_loss_val + glambie_weight * glambie_loss_val + beta * (kl_val / n_data)
