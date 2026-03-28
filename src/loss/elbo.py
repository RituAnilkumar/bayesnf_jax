"""
ELBO assembly and beta annealing schedule.

Stage 1 ELBO:  L_oggm  - beta * KL(q || N(0,1))
Stage 2 ELBO:  L_hugonnet + L_glambie - beta * KL(q || q_pretrained)

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
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 1 ELBO
# ---------------------------------------------------------------------------

def pretrain_elbo(
    oggm_loss_val,   # scalar
    kl_val,          # scalar
    beta: float,
) -> jnp.ndarray:
    """
    ELBO for Stage 1 pretraining.

    ELBO = L_oggm - beta * KL(q || N(0,1))

    Args:
        oggm_loss_val: Scalar OGGM MSE loss (lower = better)
        kl_val:        Scalar KL divergence against standard normal prior
        beta:          Current annealing coefficient in [0, 1]

    Returns:
        Scalar ELBO (to be minimised — sign convention: return positive loss)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2 ELBO
# ---------------------------------------------------------------------------

def finetune_elbo(
    hugonnet_loss_val,  # scalar
    glambie_loss_val,   # scalar (0.0 if no GLaMBIE data)
    kl_val,             # scalar
    beta: float,
) -> jnp.ndarray:
    """
    ELBO for Stage 2 finetuning.

    ELBO = L_hugonnet + L_glambie - beta * KL(q || q_pretrained)

    Args:
        hugonnet_loss_val: Scalar Hugonnet inverse-variance MSE
        glambie_loss_val:  Scalar GLaMBIE inverse-variance MSE (0.0 if absent)
        kl_val:            Scalar KL divergence against pretrained posterior
        beta:              Current annealing coefficient in [0, 1]

    Returns:
        Scalar ELBO (to be minimised)
    """
    # TODO: implement
    raise NotImplementedError
