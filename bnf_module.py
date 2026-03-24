"""
Bayesian Neural Field (BNF) module implemented in Flax.

Architecture:
    - Fixed random Fourier feature encoding of the time coordinate (year index)
    - All other features (spatial, climate, geometry) concatenated directly
    - MLP with mean-field variational inference over all weights
    - One model instance per RGI region

Temporal index convention:
    time_index = year - T_MIN, where T_MIN = 1941
    This ensures consistent indexing across training (2000-2019),
    historical inference (1941-present), and future scenarios (to ~2100).

Fourier encoding:
    Fixed random frequency matrix W drawn once at model initialisation
    (non-learned, following BayesNF convention).
    Encoding: [sin(2π·W·t), cos(2π·W·t)] concatenated → 2*n_fourier features.
    Frequencies are log-uniformly spaced to cover periods from 2 years
    (Nyquist) up to the full temporal range (T_MAX - T_MIN years).

Mean-field VI:
    Each weight and bias in the MLP has a learned mean (mu) and
    log standard deviation (log_sigma). During the forward pass, weights
    are sampled via the reparameterisation trick:
        w = mu + sigma * eps, eps ~ N(0, 1)
    This gives stochastic forward passes for ELBO training and
    Monte Carlo predictive distributions at inference time.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import glorot_uniform, zeros_init
from typing import Sequence
import numpy as np

# ---------------------------------------------------------------------------
# Global temporal index constants
# ---------------------------------------------------------------------------
T_MIN: int = 1941          # year index 0
T_MAX: int = 2100          # year index 159 — covers future scenarios
T_RANGE: int = T_MAX - T_MIN  # 159


# ---------------------------------------------------------------------------
# Fourier time encoder
# ---------------------------------------------------------------------------
class FourierTimeEncoder(nn.Module):
    """
    Fixed random Fourier feature encoding of a scalar time index.

    Frequencies are sampled log-uniformly to cover periods from
    2 years (Nyquist for annual data) up to T_RANGE years.
    The frequency matrix W is fixed at initialisation and never updated.

    Args:
        n_fourier: Number of Fourier features. Output dimension = 2 * n_fourier
                   (sin and cos for each frequency).
        seed:      RNG seed for the fixed frequency matrix.
    """
    n_fourier: int = 8    # default matches BayesNF's small default
    seed: int = 42

    def setup(self):
        # Sample frequencies log-uniformly between 1/T_RANGE and 1/2 (Nyquist)
        # These are in units of cycles per year
        rng = np.random.default_rng(self.seed)
        log_low  = np.log(1.0 / T_RANGE)
        log_high = np.log(1.0 / 2.0)
        freqs = np.exp(rng.uniform(log_low, log_high, size=(self.n_fourier,)))
        # W shape: (n_fourier,) — scalar time input, so no projection matrix needed
        self.W = jnp.array(freqs, dtype=jnp.float32)

    def __call__(self, time_index: jax.Array) -> jax.Array:
        """
        Args:
            time_index: shape (batch,) or scalar — integer year indices (year - T_MIN)

        Returns:
            Fourier features of shape (batch, 2 * n_fourier)
        """
        # Ensure float and add feature dim: (batch, 1)
        t = jnp.asarray(time_index, dtype=jnp.float32)
        if t.ndim == 0:
            t = t[None]  # scalar → (1,)
        t = t[:, None]   # (batch, 1)

        # Compute 2π·W·t: (batch, n_fourier)
        angles = 2.0 * jnp.pi * t * self.W[None, :]  # broadcast over batch

        # Concatenate sin and cos: (batch, 2 * n_fourier)
        return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


# ---------------------------------------------------------------------------
# Mean-field variational dense layer
# ---------------------------------------------------------------------------
class VIDense(nn.Module):
    """
    Dense layer with mean-field variational inference over weights and biases.

    Parameters mu and log_sigma are learned. During the forward pass,
    weights are sampled via the reparameterisation trick so gradients
    flow through mu and log_sigma.

    Args:
        features:    Output dimension.
        use_bias:    Whether to include a bias term (default True).
    """
    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, rng: jax.Array) -> jax.Array:
        """
        Args:
            x:   Input array of shape (batch, in_features)
            rng: JAX PRNGKey for weight sampling

        Returns:
            Output array of shape (batch, features)
        """
        in_features = x.shape[-1]

        # Weight variational parameters: mu and log_sigma
        # Initialise mu with Glorot uniform (standard), log_sigma to small negative
        # value so initial sigma ≈ exp(-3) ≈ 0.05 (tight initial posterior)
        w_mu = self.param(
            'w_mu',
            glorot_uniform(),
            (in_features, self.features)
        )
        w_log_sigma = self.param(
            'w_log_sigma',
            nn.initializers.constant(-3.0),
            (in_features, self.features)
        )

        # Sample weights via reparameterisation
        rng, rng_w = jax.random.split(rng)
        eps_w = jax.random.normal(rng_w, shape=(in_features, self.features))
        w = w_mu + jnp.exp(w_log_sigma) * eps_w

        # Linear transform
        out = x @ w

        if self.use_bias:
            b_mu = self.param(
                'b_mu',
                zeros_init(),
                (self.features,)
            )
            b_log_sigma = self.param(
                'b_log_sigma',
                nn.initializers.constant(-3.0),
                (self.features,)
            )
            rng, rng_b = jax.random.split(rng)
            eps_b = jax.random.normal(rng_b, shape=(self.features,))
            b = b_mu + jnp.exp(b_log_sigma) * eps_b
            out = out + b

        return out


# ---------------------------------------------------------------------------
# Bayesian Neural Field
# ---------------------------------------------------------------------------
class BayesianNeuralField(nn.Module):
    """
    Bayesian Neural Field for glacier mass balance prediction.

    The network takes a time index and a vector of glacier covariates,
    encodes the time index via fixed Fourier features, concatenates with
    the covariates, and passes through a mean-field VI MLP to produce
    a scalar mass balance prediction (MWE/yr).

    Args:
        hidden_sizes: Sequence of hidden layer widths, e.g. (64, 64)
        n_fourier:    Number of Fourier time features (output: 2*n_fourier)
        fourier_seed: RNG seed for fixed frequency matrix
    """
    hidden_sizes: Sequence[int] = (64, 64)
    n_fourier: int = 8
    fourier_seed: int = 42

    def setup(self):
        self.time_encoder = FourierTimeEncoder(
            n_fourier=self.n_fourier,
            seed=self.fourier_seed
        )
        # Build VI dense layers: hidden layers + scalar output layer
        self.hidden_layers = [
            VIDense(features=h) for h in self.hidden_sizes
        ]
        self.output_layer = VIDense(features=1)

    def __call__(
        self,
        time_index: jax.Array,   # shape (batch,)
        covariates: jax.Array,    # shape (batch, n_covariates)
        rng: jax.Array,           # PRNGKey for weight sampling
    ) -> jax.Array:
        """
        Forward pass — single weight sample (one reparameterisation draw).

        Args:
            time_index: Integer year indices, shape (batch,)
            covariates: Non-time input features, shape (batch, n_covariates)
            rng:        PRNGKey; consumed layer by layer

        Returns:
            Predicted mass balance, shape (batch,)
        """
        # Encode time coordinate
        time_features = self.time_encoder(time_index)  # (batch, 2*n_fourier)

        # Concatenate time encoding with covariates
        x = jnp.concatenate([time_features, covariates], axis=-1)  # (batch, 2*n_fourier + n_cov)

        # Pass through hidden layers with ReLU activations
        for layer in self.hidden_layers:
            rng, rng_layer = jax.random.split(rng)
            x = nn.relu(layer(x, rng_layer))

        # Output layer — no activation (regression)
        rng, rng_out = jax.random.split(rng)
        out = self.output_layer(x, rng_out)  # (batch, 1)

        return out.squeeze(-1)  # (batch,)

    def mc_predict(
        self,
        time_index: jax.Array,
        covariates: jax.Array,
        rng: jax.Array,
        n_samples: int = 100,
    ) -> jax.Array:
        """
        Monte Carlo predictive distribution via repeated forward passes,
        each with an independent weight sample.

        Args:
            time_index: shape (batch,)
            covariates: shape (batch, n_covariates)
            rng:        PRNGKey
            n_samples:  Number of MC samples

        Returns:
            Predictions of shape (n_samples, batch)
        """
        rngs = jax.random.split(rng, n_samples)

        def single_sample(rng_i):
            return self.__call__(time_index, covariates, rng_i)

        return jax.vmap(single_sample)(rngs)  # (n_samples, batch)


# ---------------------------------------------------------------------------
# KL divergence utilities
# ---------------------------------------------------------------------------
def kl_gaussian(
    mu_q: jax.Array,
    log_sigma_q: jax.Array,
    mu_p: jax.Array,
    log_sigma_p: jax.Array,
) -> jax.Array:
    """
    Analytic KL divergence KL(q || p) for diagonal Gaussians.

    KL(N(mu_q, sigma_q²) || N(mu_p, sigma_p²)) =
        log(sigma_p/sigma_q) + (sigma_q² + (mu_q - mu_p)²) / (2*sigma_p²) - 1/2

    Args:
        mu_q, log_sigma_q: Variational posterior parameters
        mu_p, log_sigma_p: Prior parameters
                           For Stage 1: mu_p=0, log_sigma_p=0 (standard normal)
                           For Stage 2: mu_p, log_sigma_p from pretrained posterior

    Returns:
        Scalar total KL (summed over all parameters)
    """
    sigma_q = jnp.exp(log_sigma_q)
    sigma_p = jnp.exp(log_sigma_p)

    kl = (
        log_sigma_p - log_sigma_q
        + (sigma_q**2 + (mu_q - mu_p)**2) / (2.0 * sigma_p**2)
        - 0.5
    )
    return jnp.sum(kl)


def compute_total_kl(
    params: dict,
    prior_mu: dict,
    prior_log_sigma: dict,
) -> jax.Array:
    """
    Compute total KL divergence over all VI parameters in the model.

    Traverses the params pytree and pairs each (w_mu, w_log_sigma) with
    the corresponding prior parameters.

    Args:
        params:           Current variational parameters (flax params dict)
        prior_mu:         Prior means (same pytree structure as params)
        prior_log_sigma:  Prior log-sigmas (same pytree structure as params)

    Returns:
        Scalar total KL
    """
    # Flatten pytrees to lists of leaves with matching structure
    leaves_q    = jax.tree_util.tree_leaves(params)
    leaves_p_mu = jax.tree_util.tree_leaves(prior_mu)
    leaves_p_ls = jax.tree_util.tree_leaves(prior_log_sigma)

    total_kl = jnp.array(0.0)
    # Parameters come in pairs: (mu, log_sigma) interleaved by flax's param ordering
    # We iterate over matching leaves assuming pytree structure is identical
    # Each VIDense layer contributes: w_mu, w_log_sigma, b_mu, b_log_sigma
    # We compute KL for mu-paired leaves (mu_q vs mu_p) and log_sigma-paired leaves
    for q_leaf, p_mu_leaf, p_ls_leaf in zip(leaves_q, leaves_p_mu, leaves_p_ls):
        # Each leaf is either a mu or log_sigma array — KL treats them as a
        # Gaussian parameter pair at the array level.
        # Since flax stores mu and log_sigma as separate named params,
        # we compute KL only on mu leaves paired with their log_sigma leaves.
        # compute_total_kl is called with the full params dict split into
        # mu and log_sigma sub-dicts by extract_vi_params() below.
        total_kl = total_kl + kl_gaussian(q_leaf, p_ls_leaf, p_mu_leaf, p_ls_leaf)

    return total_kl


def extract_vi_params(params: dict) -> tuple[dict, dict]:
    """
    Split a BayesianNeuralField params dict into separate mu and log_sigma dicts.

    Flax stores VI parameters as named leaves (w_mu, w_log_sigma, b_mu, b_log_sigma).
    This function separates them so KL can be computed cleanly.

    Args:
        params: Full flax params dict from model.init()

    Returns:
        (mu_dict, log_sigma_dict) with the same pytree structure,
        where mu_dict contains only *_mu leaves and log_sigma_dict
        contains only *_log_sigma leaves.
    """
    def _select(params, suffix):
        """Recursively filter leaves whose key ends with suffix."""
        if isinstance(params, dict):
            return {
                k: _select(v, suffix)
                for k, v in params.items()
                if isinstance(v, dict) or k.endswith(suffix)
            }
        return params

    mu_dict       = _select(params, '_mu')
    log_sigma_dict = _select(params, '_log_sigma')
    return mu_dict, log_sigma_dict


# ---------------------------------------------------------------------------
# Convenience: standard normal prior (for Stage 1 pretraining)
# ---------------------------------------------------------------------------
def make_standard_normal_prior(params: dict) -> tuple[dict, dict]:
    """
    Construct a standard normal prior (mu=0, log_sigma=0) matching
    the structure of the given params dict.

    Used in Stage 1 pretraining where the prior is N(0, 1).

    Args:
        params: Full flax params dict

    Returns:
        (prior_mu, prior_log_sigma) with all zeros, matching params structure
    """
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return zeros, zeros  # mu=0, log_sigma=0 → sigma=exp(0)=1


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import jax

    # Instantiate model
    model = BayesianNeuralField(
        hidden_sizes=(64, 64),
        n_fourier=8,
    )

    # Dummy data — 10 glaciers, 8 covariates
    rng = jax.random.PRNGKey(0)
    rng, rng_init, rng_fwd = jax.random.split(rng, 3)

    batch_size   = 10
    n_covariates = 8  # CenLon, CenLat, Slope, Area, Aspect, Zmed, Zmin, Zmax

    time_index = jnp.array([59, 60, 61, 62, 63, 64, 65, 66, 67, 68])  # years 2000-2009
    covariates = jax.random.normal(rng, shape=(batch_size, n_covariates))

    # Initialise parameters
    params = model.init(rng_init, time_index, covariates, rng_fwd)
    print("Parameter structure:")
    print(jax.tree_util.tree_map(lambda x: x.shape, params))

    # Single forward pass
    rng, rng_fwd2 = jax.random.split(rng)
    preds = model.apply(params, time_index, covariates, rng_fwd2)
    print(f"\nSingle forward pass output shape: {preds.shape}")
    print(f"Predictions: {preds}")

    # MC predictions
    rng, rng_mc = jax.random.split(rng)
    mc_preds = model.apply(
        params, time_index, covariates, rng_mc, n_samples=50,
        method=model.mc_predict
    )
    print(f"\nMC predictions shape: {mc_preds.shape}")  # (50, 10)
    print(f"Predictive mean:  {jnp.mean(mc_preds, axis=0)}")
    print(f"Predictive std:   {jnp.std(mc_preds, axis=0)}")

    # KL with standard normal prior
    prior_mu, prior_log_sigma = make_standard_normal_prior(params['params'])
    mu_dict, log_sigma_dict   = extract_vi_params(params['params'])
    print(f"\nParameter pytree keys: {list(params['params'].keys())}")
