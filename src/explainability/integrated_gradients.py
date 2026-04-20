"""
Integrated Gradients attribution for BayesianNeuralField.

Computes input attributions by integrating gradients along a straight path from
a baseline input to the actual input.  For each data point we average across
n_mc independent weight samples (different rng keys), giving both a mean and a
std for each attribution — the std captures how much the importance estimate
itself varies under weight uncertainty.

Feature ordering: [year, *covariate_cols]
  - "year" is attributed as a single scalar (gradients flow through the Fourier
    time encoder, collapsing 2*n_fourier Fourier features into one number).
  - Each covariate column gets its own attribution.

Reference: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
"""

import jax
import jax.numpy as jnp
import numpy as np


def _make_grad_fn(model, heteroscedastic: bool):
    """
    Returns jax.grad of the scalar model output w.r.t. (time_scalar, cov_1d).
    Params and rng are treated as constants (not differentiated).
    """
    def _fwd(params, t_scalar, cov_1d, rng):
        t_b = jnp.atleast_1d(jnp.asarray(t_scalar, dtype=jnp.float32))
        c_b = cov_1d[None, :]  # (1, n_cov)
        out = model.apply(params, t_b, c_b, rng)
        if heteroscedastic:
            mu, _ = out
            return mu.squeeze()
        return out.squeeze()

    return jax.grad(_fwd, argnums=(1, 2))


def make_ig_batch_fn(model, n_steps: int, heteroscedastic: bool):
    """
    Build a JIT-compiled function that computes IG attributions for a batch of
    data points under a single MC weight sample (one rng key).

    Args:
        model:           BayesianNeuralField instance.
        n_steps:         Number of Riemann integration steps (50 is usually enough).
        heteroscedastic: Whether the model outputs (mu, sigma).

    Returns:
        ig_batch: jit-compiled fn with signature
            (params, t_vals, c_vals, rng_mc, t_base, c_base) -> (B, 1+n_cov)
    """
    _grad_fn = _make_grad_fn(model, heteroscedastic)

    def _ig_one_point(params, t_val, cov_1d, rng_mc, t_base, c_base):
        alphas = jnp.linspace(0.0, 1.0, n_steps + 1)

        def _step(alpha):
            t_i = t_base + alpha * (t_val - t_base)
            c_i = c_base + alpha * (cov_1d - c_base)
            g_t, g_c = _grad_fn(params, t_i, c_i, rng_mc)
            return g_t, g_c

        g_t_all, g_c_all = jax.vmap(_step)(alphas)          # (S,), (S, n_cov)
        attr_t = jnp.mean(g_t_all) * (t_val - t_base)       # scalar
        attr_c = jnp.mean(g_c_all, axis=0) * (cov_1d - c_base)  # (n_cov,)
        return jnp.concatenate([jnp.array([attr_t]), attr_c])    # (1+n_cov,)

    @jax.jit
    def ig_batch(params, t_vals, c_vals, rng_mc, t_base, c_base):
        """
        Args:
            params:   Flax params dict {"params": ...}
            t_vals:   (B,) float32 time indices
            c_vals:   (B, n_cov) float32 covariates
            rng_mc:   PRNGKey — determines this MC weight sample
            t_base:   () scalar baseline for time
            c_base:   (n_cov,) baseline for covariates
        Returns:
            (B, 1+n_cov) attributions for this weight sample
        """
        def _one(t_val, cov_1d):
            return _ig_one_point(params, t_val, cov_1d, rng_mc, t_base, c_base)

        return jax.vmap(_one)(t_vals, c_vals)

    return ig_batch


def _subsample(time_index, covariates, max_points, seed):
    """Return (t_sub, c_sub, sample_idx) subsampled to at most max_points."""
    N = len(time_index)
    rng_np = np.random.default_rng(seed)
    if N > max_points:
        idx = np.sort(rng_np.choice(N, max_points, replace=False))
    else:
        idx = np.arange(N)
    return time_index[idx].astype(np.float32), covariates[idx].astype(np.float32), idx


def _make_baselines(t_sub, c_sub, baseline):
    n_cov = c_sub.shape[1]
    if baseline == "mean":
        return (
            jnp.array(float(t_sub.mean()), dtype=jnp.float32),
            jnp.array(c_sub.mean(axis=0), dtype=jnp.float32),
        )
    return jnp.array(0.0, dtype=jnp.float32), jnp.zeros(n_cov, dtype=jnp.float32)


def _run_ig_for_params(
    ig_batch_fn,
    params: dict,
    t_arr: "jax.Array",
    c_arr: "jax.Array",
    rng: "jax.Array",
    n_mc: int,
    chunk_size: int,
    t_base: "jax.Array",
    c_base: "jax.Array",
    label: str = "",
) -> np.ndarray:
    """
    Collect raw per-MC-sample attributions for one params dict.

    Returns: (n_mc, N', 1+n_cov) numpy array.
    """
    n_sub = len(t_arr)
    all_mc: list[np.ndarray] = []
    for mc_i in range(n_mc):
        rng, rng_mc = jax.random.split(rng)
        chunks = []
        for start in range(0, n_sub, chunk_size):
            end = min(start + chunk_size, n_sub)
            chunks.append(np.array(
                ig_batch_fn(params, t_arr[start:end], c_arr[start:end], rng_mc, t_base, c_base)
            ))
        all_mc.append(np.concatenate(chunks, axis=0))
        print(f"  IG{label}: MC sample {mc_i + 1}/{n_mc} done")
    return np.stack(all_mc)   # (n_mc, N', 1+n_cov)


def compute_attributions(
    model,
    params: dict,
    time_index: np.ndarray,
    covariates: np.ndarray,
    rng: "jax.Array",
    n_mc: int = 20,
    n_steps: int = 50,
    heteroscedastic: bool = False,
    baseline: str = "zero",
    target_scaler=None,
    max_points: int = 2000,
    chunk_size: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MC-averaged Integrated Gradients for up to max_points data points.

    Returns:
        mean_attrs:  (N', 1+n_cov) mean attribution across MC samples per point.
        std_attrs:   (N', 1+n_cov) std of attribution across MC samples per point.
        sample_idx:  (N',) indices into the original arrays.
    """
    t_sub, c_sub, sample_idx = _subsample(time_index, covariates, max_points, seed)
    t_base, c_base = _make_baselines(t_sub, c_sub, baseline)
    ig_batch = make_ig_batch_fn(model, n_steps, heteroscedastic)

    raw = _run_ig_for_params(
        ig_batch, params,
        jnp.array(t_sub), jnp.array(c_sub),
        rng, n_mc, chunk_size, t_base, c_base,
    )                               # (n_mc, N', 1+n_cov)

    mean_attrs = raw.mean(axis=0)
    std_attrs  = raw.std(axis=0)

    if target_scaler is not None:
        _, t_std = target_scaler
        mean_attrs = mean_attrs * float(t_std)
        std_attrs  = std_attrs  * float(t_std)

    return mean_attrs, std_attrs, sample_idx


def compute_ensemble_attributions(
    model,
    all_params: list[dict],
    time_index: np.ndarray,
    covariates: np.ndarray,
    rng: "jax.Array",
    n_mc_per_model: int = 5,
    n_steps: int = 50,
    heteroscedastic: bool = False,
    baseline: str = "zero",
    target_scaler=None,
    max_points: int = 2000,
    chunk_size: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute IG attributions across an ensemble of models (e.g. multirun subdirs).

    For each model in all_params, draws n_mc_per_model independent weight samples,
    giving n_models * n_mc_per_model total attribution samples per data point.
    The resulting mean/std therefore capture both:
      - VI weight uncertainty  (n_mc_per_model draws within each run's posterior)
      - Structural uncertainty (variation across the ensemble of trained models)

    Args:
        all_params:      List of Flax params dicts, one per ensemble member.
        n_mc_per_model:  MC weight samples per model (total = n_models × this).
        (other args same as compute_attributions)

    Returns:
        mean_attrs:  (N', 1+n_cov) mean attribution across all (model × MC) samples.
        std_attrs:   (N', 1+n_cov) std  across all (model × MC) samples.
        sample_idx:  (N',) indices into the original arrays.
    """
    t_sub, c_sub, sample_idx = _subsample(time_index, covariates, max_points, seed)
    t_base, c_base = _make_baselines(t_sub, c_sub, baseline)
    ig_batch = make_ig_batch_fn(model, n_steps, heteroscedastic)

    t_arr = jnp.array(t_sub)
    c_arr = jnp.array(c_sub)

    all_raw: list[np.ndarray] = []
    for model_i, params in enumerate(all_params):
        rng, rng_model = jax.random.split(rng)
        raw = _run_ig_for_params(
            ig_batch, params, t_arr, c_arr,
            rng_model, n_mc_per_model, chunk_size, t_base, c_base,
            label=f" [model {model_i + 1}/{len(all_params)}]",
        )                           # (n_mc_per_model, N', 1+n_cov)
        all_raw.append(raw)

    combined = np.concatenate(all_raw, axis=0)  # (n_models*n_mc, N', 1+n_cov)
    mean_attrs = combined.mean(axis=0)
    std_attrs  = combined.std(axis=0)

    if target_scaler is not None:
        _, t_std = target_scaler
        mean_attrs = mean_attrs * float(t_std)
        std_attrs  = std_attrs  * float(t_std)

    return mean_attrs, std_attrs, sample_idx
