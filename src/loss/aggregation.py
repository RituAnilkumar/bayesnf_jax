"""
Aggregation helpers and unit conversion constants.

All values in the pipeline are expressed in MWE/yr (metres water equivalent per year).
Conversions happen once here — never inline in loss functions.

segment_mean() and glacier_annual_mean() must stay inside the JIT boundary.
The integer segment codes (from pd.factorize) are computed once outside the
training loop and passed as static arrays.
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
MM_TO_MWE: float = 1e-3      # OGGM outputs mm/yr → MWE/yr
# All other inputs (temporal_avg, GLaMBIE) are already in MWE/yr after data_utils loading


# ---------------------------------------------------------------------------
# segment_sum / segment_mean helpers
# ---------------------------------------------------------------------------

def segment_mean(
    values: jax.Array,
    segment_ids: jax.Array,
    num_segments: int,
) -> jax.Array:
    """
    Compute the mean of `values` within each segment defined by `segment_ids`.

    Equivalent to a grouped mean over integer codes from pd.factorize().
    Stays inside JIT — no Python control flow over dynamic shapes.

    Args:
        values:       1-D array of shape (N,)
        segment_ids:  1-D integer array of shape (N,), values in [0, num_segments)
        num_segments: total number of segments (static)

    Returns:
        Array of shape (num_segments,) with per-segment means.
    """
    sums   = jax.ops.segment_sum(values,      segment_ids, num_segments)
    counts = jax.ops.segment_sum(jnp.ones_like(values), segment_ids, num_segments)
    return sums / counts


def glacier_annual_mean(
    preds: jax.Array,
    glacier_ids: jax.Array,
    n_glaciers: int,
) -> jax.Array:
    """
    Average per-glacier predictions across all time steps present.

    Used for the temporal-avg loss: given predictions for all
    (glacier × year) rows in the period window, return the per-glacier
    temporal mean.

    Args:
        preds:       Predictions array, shape (N,)  [MWE/yr]
        glacier_ids: Integer glacier codes, shape (N,), values in [0, n_glaciers)
        n_glaciers:  Total number of glaciers (static)

    Returns:
        Per-glacier mean predictions, shape (n_glaciers,)  [MWE/yr]
    """
    return segment_mean(preds, glacier_ids, n_glaciers)


def regional_annual_mean(
    preds: jax.Array,
    year_ids: jax.Array,
    n_years: int,
) -> jax.Array:
    """
    Average per-glacier predictions across all glaciers for each year.

    Used for the GLaMBIE annual mean: given predictions for all
    (glacier × year) rows, return the cross-glacier mean per year.

    Args:
        preds:    Predictions array, shape (N,)  [MWE/yr]
        year_ids: Integer year codes, shape (N,), values in [0, n_years)
        n_years:  Total number of distinct years (static)

    Returns:
        Per-year regional mean predictions, shape (n_years,)  [MWE/yr]
    """
    return segment_mean(preds, year_ids, n_years)
