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
MM_TO_MWE: float = 1e-3      # OGGM outputs mm/yr  → MWE/yr
KGM2_TO_MWE: float = 1e-3    # Hugonnet kg/m²/yr   → MWE/yr
# GLaMBIE: Gt/yr regional sum → MWE/yr regional mean handled in data loading
# by dividing by N_glaciers (done once before training, not a constant here)


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
        values:      1-D array of shape (N,)
        segment_ids: 1-D integer array of shape (N,), values in [0, num_segments)
        num_segments: total number of segments (static)

    Returns:
        Array of shape (num_segments,) with per-segment means.
    """
    # TODO: implement using jax.ops.segment_sum
    raise NotImplementedError


def glacier_annual_mean(
    preds: jax.Array,
    glacier_ids: jax.Array,
    n_glaciers: int,
) -> jax.Array:
    """
    Average per-glacier predictions across all time steps present.

    Used for the Hugonnet 20-year mean: given predictions for all
    (glacier × year) rows in the 2000-2019 window, return the per-glacier
    temporal mean.

    Args:
        preds:       Predictions array, shape (N,)  [MWE/yr]
        glacier_ids: Integer glacier codes, shape (N,), values in [0, n_glaciers)
        n_glaciers:  Total number of glaciers (static)

    Returns:
        Per-glacier mean predictions, shape (n_glaciers,)  [MWE/yr]
    """
    # TODO: segment_mean over glacier_ids
    raise NotImplementedError


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
    # TODO: segment_mean over year_ids
    raise NotImplementedError
