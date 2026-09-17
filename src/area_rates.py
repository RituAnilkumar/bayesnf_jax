"""
src/area_rates.py

Per-region linear glacier-area change rates, derived from GLaMBIE's own
`calendar_years` regional area series (GlaMBIE_Data_DOI_10.5904_wgms-glambie-
2024-07/glambie_results_20240716/calendar_years/{n}_{name}.csv, 2000-2024
annual `glacier_area` per RGI region), fitted once and embedded directly
below (_AREA_RATES) rather than read from a data file, so downstream code
does not depend on that external path (a local OneDrive mount, not reachable
from the HPC/scratch environment the actual pipeline runs in) and isn't
affected by data_for_model/ being blanket-gitignored for the pipeline's
actual (large, regenerable) data files.

Used to build a "variable area" alternative to the pipeline's usual "fixed
area" MWE->Gt conversion (which uses a single constant area per region, from
main_features_{region}.csv, applied to every year). Every site that produces
a Gt series from an MWE series now also produces a second, variable-area
version using dynamic_area_km2() below, plus a comparison plot — see
CONTEXT.md, "Fixed vs. variable area Gt conversion" for the full reasoning
trail and the assumptions listed there.

IMPORTANT — read before trusting these rates as independent information:
GLaMBIE's own glacier_area series is a perfectly linear function of year for
every region (R²=1.000 in the fit below) — this is not us fitting a noisy
independent measurement, it's us reading off a linear relationship GLaMBIE's
own methodology already assumes (almost certainly a two-point interpolation/
extrapolation between RGI inventory snapshots, not an independent yearly
area measurement). Treat "the GLaMBIE-prescribed rate", not "our estimate".

Assumptions and known limitations (explicitly acknowledged, not silently
extrapolated past their validity):
  - The fit window is 2000-2024. Extrapolating forward or backward assumes
    the same linear rate held outside that window. Checked empirically: no
    region's area goes non-positive before ~2085 (r16, the earliest),
    comfortably beyond this pipeline's actual prediction range (~2025,
    extended to at most 2050 by a separate, deferred "quadratic extension"
    analysis) — so no floor/clamp logic is implemented.
  - Extrapolating backward to 1940 is an acknowledged limitation, not a
    validated assumption: real glacier retreat has generally accelerated
    over time (particularly a step change around the 1980s-1990s in most
    regions), so a constant 2000-2024 rate applied back to 1940 likely
    overstates cumulative area loss in the earlier decades. This is exactly
    why every Gt output gets BOTH a fixed-area and a variable-area version
    side by side, rather than only the variable-area one — the true answer
    is expected to lie somewhere between the two, not to equal either.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

MWE_TO_GT = 1e-3   # Gt per (MWE/yr * km^2) — same constant used throughout the repo

# Fitted linear area-change rate per region: area(year) = slope*year + intercept.
# Fit against each GLaMBIE calendar-year period's midpoint, 2000-2024 (see module
# docstring for source and the R²=1.000-for-every-region caveat). Embedded directly
# here (not a data_for_model/ CSV) because that whole directory is blanket-gitignored
# for the pipeline's actual (large, regenerable) data files, and git's ignore
# semantics don't allow excepting a single file out of an already-ignored directory
# once the directory itself is matched by a trailing-slash pattern — embedding this
# tiny, fixed, 19-row table avoids that entirely and guarantees it travels with the
# code to the HPC/scratch environment via git, same as any other source file.
_AREA_RATES: dict[str, tuple[float, float]] = {
    # region: (slope_km2_per_yr, intercept_km2)
    "r01": (-416.28,    922615.24),
    "r02": (-78.4296,   171461.6296),
    "r03": (-73.5777,   252192.8223),
    "r04": (-32.7104,   106308.80),
    "r05": (-735.6794,  1561075.80),
    "r06": (-39.816,    90652.184),
    "r07": (-88.2934,   211075.5604),
    "r08": (-7.9623,    18889.5246),
    "r09": (-41.2736,   134180.4736),
    "r10": (-10.363,    23218.904),
    "r11": (-19.4556,   41061.5668),
    "r12": (-6.9271,    15140.4187),
    "r13": (-88.7454,   227237.527),
    "r14": (-120.8448,  275257.60),
    "r15": (-69.2498,   153441.3494),
    "r16": (-27.8579,   58084.6579),
    "r17": (-52.9722,   135373.40),
    "r18": (-8.0178,    17021.2084),
    "r19": (-358.7409,  845326.4274),
}
AREA_RATES_FIT_YEARS = (2000.0, 2024.0)


@lru_cache(maxsize=1)
def load_area_rates() -> pd.DataFrame:
    """The per-region linear area-change-rate table, indexed by region, as a DataFrame."""
    df = pd.DataFrame(
        [(r, s, i) for r, (s, i) in _AREA_RATES.items()],
        columns=["region", "slope_km2_per_yr", "intercept_km2"],
    )
    return df.set_index("region")


def dynamic_area_km2(region: str, years: np.ndarray) -> np.ndarray:
    """
    Linearly-extrapolated regional glacier area (km^2) for each year in `years`,
    using the GLaMBIE-prescribed rate for `region` (e.g. "r13").

    No floor/clamp is applied — see module docstring for why that's safe over
    this pipeline's actual year range, and why extrapolation before 2000 is an
    acknowledged (not corrected) limitation.
    """
    if region not in _AREA_RATES:
        raise KeyError(f"No GLaMBIE area-change rate available for region '{region}'. "
                        f"Available regions: {sorted(_AREA_RATES)}")
    slope, intercept = _AREA_RATES[region]
    years = np.asarray(years, dtype=float)
    return slope * years + intercept


def variable_area_scale(region: str, years: np.ndarray) -> np.ndarray:
    """
    dynamic_area_km2(region, years) * MWE_TO_GT — drop-in replacement for the
    `total_area_km2 * 1e-3` fixed-area scale factor used throughout the repo's
    MWE-to-Gt conversions.
    """
    return dynamic_area_km2(region, years) * MWE_TO_GT


def plot_fixed_vs_variable_area(
    years: np.ndarray,
    median_fixed: np.ndarray,
    std_fixed: np.ndarray,
    median_variable: np.ndarray,
    std_variable: np.ndarray,
    output_path: Path,
    title: str = "Fixed vs. variable glacier area — Gt/yr",
    ylabel: str = "Gt/yr",
    sigma_mult: float = 2.0,
) -> None:
    """
    Two-band overlay comparing the fixed-area (current pipeline default) and
    variable-area (GLaMBIE-prescribed linear area-change rate) Gt conversions.
    The true value is expected to lie somewhere between the two bands, not to
    equal either — see area_rates.py module docstring.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(years, median_fixed - sigma_mult * std_fixed, median_fixed + sigma_mult * std_fixed,
                    alpha=0.20, color="steelblue", label=f"Fixed area (±{sigma_mult:g}σ)")
    ax.plot(years, median_fixed, color="steelblue", lw=1.6)
    ax.fill_between(years, median_variable - sigma_mult * std_variable, median_variable + sigma_mult * std_variable,
                    alpha=0.20, color="darkorange", label=f"Variable area (±{sigma_mult:g}σ)")
    ax.plot(years, median_variable, color="darkorange", lw=1.6)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")
