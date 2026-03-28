"""
Data loading and preprocessing utilities for the bayesnf_jax pipeline.

Responsibilities:
- Load and normalise OGGM targets (mm/yr → MWE/yr)
- Load temporal-average targets (already MWE/yr, per-glacier 2001-2020 means)
- Load and reshape GLaMBIE targets (wide → long, source selection logic)
- Build model-ready feature arrays (drop GLIMSId, compute time index)
- Generate CV splits (loyo, logo, loygo, train, full) from the full dataset

CV split definitions:
  held_years    — 10% of years, chosen globally (same across all regions)
  held_glaciers — 10% of glaciers, chosen per region
  loygo         — rows where year in held_years AND glacier in held_glaciers
  loyo          — rows where year in held_years AND glacier NOT in held_glaciers
  logo          — rows where year NOT in held_years AND glacier in held_glaciers
  train         — rows where year NOT in held_years AND glacier NOT in held_glaciers
  full          — all rows

Pass held_years from select_held_years() to make_cv_splits() for all regions to
ensure the same years are held out everywhere.

All temporal indices are computed as  year - T_MIN  (consistent with bnf_module).
"""

import numpy as np
import pandas as pd

from src.model.bnf_module import T_MIN

# ---------------------------------------------------------------------------
# Feature columns — fixed order used across all training/inference code
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    "CenLat", "CenLon",
    "t2m_abl_mean", "t2m_abl_std",
    "t2m_acc_mean", "t2m_acc_std",
    "tp_abl_sum",   "tp_acc_sum",
    "ssrd_abl_sum", "ssrd_acc_sum",
    "Aspect", "Zmax", "Zmed", "Slope", "Zmin", "Area",
]


# ---------------------------------------------------------------------------
# OGGM (Stage 1 — pretrain)
# ---------------------------------------------------------------------------
def load_oggm(path: str) -> pd.DataFrame:
    """
    Load OGGM targets and convert mass_balance from mm/yr to MWE/yr.

    Input columns expected: year, rgi_id, mass_balance (mm/yr)
    Returns columns:        rgi_id, year, time_index, mass_balance_mwe
    """
    df = pd.read_csv(path)
    df["mass_balance_mwe"] = df["mass_balance"] / 1000.0
    df["time_index"] = df["year"] - T_MIN
    return df[["rgi_id", "year", "time_index", "mass_balance_mwe"]]


# ---------------------------------------------------------------------------
# Features (both stages)
# ---------------------------------------------------------------------------
def load_features(
    path: str,
    feature_cols: list[str] = FEATURE_COLS,
) -> pd.DataFrame:
    """
    Load main features, drop GLIMSId, and compute time_index.

    Input columns: rgi_id, CenLat, CenLon, year, <climate cols>,
                   Aspect, Zmax, GLIMSId, Zmed, Slope, Zmin, Area
    Returns columns: rgi_id, year, time_index, <feature_cols present in file>
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["GLIMSId"], errors="ignore")
    df["time_index"] = df["year"] - T_MIN
    cols_present = [c for c in feature_cols if c in df.columns]
    return df[["rgi_id", "year", "time_index"] + cols_present]


# ---------------------------------------------------------------------------
# Temporal-average targets (Stage 2 — finetune, replaces Hugonnet)
# ---------------------------------------------------------------------------
def load_temporal_avg(path: str) -> pd.DataFrame:
    """
    Load per-glacier temporal-average mass balance targets.

    Values are already in MWE/yr. Covers the period 2001-2020 per glacier.

    Input columns expected: rgi_id, start_date, end_date,
                            avg_mb_mwe, avg_mb_gt, uncertainty_mwe, uncertainty_gt
    Returns columns:        rgi_id, start_date, end_date, avg_mb_mwe, uncertainty_mwe
    """
    df = pd.read_csv(path)
    return df[["rgi_id", "start_date", "end_date", "avg_mb_mwe", "uncertainty_mwe"]]


# ---------------------------------------------------------------------------
# GLaMBIE regional targets (Stage 2 — finetune)
# ---------------------------------------------------------------------------
def load_glambie(path: str) -> pd.DataFrame:
    """
    Load GLaMBIE regional targets, reshaping from wide to long format.

    Source selection:
      - Use altimetry and/or gravimetry wherever the value + error are both non-NaN.
      - Fall back to combined for ALL rows only if the entire file contains zero
        valid altimetry and zero valid gravimetry datapoints. A single valid primary
        observation anywhere in the file disables the combined fallback entirely.
      - Rows with no valid source are dropped.

    Year is derived from end_date by flooring the fractional year to an integer.

    Input columns: region, start_date, end_date,
                   combined_gt, combined_gt_errors,
                   altimetry_gt, altimetry_gt_errors,
                   gravimetry_gt, gravimetry_gt_errors
    (column suffix '_gt' is historical; values are in MWE/yr)

    Returns columns: region, year, source, value_mwe, error_mwe
    """
    df = pd.read_csv(path)
    df["year"] = np.floor(df["end_date"]).astype(int)

    # Decide once whether any primary source data exists across the whole file.
    # Combined is only used as a last resort when the entire file has zero
    # altimetry and zero gravimetry datapoints.
    any_primary = (
        (df["altimetry_gt"].notna() & df["altimetry_gt_errors"].notna()).any()
        or
        (df["gravimetry_gt"].notna() & df["gravimetry_gt_errors"].notna()).any()
    )

    records = []
    for _, row in df.iterrows():
        alt_ok  = pd.notna(row.get("altimetry_gt"))  and pd.notna(row.get("altimetry_gt_errors"))
        grav_ok = pd.notna(row.get("gravimetry_gt")) and pd.notna(row.get("gravimetry_gt_errors"))

        if alt_ok:
            records.append({
                "region": row["region"], "year": row["year"],
                "source": "altimetry",
                "value_mwe": row["altimetry_gt"], "error_mwe": row["altimetry_gt_errors"],
            })
        if grav_ok:
            records.append({
                "region": row["region"], "year": row["year"],
                "source": "gravimetry",
                "value_mwe": row["gravimetry_gt"], "error_mwe": row["gravimetry_gt_errors"],
            })
        if not any_primary and pd.notna(row.get("combined_gt")) and pd.notna(row.get("combined_gt_errors")):
            records.append({
                "region": row["region"], "year": row["year"],
                "source": "combined",
                "value_mwe": row["combined_gt"], "error_mwe": row["combined_gt_errors"],
            })

    return pd.DataFrame(
        records,
        columns=["region", "year", "source", "value_mwe", "error_mwe"],
    )


# ---------------------------------------------------------------------------
# CV split generation
# ---------------------------------------------------------------------------

def select_held_years(
    all_years: list[int],
    frac: float = 0.1,
    seed: int = 42,
) -> list[int]:
    """
    Select a globally consistent set of held-out years for loyo evaluation.

    Call this once with the union of years across all regions, then pass the
    result to make_cv_splits() for every region so the same years are held out
    everywhere.

    Args:
        all_years: Sorted list of all integer years present in the dataset.
        frac:      Fraction of years to hold out (default 0.1 = 10%).
        seed:      RNG seed — fix this across the project for reproducibility.

    Returns:
        Sorted list of held-out year integers.
    """
    rng = np.random.default_rng(seed)
    years = np.array(sorted(set(all_years)))
    n_held = max(1, int(round(len(years) * frac)))
    held = rng.choice(years, size=n_held, replace=False)
    return sorted(held.tolist())


def make_cv_splits(
    merged_df: pd.DataFrame,
    held_years: list[int],
    held_glacier_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Generate CV splits from a merged (features + OGGM targets) DataFrame.

    Args:
        merged_df:          DataFrame with at least columns: rgi_id, year.
                            Typically the result of merging load_features() and
                            load_oggm() on (rgi_id, year).
        held_years:         Output of select_held_years() — same list for all regions.
        held_glacier_frac:  Fraction of glaciers to hold out for logo (default 0.1).
        seed:               RNG seed for glacier selection.

    Returns:
        Dict with keys 'train', 'loyo', 'logo', 'loygo', 'full'.
        Each value is a subset of merged_df (same columns, reset index).

    Split definitions:
        loygo = held_years ∩ held_glaciers  (both held out)
        loyo  = held_years only             (year held, glacier not)
        logo  = held_glaciers only          (glacier held, year not)
        train = neither held
        full  = all rows (no filtering)
    """
    rng = np.random.default_rng(seed)
    all_glaciers = np.array(sorted(merged_df["rgi_id"].unique()))
    n_held_g = max(1, int(round(len(all_glaciers) * held_glacier_frac)))
    held_glaciers = set(rng.choice(all_glaciers, size=n_held_g, replace=False).tolist())
    held_years_set = set(held_years)

    year_held    = merged_df["year"].isin(held_years_set)
    glacier_held = merged_df["rgi_id"].isin(held_glaciers)

    return {
        "full":  merged_df.reset_index(drop=True),
        "train": merged_df[ ~year_held & ~glacier_held].reset_index(drop=True),
        "loygo": merged_df[  year_held &  glacier_held].reset_index(drop=True),
        "loyo":  merged_df[  year_held & ~glacier_held].reset_index(drop=True),
        "logo":  merged_df[ ~year_held &  glacier_held].reset_index(drop=True),
    }


def merge_features_targets(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join features and OGGM targets on (rgi_id, year).

    Args:
        features_df: Output of load_features()
        targets_df:  Output of load_oggm()

    Returns:
        Merged DataFrame with all feature columns plus mass_balance_mwe.
        time_index is taken from features_df (both should agree).
    """
    return features_df.merge(
        targets_df[["rgi_id", "year", "mass_balance_mwe"]],
        on=["rgi_id", "year"],
        how="inner",
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assemble model-ready numpy arrays
# ---------------------------------------------------------------------------
def build_model_inputs(
    features_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Convert a features DataFrame into numpy arrays for model input.

    Args:
        features_df:  Output of load_features() — must contain time_index,
                      rgi_id, and the requested feature columns.
        feature_cols: Ordered list of covariate columns to include.

    Returns:
        time_index       shape (N,)         int32
        covariates       shape (N, n_feats) float32
        rgi_ids          shape (N,)         object  (rgi_id strings)
        feature_cols_used                   list of columns actually present
    """
    cols_present = [c for c in feature_cols if c in features_df.columns]
    time_index = features_df["time_index"].to_numpy(dtype=np.int32)
    covariates = features_df[cols_present].to_numpy(dtype=np.float32)
    rgi_ids    = features_df["rgi_id"].to_numpy()
    return time_index, covariates, rgi_ids, cols_present
