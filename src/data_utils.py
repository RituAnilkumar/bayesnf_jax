"""
Data loading and preprocessing utilities for the bayesnf_jax pipeline.

Responsibilities:
- Load and normalise OGGM targets (mm/yr → MWE/yr)
- Load temporal-average targets (already MWE/yr, per-glacier 2001-2020 means)
- Load and reshape GLaMBIE targets (wide → long, source selection logic)
- Build model-ready feature arrays (drop GLIMSId, compute time index)

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

    Source selection per row:
      - Use altimetry and/or gravimetry wherever the value + error are both non-NaN.
      - Fall back to combined ONLY if both altimetry and gravimetry are NaN for that row.
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

    records = []
    for _, row in df.iterrows():
        alt_val  = row.get("altimetry_gt")
        alt_err  = row.get("altimetry_gt_errors")
        grav_val = row.get("gravimetry_gt")
        grav_err = row.get("gravimetry_gt_errors")
        comb_val = row.get("combined_gt")
        comb_err = row.get("combined_gt_errors")

        alt_ok  = pd.notna(alt_val)  and pd.notna(alt_err)
        grav_ok = pd.notna(grav_val) and pd.notna(grav_err)

        if alt_ok or grav_ok:
            if alt_ok:
                records.append({
                    "region": row["region"], "year": row["year"],
                    "source": "altimetry",
                    "value_mwe": alt_val, "error_mwe": alt_err,
                })
            if grav_ok:
                records.append({
                    "region": row["region"], "year": row["year"],
                    "source": "gravimetry",
                    "value_mwe": grav_val, "error_mwe": grav_err,
                })
        elif pd.notna(comb_val) and pd.notna(comb_err):
            records.append({
                "region": row["region"], "year": row["year"],
                "source": "combined",
                "value_mwe": comb_val, "error_mwe": comb_err,
            })
        # else: no valid source for this row — skip

    return pd.DataFrame(
        records,
        columns=["region", "year", "source", "value_mwe", "error_mwe"],
    )


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
