"""
src/ensemble_uncertainty_time_encoding.py

Variant of ensemble_uncertainty_split.py for sweeps that include
model.use_time_encoding=true,false as a sweep axis, with model.heteroscedastic=true fixed.

Reads use_time_encoding from each run's .hydra/config.yaml, partitions all completed
runs into two groups, selects the top-N (default 5) within each group by a configurable
selection metric (default: glambie_rmse; alternatives: loyo_rmse, loyo_r2), and writes
separate ensemble outputs using equal weights across the top-N:

    {output_dir}/time_encoding/    — runs where model.use_time_encoding=true
    {output_dir}/no_time_encoding/ — runs where model.use_time_encoding=false

Within each group, full uncertainty decomposition is computed:
    std_epistemic  = sqrt( Σ_k (1/K) * epistemic_std_k² )
    std_aleatoric  = sqrt( Σ_k (1/K) * aleatoric_std_k² )   (always present; heteroscedastic=true)
    std_structural = sqrt( Σ_k (1/K) * (mean_k - mu_ensemble)² )
    std_total      = sqrt( std_epistemic² + std_aleatoric² + std_structural² )

Usage:
    python src/ensemble_uncertainty_time_encoding.py \\
        --multirun_root /scratch/.../multirun/r03_12345 \\
        --output_dir outputs/ensemble/r03

    python src/ensemble_uncertainty_time_encoding.py \\
        --config conf/config_ensemble_uncertainty.yaml \\
        --multirun_root /scratch/.../multirun/r03_12345 \\
        --output_dir outputs/ensemble/r03
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

from src.ensemble_common import _run_top_n_group, _minmax_norm_series
from src.hyperparam_tuning import build_results_df


# ---------------------------------------------------------------------------
# Read use_time_encoding flag from a run's Hydra config
# ---------------------------------------------------------------------------

def _read_use_time_encoding(run_dir: Path) -> bool | None:
    """
    Read model.use_time_encoding from .hydra/config.yaml for a single run.

    Returns True/False if found, None if config is missing or the key is absent
    (treated as True — time encoding on by default).
    """
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path) as fh:
            full_cfg = yaml.safe_load(fh)
        val = full_cfg.get("model", {}).get("use_time_encoding", True)
        return bool(val)
    except Exception as exc:
        warnings.warn(f"Could not read {cfg_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ensemble_time_encoding(cfg: dict) -> None:
    """
    Split a sweep by use_time_encoding and run top-N ensemble for each group.

    Reads model.use_time_encoding from each run's .hydra/config.yaml, partitions
    runs into two groups (True/False), selects top_n by glambie_rmse within each,
    and runs the full ensemble pipeline independently for each group.
    """
    multirun_root = Path(cfg["multirun_root"])
    output_dir    = Path(cfg["output_dir"])

    test_years                = list(cfg.get("glambie_test_years", [2020, 2021, 2022, 2023, 2024]))
    min_runs                  = int(cfg.get("min_runs_per_region", 1))
    top_n                     = int(cfg.get("top_n", 5))
    selection_metric          = str(cfg.get("selection_metric", "glambie_rmse"))
    loyo_r2_min               = cfg.get("loyo_r2_min", 0.1)
    logo_r2_min               = cfg.get("logo_r2_min", 0.0)
    composite_loyo_r2_weight  = float(cfg.get("composite_loyo_r2_weight", 0.5))
    if loyo_r2_min is not None:
        loyo_r2_min = float(loyo_r2_min)
    if logo_r2_min is not None:
        logo_r2_min = float(logo_r2_min)

    valid_metrics = {"glambie_rmse", "loyo_rmse", "loyo_r2", "composite"}
    if selection_metric not in valid_metrics:
        raise ValueError(f"selection_metric must be one of {valid_metrics}, got '{selection_metric}'")

    print(f"\n=== Ensemble split by use_time_encoding: {multirun_root.name} ===")
    print(f"  Selecting top {top_n} runs per group by {selection_metric}"
          + (f" over GLaMBIE test years {test_years}" if "glambie" in selection_metric else ""))
    print(f"  LOYO R² gate    : >= {loyo_r2_min}" if loyo_r2_min is not None else "  LOYO R² gate    : disabled")
    print(f"  LOGO R² gate    : >= {logo_r2_min}" if logo_r2_min is not None else "  LOGO R² gate    : disabled")

    results_df = build_results_df(multirun_root, test_years, min_runs_per_region=1)
    print(f"  {len(results_df)} total runs loaded.")

    # Read use_time_encoding for every run
    results_df["use_time_encoding"] = [
        _read_use_time_encoding(Path(d)) for d in results_df["run_dir"].values
    ]
    n_unknown = results_df["use_time_encoding"].isna().sum()
    if n_unknown:
        warnings.warn(
            f"  {n_unknown} runs had unreadable Hydra configs — "
            "use_time_encoding flag treated as True for those runs."
        )
        results_df["use_time_encoding"] = results_df["use_time_encoding"].fillna(True)

    groups = [
        (True,  "time_encoding",    output_dir / "time_encoding"),
        (False, "no_time_encoding", output_dir / "no_time_encoding"),
    ]

    skipped = []
    for flag_val, label, group_output_dir in groups:
        group = results_df[results_df["use_time_encoding"] == flag_val].copy().reset_index(drop=True)
        print(f"\n--- Group: {label} (use_time_encoding={flag_val})  —  {len(group)} runs ---")
        if group.empty:
            print("  No runs found — skipping.")
            skipped.append(label)
            continue
        ok = _run_top_n_group(
            group, group_output_dir, top_n, min_runs, selection_metric,
            loyo_r2_min, logo_r2_min, composite_loyo_r2_weight,
        )
        if not ok:
            skipped.append(label)

    if skipped:
        print(f"\n  WARNING: the following groups were skipped due to insufficient qualifying runs: {skipped}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble uncertainty split by model.use_time_encoding flag, top-N by GLaMBIE RMSE."
    )
    parser.add_argument("--config", default="conf/config_ensemble_uncertainty.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--multirun_root", default=None,
                        help="Override multirun_root from config.")
    parser.add_argument("--output_dir", default=None,
                        help="Override output_dir from config.")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Override top_n from config (default 5).")
    parser.add_argument("--loyo_r2_min", type=float, default=None,
                        help="Minimum LOYO R² a run must achieve to be eligible for the "
                             "ensemble (default 0.1). Pass --loyo_r2_min=-inf to disable.")
    parser.add_argument("--logo_r2_min", type=float, default=None,
                        help="Minimum LOGO R² a run must achieve (default 0.0). "
                             "Pass --logo_r2_min=-inf to disable.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if args.multirun_root is not None:
        cfg["multirun_root"] = args.multirun_root
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.top_n is not None:
        cfg["top_n"] = args.top_n
    if args.loyo_r2_min is not None:
        cfg["loyo_r2_min"] = args.loyo_r2_min
    if args.logo_r2_min is not None:
        cfg["logo_r2_min"] = args.logo_r2_min

    run_ensemble_time_encoding(cfg)


if __name__ == "__main__":
    main()
