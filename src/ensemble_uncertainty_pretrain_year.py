"""
src/ensemble_uncertainty_pretrain_year.py

Variant of ensemble_uncertainty_time_encoding.py for sweeps that vary
model.pretrain_year_min (e.g. 1940, 1960, 1980, 2000) with
model.use_time_encoding=false and model.heteroscedastic=true fixed.

Reads model.pretrain_year_min from each run's .hydra/config.yaml, partitions
all completed runs into one group per unique pretrain_year_min value, selects
the top-N (default 5) within each group by a configurable selection metric
(default: glambie_rmse), and writes separate ensemble outputs:

    {output_dir}/pt1940/    — runs where model.pretrain_year_min=1940
    {output_dir}/pt1960/    — runs where model.pretrain_year_min=1960
    {output_dir}/pt1980/    — runs where model.pretrain_year_min=1980
    {output_dir}/pt2000/    — runs where model.pretrain_year_min=2000

Within each group, full uncertainty decomposition is computed:
    std_epistemic  = sqrt( sum_k (1/K) * epistemic_std_k^2 )
    std_aleatoric  = sqrt( sum_k (1/K) * aleatoric_std_k^2 )   (heteroscedastic=true)
    std_structural = sqrt( sum_k (1/K) * (mean_k - mu_ensemble)^2 )
    std_total      = sqrt( std_epistemic^2 + std_aleatoric^2 + std_structural^2 )

Usage:
    python src/ensemble_uncertainty_pretrain_year.py \\
        --multirun_root /scratch/.../multirun/r03_12345 \\
        --output_dir outputs/ensemble_pretrain_year/r03

    python src/ensemble_uncertainty_pretrain_year.py \\
        --config conf/config_ensemble_uncertainty.yaml \\
        --multirun_root /scratch/.../multirun/r03_12345 \\
        --output_dir outputs/ensemble_pretrain_year/r03 \\
        --pretrain_years 1940 1960 1980 2000

Batch usage (one call per region):
    for i in $(seq -w 1 19); do
        python src/ensemble_uncertainty_pretrain_year.py \\
            --multirun_root /scratch/.../multirun/r${i}_${SLURM_JOB_ID} \\
            --output_dir outputs/ensemble_pretrain_year/r${i}
    done
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.ensemble_uncertainty_time_encoding import _run_top_n_group
from src.hyperparam_tuning import build_results_df

_DEFAULT_PRETRAIN_YEARS = [1940, 1960, 1980, 2000]


def _read_pretrain_year_min(run_dir: Path) -> int | None:
    """
    Read model.pretrain_year_min from .hydra/config.yaml for a single run.

    Returns the integer value if found, None if the config is missing or the
    key is absent.
    """
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path) as fh:
            full_cfg = yaml.safe_load(fh)
        val = full_cfg.get("model", {}).get("pretrain_year_min")
        if val is None:
            return None
        return int(val)
    except Exception as exc:
        warnings.warn(f"Could not read {cfg_path}: {exc}")
        return None


def run_ensemble_pretrain_year(cfg: dict) -> None:
    """
    Split a sweep by pretrain_year_min and run top-N ensemble for each group,
    then combine the best runs from all groups into a single ensemble whose
    structural uncertainty spans nhidden, nlayers, AND pretrain_year_min.

    Per-group outputs:
        {output_dir}/pt1940/   — top_n best runs with pretrain_year_min=1940
        {output_dir}/pt1960/
        {output_dir}/pt1980/
        {output_dir}/pt2000/

    Combined output:
        {output_dir}/combined/ — top k_per_group runs from EACH group pooled
                                 together (default k_per_group=2, so 4 groups
                                 × 2 = 8 models in the combined ensemble)
    """
    multirun_root    = Path(cfg["multirun_root"])
    output_dir       = Path(cfg["output_dir"])
    test_years       = list(cfg.get("glambie_test_years", [2020, 2021, 2022, 2023, 2024]))
    min_runs         = int(cfg.get("min_runs_per_region", 1))
    top_n            = int(cfg.get("top_n", 5))
    k_per_group      = int(cfg.get("k_per_group", 2))
    selection_metric = str(cfg.get("selection_metric", "glambie_rmse"))
    pretrain_years   = [int(y) for y in cfg.get("pretrain_years", _DEFAULT_PRETRAIN_YEARS)]
    loyo_r2_min      = cfg.get("loyo_r2_min", 0.1)
    if loyo_r2_min is not None:
        loyo_r2_min = float(loyo_r2_min)

    valid_metrics = {"glambie_rmse", "loyo_rmse", "loyo_r2"}
    if selection_metric not in valid_metrics:
        raise ValueError(f"selection_metric must be one of {valid_metrics}, got '{selection_metric}'")

    ascending = selection_metric != "loyo_r2"

    print(f"\n=== Ensemble split by pretrain_year_min: {multirun_root.name} ===")
    print(f"  Pretrain years  : {pretrain_years}")
    print(f"  Per-group top_n : {top_n}  |  k_per_group for combined: {k_per_group}")
    print(f"  Selection metric: {selection_metric}"
          + (f" over GLaMBIE test years {test_years}" if selection_metric == "glambie_rmse" else ""))
    print(f"  LOYO R² gate    : >= {loyo_r2_min}" if loyo_r2_min is not None else "  LOYO R² gate    : disabled")

    results_df = build_results_df(multirun_root, test_years, min_runs_per_region=1)
    print(f"  {len(results_df)} total runs loaded.")

    results_df["pretrain_year_min"] = [
        _read_pretrain_year_min(Path(d)) for d in results_df["run_dir"].values
    ]

    n_unknown = results_df["pretrain_year_min"].isna().sum()
    if n_unknown:
        warnings.warn(
            f"  {n_unknown} runs had unreadable Hydra configs — "
            "pretrain_year_min unknown; those runs will be skipped by all groups."
        )

    # ----------------------------------------------------------------
    # Per-group ensembles
    # ----------------------------------------------------------------
    combined_parts = []   # collect top-k_per_group from each group for combined
    skipped = []

    for year in pretrain_years:
        label        = f"pt{year}"
        group_outdir = output_dir / label
        group = (
            results_df[results_df["pretrain_year_min"] == year]
            .copy()
            .reset_index(drop=True)
        )
        print(f"\n--- Group: {label} (pretrain_year_min={year})  —  {len(group)} runs ---")
        if group.empty:
            print("  No runs found — skipping.")
            skipped.append(label)
            continue
        ok = _run_top_n_group(group, group_outdir, top_n, min_runs, selection_metric, loyo_r2_min)
        if not ok:
            skipped.append(label)
            continue

        # Collect top-k_per_group for the combined ensemble (same R² gate applied)
        valid = group.dropna(subset=[selection_metric])
        if loyo_r2_min is not None and "loyo_r2" in valid.columns:
            valid = valid[valid["loyo_r2"] >= loyo_r2_min]
        valid = valid.sort_values(selection_metric, ascending=ascending)
        combined_parts.append(valid.head(k_per_group))

    # ----------------------------------------------------------------
    # Combined ensemble — structural uncertainty spans pretrain years
    # ----------------------------------------------------------------
    if len(combined_parts) >= 2:
        import pandas as pd
        combined_df = pd.concat(combined_parts).reset_index(drop=True)
        n_combined  = len(combined_df)
        print(f"\n--- Combined ensemble ({n_combined} models: "
              f"top-{k_per_group} × {len(combined_parts)} pretrain years) ---")
        _run_top_n_group(
            combined_df,
            output_dir / "combined",
            top_n=n_combined,        # use all pooled models, no further filtering
            min_runs=min_runs,
            selection_metric=selection_metric,
            loyo_r2_min=None,        # already gated above; don't double-filter
        )
    else:
        print(f"\n  WARNING: combined ensemble skipped — only {len(combined_parts)} group(s) "
              f"had qualifying runs (need >= 2). Check loyo_r2 >= {loyo_r2_min} threshold.")

    if skipped:
        print(f"\n  WARNING: the following groups were skipped due to insufficient runs "
              f"passing loyo_r2 >= {loyo_r2_min}: {skipped}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble uncertainty split by model.pretrain_year_min, top-N by GLaMBIE RMSE."
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
    parser.add_argument("--k_per_group", type=int, default=None,
                        help="Runs taken from each pretrain year group for the "
                             "combined ensemble (default 2; total = k_per_group × n_groups).")
    parser.add_argument("--pretrain_years", type=int, nargs="+", default=None,
                        help="Which pretrain_year_min values to process "
                             "(default: 1940 1960 1980 2000).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        cfg = {}

    if args.multirun_root is not None:
        cfg["multirun_root"] = args.multirun_root
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.top_n is not None:
        cfg["top_n"] = args.top_n
    if args.loyo_r2_min is not None:
        cfg["loyo_r2_min"] = args.loyo_r2_min
    if args.k_per_group is not None:
        cfg["k_per_group"] = args.k_per_group
    if args.pretrain_years is not None:
        cfg["pretrain_years"] = args.pretrain_years

    if "multirun_root" not in cfg:
        raise ValueError("multirun_root must be set in config or via --multirun_root")
    if "output_dir" not in cfg:
        raise ValueError("output_dir must be set in config or via --output_dir")

    run_ensemble_pretrain_year(cfg)


if __name__ == "__main__":
    main()
