"""
src/validate_per_glacier_te.py

Variant of validate_per_glacier.py for ensemble outputs produced by
ensemble_uncertainty_time_encoding.py, which writes into nested subdirectories:

    {ensemble_base_dir}/{region}/time_encoding/
    {ensemble_base_dir}/{region}/no_time_encoding/

Requires one extra config key:
    te_group: "time_encoding"     # or "no_time_encoding"

All plotting, metrics, and config options are identical to validate_per_glacier.py.

Usage:
    python src/validate_per_glacier_te.py \\
        --te_group time_encoding \\
        --ensemble_base_dir /hpc/path/ensemble_te \\
        --output_dir outputs/validation_per_glacier_te
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml

from src.validate_per_glacier import (
    REF_CSV,
    MM_TO_MWE,
    _rgi_to_subdir,
    _metrics,
    plot_timeseries,
    plot_scatter,
    plot_residuals,
    plot_individual,
)

import numpy as np


def _load_glacier_csv_te(ensemble_base: Path, subdir: str, te_group: str,
                         glacier_filename: str = "ensemble_glacier.csv") -> pd.DataFrame | None:
    path = ensemble_base / subdir / te_group / glacier_filename
    if not path.exists():
        warnings.warn(f"top_models_glacier.csv not found: {path}")
        return None
    df = pd.read_csv(path)
    rename = {
        "std_total":      "total_std",
        "std_epistemic":  "epistemic_std",
        "std_structural": "structural_std",
        "std_aleatoric":  "aleatoric_std",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def build_joined(ref: pd.DataFrame, ensemble_base: Path, te_group: str,
                 glacier_filename: str = "ensemble_glacier.csv") -> pd.DataFrame:
    region_cache: dict[str, pd.DataFrame] = {}

    model_rows = []
    for _, row in ref.iterrows():
        subdir = _rgi_to_subdir(row["rgi_id"])
        if subdir not in region_cache:
            gl = _load_glacier_csv_te(ensemble_base, subdir, te_group, glacier_filename)
            region_cache[subdir] = gl

        gl = region_cache[subdir]
        if gl is None:
            model_rows.append({})
            continue

        match = gl[(gl["rgi_id"] == row["rgi_id"]) & (gl["year"] == row["YEAR"])]
        if match.empty:
            model_rows.append({})
            continue

        r = match.iloc[0]
        model_rows.append({
            "model_mean_mwe":       r.get("median_mwe", r.get("mean_mwe", np.nan)),
            "model_total_std":      r.get("total_std",      np.nan),
            "model_epistemic_std":  r.get("epistemic_std",  np.nan),
            "model_structural_std": r.get("structural_std", np.nan),
            "model_aleatoric_std":  r.get("aleatoric_std",  np.nan),
        })

    model_cols = ["model_mean_mwe", "model_total_std", "model_epistemic_std",
                  "model_structural_std", "model_aleatoric_std"]
    model_df = pd.DataFrame(model_rows, index=ref.index).reindex(columns=model_cols)
    return pd.concat([ref, model_df], axis=1)


def run(cfg: dict) -> None:
    ensemble_base = Path(cfg["ensemble_base_dir"])
    te_group      = cfg["te_group"]
    ref_csv       = Path(cfg.get("ref_csv", str(REF_CSV)))
    output_dir    = Path(cfg.get("per_glacier_output_dir",
                                  "outputs/validation_per_glacier_te"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Per-glacier validation (TE group: {te_group}) ===")
    print(f"  Ensemble base : {ensemble_base}")
    print(f"  Reference CSV : {ref_csv}")
    print(f"  Output dir    : {output_dir}\n")

    glacier_filename = cfg.get("glacier_filename", "ensemble_glacier.csv")

    ref = pd.read_csv(ref_csv)
    ref["obs_mwe"] = ref["ANNUAL_BALANCE"] * MM_TO_MWE

    print(f"  Reference: {len(ref)} rows | "
          f"{ref['rgi_id'].nunique()} glaciers | "
          f"{int(ref['YEAR'].min())}–{int(ref['YEAR'].max())}")

    joined = build_joined(ref, ensemble_base, te_group, glacier_filename)
    n_matched = joined["model_mean_mwe"].notna().sum()
    print(f"  Matched {n_matched}/{len(joined)} rows to model predictions")

    out_csv = output_dir / "ref_mb_with_model.csv"
    joined.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"  Saved ref_mb_with_model.csv")

    metrics_rows = []
    for rgi in sorted(joined["rgi_id"].unique()):
        sub  = joined[joined["rgi_id"] == rgi]
        name = sub["NAME"].iloc[0]
        m    = _metrics(sub)
        yr_min = int(sub.dropna(subset=["model_mean_mwe"])["YEAR"].min()) \
            if sub["model_mean_mwe"].notna().any() else np.nan
        yr_max = int(sub.dropna(subset=["model_mean_mwe"])["YEAR"].max()) \
            if sub["model_mean_mwe"].notna().any() else np.nan
        metrics_rows.append({"rgi_id": rgi, "name": name,
                              "year_min": yr_min, "year_max": yr_max, **m})
        print(f"  [{rgi}] {name:20s}  "
              f"n={m.get('n_years','?'):3}  "
              f"RMSE={m.get('rmse', float('nan')):.3f}  "
              f"bias={m.get('bias', float('nan')):+.3f}  "
              f"r={m.get('corr', float('nan')):.2f}  "
              f"cov1σ={m.get('coverage_pct', float('nan')):.0f}%")

    pd.DataFrame(metrics_rows).to_csv(
        output_dir / "per_glacier_metrics.csv", index=False, float_format="%.4f"
    )
    print(f"\n  Saved per_glacier_metrics.csv")

    print("\n  Generating plots...")
    plot_timeseries(joined, output_dir / "timeseries_per_glacier.png")
    plot_scatter(joined,    output_dir / "scatter_per_glacier.png")
    plot_residuals(joined,  output_dir / "residuals_per_glacier.png")
    plot_individual(joined, output_dir)

    print(f"\nDone. Outputs written to {output_dir}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate TE-split per-glacier ensemble vs. WGMS in-situ reference."
    )
    p.add_argument("--config", default="conf/config_validate_regional.yaml")
    p.add_argument("--te_group", default=None,
                   choices=["time_encoding", "no_time_encoding"],
                   help="Which TE subfolder to validate (overrides config).")
    p.add_argument("--ensemble_base_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--ref_csv", default=None)
    p.add_argument("--glacier_filename", default=None,
                   help="Glacier CSV filename inside each region/te_group/ dir "
                        "(default: ensemble_glacier.csv).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}

    if args.te_group is not None:       cfg["te_group"]              = args.te_group
    if args.ensemble_base_dir:          cfg["ensemble_base_dir"]     = args.ensemble_base_dir
    if args.output_dir:                 cfg["per_glacier_output_dir"] = args.output_dir
    if args.ref_csv:                    cfg["ref_csv"]               = args.ref_csv
    if args.glacier_filename:           cfg["glacier_filename"]      = args.glacier_filename

    if "te_group" not in cfg:
        raise ValueError("te_group must be set in config or via --te_group "
                         "(choose 'time_encoding' or 'no_time_encoding')")

    run(cfg)


if __name__ == "__main__":
    main()
