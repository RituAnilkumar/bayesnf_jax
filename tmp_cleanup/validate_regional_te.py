"""
src/validate_regional_te.py

Variant of validate_regional.py for ensemble outputs produced by
ensemble_uncertainty_time_encoding.py, which writes into nested subdirectories:

    {ensemble_base_dir}/{region}/time_encoding/
    {ensemble_base_dir}/{region}/no_time_encoding/

Requires one extra config key:
    te_group: "time_encoding"     # or "no_time_encoding"

All plotting, metrics, and config options are identical to validate_regional.py.

Usage:
    python src/validate_regional_te.py \\
        --config conf/config_validate_regional.yaml \\
        --te_group time_encoding \\
        --ensemble_base_dir /hpc/path/ensemble_te \\
        --output_dir outputs/validation_regional_te
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml

from src.validate_regional import (
    _load_ensemble_mwe,
    _load_validation_mwe,
    _merge_on_overlap,
    _compute_metrics,
    _plot_timeseries,
    _plot_scatter_region,
    _plot_multipanel,
    _plot_residuals,
)


def run_validation(cfg: dict) -> None:
    ensemble_base = Path(cfg["ensemble_base_dir"])
    val_dir       = Path(cfg["validation_dir"])
    output_dir    = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    te_group     = cfg["te_group"]           # "time_encoding" or "no_time_encoding"
    region_map   = cfg["region_map"]
    region_names = cfg.get("region_names", {})
    ens_filename = cfg.get("ensemble_filename", "ensemble_regional_mwe.csv")
    sigma_mult   = float(cfg.get("sigma_mult", 1))
    min_overlap  = int(cfg.get("min_overlap_years", 5))

    print(f"\n=== Regional MWE validation (TE group: {te_group}) ===")
    print(f"  Ensemble base : {ensemble_base}")
    print(f"  Validation dir: {val_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Sigma mult    : {sigma_mult}  |  Min overlap years: {min_overlap}\n")

    region_data: dict[str, pd.DataFrame] = {}
    rgi_id: dict[str, str] = {}
    metrics_rows: list[dict] = []

    for abbrev, subdir in sorted(region_map.items()):
        # Insert the te_group subfolder between region and filename
        region_dir = ensemble_base / subdir / te_group
        ens = _load_ensemble_mwe(region_dir, ens_filename)
        if ens is None:
            print(f"  [{abbrev}/{subdir}/{te_group}] SKIP — ensemble file missing")
            continue

        val = _load_validation_mwe(val_dir, abbrev)
        if val is None:
            print(f"  [{abbrev}/{subdir}] SKIP — validation file missing")
            continue

        merged = _merge_on_overlap(ens, val, min_overlap)
        if merged is None:
            print(f"  [{abbrev}/{subdir}] SKIP — fewer than {min_overlap} overlapping years")
            continue

        name   = region_names.get(abbrev, abbrev)
        yr_rng = f"{int(merged['year'].min())}–{int(merged['year'].max())}"
        metrics = _compute_metrics(merged, sigma_mult)
        print(f"  [{abbrev}/{subdir}] {name:30s}  {yr_rng}  "
              f"n={metrics['n_years']:3d}  "
              f"RMSE={metrics['rmse']:.3f}  bias={metrics['bias']:+.3f}  "
              f"r={metrics['corr']:.2f}  cov={metrics['coverage_pct']:.0f}%")

        region_data[abbrev] = merged
        rgi_id[abbrev]      = subdir
        metrics_rows.append({
            "region": abbrev, "rgi": subdir, "name": name,
            **metrics,
            "year_min": int(merged["year"].min()),
            "year_max": int(merged["year"].max()),
        })

        _plot_timeseries(merged, name, sigma_mult,
                         output_dir / f"timeseries_{subdir}.png")
        _plot_scatter_region(merged, name, sigma_mult,
                             output_dir / f"scatter_{subdir}.png")

    if not region_data:
        print("\nNo regions matched — check ensemble_base_dir, te_group, and region_map.")
        return

    print(f"\n  Generating multi-panel summary ({len(region_data)} regions)...")
    _plot_multipanel(region_data, region_names, sigma_mult,
                     output_dir / "summary_multipanel.png")

    print("  Generating residual time series...")
    _plot_residuals(region_data, region_names, sigma_mult,
                    output_dir / "residuals_all_regions.png")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse").reset_index(drop=True)
    metrics_df.to_csv(output_dir / "validation_metrics.csv", index=False,
                      float_format="%.4f")
    print(f"\n  Saved validation_metrics.csv ({len(metrics_df)} regions)")
    print(f"\nDone. Outputs written to {output_dir}/")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TE-split ensemble regional MWE against WGMS/Dussaillant data."
    )
    parser.add_argument("--config", default="conf/config_validate_regional.yaml")
    parser.add_argument("--te_group", default=None,
                        choices=["time_encoding", "no_time_encoding"],
                        help="Which TE subfolder to validate (overrides config).")
    parser.add_argument("--ensemble_base_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--ensemble_filename", default=None,
                        help="CSV filename to load from each region/te_group/ dir "
                             "(default: ensemble_regional_mwe.csv).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if args.te_group is not None:
        cfg["te_group"] = args.te_group
    if args.ensemble_base_dir is not None:
        cfg["ensemble_base_dir"] = args.ensemble_base_dir
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.ensemble_filename is not None:
        cfg["ensemble_filename"] = args.ensemble_filename

    if "te_group" not in cfg:
        raise ValueError("te_group must be set in config or via --te_group "
                         "(choose 'time_encoding' or 'no_time_encoding')")

    run_validation(cfg)


if __name__ == "__main__":
    main()
