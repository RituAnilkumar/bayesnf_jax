"""
src/refit_final.py

Final refit of the selected ensemble members on (almost) all available data.

Model/ensemble selection is already done (top_runs_info.csv of the combined
ensemble, chosen by WGMS validation-split correlation — see
ensemble_uncertainty_pretrain_year.py). This script does NOT re-select. For
each selected member it re-runs Stage 2 + inference with model.final_refit=true,
starting from that member's own pretrained_params.pkl (Stage 1 is unchanged), then
combines the refit members into a new ensemble with the same equal weights.

Final refit adds to the Stage 2 loss (see src/model/finetune.py, src/refit_data.py):
  - GLaMBIE 2020+ (previously held out as test)
  - Dussaillant regional MB, years <2000 with in-situ support
  - WGMS reference-list glaciers, glaciological annual MB

Every member's exact sweep config is reproduced from its .hydra/config.yaml;
only final_refit, the paths, and any --set overrides change.

Consequence: GLaMBIE 2020+, the WGMS reference glaciers (validation AND testing
split) and pre-2000 Dussaillant are no longer independent of the refit model, so
Step 6 testing metrics computed against the refit ensemble are in-sample for
those sources. Report test metrics from the original (pre-refit) ensemble.

Usage (from the repo root, where the sweep was launched, so relative data paths
in the stored configs resolve the same way):

    # all members sequentially, then combine
    python src/refit_final.py \\
        --ensemble_dir  outputs/ensemble_pretrain_year/r11/combined \\
        --output_root   /scratch/.../refit_final/r11 \\
        --ensemble_out  outputs/ensemble_final_refit/r11/combined

    # one member per SLURM array task, then a final combine-only call
    python src/refit_final.py ... --member ${SLURM_ARRAY_TASK_ID}
    python src/refit_final.py ... --combine_only

    # change refit settings for all members
    python src/refit_final.py ... --set model.duss_min_insitu=3 model.wgms_exclude_window=true
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from omegaconf import OmegaConf

REQUIRED_PRETRAIN_FILES = ("pretrained_params.pkl", "scaler.pkl", "target_scaler.pkl")


def build_member_cfg(run_dir: Path, out_dir: Path, overrides: list[str]):
    """The member's stored sweep config, switched to final-refit mode."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.model.final_refit = True
    cfg.model.pretrained_params_path = str(run_dir / "pretrained_params.pkl")
    cfg.model.output_dir = str(out_dir)
    cfg.model.finetuned_params_path = str(out_dir / "finetuned_params.pkl")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def refit_member(run_dir: Path, out_dir: Path, overrides: list[str], skip_existing: bool) -> None:
    from src.model.finetune import run_finetune
    from src.inference.predict import run_predict

    if skip_existing and (out_dir / "preds_full.csv").exists():
        print(f"  [skip] {out_dir} already has preds_full.csv")
        return
    missing = [f for f in REQUIRED_PRETRAIN_FILES if not (run_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing {missing} — cannot refit this member")

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_member_cfg(run_dir, out_dir, overrides)
    # ensemble_common reads each run's model config from .hydra/config.yaml
    (out_dir / ".hydra").mkdir(exist_ok=True)
    OmegaConf.save(cfg, out_dir / ".hydra" / "config.yaml")

    print(f"\n=== Refit member {run_dir.name}  "
          f"(nhidden={cfg.model.model_nhidden}, nlayers={cfg.model.model_nlayers}, "
          f"pretrain_year_min={cfg.model.get('pretrain_year_min')}) → {out_dir}")
    run_finetune(cfg)
    run_predict(cfg)


def combine(top: pd.DataFrame, output_root: Path, ensemble_out: Path) -> None:
    from src.ensemble_common import _combine_runs

    refit = top.copy()
    refit["original_run_dir"] = refit["run_dir"]
    refit["run_dir"] = [str(output_root / str(rid)) for rid in refit["run_id"]]
    done = refit["run_dir"].map(lambda d: (Path(d) / "preds_full.csv").exists())
    if not done.all():
        missing = refit.loc[~done, "run_id"].tolist()
        raise RuntimeError(f"Refit not finished for members {missing} — run them before --combine_only")

    ensemble_out.mkdir(parents=True, exist_ok=True)
    refit.to_csv(ensemble_out / "top_runs_info.csv", index=False)
    if not _combine_runs(refit, ensemble_out, min_runs=len(refit)):
        raise RuntimeError("Ensemble combination failed — see warnings above")
    print(f"\nRefit ensemble ({len(refit)} members) written to {ensemble_out}/")


def main() -> None:
    p = argparse.ArgumentParser(description="Refit selected ensemble members with final_refit=true.")
    p.add_argument("--ensemble_dir", required=True,
                   help="Existing ensemble dir containing top_runs_info.csv (e.g. .../r11/combined)")
    p.add_argument("--output_root", required=True, help="Where per-member refit run dirs are written")
    p.add_argument("--ensemble_out", required=True, help="Where the refit ensemble CSVs are written")
    p.add_argument("--member", type=int, default=None,
                   help="Refit only the member at this 0-based row of top_runs_info.csv (no combine)")
    p.add_argument("--combine_only", action="store_true", help="Skip training; only combine refit members")
    p.add_argument("--skip_existing", action="store_true", help="Skip members that already have preds_full.csv")
    p.add_argument("--set", nargs="*", default=[], dest="overrides",
                   help="Extra dotlist overrides applied to every member, e.g. model.duss_weight=2.0")
    args = p.parse_args()

    top = pd.read_csv(Path(args.ensemble_dir) / "top_runs_info.csv")
    output_root, ensemble_out = Path(args.output_root), Path(args.ensemble_out)
    print(f"{len(top)} selected members in {args.ensemble_dir}")

    if not args.combine_only:
        rows = top if args.member is None else top.iloc[[args.member]]
        for _, row in rows.iterrows():
            refit_member(Path(row["run_dir"]), output_root / str(row["run_id"]),
                         args.overrides, args.skip_existing)
        if args.member is not None:
            return

    combine(top, output_root, ensemble_out)


if __name__ == "__main__":
    main()
