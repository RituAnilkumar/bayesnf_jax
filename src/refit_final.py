"""
src/refit_final.py

Refit of the selected ensemble members on previously held-out data, in two modes:

  --mode validation   Adds only data that was already used for selection: WGMS
                      validation-split reference glaciers. GLaMBIE 2020+, WGMS
                      testing-split glaciers and Dussaillant stay held out, so the
                      Step 6 test scripts give an honest score for the refit procedure
                      (point them at --ensemble_out's parent as --ensemble_base).
  --mode full         Final global product: everything below, nothing held out.

Model/ensemble selection is already done (top_runs_info.csv of the combined
ensemble, chosen by WGMS validation-split correlation — see
ensemble_uncertainty_pretrain_year.py). This script does NOT re-select. For
each selected member it re-runs Stage 2 + inference with model.final_refit=true,
starting from that member's own pretrained_params.pkl (Stage 1 is unchanged), then
combines the refit members into a new ensemble with the same equal weights.

Full mode adds to the Stage 2 loss (see src/model/finetune.py, src/refit_data.py):
  - GLaMBIE 2020+ (previously held out as test)
  - Dussaillant regional (not per-glacier) MB, years <2000, only region-years
    with >=1 WGMS in-situ measured glacier
  - WGMS reference-list glaciers (both splits), glaciological annual MB

Every member's exact sweep config is reproduced from its .hydra/config.yaml;
only final_refit, the paths, and any --set overrides change.

Consequence of full mode: GLaMBIE 2020+, all WGMS reference glaciers and
pre-2000 Dussaillant are no longer independent of the refit model, so Step 6
metrics against the full-refit ensemble are in-sample for those sources. Report
test metrics from the original and the validation-mode ensembles instead.

Usage (from the repo root, where the sweep was launched, so relative data paths
in the stored configs resolve the same way):

    # 1) validation-only refit, then Step 6 testing on outputs/ensemble_refit_val
    python src/refit_final.py --mode validation \\
        --ensemble_dir  outputs/ensemble_pretrain_year/r11/combined \\
        --output_root   /scratch/.../refit_val/r11 \\
        --ensemble_out  outputs/ensemble_refit_val/r11/combined

    # 2) full refit for the global results
    python src/refit_final.py --mode full \\
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

# Model overrides per refit mode; --set overrides are applied after these.
MODE_OVERRIDES = {
    "validation": [
        "model.use_duss=false",
        "model.wgms_splits=[validation]",
        "model.refit_glambie_all_years=false",
    ],
    "full": [
        "model.use_duss=true",
        "model.wgms_splits=null",
        "model.refit_glambie_all_years=true",
    ],
}


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
    p.add_argument("--mode", choices=sorted(MODE_OVERRIDES), required=True,
                   help="validation: add WGMS validation-split glaciers only (test sets stay held out); "
                        "full: add everything for the final global product")
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
    overrides = MODE_OVERRIDES[args.mode] + args.overrides
    print(f"{len(top)} selected members in {args.ensemble_dir}  (mode={args.mode})")

    if not args.combine_only:
        rows = top if args.member is None else top.iloc[[args.member]]
        for _, row in rows.iterrows():
            refit_member(Path(row["run_dir"]), output_root / str(row["run_id"]),
                         overrides, args.skip_existing)
        if args.member is not None:
            return

    combine(top, output_root, ensemble_out)


if __name__ == "__main__":
    main()
