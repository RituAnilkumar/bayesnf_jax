"""
Full pipeline entry point — runs pretrain → finetune → predict in sequence.

Runs Stage 1 (full data), Stage 2, and inference for a single region
in a single process. Useful for local runs and debugging. For HPC,
prefer the individual SLURM scripts in slurm_calls/.

Usage:
    python main_pipeline.py model=bnf_regional_seasonal/r01

    # Run only from finetune onward (skip pretrain, e.g. params already exist):
    python main_pipeline.py model=bnf_regional_seasonal/r01 pipeline.start_stage=finetune

    # Run only CV pretrain (no finetune/predict):
    python main_pipeline.py model=bnf_regional_seasonal/r01 pipeline.stages=[pretrain_cv]

Stages run in order:
    pretrain_cv   → train on 'train' split, generate loyo/logo/loygo metrics
    pretrain_full → train on 'full' split, save pretrained_params.pkl
    finetune      → train on Hugonnet + GLaMBIE, save finetuned_params.pkl
    predict       → MC inference, save predictions.csv

Note: pretrain_cv and pretrain_full write to separate output directories so
they do not overwrite each other.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.model.pretrain import run_pretrain
from src.model.finetune import run_finetune
from src.inference.predict import run_predict


STAGE_ORDER = ["pretrain_cv", "pretrain_full", "finetune", "predict"]


def _mutable_copy(cfg: DictConfig) -> DictConfig:
    """Return a fully mutable deep copy of cfg (safe across Hydra struct/readonly modes)."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))


@hydra.main(config_path="conf", config_name="config_pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    stages = list(cfg.pipeline.stages)
    assert all(s in STAGE_ORDER for s in stages), \
        f"Unknown stage(s): {set(stages) - set(STAGE_ORDER)}"

    # Work with a fully mutable copy so OmegaConf.update and direct assignment
    # are safe regardless of Hydra's struct/readonly mode in single-run or multirun.
    cfg = _mutable_copy(cfg)

    if "pretrain_cv" in stages:
        cfg_cv = _mutable_copy(cfg)
        cfg_cv.model.train_split = "train"
        cfg_cv.model.output_dir = cfg.model.output_dir + "_cv"
        run_pretrain(cfg_cv)
        # cfg.model.output_dir is untouched — cfg_cv is an independent copy

    if "pretrain_full" in stages:
        cfg_full = _mutable_copy(cfg)
        cfg_full.model.train_split = "full"
        run_pretrain(cfg_full)
        # Override pretrained_params_path to where pretrain_full actually wrote it.
        # This takes priority over whatever the region yaml specified, which may
        # point to a different (standalone pretrain) output directory.
        actual_pretrained = os.path.abspath(
            os.path.join(cfg.model.output_dir, "pretrained_params.pkl")
        )
        cfg.model.pretrained_params_path = actual_pretrained

    if "finetune" in stages:
        run_finetune(cfg)
        # Override finetuned_params_path to where finetune actually wrote it.
        actual_finetuned = os.path.abspath(
            os.path.join(cfg.model.output_dir, "finetuned_params.pkl")
        )
        cfg.model.finetuned_params_path = actual_finetuned

    if "predict" in stages:
        run_predict(cfg)


if __name__ == "__main__":
    main()
