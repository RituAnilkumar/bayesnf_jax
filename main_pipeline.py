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

import hydra
from omegaconf import DictConfig

from src.model.pretrain import run_pretrain
from src.model.finetune import run_finetune
from src.inference.predict import run_predict


STAGE_ORDER = ["pretrain_cv", "pretrain_full", "finetune", "predict"]


@hydra.main(config_path="conf", config_name="config_pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    stages = list(cfg.pipeline.stages)
    assert all(s in STAGE_ORDER for s in stages), \
        f"Unknown stage(s): {set(stages) - set(STAGE_ORDER)}"

    if "pretrain_cv" in stages:
        cfg_cv = cfg.copy()
        cfg_cv.model.train_split = "train"
        cfg_cv.model.output_dir = cfg.model.output_dir + "_cv"
        run_pretrain(cfg_cv)

    if "pretrain_full" in stages:
        cfg_full = cfg.copy()
        cfg_full.model.train_split = "full"
        run_pretrain(cfg_full)

    if "finetune" in stages:
        # Derive pretrained_params_path from the pretrain output dir if not set
        if not cfg.model.get("pretrained_params_path"):
            import os
            cfg.model.pretrained_params_path = os.path.join(
                cfg.model.output_dir, "pretrained_params.pkl"
            )
        run_finetune(cfg)

    if "predict" in stages:
        # Derive finetuned_params_path from the finetune output dir if not set
        if not cfg.model.get("finetuned_params_path"):
            import os
            cfg.model.finetuned_params_path = os.path.join(
                cfg.model.output_dir, "finetuned_params.pkl"
            )
        run_predict(cfg)


if __name__ == "__main__":
    main()
