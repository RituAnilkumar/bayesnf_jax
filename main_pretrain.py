"""
Stage 1 entry point — pretrain on OGGM point-level mass balance estimates.

Usage:
    # CV run (evaluates loyo/logo/loygo out-of-sample):
    python main_pretrain.py model=bnf_regional_seasonal/r01 model.train_split=train

    # Full run (trains on all data → pretrained_params.pkl for Stage 2):
    python main_pretrain.py model=bnf_regional_seasonal/r01 model.train_split=full

    # Monthly input type:
    python main_pretrain.py model=bnf_regional_monthly/r01 model.train_split=full

Hydra writes outputs to cfg.model.output_dir (set in config_pretrain.yaml).
"""

import hydra
from omegaconf import DictConfig

from src.model.pretrain import run_pretrain


@hydra.main(config_path="conf", config_name="config_pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    run_pretrain(cfg)


if __name__ == "__main__":
    main()
