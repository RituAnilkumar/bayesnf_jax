"""
Stage 2 entry point — finetune on Hugonnet + GLaMBIE observational aggregates.

Requires pretrained_params.pkl from Stage 1 (path set in region config or overridden).

Usage:
    python main_finetune.py model=bnf_regional_seasonal/r01

    # Override pretrained params path explicitly:
    python main_finetune.py model=bnf_regional_seasonal/r01 \
        model.pretrained_params_path=outputs/pretrain/r01_seasonal/pretrained_params.pkl

Outputs:
    finetuned_params.pkl, training_loss.png, metrics_glambie_test.csv
"""

import hydra
from omegaconf import DictConfig

from src.model.finetune import run_finetune


@hydra.main(config_path="conf", config_name="config_finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    run_finetune(cfg)


if __name__ == "__main__":
    main()
