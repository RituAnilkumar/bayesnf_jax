"""
Inference entry point — generate mass balance predictions from finetuned model.

Runs MC forward passes over the full (glacier × year) prediction grid (for_preds.csv).

Usage:
    python main_predict.py model=bnf_regional_seasonal/r01

    # Override number of MC samples:
    python main_predict.py model=bnf_regional_seasonal/r01 model.model_nensemble=200

Output:
    predictions.csv with columns: glacier_id, year, mass_balance_mwe, uncertainty_mwe
"""

import hydra
from omegaconf import DictConfig

from src.inference.predict import run_predict


@hydra.main(config_path="conf", config_name="config_predict", version_base=None)
def main(cfg: DictConfig) -> None:
    run_predict(cfg)


if __name__ == "__main__":
    main()
