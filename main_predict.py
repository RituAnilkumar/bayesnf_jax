"""
Inference entry point — generate mass balance predictions from finetuned model.

Runs MC forward passes over the full (glacier × year) prediction grid
(main_features_{region}.csv).

Usage:
    python main_predict.py model=bnf_regional_seasonal/r01

    # Override number of MC samples:
    python main_predict.py model=bnf_regional_seasonal/r01 model.model_nensemble=200

Key outputs (written to cfg.model.output_dir):
    preds_full.csv               — per (glacier, year): rgi_id, year, p2_5, p50, p97_5, mean, std [MWE/yr]
    regional_annual_mwe.csv      — area-weighted regional mean MWE/yr per year with MC quantiles
    regional_annual_gt.csv       — regional Gt/yr per year with MC quantiles
    regional_mass_loss.png       — annual Gt/yr: model vs GLaMBIE sources + OGGM
    regional_mwe_comparison.png  — pretrain vs finetune MWE/yr time series
    cumulative_gt.png            — cumulative Gt: pretrain, finetune, OGGM vs GLaMBIE
    hugonnet_scatter_*.png       — per-glacier Hugonnet period scatter plots
"""

import hydra
from omegaconf import DictConfig

from src.inference.predict import run_predict


@hydra.main(config_path="conf", config_name="config_predict", version_base=None)
def main(cfg: DictConfig) -> None:
    run_predict(cfg)


if __name__ == "__main__":
    main()
