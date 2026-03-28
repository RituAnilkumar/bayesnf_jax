#!/bin/bash
#SBATCH --job-name=predict_r01_s
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_calls/logs/%x_%j.out
#SBATCH --error=slurm_calls/logs/%x_%j.err

cd $SLURM_SUBMIT_DIR

# Requires: outputs/finetune/r01_seasonal/finetuned_params.pkl
python main_predict.py \
    +model/bnf_regional_seasonal=r01
