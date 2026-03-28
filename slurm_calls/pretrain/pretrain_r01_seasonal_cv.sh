#!/bin/bash
#SBATCH --job-name=pretrain_cv_r01_s
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_calls/logs/%x_%j.out
#SBATCH --error=slurm_calls/logs/%x_%j.err

cd $SLURM_SUBMIT_DIR

# CV run: train on 'train' split, evaluate on loyo (primary), logo, loygo
# loyo performance is the key metric for Stage 1 assessment
python main_pretrain.py \
    +model/bnf_regional_seasonal=r01 \
    model.train_split=train
