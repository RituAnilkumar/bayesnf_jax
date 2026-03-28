#!/bin/bash
#SBATCH --job-name=pipeline_r02_s
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_calls/logs/%x_%j.out
#SBATCH --error=slurm_calls/logs/%x_%j.err

cd $SLURM_SUBMIT_DIR

python main_pipeline.py \
    +model/bnf_regional_seasonal=r02 \
    "pipeline.stages=[pretrain_full,finetune,predict]"
