#!/bin/bash
# One array task = one selected ensemble member of one region (src/refit_final.py --member).
# Submitted by slurm/submit_refit.sh, which sets REGION, MODE, OUT_TAG via --export.
#SBATCH --job-name=refit
#SBATCH --output=/scratch/b5at/ranil.b5at/bayesnf_jax/out_files/refit_%x_%A_%a.out
#SBATCH --error=/scratch/b5at/ranil.b5at/bayesnf_jax/err_files/refit_%x_%A_%a.err
#SBATCH --gpus=1
#SBATCH --time=06:00:00

source ~/miniforge3/bin/activate
conda activate hydra_bnf_env
cd /scratch/b5at/ranil.b5at/bayesnf_jax

python src/refit_final.py --mode "${MODE}" \
    --ensemble_dir  "outputs/ensemble_pretrain_year/${REGION}/combined" \
    --output_root   "/scratch/b5at/ranil.b5at/bayesnf_jax/${OUT_TAG}/${REGION}" \
    --ensemble_out  "outputs/ensemble_${OUT_TAG}/${REGION}/combined" \
    --member "${SLURM_ARRAY_TASK_ID}" --skip_existing
