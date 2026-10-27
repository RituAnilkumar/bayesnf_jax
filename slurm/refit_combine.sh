#!/bin/bash
# Combine the refit members of one region into the refit ensemble (runs after the array).
#SBATCH --job-name=refit_combine
#SBATCH --output=/scratch/b5at/ranil.b5at/bayesnf_jax/out_files/refit_%x_%j.out
#SBATCH --error=/scratch/b5at/ranil.b5at/bayesnf_jax/err_files/refit_%x_%j.err
#SBATCH --time=01:00:00

source ~/miniforge3/bin/activate
conda activate hydra_bnf_env
cd /scratch/b5at/ranil.b5at/bayesnf_jax

python src/refit_final.py --mode "${MODE}" \
    --ensemble_dir  "outputs/ensemble_pretrain_year/${REGION}/combined" \
    --output_root   "/scratch/b5at/ranil.b5at/bayesnf_jax/${OUT_TAG}/${REGION}" \
    --ensemble_out  "outputs/ensemble_${OUT_TAG}/${REGION}/combined" \
    --combine_only
