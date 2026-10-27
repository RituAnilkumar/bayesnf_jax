#!/bin/bash
# Submit the refit of every selected ensemble member, one SLURM array per region,
# plus a combine job per region that starts when that region's array has succeeded.
#
#   bash slurm/submit_refit.sh validation          # all regions
#   bash slurm/submit_refit.sh full r01 r13        # chosen regions
#
# validation -> members in  /scratch/.../refit_val/rXX/<run_id>,
#               ensemble in outputs/ensemble_refit_val/rXX/combined
# full       -> members in  /scratch/.../final_refit/rXX/<run_id>,
#               ensemble in outputs/ensemble_final_refit/rXX/combined
set -euo pipefail
cd /scratch/b5at/ranil.b5at/bayesnf_jax

MODE=${1:?usage: submit_refit.sh validation|full [rXX ...]}
shift
case "$MODE" in
    validation) OUT_TAG=refit_val ;;
    full)       OUT_TAG=final_refit ;;
    *) echo "mode must be validation or full"; exit 1 ;;
esac
REGIONS=${*:-$(printf "r%02d " $(seq 1 19))}

# Preflight: inputs that are not in git must already be on /scratch.
for d in validation_data/per_gla validation_data/regional_wgms_duss; do
    [ -d "$d" ] || { echo "Missing $d (rsync it from the local repo first)"; exit 1; }
done

for R in $REGIONS; do
    TOP="outputs/ensemble_pretrain_year/${R}/combined/top_runs_info.csv"
    [ -f "$TOP" ] || { echo "Missing $TOP — skipping $R"; continue; }
    N=$(( $(wc -l < "$TOP") - 1 ))
    for RUN in $(tail -n +2 "$TOP" | cut -d, -f3); do
        for f in pretrained_params.pkl scaler.pkl target_scaler.pkl .hydra/config.yaml; do
            [ -e "$RUN/$f" ] || { echo "Missing $RUN/$f — fix before submitting $R"; exit 1; }
        done
    done
    ARR=$(sbatch --parsable --job-name="refit_${MODE}_${R}" --array=0-$((N-1)) \
          --export=ALL,REGION=$R,MODE=$MODE,OUT_TAG=$OUT_TAG slurm/refit_member.sh)
    CMB=$(sbatch --parsable --job-name="combine_${MODE}_${R}" --dependency=afterok:${ARR} \
          --export=ALL,REGION=$R,MODE=$MODE,OUT_TAG=$OUT_TAG slurm/refit_combine.sh)
    echo "$R: $N members -> array $ARR, combine $CMB"
done
