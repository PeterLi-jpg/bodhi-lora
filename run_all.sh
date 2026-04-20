#!/bin/bash
# submit the full pipeline as a dependency chain
# each job waits for the previous one to finish successfully

set -e

# SBATCH --output=logs/... lines fail if logs/ doesn't exist — create it first.
mkdir -p logs
RUN_DATE="${RUN_DATE:-$(date +%F)}"
CONFIG_TAG="${CONFIG_TAG:-$(basename "${CONFIG_PATH:-configs/lora_medgemma27b.yaml}" .yaml)}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-${CONFIG_TAG}_seed${SEED}}"
RUN_DIR="${RUN_DIR:-results/${RUN_DATE}_${RUN_TAG}}"

mkdir -p "$RUN_DIR"

JOB1=$(sbatch --parsable slurm/generate_traces.sh)
echo "generate_traces submitted: $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:"$JOB1" slurm/filter_traces.sh)
echo "filter_traces submitted: $JOB2 (waits for $JOB1)"

JOB3=$(sbatch --parsable --dependency=afterok:"$JOB2" slurm/train_lora.sh "$RUN_DIR")
echo "train_lora submitted: $JOB3 (waits for $JOB2)"

JOB4=$(sbatch --parsable --dependency=afterok:"$JOB3" slurm/eval_lora.sh "$RUN_DIR")
echo "eval_lora submitted: $JOB4 (waits for $JOB3)"

echo ""
echo "full pipeline queued: $JOB1 -> $JOB2 -> $JOB3 -> $JOB4"
echo "run outputs will be archived under: $RUN_DIR"
echo "check progress with: squeue -u \$USER"
