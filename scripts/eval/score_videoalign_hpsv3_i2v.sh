#!/bin/bash
#SBATCH --job-name=score_va_hps_i2v
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=04:00:00
#SBATCH --output=./sbatch_logs/eval-%x-%j-O
#SBATCH --error=./sbatch_logs/eval-%x-%j-E

# Score a SkyReels/Wan2.2 I2V run's pre-generated mp4s (evaluation_vbench_i2v/images)
# with VideoAlign (TA/VQ/MQ/Overall) + HPSv3 + HPSv3-percentile — no regeneration.
# Reads (video,prompt) pairs from the run's evaluation_results_vbench_i2v*.jsonl.
# Usage: sbatch scripts/eval/score_videoalign_hpsv3_i2v.sh <RUN_DIR>
set -euo pipefail

RUN_DIR="${1:?provide the training run dir as first arg}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "${REPO_DIR}/config/paths.sh"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_VIDEO}"

echo "==========================================================="
echo "Run dir: ${RUN_DIR}"
nvidia-smi -L 2>&1 || true
echo "==========================================================="

cd "${REPO_DIR}"
SHARD_ARGS=""
if [ "${NUM_SHARDS:-1}" -gt 1 ]; then
    SHARD_ARGS="--num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID:-0}"
fi
# EVAL_SUBDIR selects the prompt suite: evaluation_vbench_i2v (default) or
# evaluation_vbench_i2v_extended (VL-rewritten prompts).
SUBDIR_ARGS=""
if [ -n "${EVAL_SUBDIR:-}" ]; then
    SUBDIR_ARGS="--eval_subdir ${EVAL_SUBDIR}"
fi
python scripts/eval/score_videoalign_hpsv3_i2v.py --run_dir "${RUN_DIR}" ${SHARD_ARGS} ${SUBDIR_ARGS}
echo "Done -> ${RUN_DIR}/evaluation_vbench_i2v/videoalign_hpsv3_scores*.json"
