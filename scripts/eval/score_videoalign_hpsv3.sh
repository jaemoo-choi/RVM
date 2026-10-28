#!/bin/bash
#SBATCH --job-name=score_va_hps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=04:00:00
#SBATCH --output=./sbatch_logs/eval-%x-%j-O
#SBATCH --error=./sbatch_logs/eval-%x-%j-E

# Score pre-generated VBench mp4s with VideoAlign (TA/VQ/MQ/Overall) + HPSv3.
# Usage: sbatch scripts/eval/score_videoalign_hpsv3.sh <VIDEOS_DIR> [OUTPUT_JSON]
set -euo pipefail

VIDEOS_DIR="${1:?provide videos dir as first arg}"
OUTPUT_JSON="${2:-$(dirname "${VIDEOS_DIR}")/videoalign_hpsv3_scores.json}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "${REPO_DIR}/config/paths.sh"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_VIDEO}"

echo "==========================================================="
echo "Videos:  ${VIDEOS_DIR}"
echo "Output:  ${OUTPUT_JSON}"
nvidia-smi -L 2>&1 || true
echo "==========================================================="

cd "${REPO_DIR}"
python scripts/eval/score_videoalign_hpsv3.py \
    --videos_dir "${VIDEOS_DIR}" \
    --output_json "${OUTPUT_JSON}"

echo "Done -> ${OUTPUT_JSON}"
