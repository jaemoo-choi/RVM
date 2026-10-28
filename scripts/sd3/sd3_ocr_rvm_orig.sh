#!/bin/bash
# Training recipe: config/sd3.py:sd3_ocr_multi_reward (src/train.py).
# Run on one node with NUM_GPUS GPUs:  bash scripts/sd3/sd3_ocr_rvm_orig.sh
# (for SLURM, add your cluster's #SBATCH lines and submit with sbatch)

export NUM_GPUS=8
export TASK="ocr"
export MAX_EPOCHS=70
export METHOD="rvm"
export ADV="exact"
export KL_METHOD="girsanov"
export BETA=0.0001
export ALPHA=0.001
export SCALE=1.0
export ELBO="adaptive"
export DECAY_TYPE=0
export DECAY_TAG=2
export MAX_GRAD_NORM=5.0
export GRADIENT_STEP_PER_EPOCH=1
export GUIDANCE_SCALE=1.0
export TRAIN_GUIDANCE_SCALE=1.0
export REF_GUIDANCE_SCALE=1.0
export LEARNING_RATE=3e-4
export XT_FROM_LATENTS=0
export NCCL_TIMEOUT_SECONDS=1800
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / ${NUM_GPUS:-8} ))
[ "${OMP_NUM_THREADS}" -lt 1 ] && export OMP_NUM_THREADS=1

export OCR_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} / ${NUM_GPUS:-8} ))
[ "${OCR_WORKERS}" -lt 1 ] && export OCR_WORKERS=1
export OCR_CPU_THREADS=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_DIR}/config/paths.sh"
cd "${REPO_DIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_IMAGE}"

MASTER_PORT=$((29500 + RANDOM % 1000))
torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NUM_GPUS}" src/train.py \
    --config config/sd3.py:sd3_ocr_multi_reward
