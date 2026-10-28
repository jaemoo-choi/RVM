#!/bin/bash
# Training recipe: config/wan2_1.py:wan2_1_def_cfgdrop (src/train.py).
# Run on one node with NUM_GPUS GPUs:  bash scripts/wan2_1/wan2_1_def_cfgdrop.sh
# (for SLURM, add your cluster's #SBATCH lines and submit with sbatch)

export NUM_GPUS=8
export TASK="videoalign"
export METHOD="rvm"
export SKIP_OLD_FORWARD=1
export KL_METHOD="girsanov"
export ADV="standard"
export BETA=0
export ALPHA=0.0001
export SCALE=0.1
export ELBO="adaptive"
export DECAY_TYPE=0
export MAX_GRAD_NORM=1.0
export GRADIENT_STEP_PER_EPOCH=2
export MAX_EPOCHS=90
export GUIDANCE_SCALE=1.0
export TRAIN_GUIDANCE_SCALE=1.0
export REF_GUIDANCE_SCALE=1.0
export CFG_DROP_PROB=0.05
export NP_BASE_VELOCITY=0
export NEGATIVE_PROMPT=official
export LEARNING_RATE=5e-5
export XT_FROM_LATENTS=0
export NCCL_TIMEOUT_SECONDS=7200
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / ${NUM_GPUS:-8} ))
[ "${OMP_NUM_THREADS}" -lt 1 ] && export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export VIDEOALIGN_DEBUG_DUMP_DIR="${REPO_DIR}/logs/videoalign_hangs"
source "${REPO_DIR}/config/paths.sh"
cd "${REPO_DIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_VIDEO}"

MASTER_PORT=$((29500 + RANDOM % 1000))
torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NUM_GPUS}" src/train.py \
    --config config/wan2_1.py:wan2_1_def_cfgdrop
