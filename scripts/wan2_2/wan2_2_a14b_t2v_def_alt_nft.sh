#!/bin/bash
# Training recipe: config/wan2_2_a14b_t2v.py:wan2_2_a14b_t2v_def_alt_diffusionnft (src/train_wan2_2.py).
# Run on one node with NUM_GPUS GPUs:  bash scripts/wan2_2/wan2_2_a14b_t2v_def_alt_nft.sh
# (for SLURM, add your cluster's #SBATCH lines and submit with sbatch)

export NUM_GPUS=8
export SAMPLE_BSZ=8
export TASK="videoalign"
export MAX_EPOCHS=90
export METHOD="diffusionnft"
export ADV="diffusionnft"
export BETA=0.0001
export SCALE=1.0
export KL_METHOD="girsanov"
export ALPHA=0.0001
export ELBO="adaptive"
export DECAY_TYPE=2
export MAX_GRAD_NORM=1.0
export GRADIENT_STEP_PER_EPOCH=1
export GUIDANCE_SCALE=1.0
export TRAIN_GUIDANCE_SCALE=1.0
export REF_GUIDANCE_SCALE=1.0
export LEARNING_RATE=5e-5
export XT_FROM_LATENTS=0
export DIFFUSIONNFT_BETA=0.1
export SEQUENTIAL_EXPERT_SAMPLING=1
export BOTH_EXPERTS_RESIDENT=0
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
torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NUM_GPUS}" src/train_wan2_2.py \
    --config config/wan2_2_a14b_t2v.py:wan2_2_a14b_t2v_def_alt_diffusionnft
