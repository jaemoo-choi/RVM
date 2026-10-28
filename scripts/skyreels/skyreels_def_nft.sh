#!/bin/bash
# Training recipe: config/skyreels_i2v.py:skyreels_def_diffusionnft (src/train.py).
# Run on one node with NUM_GPUS GPUs:  bash scripts/skyreels/skyreels_def_nft.sh
# (for SLURM, add your cluster's #SBATCH lines and submit with sbatch)

export NUM_GPUS=8
export VAE_TILING=1
export TASK="videoalign"
export MAX_EPOCHS=90
export METHOD="diffusionnft"
export KL_METHOD="girsanov"
export ADV="diffusionnft"
export BETA=0.0001
export ALPHA=0.0001
export SCALE=1.0
export ELBO="adaptive"
export DECAY_TYPE=2
export MAX_GRAD_NORM=1.0
export GRADIENT_STEP_PER_EPOCH=2
export GUIDANCE_SCALE=1.0
export TRAIN_GUIDANCE_SCALE=1.0
export REF_GUIDANCE_SCALE=1.0
export LEARNING_RATE=5e-5
export XT_FROM_LATENTS=0
export DIFFUSIONNFT_BETA=0.1
export DD_BUMP_LO=3.0
export DD_BUMP_C=5.0
export DD_BUMP_HI=8.0
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

SF_I2V_DIR="${REPO_DIR}/dataset/vidprom_sf_i2v_544"
META="${SF_I2V_DIR}/ref_images_train/metadata.jsonl"
EXPECTED=$(wc -l < "${SF_I2V_DIR}/train.txt")
if [ ! -s "${META}" ]; then
    echo "ERROR: ${META} missing or empty." >&2
    echo "  Run scripts/dataset/gen_vidprom_sf_i2v_544.sh to completion first." >&2
    exit 64
fi
HAVE=$(wc -l < "${META}")
if [ "${HAVE}" -ne "${EXPECTED}" ]; then
    echo "ERROR: ${META} has ${HAVE} entries, expected ${EXPECTED}." >&2
    echo "  Reference-image generation is incomplete, or metadata.jsonl was" >&2
    echo "  truncated by a --skip_existing resume. Rebuild it before training." >&2
    exit 64
fi

MASTER_PORT=$((29500 + RANDOM % 1000))
torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NUM_GPUS}" src/train.py \
    --config config/skyreels_i2v.py:skyreels_def_diffusionnft
