#!/bin/bash
#SBATCH --job-name=eval_skyreels_vbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=04:00:00
#SBATCH --output=./sbatch_logs/eval-%x-%j-O
#SBATCH --error=./sbatch_logs/eval-%x-%j-E

# Stage 1 of VBench-I2V eval for a SkyReels-I2V LoRA checkpoint: GENERATE
# videos on the 1118-prompt vbench_i2v set. Stage 2 (official VBench-I2V
# dimension scoring via scripts/eval/score_vbench_i2v.py) runs automatically
# after generation (SCORE_AFTER=1) — see the printed command at the end.
#
# Usage: sbatch scripts/eval/eval_skyreels_i2v_vbench.sh <RUN_DIR>
#   <RUN_DIR> is the training run directory; evaluation.py reads
#   <RUN_DIR>/checkpoints/{config.json,lora_ema_policy} (prefers the EMA LoRA).
#
# Note: vbench_i2v ref images are 640x400 = SkyReels native geometry, so the
# ImagePromptDataset resize+center-crop is a no-op here. evaluation.py forces
# prompt_fn=image_prompt for skyreels_i2v, so the embed checkpoint loads its
# text encoders + VAE normally (no precomputed embeddings needed).
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: provide the training run dir as the first argument." >&2
    exit 1
fi

GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_PATH}/evaluation_vbench_i2v}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Per-rank shard files make the eval resumable: a resubmit after the 4h cap
# picks up where the previous walltime cut it off. Bump the dist timeout past
# walltime so the final all_gather barrier doesn't kill a near-complete run
# while one slow rank finishes.
export DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-300}"
ulimit -n 1048576 || true
# SLURM cgroups already restrict GPU visibility; SLURM_JOB_GPUS is colon-
# separated which CUDA misreads as one device, so leave CUDA_VISIBLE_DEVICES unset.
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_DIR}/config/paths.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Tiled VAE decode. The _def family runs at SkyReels' official 544x960, where the
# untiled 3D-conv decode of 53 frames needs >79 GB and OOMs on an H100 (see
# src/models/skyreels_i2v.py). Every skyreels training script exports this; the eval
# script did not, so its decode blew up while the same geometry trained fine.
export VAE_TILING="${VAE_TILING:-1}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_VIDEO}"

cd "${REPO_DIR}"

# I2V prompt/ref-image suite. Default to vbench_i2v_a14b (the 1118-prompt
# VBench-I2V suite, with camera_motion entries). Override with DATASET=vbench_i2v.
DATASET="${DATASET:-vbench_i2v_a14b}"
NO_LORA_FLAG=""; [ "${NO_LORA:-0}" = "1" ] && NO_LORA_FLAG="--no_lora"
RANGE_FLAGS=""
[ -n "${START_IDX:-}" ] && RANGE_FLAGS="${RANGE_FLAGS} --start_prompt_idx ${START_IDX}"
[ -n "${END_IDX:-}" ] && RANGE_FLAGS="${RANGE_FLAGS} --end_prompt_idx ${END_IDX}"
META="${REPO_DIR}/dataset/${DATASET}/ref_images_test/metadata.jsonl"
if [ ! -s "${META}" ]; then
    echo "ERROR: ${META} missing or empty." >&2
    exit 64
fi

GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[plan] GPUs=${GPUS}  cfg=${GUIDANCE_SCALE}  steps=${NUM_INFERENCE_STEPS}  dataset=${DATASET}"
echo "[plan] checkpoint: ${CHECKPOINT_PATH}"
echo "[plan] output_dir: ${OUTPUT_DIR}"

torchrun --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}" \
    src/evaluation.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --model_type skyreels_i2v \
    ${NO_LORA_FLAG} ${RANGE_FLAGS} \
    --dataset "${DATASET}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --eval_batch_size 1 \
    --save_images \
    --reward_scorers "videoalign_mq:1.0" \
    --output_dir "${OUTPUT_DIR}"

# Stage 2: official VBench-I2V scoring (single GPU). Runs automatically after
# generation unless SCORE_AFTER=0. Scores all 10 I2V dims (i2v_subject,
# i2v_background, camera_motion + 7 quality) via the official leaderboard math.
if [ "${SCORE_AFTER:-1}" = "1" ]; then
    echo ""
    if [ "${SKIP_SCORING:-0}" = "1" ]; then
        echo "[stage-2] SKIP_SCORING=1 — generation-only shard, exiting before scoring."
        exit 0
    fi
    echo "[stage-2] scoring VBench-I2V on ${OUTPUT_DIR}/images ..."
    python scripts/eval/score_vbench_i2v.py \
        --videos_dir "${OUTPUT_DIR}/images" \
        --metadata_jsonl "${META}" \
        --output_dir "${OUTPUT_DIR}/vbench_i2v_eval"
else
    echo ""
    echo "[stage-2] SCORE_AFTER=0; score later with:"
    echo "  python scripts/eval/score_vbench_i2v.py --videos_dir ${OUTPUT_DIR}/images --metadata_jsonl ${META} --output_dir ${OUTPUT_DIR}/vbench_i2v_eval"
fi
