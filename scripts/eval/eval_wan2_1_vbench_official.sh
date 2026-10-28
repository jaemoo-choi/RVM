#!/bin/bash
#SBATCH --job-name=eval_wan_vbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=03:55:00
#SBATCH --output=./sbatch_logs/eval-%x-%j-O
#SBATCH --error=./sbatch_logs/eval-%x-%j-E

# VBench leaderboard-comparable evaluation for a WAN2.1 RL checkpoint.
#
# Uses the OFFICIAL VBench protocol: 946-prompt suite from VBench_full_info.json,
# `mode="vbench_standard"` so all 16 dims (incl. object_class, color, etc.)
# score correctly via auxiliary_info, official Quality / Semantic / Total
# aggregation. Sampling stays training-faithful (cfg, time_fraction, num_steps,
# solver from the run's training config).
#
# Usage:
#   sbatch scripts/eval/eval_wan2_1_vbench_official.sh <CHECKPOINT_PATH>
#
# Env knobs:
#   NUM_VIDEOS_PER_PROMPT  (default 1)  — set to 5 for full leaderboard convention.
#   START_IDX / END_IDX                 — shard the 946-prompt suite for multi-job runs.
#                                          Leave unset to do the full suite.
#   SKIP_GENERATION (1/0, default 0)    — only run scoring on existing mp4s.
#   SKIP_SCORING    (1/0, default 0)    — only generate, don't score (use across shards).
#   NEGATIVE_PROMPT — negative prompt for CFG. DEFAULT (unset) = the Wan repo's
#                                          official Chinese sample_neg_prompt (since
#                                          2026-08-23). "empty" = pre-2026-08-23 empty
#                                          NP (reproduces the old base cfg5 column);
#                                          literal text used verbatim. Only matters at
#                                          cfg>1 (at cfg=1 the negative embed is unused).
#
# Compute estimate: 946 × N videos × ~30s/H100. N=1 ≈ 6 GPU-h, fits one job.
# N=5 ≈ 32 GPU-h, needs 2 shards (each: START_IDX=0/473, END_IDX=473/946) +
# a final SKIP_GENERATION=1 scoring job over the whole videos/ dir.

set -euo pipefail

CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: provide the training run dir as the first argument." >&2
    exit 1
fi

NUM_VIDEOS_PER_PROMPT="${NUM_VIDEOS_PER_PROMPT:-1}"
# Default to the Self-Forcing extended (Qwen2.5-7B-rewritten) VBench prompts,
# generated-from / scored-against the 946-prompt all_dimension suite. Override
# with PROMPT_SOURCE=wanx_extended_qwen14b/qwen3b for the Wan2.1 prompt rewrites.
PROMPT_SOURCE="${PROMPT_SOURCE:-self_forcing_extended}"
START_IDX="${START_IDX:-0}"
END_IDX_ARG=""
if [ -n "${END_IDX:-}" ]; then
    END_IDX_ARG="--end_prompt_idx ${END_IDX}"
fi
EXTRA_FLAGS=""
# LORA_POLICY (current|ema): which saved adapter to evaluate. Wan2.1 evaluates
# the NON-EMA `current` policy (checkpoints/lora_current_policy, the policy
# training actually samples with) by default. Before
# that every eval silently used lora_ema_policy; pass LORA_POLICY=ema to
# reproduce those numbers (they live in evaluation_vbench_official_ema/).
LORA_POLICY="${LORA_POLICY:-current}"
EXTRA_FLAGS="${EXTRA_FLAGS} --lora_policy ${LORA_POLICY}"
if [ "${SKIP_GENERATION:-0}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --skip_generation"
fi
if [ "${SKIP_SCORING:-0}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --skip_scoring"
fi
if [ "${NO_LORA:-0}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --no_lora"
fi
if [ -n "${GUIDANCE_SCALE:-}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --guidance_scale ${GUIDANCE_SCALE}"
fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --num_inference_steps ${NUM_INFERENCE_STEPS}"
fi
# Custom sigma subset of the trained schedule (comma-separated, descending, no
# trailing 0). Overrides NUM_INFERENCE_STEPS; num_steps becomes len(sigmas).
if [ -n "${SIGMAS:-}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --sigmas ${SIGMAS}"
fi
# Trimmed prompt-suite support (reward_eval pipeline): a reduced full_info json
# + a PAIRED extended-prompt file of the SAME length. START/END then index into
# the trimmed suite.
if [ -n "${VBENCH_FULL_INFO:-}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --vbench_full_info_path ${VBENCH_FULL_INFO}"
fi
if [ -n "${PROMPT_AUG_FILE:-}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --prompt_aug_file ${PROMPT_AUG_FILE}"
fi
if [ -n "${SDE_ETA:-}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --sde_eta ${SDE_ETA}"
fi
if [ -n "${VIDEO_IDX_FILTER:-}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --video_idx_filter ${VIDEO_IDX_FILTER}"
fi
if [ -n "${NUM_FRAMES:-}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --num_frames ${NUM_FRAMES}"
fi
if [ "${SEMANTIC_ONLY:-0}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --semantic_only"
fi
# Allow overriding the default <ckpt>/evaluation_vbench_official path
# (useful when --no_lora and we want a separate baseline output dir).
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_PATH}/evaluation_vbench_official}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ulimit -n 1048576 || true
# SLURM cgroups already isolate GPUs to this job and expose them as
# 0..N-1 in nvidia-smi inside the cgroup. SLURM_JOB_GPUS holds the host's
# physical indices (e.g. "4,5,6,7"), which would point OUTSIDE the cgroup
# if assigned to CUDA_VISIBLE_DEVICES — leaving PyTorch with 0 GPUs.
# Don't override CUDA_VISIBLE_DEVICES here.

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_DIR}/config/paths.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "${CONDA_SH}"
conda activate "${CONDA_ENV_VIDEO}"

echo "DEBUG SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  nvidia-smi-L:"
nvidia-smi -L 2>&1 || true
# Pre-flight GPU health filter (2026-09-11): the cluster has scattered dead
# H100s that report idle (Default mode, 0 MiB, no ECC errors) but fail CUDA
# context creation with "device busy or unavailable". Probe every
# cgroup-visible device in a subprocess and keep only the working ones.
# Indices are cgroup-relative (0..N-1), so exporting CUDA_VISIBLE_DEVICES
# from them stays inside the cgroup (unlike SLURM_JOB_GPUS physical indices).
HEALTHY_GPUS=$(python - <<'PY'
import subprocess, sys, torch
ok = []
for i in range(torch.cuda.device_count()):
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             f"import torch; torch.cuda.set_device({i}); "
             f"torch.randn(8, device='cuda:{i}').sum().item()"],
            capture_output=True, timeout=180)
        if r.returncode == 0:
            ok.append(str(i))
    except subprocess.TimeoutExpired:
        pass
print(",".join(ok))
PY
)
if [ -z "${HEALTHY_GPUS}" ]; then
    echo "Pre-flight: NO healthy GPUs on $(hostname) - aborting." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${HEALTHY_GPUS}"
echo "Pre-flight healthy GPUs (cgroup indices): ${HEALTHY_GPUS}"

GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "==========================================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "GPUs: ${GPUS}"
echo "Videos/prompt: ${NUM_VIDEOS_PER_PROMPT}"
echo "Prompt source: ${PROMPT_SOURCE}"
echo "Prompt slice: [${START_IDX}, ${END_IDX:-end})"
echo "Extra flags: ${EXTRA_FLAGS}"
echo "==========================================================="

cd "${REPO_DIR}"

torchrun --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}" \
    src/evaluation.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --vbench_official \
    --prompt_source "${PROMPT_SOURCE}" \
    --num_videos_per_prompt "${NUM_VIDEOS_PER_PROMPT}" \
    --start_prompt_idx "${START_IDX}" \
    ${END_IDX_ARG} \
    --eval_batch_size 4 \
    --output_dir "${OUTPUT_DIR}" \
    ${EXTRA_FLAGS}

echo ""
echo "Done. Aggregate at: ${CHECKPOINT_PATH}/evaluation_vbench_official/average_scores_vbench_official.json"
