#!/bin/bash
#SBATCH --job-name=eval_sd3_ocr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=03:55:00
#SBATCH --output=./sbatch_logs/eval-%x-%j-O
#SBATCH --error=./sbatch_logs/eval-%x-%j-E

# SD3.5-M DrawBench evaluation — the Rethinking paper's out-of-distribution
# benchmark for the multi-reward / Pick-a-Pic task (paper §Experimental Setup:
# "training is performed on Pick-a-Pic, and evaluation is conducted on
# DrawBench").
#
# Metrics are chosen automatically by src/evaluation.py for --dataset ocr:
#     imagereward, pickscore, aesthetic, clipscore, hpsv2
# which is exactly the paper's DrawBench column set (tab:main_comp:
# "PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd").
#
# CFG: the paper evaluates DrawBench **without CFG** (main.tex:1173 — "We
# evaluate all methods on GenEval and OCR using CFG with a scale of 4.5, and on
# DrawBench without CFG"). GUIDANCE_SCALE therefore defaults to 1.0 here, which
# also matches how these runs were trained (CFG-free). Do NOT copy the 4.5 used
# for GenEval/OCR.
#
# Prompts: dataset/ocr/test.txt holds the standard 200 DrawBench prompts,
# each repeated 5x (999 lines). We generate one image per line, i.e. 5 samples
# per prompt, and let the aggregate average over all of them — more stable than
# a single sample per prompt.
#
# Usage:
#   sbatch scripts/eval/eval_sd3_drawbench.sh <RUN_DIR>          # trained LoRA
#   NO_LORA=1 sbatch scripts/eval/eval_sd3_drawbench.sh <RUN_DIR>  # base model
#
# Env overrides: GUIDANCE_SCALE, NUM_INFERENCE_STEPS, EVAL_BATCH_SIZE,
#                OUTPUT_DIR, NO_LORA, FRESH (default 1 = regenerate images).
#                DIVERSITY_SEEDS (default 0 = off; K >= 2 adds K images/prompt and writes
#                diversity.json, see README Evaluation), DIVERSITY_PROMPTS (first N prompts only).
set -euo pipefail

RUN_DIR="${1:-${RUN_DIR:-}}"
if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: provide the run dir, e.g. logs/pickscore_multi_reward/<run>" >&2
    exit 1
fi

GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"          # paper: OCR evaluated at CFG 4.5
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}" # repo eval convention (config.sample.eval_num_steps)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NO_LORA="${NO_LORA:-0}"
FRESH="${FRESH:-1}"
DIVERSITY_SEEDS="${DIVERSITY_SEEDS:-0}"
DIVERSITY_PROMPTS="${DIVERSITY_PROMPTS:-}"

TAG="ocr_cfg${GUIDANCE_SCALE}"
[ "${NO_LORA}" = "1" ] && TAG="ocr_cfg${GUIDANCE_SCALE}_base"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/evaluation_${TAG}}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${REPO_DIR}/config/paths.sh"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_IMAGE}"
cd "${REPO_DIR}"

# Eval protocol: regenerate rather than rescore. src/evaluation.py skips
# sampling when an image already exists, so a stale images/ dir would silently
# re-score old samples and report them as a fresh run.
#
# Removing images/ alone is NOT enough. evaluation.py also keeps per-rank resume
# shards (.eval_shard_rank*.jsonl) and skips every batch whose sample_ids they
# already contain. On 2026-07-29 that made a job "COMPLETE" in 4 min
# having generated nothing, re-emitting the previous run's scores bit-for-bit.
# FRESH=1 must clear the resume state too.
if [ "${FRESH}" = "1" ]; then
    if [ -d "${OUTPUT_DIR}/images" ]; then
        echo "FRESH=1 -> removing stale images at ${OUTPUT_DIR}/images"
        rm -rf "${OUTPUT_DIR}/images"
    fi
    if compgen -G "${OUTPUT_DIR}/.eval_shard_rank*.jsonl" >/dev/null; then
        echo "FRESH=1 -> removing resume shards at ${OUTPUT_DIR}/.eval_shard_rank*.jsonl"
        rm -f "${OUTPUT_DIR}"/.eval_shard_rank*.jsonl
    fi
    # Named after --dataset (always "drawbench"), not TAG: NO_LORA only changes
    # OUTPUT_DIR, so the file inside it keeps the dataset name.
    if [ -f "${OUTPUT_DIR}/evaluation_results_drawbench.jsonl" ]; then
        echo "FRESH=1 -> removing stale ${OUTPUT_DIR}/evaluation_results_drawbench.jsonl"
        rm -f "${OUTPUT_DIR}/evaluation_results_drawbench.jsonl"
    fi
    if [ -d "${OUTPUT_DIR}/.diversity_emb" ] || compgen -G "${OUTPUT_DIR}/.diversity_shard_rank*.jsonl" >/dev/null; then
        echo "FRESH=1 -> removing diversity resume state at ${OUTPUT_DIR}/.diversity_*"
        rm -rf "${OUTPUT_DIR}/.diversity_emb" "${OUTPUT_DIR}"/.diversity_shard_rank*.jsonl
    fi
fi

EXTRA_FLAGS=""
[ "${NO_LORA}" = "1" ] && EXTRA_FLAGS="--no_lora"
[ "${DIVERSITY_SEEDS}" != "0" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --diversity_seeds ${DIVERSITY_SEEDS}"
[ -n "${DIVERSITY_PROMPTS}" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --diversity_prompts ${DIVERSITY_PROMPTS}"

echo "==========================================================="
echo "Run dir      : ${RUN_DIR}"
echo "Output       : ${OUTPUT_DIR}"
echo "Guidance     : ${GUIDANCE_SCALE}  (paper: DrawBench is CFG-free)"
echo "Steps        : ${NUM_INFERENCE_STEPS}"
echo "LoRA         : $([ "${NO_LORA}" = "1" ] && echo 'DISABLED (base SD3.5-M reference)' || echo 'lora_current_policy')"
echo "Prompts      : $(wc -l < dataset/ocr/test.txt) lines ($(sort -u dataset/ocr/test.txt | wc -l) unique x5)"
echo "Diversity    : seeds=${DIVERSITY_SEEDS} prompts=${DIVERSITY_PROMPTS:-all}"
nvidia-smi -L 2>&1 || true
echo "==========================================================="

GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
MASTER_PORT="${MASTER_PORT:-$((21000 + ${SLURM_JOB_ID:-0} % 20000))}"

torchrun --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
    src/evaluation.py \
    --checkpoint_path "${RUN_DIR}" \
    --model_type sd3 \
    --dataset ocr \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --save_images \
    --output_dir "${OUTPUT_DIR}" \
    ${EXTRA_FLAGS}

echo "Done -> ${OUTPUT_DIR}/average_scores_ocr.json"
