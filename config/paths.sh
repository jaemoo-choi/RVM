#!/usr/bin/env bash
# config/paths.sh — every external path the code uses, in one place. Source it in every shell script.
# All values use ${VAR:-default}: set them in your environment or in the gitignored config/paths.local.sh.

# Machine-local overrides (gitignored): export HF_HOME=... CONDA_SH=... etc.
_PATHS_LOCAL="$(dirname "${BASH_SOURCE[0]}")/paths.local.sh"
[[ -f "${_PATHS_LOCAL}" ]] && source "${_PATHS_LOCAL}"

# ── Conda ─────────────────────────────────────────────────────
export CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
export CONDA_ENV_IMAGE="${CONDA_ENV_IMAGE:-${HOME}/anaconda3/envs/image_elbo}"     # SD3.5-M, FLUX.1
export CONDA_ENV_VIDEO="${CONDA_ENV_VIDEO:-${HOME}/anaconda3/envs/video_elbo}"     # Wan2.1, Wan2.2-A14B, SkyReels-I2V

# transformers>=4.49 pip overlay for the Qwen2.5-VL judge rewards (videoscore2 / unifiedreward2 / semantic / spectra)
export VS2_OVERLAY="${VS2_OVERLAY:-${HOME}/vs2_overlay}"

# ── HuggingFace cache ────────────────────────────────────────
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# Do NOT set TRANSFORMERS_CACHE: set to HF_HOME directly it bypasses the {HF_HOME}/hub layout and breaks
# offline loading. HF_HOME alone routes transformers to {HF_HOME}/hub.
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
# Training runs offline: download every model once first (README "Model Download").
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_PREFER_LOCAL_FILES="${HF_PREFER_LOCAL_FILES:-1}"

# ── HuggingFace token (gated models) ─────────────────────────
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HF_HOME}/token}"
if [[ -f "${HF_TOKEN_FILE}" ]]; then
    export HUGGING_FACE_HUB_TOKEN=$(cat "${HF_TOKEN_FILE}")
    export HF_TOKEN=$(cat "${HF_TOKEN_FILE}")
fi

# ── Weights & Biases (empty: your default entity) ────────────
export WANDB_ENTITY="${WANDB_ENTITY:-}"
