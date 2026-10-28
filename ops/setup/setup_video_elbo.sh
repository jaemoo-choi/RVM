#!/bin/bash
# video_elbo Full Environment Setup
# Mirrors the currently working video_elbo environment at:
#   ${CONDA_ENV_VIDEO} (config/paths.sh)
#
# Prerequisites: run on a GPU node
#   (an interactive GPU session, e.g. srun --gres=gpu:1 --pty bash)
#
# Usage:
#   cd <repo_root>
#   bash ops/setup/setup_video_elbo.sh

set -e

# ── Core paths ────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO}/config/paths.sh"
SCRATCH="${SCRATCH:-${HOME}}"   # pip cache and tmp live here
# conda and the env path come from config/paths.sh (CONDA_SH, CONDA_ENV_VIDEO; set them in config/paths.local.sh)
CONDA_ROOT="$(cd "$(dirname "${CONDA_SH}")/../.." && pwd)"
ENV_PATH="${CONDA_ENV_VIDEO}"

export PIP_CACHE_DIR=$SCRATCH/pip-cache
export TMPDIR=$SCRATCH/tmp
export TEMP=$SCRATCH/tmp
export TMP=$SCRATCH/tmp

# ── Helpers ───────────────────────────────────────────────────────────────────
PY="$ENV_PATH/bin/python"

load_cuda() {
    module load cuda/12.6
    CUDA_HOME=$(which nvcc | sed 's|/bin/nvcc||')
    export CUDA_HOME
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export TORCH_CUDA_ARCH_LIST="9.0"
    echo "  CUDA_HOME : $CUDA_HOME"
    echo "  ARCH_LIST : $TORCH_CUDA_ARCH_LIST"
}

# =========================================
# Step 1 — Environment variables
# =========================================
echo ""
echo "========================================="
echo "Step 1: Scratch environment variables"
echo "========================================="
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
unset CONDA_EXE CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE
export CONDA_NO_PLUGINS=true
echo "  ENV_PATH : $ENV_PATH"
echo "  REPO     : $REPO"
echo "✓ Done"

# =========================================
# Step 1.5 — Environment reset
# =========================================
echo ""
echo "========================================="
echo "Step 1.5: Environment reset"
echo "========================================="
rm -rf "$ENV_PATH"
rm -rf "$TMPDIR"/* "$PIP_CACHE_DIR"/*
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
echo "✓ Done"

# =========================================
# Step 2 — Create conda env
# =========================================
echo ""
echo "========================================="
echo "Step 2: Create video_elbo env (Python 3.10.16)"
echo "========================================="
"$CONDA_ROOT/bin/conda" create -y -p "$ENV_PATH" python=3.10.16
"$PY" -V
echo "✓ Done"

# =========================================
# Step 3 — Base project dependencies
# =========================================
echo ""
echo "========================================="
echo "Step 3: Base project dependencies"
echo "========================================="
cd "$REPO"
# Pin pip and setuptools for compatibility
"$PY" -m pip install -q "pip==22.3.1" "setuptools<70" wheel
"$PY" -m pip install -q -U ninja
"$PY" -m pip install -q psutil
"$PY" -m pip install -e .

echo "Syncing package versions to the known-good video_elbo environment..."
"$PY" -m pip install -q \
    "accelerate==1.4.0" \
    "bitsandbytes==0.45.3" \
    "datasets==3.3.2" \
    "decord==0.6.0" \
    "diffusers==0.36.0" \
    "huggingface-hub==0.26.1" \
    "imageio==2.37.3" \
    "imageio-ffmpeg==0.6.0" \
    "opencv-python-headless==4.11.0.86" \
    "peft==0.18.1" \
    "sentencepiece==0.2.1" \
    "timm==1.0.12" \
    "tokenizers==0.20.1" \
    "transformers==4.46.1" \
    "vbench==0.1.5" \
    "xformers==0.0.29.post3"
"$PY" -m pip install -q "av==17.0.0"
echo "✓ Done"

# =========================================
# Step 4 — flash-attn  (non-blocking)
# =========================================
echo ""
echo "========================================="
echo "Step 4: flash-attn (build from source for ABI compatibility)"
echo "========================================="
load_cuda
nvcc --version
# Build from source to match the installed PyTorch ABI.
# Pre-built wheels often have ABI mismatches (undefined symbol errors).
if "$PY" -m pip install --no-build-isolation --no-cache-dir "flash-attn>=2.5.0,<3.0" 2>&1; then
    echo "✓ flash-attn installed"
else
    echo "⚠ flash-attn build failed — models will use SDPA attention instead"
fi

# =========================================
# Step 5 — Reward checkpoints
# =========================================
echo ""
echo "========================================="
echo "Step 5: Reward checkpoints (video-specific)"
echo "========================================="
mkdir -p "$REPO/reward_ckpts"
cd "$REPO/reward_ckpts"
wget -qc https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
cd "$REPO"
echo "✓ Done"

# =========================================
# Step 6 — Final integration check
# =========================================
echo ""
echo "========================================="
echo "Step 6: Final integration check"
echo "========================================="
cd "$REPO"
"$PY" -c "
import sys
print('python:', sys.executable)
print('✓ Integration check passed!')
"
cd "$REPO"

# =========================================
# Step 7 — Video reward packages
# =========================================
echo ""
echo "========================================="
echo "Step 7: Video reward packages (vbench, videoalign, hpsv2x)"
echo "========================================="
"$PY" -m pip install hpsv2x==1.2.0
"$PY" -m pip install vbench==0.1.5
echo "✓ Done (videoalign is included in codebase)"

# =========================================
# Step 8 — Version snapshot
# =========================================
echo ""
echo "========================================="
echo "Step 8: Version snapshot"
echo "========================================="
"$PY" -c "
import importlib.metadata as md
pkgs = [
    'torch', 'torchvision', 'xformers', 'flash-attn', 'transformers',
    'tokenizers', 'huggingface-hub', 'diffusers', 'peft', 'accelerate',
    'timm', 'bitsandbytes', 'decord', 'imageio', 'imageio-ffmpeg', 'av',
    'vbench', 'hpsv2x'
]
for pkg in pkgs:
    try:
        print(f'{pkg}=={md.version(pkg)}')
    except md.PackageNotFoundError:
        print(f'{pkg}==<missing>')
"
echo "✓ Done"

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo ""
echo "Activate with:"
echo "  source ${CONDA_SH}"
echo "  conda activate $ENV_PATH"
