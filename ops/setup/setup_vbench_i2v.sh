#!/bin/bash
# Set up the VBench++ (VBench-I2V) extra dependencies in the video_elbo env.
#
# The pip `vbench2_beta_i2v` package ships the I2V dimension code but is MISSING
# two things needed for the 3 I2V-specific dims:
#   * `dreamsim`  — used by i2v_background (frame-vs-reference similarity).
#   * the vendored `third_party/cotracker` subpackage + CoTracker weights —
#     used by camera_motion (`from vbench2_beta_i2v.third_party.cotracker...`).
# i2v_subject (DINO) and the 7 quality dims already work from ~/.cache/vbench.
#
# This script (run ON A NODE WITH INTERNET — login node is fine):
#   1. pip-installs dreamsim + the cotracker package into video_elbo.
#   2. Symlinks `vbench2_beta_i2v/third_party/cotracker` -> the installed
#      `cotracker` package so the vendored import resolves.
#   3. Pre-caches the CoTracker2 + DreamSim weights (so GPU compute nodes, which
#      may be offline, don't try to download at eval time).
#   4. Best-effort fetches the official vbench2_i2v_full_info.json (reference
#      only — score_vbench_i2v.py builds its own full-info from our metadata).
#
# Usage:  bash ops/setup/setup_vbench_i2v.sh
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_DIR}/config/paths.sh"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_VIDEO}"

SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
VB_I2V="${SITE_PKGS}/vbench2_beta_i2v"
echo "[setup] site-packages: ${SITE_PKGS}"
echo "[setup] vbench2_beta_i2v: ${VB_I2V}"

# --- 1. deps. Allow online for this step even though HF is offline by default. ---
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
echo "[setup] installing dreamsim + cotracker ..."
pip install --no-input dreamsim || { echo "[setup] FAILED: pip install dreamsim"; exit 1; }
# CoTracker is not on PyPI; install from GitHub (provides the `cotracker` module).
pip install --no-input "git+https://github.com/facebookresearch/co-tracker.git" \
    || { echo "[setup] FAILED: pip install cotracker"; exit 1; }

# dreamsim pulls open-clip-torch, which bumps timm to 1.0.27 and BREAKS vbench's
# UMT human_action dim (`VisionTransformer.__init__() got unexpected kwarg
# 'cache_dir'`). vbench/pyiqa pin timm<=1.0.12 and open-clip-torch 3.3.0 works
# fine with 1.0.12, so pin it back.
pip install --no-input "timm==1.0.12" \
    || { echo "[setup] FAILED: pinning timm==1.0.12"; exit 1; }
echo "[setup] timm pinned to $(python -c 'import timm; print(timm.__version__)')"

# --- 2. wire the vendored import path expected by camera_motion.py ---
python - <<'PY'
import os, site, importlib.util
sp = site.getsitepackages()[0]
tp = os.path.join(sp, "vbench2_beta_i2v", "third_party")
os.makedirs(tp, exist_ok=True)
init = os.path.join(tp, "__init__.py")
if not os.path.exists(init):
    open(init, "a").close()
spec = importlib.util.find_spec("cotracker")
src = os.path.dirname(spec.origin)
dst = os.path.join(tp, "cotracker")
if not os.path.exists(dst):
    os.symlink(src, dst)
    print(f"[setup] symlinked {dst} -> {src}")
else:
    print(f"[setup] {dst} already present")
# sanity: the exact import camera_motion.py does
from vbench2_beta_i2v.third_party.cotracker.utils.visualizer import Visualizer  # noqa
print("[setup] vendored cotracker import OK")
PY
[ $? -ne 0 ] && { echo "[setup] FAILED wiring cotracker third_party"; exit 1; }

# --- 3. pre-cache weights to avoid downloads on offline compute nodes ---
echo "[setup] pre-caching CoTracker2 + DreamSim weights ..."
python - <<'PY'
import torch
try:
    torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    print("[setup] cotracker2 weights cached")
except Exception as e:
    print(f"[setup] WARN cotracker2 cache failed: {e}")
try:
    from dreamsim import dreamsim
    dreamsim(pretrained=True)
    print("[setup] dreamsim weights cached")
except Exception as e:
    print(f"[setup] WARN dreamsim cache failed: {e}")
PY

# --- 4. reference info json (optional; scorer doesn't depend on it) ---
INFO_JSON="${VB_I2V}/vbench2_i2v_full_info.json"
if [ ! -s "${INFO_JSON}" ]; then
    echo "[setup] fetching reference vbench2_i2v_full_info.json ..."
    wget -q -O "${INFO_JSON}" \
      "https://raw.githubusercontent.com/Vchitect/VBench/master/vbench2_beta_i2v/vbench2_i2v_full_info.json" \
      || echo "[setup] WARN: could not fetch reference info json (non-fatal)"
fi

# --- verify DINO local hub dir for i2v_subject (local=True load) ---
DINO_HUB="$(ls -d ${HOME}/.cache/vbench/dino_model/facebookresearch_dino_main 2>/dev/null)"
if [ -z "${DINO_HUB}" ]; then
    echo "[setup] NOTE: ~/.cache/vbench/dino_model/facebookresearch_dino_main not found."
    echo "        i2v_subject uses local torch.hub; if it errors, re-run a T2V"
    echo "        subject_consistency eval once to populate the DINO hub dir."
fi

echo "[setup] done. Verify with:"
echo "  python scripts_temp/verify/verify_vbench_full.py --i2v"
