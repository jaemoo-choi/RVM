#!/bin/bash
#SBATCH --job-name=dl-wan2_2_a14b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --output=./sbatch_logs/dl-wan2_2_a14b-O
#SBATCH --error=./sbatch_logs/dl-wan2_2_a14b-E

# Download Wan2.2-I2V-A14B (dual-expert: high-noise transformer +
# low-noise transformer_2). ~28B total params on disk; ~14B active per
# forward. Authoritative inference defaults from HF model_index.json and
# scheduler_config.json (verified 2026-05-02):
#   - boundary_ratio = 0.9   (NOT 0.875)
#   - scheduler      = UniPCMultistepScheduler (solver_order=2,
#                       flow_shift=3.0, prediction_type="flow_prediction")
#   - transformer.in_channels = 36   (16 noise + 20 condition)
#   - transformer.image_dim   = null (no CLIP image encoder)
# Sanity-checks the offline load and prints both expert architectures.

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${REPO_DIR}/config/paths.sh"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

source "${CONDA_SH}"
conda activate "${CONDA_ENV_VIDEO}"

cd "${REPO_DIR}"

python <<'PY'
import os
from huggingface_hub import snapshot_download

cache_dir = os.path.join(os.environ["HF_HOME"], "hub")
model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

print(f"Downloading {model_id} to {cache_dir} ...")
path = snapshot_download(model_id, cache_dir=cache_dir, repo_type="model")
print(f"Done: {path}")

# Offline sanity check.
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from diffusers import WanImageToVideoPipeline
pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print("Pipeline loaded offline OK")
print("  transformer       :", type(pipe.transformer).__name__,
      "in_channels=", getattr(pipe.transformer.config, "in_channels", None))
print("  transformer_2     :", type(pipe.transformer_2).__name__ if pipe.transformer_2 is not None else "None")
print("  vae               :", type(pipe.vae).__name__,
      "z_dim=", getattr(pipe.vae.config, "z_dim", None))
print("  text_encoder      :", type(pipe.text_encoder).__name__)
print("  scheduler         :", type(pipe.scheduler).__name__)
print("  boundary_ratio    :", getattr(pipe.config, "boundary_ratio", None))
print("  expand_timesteps  :", getattr(pipe.config, "expand_timesteps", None))

assert pipe.transformer_2 is not None, (
    "Wan2.2-I2V-A14B must populate transformer_2; got None — wrong checkpoint?"
)
# Pin the authoritative settings sourced upstream so a silent HF retag
# wouldn't go unnoticed.
br = getattr(pipe.config, "boundary_ratio", None)
assert br == 0.9, f"Expected boundary_ratio=0.9, got {br!r} — HF config drift?"
in_ch = getattr(pipe.transformer.config, "in_channels", None)
assert in_ch == 36, f"Expected transformer.in_channels=36, got {in_ch!r}"
img_dim = getattr(pipe.transformer.config, "image_dim", "MISSING")
assert img_dim is None, (
    f"Expected transformer.image_dim=None (no CLIP), got {img_dim!r}. "
    "Pipeline patch hardcodes encoder_hidden_states_image=None."
)
print("All authoritative-setting assertions passed.")
PY
