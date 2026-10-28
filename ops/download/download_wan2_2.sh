#!/bin/bash
#SBATCH --job-name=dl-wan2_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --output=./sbatch_logs/dl-wan2_2-O
#SBATCH --error=./sbatch_logs/dl-wan2_2-E

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
model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

print(f"Downloading {model_id} to {cache_dir} ...")
path = snapshot_download(model_id, cache_dir=cache_dir, repo_type="model")
print(f"Done: {path}")

# Offline sanity check
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from diffusers import WanImageToVideoPipeline
pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print("Pipeline loaded offline OK")
print("  transformer:", type(pipe.transformer).__name__)
print("  vae:", type(pipe.vae).__name__)
print("  text_encoder:", type(pipe.text_encoder).__name__)
vae_cfg = pipe.vae.config
for k in ("scale_factor_spatial", "scale_factor_temporal", "latents_mean", "latents_std"):
    if hasattr(vae_cfg, k):
        v = getattr(vae_cfg, k)
        if isinstance(v, list):
            print(f"  vae.config.{k} = list(len={len(v)})")
        else:
            print(f"  vae.config.{k} = {v}")
PY
