#!/bin/bash
#SBATCH --job-name=dl-wan2_1_14b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=02:00:00
#SBATCH --output=./sbatch_logs/dl-wan2_1_14b-O
#SBATCH --error=./sbatch_logs/dl-wan2_1_14b-E

# Download Wan-AI/Wan2.1-T2V-14B-Diffusers (~28-30 GB) to HF_HOME and run an
# offline sanity load to confirm the pipeline imports without network.

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
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

print(f"Downloading {model_id} to {cache_dir} ...")
path = snapshot_download(model_id, cache_dir=cache_dir, repo_type="model")
print(f"Done: {path}")

# Offline sanity check.
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print("Pipeline loaded offline OK")
print("  transformer       :", type(pipe.transformer).__name__,
      "in_channels=", getattr(pipe.transformer.config, "in_channels", None))
print("  vae               :", type(pipe.vae).__name__,
      "z_dim=", getattr(pipe.vae.config, "z_dim", None))
print("  text_encoder      :", type(pipe.text_encoder).__name__)
print("  scheduler         :", type(pipe.scheduler).__name__,
      "flow_shift=", pipe.scheduler.config.get("flow_shift"))
PY
