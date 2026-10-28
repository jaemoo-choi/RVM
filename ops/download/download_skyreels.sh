#!/bin/bash
#SBATCH --job-name=dl-skyreels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --output=./sbatch_logs/dl-skyreels-O
#SBATCH --error=./sbatch_logs/dl-skyreels-E

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
model_id = "xzyhku/SkyReels-V1-I2V"

print(f"Downloading {model_id} to {cache_dir} ...")
path = snapshot_download(model_id, cache_dir=cache_dir, repo_type="model")
print(f"Done: {path}")

# Offline sanity check — SkyReels-V1-I2V uses HunyuanVideoPipeline as its base class
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from diffusers import HunyuanVideoPipeline
pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print("Pipeline loaded offline OK")
print("  transformer:", type(pipe.transformer).__name__)
print("  vae:", type(pipe.vae).__name__)
print("  text_encoder:", type(pipe.text_encoder).__name__)
print("  text_encoder_2:", type(pipe.text_encoder_2).__name__)
PY
