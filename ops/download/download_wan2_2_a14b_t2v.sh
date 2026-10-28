#!/bin/bash
#SBATCH --job-name=dl-wan2_2_a14b_t2v
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=08:00:00
#SBATCH --output=./sbatch_logs/dl-wan2_2_a14b_t2v-O
#SBATCH --error=./sbatch_logs/dl-wan2_2_a14b_t2v-E

# Download Wan2.2-T2V-A14B (dual-expert MoE: high-noise `transformer` +
# low-noise `transformer_2`), the text-to-video sibling of the I2V A14B.
# ~28B total on disk, ~14B active per forward.
#
# Unlike the I2V variant we do NOT hardcode boundary_ratio / in_channels
# assertions: the T2V authoritative values differ from I2V (I2V uses
# boundary_ratio=0.9 and in_channels=36 for the 16ch noise + 20ch condition).
# T2V has no image conditioning, so in_channels should be 16. We PRINT the
# real values and assert only structural invariants, then bake the observed
# boundary_ratio into config/wan2_2_a14b_t2v.py.

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
model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

print(f"Downloading {model_id} to {cache_dir} ...", flush=True)
path = snapshot_download(model_id, cache_dir=cache_dir, repo_type="model")
print(f"Done: {path}", flush=True)

os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print("Pipeline loaded offline OK")
print("  transformer      :", type(pipe.transformer).__name__,
      "in_channels=", getattr(pipe.transformer.config, "in_channels", None))
t2 = getattr(pipe, "transformer_2", None)
print("  transformer_2    :", type(t2).__name__ if t2 is not None else "None")
print("  vae              :", type(pipe.vae).__name__,
      "z_dim=", getattr(pipe.vae.config, "z_dim", None))
print("  text_encoder     :", type(pipe.text_encoder).__name__)
print("  scheduler        :", type(pipe.scheduler).__name__)
print("  boundary_ratio   :", getattr(pipe.config, "boundary_ratio", None))
print("  expand_timesteps :", getattr(pipe.config, "expand_timesteps", None))

# Structural invariants only (values differ from I2V; see header).
assert t2 is not None, "Wan2.2-T2V-A14B must populate transformer_2 (dual-expert MoE)"
in_ch = getattr(pipe.transformer.config, "in_channels", None)
assert in_ch == 16, f"T2V expects transformer.in_channels=16 (no image cond), got {in_ch!r}"
br = getattr(pipe.config, "boundary_ratio", None)
assert br is not None and 0.0 < float(br) < 1.0, f"boundary_ratio looks wrong: {br!r}"
print(f"\nBAKE THIS INTO config/wan2_2_a14b_t2v.py -> boundary_ratio={br}")
print("Structural assertions passed.")
PY
