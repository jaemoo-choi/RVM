#!/bin/bash
#SBATCH --job-name=gen-sf-i2v-544
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=03:55:00
#SBATCH --output=./sbatch_logs/gen-sf-i2v-544-%j-O
#SBATCH --error=./sbatch_logs/gen-sf-i2v-544-%j-E

# Flux first frames for the SkyReels def family:
# 27k-subsample of the Self-Forcing prompts at the OFFICIAL 544x960.
# Resumable (--skip_existing) — resubmit until both splits complete.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / 4 ))
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${REPO_DIR}/config/paths.sh"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_VIDEO}"
cd "${REPO_DIR}"
# Pre-flight GPU health filter (same as eval_wan2_1_vbench_official.sh):
# drop cgroup GPUs that fail CUDA context creation; run on the survivors.
HEALTHY_GPUS=$(python - <<'PY'
import subprocess, sys, torch
ok = []
for i in range(torch.cuda.device_count()):
    try:
        r = subprocess.run([sys.executable, "-c",
            f"import torch; torch.cuda.set_device({i}); torch.randn(8, device='cuda:{i}').sum().item()"],
            capture_output=True, timeout=180)
        if r.returncode == 0: ok.append(str(i))
    except subprocess.TimeoutExpired: pass
print(",".join(ok))
PY
)
[ -z "${HEALTHY_GPUS}" ] && { echo "no healthy GPUs" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${HEALTHY_GPUS}"
NPROC=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Pre-flight healthy GPUs: ${HEALTHY_GPUS} (nproc=${NPROC})"
MASTER_PORT=$((21000 + ${SLURM_JOB_ID:-0} % 20000))

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    scripts/dataset/generate_skyreels_ref_images.py \
    --prompt_file dataset/vidprom_sf_i2v_544/test.txt \
    --output_dir dataset/vidprom_sf_i2v_544/ref_images_test \
    --height 544 --width 960 \
    --guidance_scale 3.5 --num_inference_steps 30 \
    --seed 42 --skip_existing

torchrun --nproc_per_node="${NPROC}" --master_port="$((MASTER_PORT + 1))" \
    scripts/dataset/generate_skyreels_ref_images.py \
    --prompt_file dataset/vidprom_sf_i2v_544/train.txt \
    --output_dir dataset/vidprom_sf_i2v_544/ref_images_train \
    --height 544 --width 960 \
    --guidance_scale 3.5 --num_inference_steps 30 \
    --seed 42 --skip_existing
