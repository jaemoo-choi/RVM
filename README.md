<h1 align="center">Scaling Reinforcement Learning for Diffusion Models via Velocity Matching</h1>

<p align="center">
  Jaemoo Choi<sup>1</sup>, Wei Guo<sup>1</sup>, Yuchen Zhu<sup>1</sup>, Arash Vahdat<sup>2</sup>, Molei Tao<sup>1</sup>, Julius Berner<sup>2</sup>, Yongxin Chen<sup>1,2</sup><br>
  <sup>1</sup>Georgia Institute of Technology, <sup>2</sup>NVIDIA
</p>

<p align="center">
  <a href="https://jaemoo-choi.github.io/RVM/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2608.23664"><img src="https://img.shields.io/badge/arXiv-2608.23664-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
</p>

This is the official implementation of **reward-based velocity matching (RVM)**. Diffusion models are trained
and sampled through velocity fields, so RVM fine-tunes them with rewards directly in velocity space: each
generated sample is noised once and regressed toward its velocity target with a reward-weighted
velocity-matching loss, plus an optional anchor regression term. It needs neither likelihood estimation nor
likelihood ratios, and no stored rollout trajectories. RVM recovers Reinforce Adjoint Matching at the
update level and contains DiffusionNFT as a special case.

The code trains SD3.5-Medium (text-to-image), Wan2.1-T2V-1.3B and Wan2.2-T2V-A14B (text-to-video) and
SkyReels-V1-I2V (image-to-video) with LoRA, against image and video reward mixtures, including our
RAFT-based **dynamic-tracking (DT) reward** for motion. DiffusionNFT is included as a baseline in the same
framework.

## Changelog

- **[2026]** Code released: RVM and DiffusionNFT for SD3.5-M, FLUX.1, Wan2.1-T2V-1.3B, Wan2.2-T2V-A14B (both
  MoE experts) and SkyReels-I2V, with VBench / VBench-I2V evaluation.

## 🤗 Model

| Base model | Task | `base_model` | Entry point | Config |
|---|---|---|---|---|
| [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | T2I (OCR, GenEval, PickScore) | `sd3` | `src/train.py` | `config/sd3.py` |
| [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | T2I (GenEval) | `flux` | `src/train.py` | `config/flux.py` |
| [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | T2V | `wan2_1` | `src/train.py` | `config/wan2_1.py` |
| [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | T2V (two MoE experts) | `wan2_2_a14b_t2v` | `src/train_wan2_2.py` | `config/wan2_2_a14b_t2v.py` |
| [xzyhku/SkyReels-V1-I2V](https://huggingface.co/xzyhku/SkyReels-V1-I2V) | I2V (544×960) | `skyreels_i2v` | `src/train.py` | `config/skyreels_i2v.py` |

## Methods

Every objective is written in one form, `weight · ‖y − sg[anchor + reward step]‖² + extra`
(`src/loss/objectives.py`, the form in `src/loss/box.py`), selected with `METHOD`:

| `METHOD` | Method | weight | y | anchor | extra |
|---|---|---|---|---|---|
| `rvm` | RVM | A / (G · sg(mean\|x̂₀ − x₀\|)) | x̂₀(v_θ) = x_t − t·v_θ | x₀ | β‖v_θ − v_anc‖² (anchor) |
| `diffusionnft` | DiffusionNFT | r·M/γ and (1 − r)·M/γ (adaptive norms) | x̂₀ of v⁺ = γv_θ + (1−γ)v_old and v⁻ = (1+γ)v_old − γv_θ | x₀ | β‖v_θ − v_ref‖² |

x̂₀ = x_t − t·v_θ, so RVM's x₀-space regression is the reward-weighted velocity-matching loss
A·t/(G·sg(·))·‖v_θ − sg[v_tgt]‖² with v_tgt = ε − x₀. A is the reward centred within each prompt group and
divided by a standard deviation; r is the DiffusionNFT optimality (A clipped to ±M and mapped to [0, 1]);
v_old is the rollout ("old") adapter. The anchor v_anc is the frozen base model (`BETA` > 0); `BETA=0`
trains without an anchor. All recipes sample and train without CFG (guidance scale 1).

## 🚀 Quick Started

### 1. Environment Set Up

Clone this repository and set the paths of your machine:

```bash
git clone https://github.com/jaemoo-choi/RVM.git
cd RVM
```

Every external location (conda, the two environments, HF cache, W&B entity) is read from
`config/paths.sh`. Fill in yours once, in the gitignored `config/paths.local.sh`:

```bash
cp config/paths.local.sh.example config/paths.local.sh
# edit: CONDA_SH, CONDA_ENV_IMAGE, CONDA_ENV_VIDEO, HF_HOME, WANDB_ENTITY
```

The setup scripts, launchers and evaluation scripts all read it.

What you set before the first run:

| Where | What | How |
|---|---|---|
| `config/paths.local.sh` | `CONDA_SH`, `CONDA_ENV_IMAGE`, `CONDA_ENV_VIDEO` | your conda install and where the two environments go |
| `config/paths.local.sh` | `HF_HOME` | where models are downloaded (~200 GB for all models) |
| `config/paths.local.sh` | `WANDB_ENTITY` (optional) | your W&B team or user; run `wandb login` once |
| `HF_HOME/token` | Hugging Face token | needed for gated models (SD3.5-M, FLUX.1): accept their licenses on the Hub, then `huggingface-cli login` |
| `scripts/<model>/*.sh` | `NUM_GPUS` (default 8) | GPUs on your node |
| `scripts/<model>/*.sh` | SLURM header (optional) | add your cluster's `#SBATCH` lines to submit with `sbatch` |

Runs write checkpoints and logs to `logs/<config name>/<run name>/` in the repository.

Two conda environments are used, built on a GPU node (MMCV / flash-attention
compile against CUDA):

| Env | Models | Build |
|---|---|---|
| image | SD3.5-M, FLUX.1 | `bash ops/setup/setup_image_elbo.sh` |
| video | Wan2.1, Wan2.2-A14B, SkyReels-I2V | `bash ops/setup/setup_video_elbo.sh` |

```bash
source config/paths.sh && source "${CONDA_SH}"
conda activate "${CONDA_ENV_IMAGE}"     # or "${CONDA_ENV_VIDEO}"
```

The setup scripts pin every package version (the video env keeps `transformers==4.46.1` for VBench); do
not upgrade packages inside these environments.

### 2. Model Download

Runs are offline (`HF_HUB_OFFLINE=1` in `config/paths.sh`), so download every model into `HF_HOME`
once (gated models need an accepted license and an HF token):

```bash
source config/paths.sh
bash ops/download/download_sd3_5.sh                                               # SD3.5-M
HF_HUB_OFFLINE=0 huggingface-cli download black-forest-labs/FLUX.1-dev            # FLUX.1
HF_HUB_OFFLINE=0 huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers        # Wan2.1
HF_HUB_OFFLINE=0 huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers        # Wan2.2-A14B
bash ops/download/download_skyreels.sh                                            # SkyReels-I2V
```

### 3. Reward Preparation

Rewards are combined by weight with `config.reward_fn = {name: weight}`; the registry is
`score_functions` in `src/reward/rewards.py:MultiScorer`. The environment setup scripts already pull
the checkpoints of the rewards below into `reward_ckpts/`; HF-hosted scorers need a one-time download
into `HF_HOME`.

#### Image rewards (image env)

`ocr` (PaddleOCR), `pickscore`, `clipscore`, `hpsv2`, `aesthetic`, `imagereward`, `geneval` (MMDetection):

```bash
HF_HUB_OFFLINE=0 huggingface-cli download openai/clip-vit-large-patch14           # aesthetic + clipscore
HF_HUB_OFFLINE=0 huggingface-cli download yuvalkirstain/PickScore_v1              # pickscore
HF_HUB_OFFLINE=0 huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K   # pickscore processor
HF_HUB_OFFLINE=0 python -c "import os, ImageReward as RM; RM.load('ImageReward-v1.0', download_root=os.path.join(os.environ['HF_HOME'], 'ImageReward'))"
```

#### Video rewards (video env)

`videoalign_vq` / `videoalign_mq` / `videoalign_ta` (VideoAlign), `video_hpsv3_general` /
`video_hpsv3_percentile` (HPSv3), the dynamic-tracking rewards `dynamic_degree_smooth` (**DT**) and
`dynamic_degree_bump` (**DT2**, a bounded motion window used for SkyReels-I2V; thresholds `DD_BUMP_LO/C/HI`
= 3 / 5 / 8), both on RAFT optical flow, and `vbench` (offline evaluation):

```bash
pip install hpsv3==1.0.0
HF_HUB_OFFLINE=0 huggingface-cli download KwaiVGI/VideoReward --local-dir reward_ckpts/VideoReward
```

#### VLM judge rewards (optional)

`videoscore2`, `unifiedreward2`, `spectra` and `semantic_v2a` / `semantic_v4c` use Qwen2.5-VL judges that
need `transformers >= 4.49`. They run in a per-rank subprocess with a separate package overlay
(`VLM_OVERLAY_PYTHONPATH`, see `src/reward/vlm_reward_worker.py`), so the training environment stays pinned.

### 4. Dataset

| Model(s) | Training prompts / conditions | Evaluation |
|---|---|---|
| SD3.5-M, FLUX.1 | `dataset/{ocr,geneval,pickscore}` | `dataset/{ocr,geneval,drawbench}` |
| Wan2.1, Wan2.2-A14B | `dataset/vidprom_filtered_extended` (VidProM prompts, extended; 248,221 train / 100 test) | `dataset/vbench` (VBench, 946 prompts) |
| SkyReels-I2V | `dataset/vidprom_sf_i2v_544` (27k prompts with FLUX.1-dev first frames at 544×960) | `dataset/vbench_i2v_a14b` (VBench-I2V) |

The prompt files are in the repository; the Wan training prompts are stored compressed:

```bash
gunzip -k dataset/vidprom_filtered_extended/train.txt.gz
```

The SkyReels-I2V images are not in git: the training first frames (`dataset/vidprom_sf_i2v_544/ref_images_{train,test}`,
16 GB) and the VBench-I2V evaluation frames (`dataset/vbench_i2v_a14b`) will be attached to the
[GitHub releases](https://github.com/jaemoo-choi/RVM/releases) (coming soon). The training first frames can also be
regenerated with FLUX.1-dev: `bash scripts/dataset/gen_vidprom_sf_i2v_544.sh`.

### 5. Start Training

Each launcher in `scripts/<model>/` is one recipe: it exports the method's environment variables (read by
`config/base.py`) and runs `torchrun` on one node with `NUM_GPUS` GPUs (8 by default). For SLURM, add your
cluster's `#SBATCH` lines and submit it with `sbatch`.
Each epoch samples, scores and trains in the same job; a checkpoint is written at the start of every
`config.save_freq`-th epoch, and a restarted job resumes from the last one.

#### RVM

```bash
bash scripts/wan2_1/wan2_1_def_epg.sh                 # Wan2.1-T2V-1.3B
bash scripts/wan2_2/wan2_2_a14b_t2v_def_alt_epg.sh    # Wan2.2-T2V-A14B (both experts)
bash scripts/skyreels/skyreels_def_epg.sh             # SkyReels-I2V
bash scripts/sd3/sd3_ocr_rvm_orig.sh                  # SD3.5-M on OCR
```

#### DiffusionNFT

```bash
bash scripts/wan2_1/wan2_1_def_nft.sh
bash scripts/wan2_2/wan2_2_a14b_t2v_def_alt_nft.sh
bash scripts/skyreels/skyreels_def_nft.sh
```

#### Wan2.2-T2V-A14B (two experts)

Both MoE experts are trained off one rollout, alternating per optimizer step
(`GRADIENT_STEP_PER_EPOCH=N` → 2N updates: high, low, …). One expert is GPU-resident at a time
(`BOTH_EXPERTS_RESIDENT=1` keeps both); checkpoints are written per expert in `checkpoints/{high,low}/`.

## 📊 Evaluation

| Script | Model | Benchmark |
|---|---|---|
| `scripts/eval/eval_wan2_1_vbench_official.sh <run_dir>` | Wan2.1-T2V | VBench (946 prompts, 16 dimensions) |
| `scripts/eval/eval_wan2_2_a14b_t2v_vbench_official.sh <run_dir>` | Wan2.2-T2V-A14B | VBench (946 prompts, 16 dimensions) |
| `scripts/eval/eval_skyreels_i2v_vbench.sh <run_dir>` | SkyReels-I2V | VBench-I2V (10 dimensions) |
| `scripts/eval/eval_sd3_ocr.sh <run_dir>`, `eval_sd3_drawbench.sh <run_dir>` | SD3.5-M | OCR / DrawBench rewards |

`src/evaluation.py --lora_policy current` (the default) evaluates `lora_current_policy`, the adapter
training samples with; `--lora_policy ema` evaluates the EMA adapter. The sampler settings (CFG scale,
negative prompt) are read from the run's `config.json`. For SD3 / FLUX, `--diversity_seeds K` also
reports within-prompt diversity (DINOv2 / CLIP pairwise distance, Vendi score).

## How to Support Other Models

1. `src/models/<model>.py`: a `TrainingUtils` subclass of `src/models/base.py` (model loading, LoRA
   config, `_branch` / `_pair` forwards, rollout).
2. `src/models/pipelines/pipeline_with_logprob_<model>.py`: the sampling loop, calling
   `src/models/guidance.py:cfg` for guidance.
3. Register it in `src/models/__init__.py:model_utils_class` and add a config in `config/<model>.py`.

The training loop is shared: `src/train.py:train(run)` runs every model through the same epoch
skeleton, and a model that needs a different training step (like Wan2.2's two experts,
`src/engine/run_experts.py`) subclasses `src/engine/run.py:Run` and overrides only those steps.

## 🏁 Multi Reward Training

```python
config.reward_fn = {
    "video_hpsv3_general": 0.1,
    "video_hpsv3_percentile": 0.1,
    "videoalign_mq": 1.0,
    "videoalign_ta": 1.5,
    "dynamic_degree_smooth": 0.7,
}
```

This is the default video mixture (VideoAlign TA and MQ, HPSv3, DT); SkyReels-I2V replaces DT by DT2 at the
same weight. The advantage is computed from the weighted sum (`reward/avg`); `config.train.whiten_rewards = True`
standardises each component over the batch before the sum (`src/train.py` models). Wan2.2 can train its two experts on different reward mixes
(`REWARD_FN_HIGH_NOISE` / `REWARD_FN_LOW_NOISE`).

## ✨ Important Hyperparameters

| Variable | Meaning |
|---|---|
| `METHOD` | `rvm` or `diffusionnft` |
| `ADV` | advantage: `standard` (SCALE·(R − mean)/std), `exact` (BETA·(R − mean)/ALPHA), `diffusionnft` |
| `SCALE`, `BETA`, `ALPHA` | advantage scale; `BETA` is also the weight of the β‖v_θ − v_ref‖² term |
| `GRADIENT_STEP_PER_EPOCH` | optimizer steps per rollout epoch |
| `DECAY_TYPE` | old-adapter update `old ← d·old + (1 − d)·current` (0: old = current every epoch) |
| `LEARNING_RATE`, `MAX_GRAD_NORM`, `MAX_EPOCHS` | optimizer settings and run length |
| `config.sample.num_image_per_prompt` | group size K (samples per prompt) |
| `GUIDANCE_SCALE` / `TRAIN_GUIDANCE_SCALE` / `REF_GUIDANCE_SCALE` | CFG scale of the rollout / trainable forward / frozen reference (1 in every recipe) |
| `XT_FROM_LATENTS` | train on the visited rollout states (1) or re-noise x₀ at the rollout's t (0) |

## 🤗 Acknowledgement

This repo extends the codebase of [Rethinking the Design Space of Reinforcement Learning for Diffusion
Models](https://github.com/jaemoo-choi/rethinking-diffusion-rl) (ICML 2026) to video models, and builds on the
diffusion RL ecosystem, including [Flow-GRPO](https://github.com/yifan123/flow_grpo),
[DiffusionNFT](https://github.com/NVlabs/DiffusionNFT), [DanceGRPO](https://github.com/XueZeyue/DanceGRPO) and
[diffusers](https://github.com/huggingface/diffusers). Reward models are adapted from
[GenEval](https://github.com/djghosh13/geneval), [PickScore](https://github.com/yuvalkirstain/PickScore),
[ImageReward](https://github.com/THUDM/ImageReward), [HPSv2](https://github.com/tgxs002/HPSv2),
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [VideoAlign](https://github.com/KwaiVGI/VideoAlign),
[HPSv3](https://github.com/MizzenAI/HPSv3) and [VBench](https://github.com/Vchitect/VBench). We thank the
authors for open-sourcing their work.

## License

The code is released under the Apache-2.0 license (`LICENSE`). The VidProM prompts are under CC BY-NC 4.0, and
the SkyReels-I2V first frames were generated with FLUX.1-dev and follow the FLUX.1 [dev] Non-Commercial License.
Pretrained models and reward models keep their own licenses.

## ⭐Citation

```bibtex
@article{choi2026scaling,
  title   = {Scaling Reinforcement Learning for Diffusion Models via Velocity Matching},
  author  = {Choi, Jaemoo and Guo, Wei and Zhu, Yuchen and Vahdat, Arash and Tao, Molei and Berner, Julius and Chen, Yongxin},
  journal = {arXiv preprint arXiv:2608.23664},
  year    = {2026}
}

@inproceedings{choi2026rethinking,
  title     = {Rethinking the Design Space of Reinforcement Learning for Diffusion Models: On the Importance of Likelihood Estimation Beyond Loss Design},
  author    = {Choi, Jaemoo and Zhu, Yuchen and Guo, Wei and Molodyk, Petr and Yuan, Bo and Bai, Jinbin and Xin, Yi and Tao, Molei and Chen, Yongxin},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026}
}
```
