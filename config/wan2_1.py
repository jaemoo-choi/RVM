import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


# Default video reward: VideoAlign TA and MQ, HPSv3 (general + percentile) and the dynamic-tracking (DT)
# reward dynamic_degree_smooth (RAFT optical flow).
_WAN2_1_REWARD_FN = {
    "video_hpsv3_general": 0.1,
    "video_hpsv3_percentile": 0.1,
    "videoalign_mq": 1.0,
    "videoalign_ta": 1.5,
    "dynamic_degree_smooth": 0.7,
}


# ============================================
# WAN2.1 ODE base config
# ============================================
def _wan2_1_base_config(reward_fn, name, frames=53, height=480, width=832, video_fps=15,
                        num_image_per_prompt=8, num_groups=32, bsz=8,
                        num_steps=16, shift=8.0,
                        pretrained_model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"):
    """Base WAN2.1 config.

    Default geometry / sampling matches the new Wan-family recipe:
    53 frames @ 480x832 / 15 fps, 16 inference steps, flow-matching shift=8.0,
    K=8 samples per prompt, 32 prompt groups per epoch.

    ``pretrained_model_id`` selects the WAN2.1 T2V variant: pass
    ``"Wan-AI/Wan2.1-T2V-14B-Diffusers"`` for the 14B checkpoint. Architecture
    is identical to 1.3B so all utils, pipelines, and reward code reuse.

    Constraints:
    - height & width must be multiples of 16 (latent alignment for WAN VAE).
    - frames should satisfy the VAE temporal constraint (typically 4k+1).
    """
    config = base.get_config()
    # negative prompt: the official Wan negative prompt, identical in the Wan2.1 and
    # Wan2.2 repos (wan/configs/shared_config.py `sample_neg_prompt`). Used only when the CFG scale is > 1.
    config.negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
        "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )
    config.base_model = "wan2_1"
    # Maximum number of outer epochs (one sampling+update round each) before
    # training stops. Per-model default; override with env MAX_EPOCHS.
    config.num_epochs = int(os.getenv("MAX_EPOCHS", "100"))
    config.dataset = os.path.join(os.getcwd(), "dataset/video")
    config.pretrained.model = pretrained_model_id

    # Precision — WAN2.1 requires bf16 for stable training.
    config.mixed_precision = "bf16"
    config.sample.num_steps = num_steps
    config.sample.eval_num_steps = num_steps
    # Flow-matching trajectory shift; consumed by Wan21TrainingUtils via
    # BaseTrainingUtils._maybe_override_flow_shift after pipeline load.
    config.sample.shift = shift

    # Video geometry (explicit, BranchGRPO-style --h / --w / --t).
    config.frames = frames
    config.height = height
    config.width = width
    config.video_fps = video_fps

    # Memory optimizations: CPU sample offload + gradient checkpointing
    config.gradient_checkpointing = True
    config.cpu_offload_samples = True
    config.offload_transformer_during_decode = True

    # Batching.
    n_gpus = int(os.getenv("NUM_GPUS"))
    gradient_step_per_epoch = int(os.getenv("GRADIENT_STEP_PER_EPOCH"))

    config.sample.num_image_per_prompt = num_image_per_prompt

    # Auto-adjust bsz for different GPU counts (e.g. H200:8 vs H100:6).
    total_samples = num_groups * num_image_per_prompt
    while bsz > 1 and (total_samples % (n_gpus * bsz) != 0 or bsz * n_gpus % num_image_per_prompt != 0):
        bsz -= 1
    assert total_samples % (n_gpus * bsz) == 0, f"No valid bsz for n_gpus={n_gpus}, num_groups={num_groups}, nipp={num_image_per_prompt}"
    assert bsz * n_gpus % num_image_per_prompt == 0, f"No valid bsz for n_gpus={n_gpus}, num_groups={num_groups}, nipp={num_image_per_prompt}"

    config.sample.train_batch_size = bsz
    config.train.batch_size = bsz
    n_batch_per_epoch = total_samples // (n_gpus * bsz)
    config.sample.num_batches_per_epoch = n_batch_per_epoch
    config.train.gradient_accumulation_steps = n_batch_per_epoch // gradient_step_per_epoch
    config.name = name
    # wandb project: shared across all Wan2.1 loss/method variants (ram,
    # diffusionnft, flow_grpo, grpo, par_noratio, scale1.0, ...). They differ
    # only in the training objective, so they belong on one project page,
    # distinguished by run_name. config.name still drives the checkpoint dir
    # (logs/{name}/{run_name}), so per-variant save dirs are unchanged.
    config.project = "wan2_1"

    config.prompt_fn = "text"
    config.sample.test_batch_size = 2
    config.reward_fn = reward_fn
    # Log train sample videos to wandb every 4 epochs (was 10 in base.py).
    # Promoted from the wan_videoalign_hpsv3_smooth override on 2026-05-20.
    config.sample_log_freq = 4

    config.decay_type = int(os.getenv("DECAY_TYPE"))
    config.train.adv_mode = "all"

    _solver = os.getenv("SAMPLE_SOLVER", "dpm2")
    _noise_level = float(os.getenv("NOISE_LEVEL", "0.7"))
    _deterministic = os.getenv("DETERMINISTIC", "1") == "1"
    config.sample.deterministic = _deterministic
    config.sample.solver = _solver
    config.sample.noise_level = _noise_level

    _solver_tag = f"_{_solver}" if _solver != "dpm2" else ""
    _eta_tag = f"_eta{_noise_level}" if not _deterministic else ""
    config.run_name = config.run_name + _solver_tag + _eta_tag
    config.save_dir = f"logs/{name}/{config.run_name}"

    return config


# ---------- WAN2.1 ODE configs ----------


def wan2_1_default():
    """Wan2.1-T2V-1.3B with the default reward (see ``_WAN2_1_REWARD_FN``).

    Dataset: ``dataset/video_wan_extended_qwen3b`` (Qwen2.5-14B-extended
    VidProM/DanceGRPO prompt pool).
    """
    config = _wan2_1_base_config(
        reward_fn=dict(_WAN2_1_REWARD_FN),
        name="wan2_1_default",
    )
    config.dataset = os.path.join(os.getcwd(), "dataset/video_wan_extended_qwen3b")
    return config


# ---------- DEFAULT-conform family ----------
# Wan2.1 declared defaults (2026-09-12): Self-Forcing dataset + DECAY_TYPE=2
# (set in the scripts). Reward/geometry/optimizer identical to the historical
# runs. Separate names -> separate logs trees; scripts/wan2_1/wan2_1_def_*.sh.
def _wan2_1_def_dataset(config):
    config.dataset = os.path.join(os.getcwd(), "dataset/vidprom_filtered_extended")
    return config


def wan2_1_def():
    """RVM on Wan2.1-T2V-1.3B (scripts/wan2_1/wan2_1_def_epg.sh)."""
    config = wan2_1_default()
    config.name = "wan2_1_def"
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return _wan2_1_def_dataset(config)


def wan2_1_def_cfgdrop():
    """``wan2_1_def`` with condition dropout.

    Standard cfg=1 rollouts and the unchanged loss, except that each loss-side
    sample swaps its prompt embedding for the run's negative-prompt embedding with
    probability CFG_DROP_PROB (default 0.05), so the uncond branch used by CFG at
    inference is trained by the same objective. Launch with NP_BASE_VELOCITY=0 so
    a cfg>1 evaluation takes that branch from the trained adapter. NEGATIVE_PROMPT
    picks the dropped-to embedding (official NP default; "empty" = literal text
    removal) and must match at evaluation.
    """
    config = wan2_1_def()
    config.train.cfg_drop_prob = float(os.getenv("CFG_DROP_PROB", "0.05"))
    assert 0.0 < config.train.cfg_drop_prob < 1.0, "CFG_DROP_PROB must be in (0, 1)"
    _np = os.getenv("NEGATIVE_PROMPT", "official").strip().lower()
    _np_tag = "" if _np == "official" else ("_npempty" if _np in ("empty", "none") else "_npcustom")
    config.name = "wan2_1_def_cfgdrop"
    config.run_name = config.run_name + f"_cfgdrop{config.train.cfg_drop_prob}{_np_tag}"
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return config


def wan2_1_def_diffusionnft():
    """DiffusionNFT on Wan2.1-T2V-1.3B (scripts/wan2_1/wan2_1_def_nft.sh)."""
    config = wan2_1_diffusionnft()
    config.name = "wan2_1_def_diffusionnft"
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return _wan2_1_def_dataset(config)


def wan2_1_diffusionnft():
    """Wan2.1-T2V-1.3B trained with the DiffusionNFT contrastive flow-matching loss.

    Same dataset / reward / geometry as ``wan2_1_default``; run with ``METHOD=diffusionnft
    ADV=diffusionnft`` (reward mapped to r ∈ [0, 1]), ``DIFFUSIONNFT_BETA`` = the guidance strength γ and
    ``BETA`` = the reference-term weight (≈1e-4, as in the reference repo).
    """
    config = wan2_1_default()
    config.name = "wan2_1_diffusionnft"
    config.train.diffusionnft_beta = float(os.getenv("DIFFUSIONNFT_BETA", "0.1"))
    config.run_name = config.run_name + f"_nftg{config.train.diffusionnft_beta}"
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return config
