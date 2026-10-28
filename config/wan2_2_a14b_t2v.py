import imp
import os

# Wan2.2 default: 1 optimizer update per epoch. Must be set
# BEFORE base.py loads -- base.py reads GRADIENT_STEP_PER_EPOCH at import time
# with no fallback (and bakes it into run_name). Scripts' exports still win.
os.environ.setdefault("GRADIENT_STEP_PER_EPOCH", "1")

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


# Main Wan2.2-A14B-T2V reward: the repo's
# standard video mix — VideoAlign TA + MQ, both HPSv3 terms, and the
# Wan2.1's video reward (VideoAlign TA and MQ, HPSv3, DT); the Wan2.2 recipes lower DT to 0.4 (_DD04).
_WAN2_2_A14B_T2V_REWARD_FN = {
    "video_hpsv3_general": 0.1,
    "video_hpsv3_percentile": 0.1,
    "videoalign_mq": 1.0,
    "videoalign_ta": 1.5,
    "dynamic_degree_smooth": 0.7,
}
_WAN2_2_A14B_T2V_REWARD_FN_DD04 = {**_WAN2_2_A14B_T2V_REWARD_FN, "dynamic_degree_smooth": 0.4}


# ============================================
# Wan2.2 T2V-A14B ODE base config (Text-to-Video, Dual-Expert MoE)
# ============================================
def _wan2_2_a14b_t2v_base_config(
    reward_fn, name, frames=53, height=480, width=832,
    num_image_per_prompt=8, num_groups=24, bsz=4,
    num_steps=16, video_fps=15, shift=8.0,
    boundary_ratio=None, guidance_scale_2=None,
):
    """Wan2.2-T2V-A14B (HF: ``Wan-AI/Wan2.2-T2V-A14B-Diffusers``).

    Two experts (``transformer`` high-noise, ``transformer_2`` low-noise), both LoRA-trained in
    alternating steps off one rollout by src/train_wan2_2.py.

    - No image conditioning: ``transformer.in_channels == 16`` (vs 36), no
      reference images, no Flux generator, ``prompt_fn = "text"``.
    - Dataset is ``dataset/video_wan_extended_qwen3b`` — the SAME
      Qwen-extended VidProM prompt pool the Wan2.1 runs use, so T2V results
      line up with the Wan2.1 family.
    - ``boundary_ratio`` is read at runtime from
      ``pipeline.config.boundary_ratio`` rather than hardcoded. Verified
      2026-07-24 against the downloaded checkpoint: **T2V-A14B ships 0.875**,
      whereas the I2V-A14B sibling ships 0.9 — do not copy the I2V value.
      Pass ``boundary_ratio=`` only to deliberately override; leaving it
      ``None`` (the default) is correct.

    Geometry matches the Wan2.1 recipe (480×832 / 53 frames / 15 fps /
    16 steps / shift=8.0) so the two families are comparable.

    Batch geometry: ``shift=8.0``, ``num_image_per_prompt=8`` (group size K),
    ``num_groups=24``, ``bsz=4`` — i.e. 192 samples/epoch, not the 256 this
    config originally used. Divisibility holds at both dispatchable GPU
    counts: 24*8 % (8*4) == 0 and 24*8 % (4*4) == 0, with
    bsz*n_gpus % K == 0 in both cases.

    Memory: two 14B transformers on disk (~28B), one active per forward.
    bf16 + gradient checkpointing + the Phase 1/2/3 swap in
    ``src/train_wan2_2.py`` keeps this at roughly 60–75 GB/GPU — fits
    H200 (141 GB) comfortably, tight on H100 (80 GB).
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
    config.base_model = "wan2_2_a14b_t2v"
    config.num_epochs = int(os.getenv("MAX_EPOCHS", "100"))
    # wan2.1_extended prompt pool — same file the Wan2.1
    # runs train on, so the two families share a prompt distribution.
    config.dataset = os.path.join(os.getcwd(), "dataset/video_wan_extended_qwen3b")
    config.pretrained.model = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    config.mixed_precision = "bf16"
    config.sample.num_steps = num_steps
    config.sample.eval_num_steps = num_steps
    # Flow-matching trajectory shift; consumed via
    # BaseTrainingUtils._maybe_override_flow_shift after pipeline load.
    config.sample.shift = shift
    if guidance_scale_2 is not None:
        config.sample.guidance_scale_2 = guidance_scale_2
    # Per-band CFG env knobs (2026-09-12). _2 = LOW-noise band; unset -> the
    # single-value knob applies to both bands (backward compatible). Official
    # Wan2.2 T2V-A14B: sample_guide_scale=(3.0, 4.0) low/high.
    if os.getenv("GUIDANCE_SCALE_2"):
        config.sample.guidance_scale_2 = float(os.getenv("GUIDANCE_SCALE_2"))
    config.train.guidance_scale_2 = (
        float(os.getenv("TRAIN_GUIDANCE_SCALE_2"))
        if os.getenv("TRAIN_GUIDANCE_SCALE_2") else None
    )
    config.train.ref_guidance_scale_2 = (
        float(os.getenv("REF_GUIDANCE_SCALE_2"))
        if os.getenv("REF_GUIDANCE_SCALE_2") else None
    )

    config.frames = frames
    config.height = height
    config.width = width
    config.video_fps = video_fps
    # Only set when explicitly overridden — otherwise the utils property
    # falls back to the checkpoint's own boundary_ratio.
    if boundary_ratio is not None:
        config.boundary_ratio = boundary_ratio

    # Both experts train, alternating (src/train_wan2_2.py); "_alt" names the run dirs.
    config.run_name = f"{config.run_name}_alt"

    config.gradient_checkpointing = True
    config.cpu_offload_samples = True
    config.sample.offload_text_encoder = True   # text encoder on GPU only while it encodes
    # Enabled 2026-07-25 after w22t2v_diffusionnft_high OOM'd at 139.13 GiB of
    # a 139.80 GiB H200. At 4 GPUs
    # the epoch's 192 samples split across half as many ranks, so per-GPU
    # sample storage doubles (48 vs 24) on top of ~56 GB of dual-expert weights
    # plus resident VideoAlign + HPSv3 — leaving nothing for the VAE decode of
    # a 53-frame video. This flag moves the transformer to CPU during decode,
    # freeing ~28 GB at the peak. (The I2V sibling defaults it False only
    # because it fits without it; the historical crash that made this flag
    # unsafe was fixed when train.py started reloading after every batch.)
    config.offload_transformer_during_decode = True

    n_gpus = int(os.getenv("NUM_GPUS"))
    # Optimizer updates per epoch (per expert). Defaults to 1; the launchers may export GRADIENT_STEP_PER_EPOCH.
    gradient_step_per_epoch = int(os.getenv("GRADIENT_STEP_PER_EPOCH", "1"))

    config.sample.num_image_per_prompt = num_image_per_prompt
    # Auto-adjust bsz so the K-repeat sampler divides cleanly at 8 or 4 GPUs
    # (6 GPUs would not divide the 192 samples).
    total_samples = num_groups * num_image_per_prompt
    while bsz > 1 and (
        total_samples % (n_gpus * bsz) != 0 or bsz * n_gpus % num_image_per_prompt != 0
    ):
        bsz -= 1
    assert total_samples % (n_gpus * bsz) == 0, (
        f"No valid bsz for n_gpus={n_gpus}, num_groups={num_groups}, "
        f"nipp={num_image_per_prompt}"
    )
    assert bsz * n_gpus % num_image_per_prompt == 0, (
        f"No valid bsz for n_gpus={n_gpus}, num_groups={num_groups}, "
        f"nipp={num_image_per_prompt}"
    )

    # SAMPLE_BSZ: the sampling batch per GPU may be larger than the training micro-batch: sampling peaks
    # ~57 GiB at bsz 4 (~72 GiB at bsz 8) while training is at its memory ceiling with bsz 4.
    sample_bsz = int(os.getenv("SAMPLE_BSZ", bsz))
    while sample_bsz > bsz and (
        total_samples % (n_gpus * sample_bsz) != 0
        or sample_bsz * n_gpus % num_image_per_prompt != 0
    ):
        sample_bsz -= 1
    sample_bsz = max(sample_bsz, bsz)
    config.sample.train_batch_size = sample_bsz
    config.train.batch_size = bsz
    config.sample.num_batches_per_epoch = total_samples // (n_gpus * sample_bsz)
    # Optimizer-step count must be anchored to the TRAIN micro-batch count,
    # not the sample batch count — otherwise raising SAMPLE_BSZ would double
    # the updates/epoch (recipe change). With SAMPLE_BSZ unset this equals the
    # old n_batch_per_epoch // gradient_step exactly.
    n_train_micro = total_samples // (n_gpus * bsz)
    config.train.gradient_accumulation_steps = n_train_micro // gradient_step_per_epoch
    config.name = name
    # Shared wandb project across all Wan2.2-A14B-T2V variants; run_name
    # distinguishes them. config.name still drives the checkpoint dir.
    config.project = "wan2_2_a14b_t2v"

    # T2V: plain text prompts, no reference images.
    config.prompt_fn = "text"
    config.sample.test_batch_size = 1
    config.save_dir = f"logs/{name}/{config.run_name}"
    config.reward_fn = reward_fn

    config.decay_type = int(os.getenv("DECAY_TYPE"))
    config.train.adv_mode = "all"

    config.sample.deterministic = True
    # dpm2 to match the Wan2.1-1.3B and A14B-I2V RL recipes (deterministic,
    # single-eval-per-step flow-ODE solver, so the per-timestep dual-expert
    # dispatch and the trajectory filter are unaffected).
    config.sample.solver = "dpm2"
    config.sample.noise_level = 0.7

    config.sample_log_freq = 2
    return config


# ============================================================================
# Alternating (joint high+low) configs — src/train_wan2_2.py
# ============================================================================
# Both experts train off a SINGLE rollout, alternating optimizer steps (high, low, high, low, ...),
# so GRADIENT_STEP_PER_EPOCH=N yields 2N updates per epoch. Both experts regress toward x_0
# (`latents_clean`) and both carry a LoRA during sampling (the rollout is the joint current policy).


def _alt_reward_weights(base_reward_fn):
    """Return the (high_noise, low_noise) reward weight dicts.

    Both default to a COPY of the run's main reward mix, i.e. the split is an
    identity until it is deliberately changed — so the first alternating run
    isolates the alternating + x_0 change and nothing else.

    To differentiate the bands, either edit the two dicts here or override the
    individual weights from the train script via the env vars below. Keys must
    be a subset of `config.reward_fn` (only scored components can be
    re-weighted; `src/train_wan2_2.py` raises a clear KeyError
    otherwise).

    Env overrides, both of the form "name:weight,name:weight":
        REWARD_FN_HIGH_NOISE, REWARD_FN_LOW_NOISE
    e.g. REWARD_FN_HIGH_NOISE="videoalign_ta:3.0,dynamic_degree_smooth:0.7"
    """
    def _parse(env_name, default):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            return dict(default)
        out = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    f"{env_name} entry '{item}' must be 'name:weight'"
                )
            k, v = item.rsplit(":", 1)
            out[k.strip()] = float(v)
        if not out:
            raise ValueError(f"{env_name} parsed to an empty dict: '{raw}'")
        return out

    return (
        _parse("REWARD_FN_HIGH_NOISE", base_reward_fn),
        _parse("REWARD_FN_LOW_NOISE", base_reward_fn),
    )


def _wan2_2_a14b_t2v_alt_base_config(reward_fn, name, **kwargs):
    """Wan2.2-T2V-A14B, both experts trained off one rollout.

    Identical model / dataset / geometry / batch shape to
    ``_wan2_2_a14b_t2v_base_config`` — the differences are the training
    schedule (alternating updates) and the per-band reward weighting.
    Launch with ``src/train_wan2_2.py``.
    """
    # Rollout-size overrides, for smoke tests. A full epoch is 24 groups x 8
    # samples = 192 videos; a 1-epoch shakedown of the alternating machinery
    # does not need that. Unset in every real run.
    if os.getenv("NUM_GROUPS"):
        kwargs["num_groups"] = int(os.getenv("NUM_GROUPS"))
    if os.getenv("NUM_IMAGE_PER_PROMPT"):
        kwargs["num_image_per_prompt"] = int(os.getenv("NUM_IMAGE_PER_PROMPT"))
    if os.getenv("NUM_STEPS"):
        kwargs["num_steps"] = int(os.getenv("NUM_STEPS"))

    config = _wan2_2_a14b_t2v_base_config(reward_fn=reward_fn, name=name, **kwargs)

    config.reward_fn_high_noise, config.reward_fn_low_noise = _alt_reward_weights(
        reward_fn
    )
    # Boundary between the two reward regions, as a fraction of
    # num_train_timesteps. Defaults to the T2V-A14B checkpoint's own
    # boundary_ratio (0.875) so the reward split lines up EXACTLY with the
    # expert split — each expert's steps then all read the same region
    # advantage. Do not copy the I2V value (0.9).
    config.train.high_noise_threshold = float(
        os.getenv("HIGH_NOISE_THRESHOLD", "0.875")
    )
    # Keep BOTH 14B experts resident on GPU through the training phase instead
    # of swapping the inactive one to CPU between the high and low steps.
    # Saves 2 x GRADIENT_STEP_PER_EPOCH transfers of ~28 GB per epoch, but
    # needs the headroom: measure the reported Phase-3 peak first. Off by
    # default (safe on both H100:80 and H200:141).
    config.a14b_both_experts_resident = (
        os.getenv("BOTH_EXPERTS_RESIDENT", "0") == "1"
    )
    # Page one expert onto GPU at a time during sampling too (the schedule crosses boundary_ratio
    # once: one expert transfer per sampled batch). On by default; ignored when BOTH_EXPERTS_RESIDENT=1.
    config.a14b_sequential_expert_sampling = (
        os.getenv("SEQUENTIAL_EXPERT_SAMPLING", "1") == "1"
    )
    return config


def wan2_2_a14b_t2v_def_alt():
    """RVM on Wan2.2-T2V-A14B, both experts alternating (scripts/wan2_2/wan2_2_a14b_t2v_def_alt_epg.sh)."""
    config = _wan2_2_a14b_t2v_alt_base_config(
        reward_fn=dict(_WAN2_2_A14B_T2V_REWARD_FN_DD04),
        name="wan2_2_a14b_t2v_def_alt",
    )
    config.lora_r = 128
    config.lora_alpha = 64
    # Default dataset: Self-Forcing training prompts —
    # 248,221 VidProm-filtered extended prompts. Better VBench alignment than
    # video_wan_extended_qwen3b (actions 99/100 vs 95/100 verb-level; closer
    # camera-command register) and 5x the diversity. See
    #  All pre-2026-09-12 runs (both
    # families) used video_wan_extended_qwen3b.
    config.dataset = os.path.join(os.getcwd(), "dataset/vidprom_filtered_extended")
    return config


def wan2_2_a14b_t2v_def_alt_diffusionnft():
    """DiffusionNFT on Wan2.2-T2V-A14B (scripts/wan2_2/wan2_2_a14b_t2v_def_alt_nft.sh)."""
    config = wan2_2_a14b_t2v_def_alt()
    config.name = "wan2_2_a14b_t2v_def_alt_diffusionnft"
    config.train.diffusionnft_beta = float(os.getenv("DIFFUSIONNFT_BETA", "0.1"))
    config.run_name = config.run_name + f"_nftg{config.train.diffusionnft_beta}"
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return config


