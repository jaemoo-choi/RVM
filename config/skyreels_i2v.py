import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

# SkyReels-I2V reward: Wan2.1's mixture with DT2 (dynamic_degree_bump, a bounded motion window: 0 for
# static or background-churn motion) in place of DT, and VideoAlign MQ at 0.5.
_SKYREELS_REWARD_FN_BUMP = {
    "video_hpsv3_general": 0.1,
    "video_hpsv3_percentile": 0.1,
    "videoalign_mq": 0.5,
    "videoalign_ta": 1.5,
    "dynamic_degree_bump": 0.7,
}


def get_config(name):
    return globals()[name]()


# ============================================
# SkyReels-I2V ODE base config (Image-to-Video)
# ============================================
def _skyreels_i2v_base_config(reward_fn, name, frames=53, height=400, width=640,
                               num_image_per_prompt=8, num_groups=16, bsz=4,
                               num_steps=16, video_fps=15, use_cached_ref_images=True):
    """SkyReels-I2V config adapted from DanceGRPO.

    Default geometry: 53 frames at 400x640 (DanceGRPO default), 15 fps for
    saved videos. Uses PRE-GENERATED Flux reference images
    (``dataset/skyreels_i2v/ref_images_{train,test}/``); generate them once
    with ``scripts/dataset/generate_skyreels_ref_images.sh``.
    """
    config = base.get_config()
    # negative prompt: the official SkyReels-V1 negative prompt (video_generate.py in
    # SkyworkAI/SkyReels-V1). Used only when the CFG scale is > 1.
    config.negative_prompt = (
        "Aerial view, aerial view, overexposed, low quality, deformation, "
        "a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
    )
    config.base_model = "skyreels_i2v"
    # Maximum number of outer epochs (one sampling+update round each) before
    # training stops. Per-model default; override with env MAX_EPOCHS.
    config.num_epochs = int(os.getenv("MAX_EPOCHS", "20"))
    config.dataset = os.path.join(os.getcwd(), "dataset/skyreels_i2v")
    config.pretrained.model = "xzyhku/SkyReels-V1-I2V"

    # Flux only matters when generating ref images on the fly
    config.reference_model = "black-forest-labs/FLUX.1-dev"
    config.use_cached_ref_images = use_cached_ref_images

    config.mixed_precision = "bf16"
    config.sample.num_steps = num_steps
    config.sample.eval_num_steps = num_steps

    config.frames = frames
    config.height = height
    config.width = width
    config.video_fps = video_fps

    config.gradient_checkpointing = True
    config.cpu_offload_samples = True
    config.offload_transformer_during_decode = True

    n_gpus = int(os.getenv("NUM_GPUS"))
    gradient_step_per_epoch = int(os.getenv("GRADIENT_STEP_PER_EPOCH"))

    config.sample.num_image_per_prompt = num_image_per_prompt

    # Adjust bsz downward for fewer GPUs so divisibility constraints hold.
    total_samples = num_groups * num_image_per_prompt
    while bsz > 1 and (total_samples % (n_gpus * bsz) != 0 or bsz * n_gpus % num_image_per_prompt != 0):
        bsz -= 1
    assert total_samples % (n_gpus * bsz) == 0, f"No valid bsz for n_gpus={n_gpus}, num_groups={num_groups}, nipp={num_image_per_prompt}"
    assert bsz * n_gpus % num_image_per_prompt == 0, f"No valid bsz for n_gpus={n_gpus}, num_groups={num_groups}, nipp={num_image_per_prompt}"

    # SAMPLE_BSZ: the sampling batch per GPU may be larger than the training micro-batch (sampling
    # peaks ~60 GiB at bsz 4 with the tiled per-sample decode).
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
    # wandb project: shared across all SkyReels-I2V variants (ram, diffusionnft,
    # flow_grpo, embed, wan_extended_a14b, ...). They log to one project page,
    # distinguished by run_name. config.name still drives the checkpoint dir.
    config.project = "skyreels"
    config.prompt_fn = "image_prompt" if use_cached_ref_images else "text"
    config.sample.test_batch_size = 1
    config.save_dir = f"logs/{name}/{config.run_name}"
    config.reward_fn = reward_fn

    config.decay_type = int(os.getenv("DECAY_TYPE"))
    config.train.adv_mode = "all"

    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    config.sample.noise_level = 0.7

    config.sample_log_freq = 2

    return config


# skyreels_def: 544x960 (the official SkyReels-V1 size), 53 frames at 15 fps, per-component reward whitening,
# prompts from dataset/vidprom_sf_i2v_544 with pre-generated FLUX.1-dev first frames (use_cached_ref_images).
# 544x960 needs the tiled VAE decode (VAE_TILING=1, set by the launchers and the eval script): an untiled
# 53-frame decode asks for a single ~114 GiB allocation.
def skyreels_def():
    """RVM on SkyReels-I2V (scripts/skyreels/skyreels_def_epg.sh)."""
    config = _skyreels_i2v_base_config(
        reward_fn=dict(_SKYREELS_REWARD_FN_BUMP),
        name="skyreels_def",
        height=544,
        width=960,
        # CACHED refs. One-time
        # pre-generation: scripts/dataset/gen_vidprom_sf_i2v_544.sh fills
        # ref_images_{train,test}/ (+metadata.jsonl) at 544x960, seed 42.
        use_cached_ref_images=True,
    )
    config.dataset = os.path.join(os.getcwd(), "dataset/vidprom_sf_i2v_544")
    config.train.whiten_rewards = True
    return config


def skyreels_def_diffusionnft():
    """DiffusionNFT on SkyReels-I2V (scripts/skyreels/skyreels_def_nft.sh)."""
    config = skyreels_def()
    config.name = "skyreels_def_diffusionnft"
    config.train.diffusionnft_beta = float(os.getenv("DIFFUSIONNFT_BETA", "0.1"))
    config.run_name = config.run_name + f"_nftg{config.train.diffusionnft_beta}"
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return config


