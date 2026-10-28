import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


# ============================================
# SD3 ODE base config
# ============================================
def _sd3_ode_base_config(n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name="", num_image_per_prompt=24):
    config = base.get_config()
    config.negative_prompt = ""   # negative prompt: empty. Used only when the CFG scale is > 1.
    assert dataset in ["pickscore", "ocr", "geneval", "geneval_unseen_objects"]

    config.base_model = "sd3"
    # Image fine-tuning default: 360 epochs. Override per-run with env MAX_EPOCHS
    # (e.g. the 4-step runs set MAX_EPOCHS=10000).
    config.num_epochs = int(os.getenv("MAX_EPOCHS", "360"))
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    # per-GPU batch so that n_gpus * bsz is a multiple of K = 24 (DistributedKRepeatSampler):
    # 8 GPUs -> 9 (72), 6 -> 8 (48), 4 -> 6 (24, added 2026-10-20 so SD3 runs fit 4-GPU slots;
    # samples per epoch stay 48 groups x 24 = 1152 and one update per rollout is unchanged).
    bsz = 9 if n_gpus % 8 == 0 else (6 if n_gpus == 4 else 8)

    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.resolution = 512

    config.sample.num_image_per_prompt = num_image_per_prompt
    num_groups = 48
    n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
    assert n_batch_per_epoch % gradient_step_per_epoch == 0
    config.sample.train_batch_size = bsz
    config.sample.num_batches_per_epoch = n_batch_per_epoch
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // gradient_step_per_epoch
    # TRAIN_MICRO_BSZ (2026-10-20, user): memory-only knob -- train on micro-batches of the largest divisor of
    # bsz that is <= TRAIN_MICRO_BSZ and accumulate proportionally more of them, so every optimizer step still
    # averages the same samples (identical gradient; K and samples/epoch unchanged). Used by the CFG_UNCOND_GRAD
    # arm, whose cond + uncond forwards both keep activations and OOM an H100 at bsz 8.
    _micro = int(os.getenv("TRAIN_MICRO_BSZ", "0"))
    if 0 < _micro < bsz:
        _mb = max(d for d in range(1, _micro + 1) if bsz % d == 0)
        config.train.batch_size = _mb
        config.train.gradient_accumulation_steps = (
            config.sample.num_batches_per_epoch * (bsz // _mb) // gradient_step_per_epoch)

    if dataset in ["geneval", "geneval_unseen_objects"]:
        config.sample.test_batch_size = 14
    else:
        config.sample.test_batch_size = 16
    config.prompt_fn = "geneval" if dataset in ["geneval", "geneval_unseen_objects"] else "text"

    config.name = name
    config.save_dir = f"logs/{name}/{config.run_name}"
    config.reward_fn = reward_fn

    config.decay_type = int(os.getenv("DECAY_TYPE"))
    config.train.adv_mode = "all"

    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    config.sample.noise_level = 0.7

    return config


# ---------- SD3 ODE configs ----------


def sd3_ocr_multi_reward():
    """OCR task exactly as specified in the Rethinking paper (tab:impl_details r.13).

    Two deliberate differences from `sd3_ocr_general_reward`, which is kept
    untouched so existing runs stay reproducible:
      * num_steps 10, not 25 — the paper gives 10 for OCR in both the impl table
        and the other-benchmarks table; the 25 there looks copied from
        sd3_multi_reward.
      * name "ocr_multi_reward", not "multi_reward" — the latter collides with
        sd3_multi_reward's save_dir (logs/multi_reward/).
    Reward mix follows the paper: OCR + PickScore + CLIPScore + HPSv2.1 at equal
    weight (PickScore is already rescaled by 1/26 inside pickscore_scorer.py).
    """
    config = _sd3_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="ocr",
        reward_fn={"pickscore": 1.0, "hpsv2": 1.0, "clipscore": 1.0, "ocr": 1.0},
        name="ocr_multi_reward",
    )
    config.sample.num_steps = 10
    config.sample_log_freq = 1
    return config


# ---------- SD3 SDE configs ----------


