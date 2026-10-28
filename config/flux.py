import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


# ============================================
# Flux ODE base config
# ============================================
def _flux_ode_base_config(n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name="", num_image_per_prompt=24):
    config = base.get_config()
    config.negative_prompt = ""   # negative prompt: empty. Used only when the CFG scale is > 1.
    assert dataset in ["pickscore", "ocr", "geneval", "geneval_unseen_objects"]

    config.base_model = "flux"
    # Image fine-tuning default: 360 epochs. Override per-run with env MAX_EPOCHS.
    config.num_epochs = int(os.getenv("MAX_EPOCHS", "360"))
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    bsz = 3 if n_gpus % 8 == 0 else 4

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


# ---------- Flux ODE configs ----------

def flux_geneval():
    config = _flux_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="geneval",
        reward_fn={"geneval": 1.0},
        name="geneval",
    )
    return config


def flux_geneval_bf16():
    config = _flux_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="geneval",
        reward_fn={"geneval": 1.0},
        name="geneval",
    )
    config.mixed_precision = "bf16"
    return config
