"""The models: one TrainingUtils class per base model (base.py = the shared interface), the CFG / negative-prompt
code (guidance.py), the LoRA adapter roles (adapters.py) and the sampling pipelines (pipelines/)."""


def model_utils_class(base_model):
    """The TrainingUtils class for ``config.base_model``.

    Imports are lazy so the heavy diffusers dependencies of unused models are not pulled in at
    training start.
    """
    if base_model == "sd3":
        from .sd3 import SD3TrainingUtils as cls
    elif base_model == "wan2_1":
        from .wan2_1 import Wan21TrainingUtils as cls
    elif base_model == "flux":
        from .flux import FluxTrainingUtils as cls
    elif base_model == "skyreels_i2v":
        from .skyreels_i2v import SkyReelsI2VTrainingUtils as cls
    elif base_model == "wan2_2_a14b_t2v":
        from .wan2_2_a14b_t2v import Wan22A14BT2VTrainingUtils as cls
    else:
        raise ValueError(f"Unsupported config.base_model={base_model}.")
    return cls


def build_model_utils(base_model, *, config, logger, wandb_module, device, rank, world_size):
    """Instantiate the TrainingUtils class for ``config.base_model``."""
    return model_utils_class(base_model)(
        config=config,
        logger=logger,
        wandb_module=wandb_module,
        device=device,
        rank=rank,
        world_size=world_size,
    )
