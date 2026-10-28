"""Base class for all model training utilities.

Captures the common skeleton shared across SD3, Flux, WAN2.1, HunyuanVideo,
SkyReels-I2V.  Derived classes override model-specific methods
(get_pretrained_model, rollout, the velocity hooks, text-embedding
helpers, logging, eval_fn) and provide their LoRA configuration via
get_lora_config().

The loss itself is not here: src/loss/step.py:compute_loss(model_utils, ...) calls this class's
velocity hooks (v_pred_fn and the _forward_ctx / _conds / _branch / _pair hooks below).
"""

from abc import ABC, abstractmethod

import torch
from torch.cuda.amp import autocast as torch_autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, PeftModel, get_peft_model


class BaseTrainingUtils(ABC):
    """Shared training utility base class for all diffusion models."""

    def __init__(self, config, logger, wandb_module, device, rank, world_size):
        self.config = config
        self.logger = logger
        self.wandb = wandb_module
        self.device = device
        self.rank = rank
        self.world_size = world_size

    # ------------------------------------------------------------------
    # Abstract methods – must be implemented by every derived class
    # ------------------------------------------------------------------

    @abstractmethod
    def get_pretrained_model(self, mixed_precision_dtype, enable_amp):
        """Load the pretrained pipeline.

        Returns:
            (pipeline, text_encoders, tokenizers)
        """
        ...

    @abstractmethod
    def get_lora_config(self):
        """Return a ``peft.LoraConfig`` appropriate for this model."""
        ...

    def _maybe_override_flow_shift(self, scheduler):
        """Re-instantiate ``scheduler`` with ``config.sample.shift`` if set.

        Flow-matching schedulers expose the trajectory-warping shift under
        either ``"shift"`` (FlowMatchEulerDiscreteScheduler) or ``"flow_shift"``
        (UniPCMultistepScheduler with ``use_flow_sigmas=True``). When the
        per-model config sets ``config.sample.shift``, override whichever key
        the loaded scheduler exposes; otherwise return the scheduler unchanged.
        """
        shift = getattr(self.config.sample, "shift", None)
        if shift is None:
            return scheduler
        cfg = dict(scheduler.config)
        overrides = {}
        if "shift" in cfg:
            overrides["shift"] = float(shift)
        if "flow_shift" in cfg:
            overrides["flow_shift"] = float(shift)
        if not overrides:
            return scheduler
        return type(scheduler).from_config(scheduler.config, **overrides)

    @abstractmethod
    def compute_text_embeddings(self, prompts, text_encoders, tokenizers, max_sequence_length=128):
        """Encode text prompts into embeddings."""
        ...

    @abstractmethod
    def get_negative_prompt_embeddings(self, text_encoders, tokenizers):
        """Return negative-prompt embeddings (neg_embeds, neg_pooled_embeds)."""
        ...

    @abstractmethod
    def get_prompt_embeddings(self, prompts, text_encoders, tokenizers):
        """Return prompt embeddings (embeds, pooled_embeds)."""
        ...

    @abstractmethod
    def build_sample_kwargs(self, prompt_embeds, pooled_prompt_embeds,
                            sample_neg_prompt_embeds, sample_neg_pooled_prompt_embeds,
                            batch_size):
        """Build kwargs dict for ``rollout``."""
        ...

    def _compute_shared_uncond(self, transformer_ddp, xt, samples, j_idx,
                               neg_embeds, neg_pooled_embeds):
        """Compute one CFG uncond pass to share across old/current/ref.

        Default: return ``None`` (no sharing; each v_pred_fn computes its
        own uncond). Override per-model when the transformer signature and
        CFG formulation allow a single shared forward. Only the base-model NP
        (adapters disabled) may be shared: an adapter NP differs per adapter,
        and a no-grad forward of the trainable adapter ahead of its gradient
        forward loses the LoRA gradient under autocast (src/models/wan2_1.py).
        """
        return None

    @abstractmethod
    def rollout(self, pipeline, **kwargs):
        """Run the patched sampling pipeline -> ``src.rollout.rollout.Rollout`` (media, latents, extras)."""
        ...

    @abstractmethod
    def decode_latents(self, pipeline, latents):
        """Decode raw latents into images/videos using the pipeline's VAE.

        This mirrors the VAE decode logic from the patched pipeline, allowing
        decode to happen after the transformer has been offloaded from GPU.

        Args:
            pipeline: The diffusion pipeline (VAE must be on GPU).
            latents: Raw latents from the last denoising step.

        Returns:
            Decoded images (list of PIL images) or video tensors.
        """
        ...

    @abstractmethod
    def save_sample(self, data, output_path):
        """Save a single image/video sample to disk."""
        ...

    @abstractmethod
    def log_train_samples(self, images, prompts, rewards_to_log, global_step, log_step):
        """Log training samples to wandb."""
        ...

    @abstractmethod
    def _log_eval_outputs(self, final_rewards, images, prompts, global_step):
        """Log evaluation outputs to wandb."""
        ...

    @abstractmethod
    def eval_fn(self, pipeline, test_dataloader, text_encoders, tokenizers,
                global_step, reward_fn, executor, mixed_precision_dtype,
                ema, transformer_trainable_parameters, tqdm_fn):
        """Run evaluation loop."""
        ...

    # ------------------------------------------------------------------
    # Concrete shared methods
    # ------------------------------------------------------------------

    def get_tokenizer_max_length(self):
        return 256

    # ------------------------------------------------------------------
    # Hooks for the training loop
    #
    # These hide model-specific dataloading and sampling-artifact handling
    # from train.py. The default implementations cover the simple
    # text-only path (SD3 / Flux / WAN2.1); models with extra
    # conditioning (precomputed embeddings, reference images, action ids,
    # I2V conditioning tensors, etc.) override one or both.
    # ------------------------------------------------------------------

    def on_epoch_start(self, epoch):
        """Called by the trainer at the start of every epoch, before sampling (no-op by default)."""


    def extract_dataloader_metadata(self, prompts, prompt_metadata, text_encoders, tokenizers):
        """Extract per-batch model inputs from a dataloader yield.

        Encodes prompts (or unpacks pre-computed embeddings from
        ``prompt_metadata``) and stashes any model-specific conditioning
        on ``self`` so that the subsequent ``build_sample_kwargs`` /
        ``rollout`` calls can pick it up.

        Returns:
            dict with ``prompt_embeds``, ``pooled_prompt_embeds``,
            ``prompt_ids`` (the latter is ``None`` when prompts are not
            tokenized — e.g. precomputed-embedding mode).
        """
        prompt_embeds, pooled_prompt_embeds = self.get_prompt_embeddings(
            prompts, text_encoders, tokenizers,
        )
        prompt_ids = tokenizers[0](
            prompts,
            padding="max_length",
            max_length=self.get_tokenizer_max_length(),
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_ids": prompt_ids,
        }

    def collect_sample_artifacts(self, sample_data, rollout=None):
        """Append model-specific fields to ``sample_data`` after sampling.

        Default: no-op. Override in models that need to forward
        per-sample conditioning produced by ``rollout`` (``rollout.extras``)
        (first-frame latents, I2V condition tensors, attention masks,
        action ids, ...) into the per-sample dict consumed by training.
        """
        return

    # ------------------------------------------------------------------
    # Phase 1: sampling (the epoch's rollouts under the old adapter)
    # ------------------------------------------------------------------

    def sampling_to_device(self, pipeline):
        """Before sampling: what the sampler needs goes to GPU."""
        pipeline.vae.to(self.device)
        if getattr(pipeline, "text_encoder", None) is not None:
            pipeline.text_encoder.to(self.device)
        if getattr(pipeline, "transformer", None) is not None:
            pipeline.transformer.to(self.device)
        pipeline.transformer.eval()

    def foundation_to_cpu(self, pipeline):
        """Before scoring: the generator's large modules leave the GPU for the reward models."""
        for name in ("transformer", "vae", "text_encoder"):
            if getattr(pipeline, name, None) is not None:
                getattr(pipeline, name).to("cpu")

    def sampling_inputs(self, pipeline, prompts, prompt_metadata, text_encoders, tokenizers):
        """One batch's model inputs (``extract_dataloader_metadata``). ``config.sample.offload_text_encoder``:
        the text encoder sits on GPU only while it encodes."""
        offload = bool(getattr(self.config.sample, "offload_text_encoder", False))
        text_encoder = getattr(pipeline, "text_encoder", None)
        if offload and text_encoder is not None:
            text_encoder.to(self.device)
        inputs = self.extract_dataloader_metadata(prompts, prompt_metadata, text_encoders, tokenizers)
        if offload:
            if text_encoder is not None:
                text_encoder.to("cpu")
            torch.cuda.empty_cache()
        return inputs

    def decode_offloaded(self, pipeline, latents_clean, last_batch):
        """``offload_transformer_during_decode``: decode with the transformer on CPU, then bring it back."""
        pipeline.transformer.to("cpu")
        torch.cuda.empty_cache()
        with torch.no_grad():
            images = self.decode_latents(pipeline, latents_clean)
        pipeline.transformer.to(self.device)
        torch.cuda.empty_cache()
        return images


    def set_adapter(self, pipeline, local_rank):
        """Configure LoRA adapters (default + old) and wrap in DDP."""
        if not self.config.use_lora:
            raise NotImplementedError("Current training loop expects LoRA adapters.")

        transformer = pipeline.transformer.to(self.device)
        transformer_lora_config = self.get_lora_config()

        if self.config.train.lora_path:
            transformer = PeftModel.from_pretrained(
                transformer, self.config.train.lora_path,
            )
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)

        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")

        transformer_ddp = DDP(
            transformer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        transformer_ddp.module.set_adapter("default")
        transformer_trainable_parameters = list(
            filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
        )
        transformer_ddp.module.set_adapter("old")
        old_transformer_trainable_parameters = list(
            filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
        )
        transformer_ddp.module.set_adapter("default")

        return (
            transformer_ddp,
            transformer_trainable_parameters,
            old_transformer_trainable_parameters,
            transformer_lora_config,
        )

    def save_extra(self, save_dir, ckpt_subdir):
        """What this model saves next to the engine's checkpoint (src/engine/checkpoint.py:save) in
        <save_dir>/<ckpt_subdir>; nothing by default."""

    # ------------------------------------------------------------------
    # Velocity prediction: one generic CFG path (src/models/guidance.py). A model provides
    #   _forward_ctx(transformer_ddp, xt, samples, index) -> ctx   (timesteps, latent input, ...)
    #   _conds(samples, embeds, pooled, neg, neg_pooled)  -> (cond, uncond)
    #   _branch(transformer_ddp, ctx, cond, embed_scale)  -> v   (one conditioning, no CFG)
    #   _pair(transformer_ddp, ctx, cond, uncond)         -> (v_c, v_np)  (doubled batch)
    #   _cfg_options(kind)                                 -> guidance.cfg keywords
    # ------------------------------------------------------------------
    def _guidance_scale(self, kind):
        if kind == "train":
            return self.config.train.guidance_scale
        if kind == "sample":
            return self.config.sample.guidance_scale
        if kind == "ref":
            return self.config.train.ref_guidance_scale
        raise ValueError(f"Unknown sample_type: {kind}")

    def _cfg_options(self, kind):
        from src.models.guidance import cfg_options
        return cfg_options(self.config, kind)

    def _forward_ctx(self, transformer_ddp, xt, samples, index):
        return {"xt": xt, "timesteps": samples["timesteps"][:, index]}

    def _conds(self, samples, embeds, pooled_embeds, neg_embeds, neg_pooled_embeds):
        cond = {"embeds": embeds, "pooled": pooled_embeds}
        uncond = None if neg_embeds is None else {"embeds": neg_embeds, "pooled": neg_pooled_embeds}
        return cond, uncond

    def _branch(self, transformer_ddp, ctx, cond, embed_scale):
        raise NotImplementedError

    def _pair(self, transformer_ddp, ctx, cond, uncond):
        raise NotImplementedError(f"{type(self).__name__} has no doubled-batch CFG forward")

    def v_pred_fn(self, transformer_ddp, xt, samples, index, embeds, pooled_embeds,
                  neg_embeds, neg_pooled_embeds, sample_type="train", cached_uncond=None):
        """Guided velocity of the active adapter for one forward kind ("train" / "sample" / "ref")."""
        from src.models.guidance import cfg
        options = self._cfg_options(sample_type)
        scale = self._guidance_scale(sample_type)
        ctx = self._forward_ctx(transformer_ddp, xt, samples, index)
        cond, uncond = self._conds(samples, embeds, pooled_embeds, neg_embeds, neg_pooled_embeds)
        return cfg(
            lambda c, embed_scale: self._branch(transformer_ddp, ctx, c, embed_scale),
            lambda c, u: self._pair(transformer_ddp, ctx, c, u),
            transformer_ddp.module, cond, uncond, scale, kind=sample_type, cached_uncond=cached_uncond,
            **options,
        )


