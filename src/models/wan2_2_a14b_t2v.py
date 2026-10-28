"""Training utilities for Wan2.2-T2V-A14B (two experts selected by timestep).

The checkpoint ``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` (``diffusers.WanPipeline``) carries two dense
transformers: ``transformer`` (high-noise expert, timesteps ``>= boundary_ratio * 1000``) and
``transformer_2`` (low-noise expert, the rest). Both are LoRA-wrapped and trained off one rollout
in alternating optimizer steps by ``src/train_wan2_2.py``, which calls ``set_active_expert``
around each step; ``expert_step_mask`` splits the trajectory into the two bands, and the loss
forward checks that every step belongs to the active expert's band.
"""

from collections import defaultdict
import os
import tempfile
import time

import imageio
import numpy as np
import contextlib
from src.models.pipelines.lora_ctx import maybe_disable_lora
import torch
import torch.distributed as dist
from torch import autocast as torch_autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from diffusers import WanPipeline
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.pipelines.pipeline_with_logprob_wan2_2_a14b_t2v import (
    pipeline_with_logprob as patched_pipeline_with_logprob,
)
from src.utils.distributed import gather_tensor_to_all, is_main_process
from src.models.guidance import resolve_negative_prompt_text
from src.rollout.rollout import Rollout
from .base import BaseTrainingUtils


class Wan22A14BT2VTrainingUtils(BaseTrainingUtils):
    def __init__(self, config, logger, wandb_module, device, rank, world_size):
        super().__init__(config, logger, wandb_module, device, rank, world_size)
        self.pipeline = None
        self._active_expert = None   # "high" / "low": the expert being optimized (set_active_expert)
        self.expert_ddps = {}         # {"high", "low"} -> DDP-wrapped expert (set_adapter_both)
        self._sampling_resident = None   # expert(s) on GPU while sampling: "high" / "low" / "both" / None
        self._sampling_swap = None       # expert_swap_fn for the rollout (sequential expert sampling)

    @property
    def video_fps(self):
        return getattr(self.config, "video_fps", 16)

    @property
    def boundary_ratio(self):
        # Pipeline config takes precedence (set by the HF checkpoint).
        # Verified 2026-07-24 from the downloaded checkpoint: T2V-A14B ships
        # boundary_ratio=0.875 (the I2V-A14B sibling ships 0.9 — do NOT copy
        # that value here). The 0.875 fallback only applies before the
        # pipeline is loaded.
        if self.pipeline is not None:
            br = getattr(self.pipeline.config, "boundary_ratio", None)
            if br is not None:
                return br
        return getattr(self.config, "boundary_ratio", 0.875)

    @property
    def active_expert(self):
        if self._active_expert is None:
            raise ValueError("no active expert: call set_active_expert('high'|'low') first")
        return self._active_expert

    def set_active_expert(self, name):
        """Set the expert being optimized (``"high"`` / ``"low"``), or clear it (``None``)."""
        if name not in ("high", "low", None):
            raise ValueError(f"set_active_expert expects 'high', 'low' or None (got {name!r})")
        self._active_expert = name

    @property
    def _trainable_attr(self):
        return "transformer" if self.active_expert == "high" else "transformer_2"

    def get_pretrained_model(self, mixed_precision_dtype, enable_amp):
        model_dtype = mixed_precision_dtype if enable_amp else torch.float32
        pipeline = WanPipeline.from_pretrained(
            self.config.pretrained.model, torch_dtype=model_dtype,
        )
        pipeline.scheduler = self._maybe_override_flow_shift(pipeline.scheduler)
        if pipeline.transformer_2 is None:
            raise RuntimeError(
                f"{self.config.pretrained.model} loaded without a "
                "`transformer_2`. Wan2.2-T2V-A14B requires both experts; "
                "make sure you downloaded the A14B (not the TI2V-5B) "
                "checkpoint."
            )

        pipeline.vae.requires_grad_(False)
        if pipeline.text_encoder is not None:
            pipeline.text_encoder.requires_grad_(False)
        # Both experts are LoRA-trained: the base weights stay frozen.
        if not self.config.use_lora:
            raise NotImplementedError("Wan2.2-T2V-A14B requires LoRA training (config.use_lora=True).")
        pipeline.transformer.requires_grad_(False)
        pipeline.transformer_2.requires_grad_(False)

        self.pipeline = pipeline

        text_encoders = [pipeline.text_encoder] if pipeline.text_encoder is not None else []
        tokenizers = [pipeline.tokenizer] if pipeline.tokenizer is not None else []

        if hasattr(pipeline, "safety_checker"):
            pipeline.safety_checker = None

        pipeline.set_progress_bar_config(
            position=1,
            disable=not is_main_process(self.rank),
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        pipeline.to(self.device)
        pipeline.vae.to(self.device, dtype=model_dtype)
        if pipeline.text_encoder is not None:
            pipeline.text_encoder.to(self.device, dtype=model_dtype)
        pipeline.transformer.to(self.device, dtype=model_dtype)
        pipeline.transformer_2.to(self.device, dtype=model_dtype)

        if is_main_process(self.rank):
            self.logger.info(f"Wan2.2-T2V-A14B loaded. boundary_ratio={getattr(pipeline.config, 'boundary_ratio', None)}; "
                             "LoRA on both experts, trained in alternating steps")

        # T2V: no reference images, no Flux generator (see module docstring).

        return pipeline, text_encoders, tokenizers

    def get_lora_config(self):
        # LoRA on every linear in self-attention (attn1: to_q/k/v/out),
        # cross-attention (attn2: to_q/k/v/out), and FFN (net.0.proj, net.2).
        # PEFT matches by name suffix so the strings cover both attention
        # blocks at once. Default r=32, alpha=64 — matches SD3.5 defaults.
        # Both are config-overridable (``config.lora_r`` / ``config.lora_alpha``)
        # so a config function can match the Wan2.1-1.3B recipe (r=128).
        r = getattr(self.config, "lora_r", 32)
        alpha = getattr(self.config, "lora_alpha", 64)
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",
                "ffn.net.0.proj", "ffn.net.2",
            ],
        )

    def set_adapter(self, pipeline, local_rank):
        """LoRA- and DDP-wrap the active expert (transformer for high, transformer_2 for low)."""
        if not self.config.use_lora:
            raise NotImplementedError("Wan2.2-T2V-A14B requires LoRA training.")

        attr = self._trainable_attr
        transformer = getattr(pipeline, attr).to(self.device)
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
        # PEFT mutates the trainable expert in place — its attention/FFN
        # submodules are now LoraLinear, so the patched sampling pipeline's
        # `pipeline.transformer(_2)` call routes through LoRA without
        # reassigning the pipeline attribute. ``set_adapter("default")``
        # propagates the active adapter to those in-place LoraLinear
        # modules, so sampling sees the same weights training updates.

        # broadcast_buffers=False:
        # Default DDP runs `_sync_module_buffers` at the start of every
        # forward, broadcasting all module buffers from rank 0. A14B's
        # sequential-offload scheme moves the LoRA-wrapped expert
        # CPU↔GPU between phases, which can leave a non-persistent
        # buffer on CPU even after `transformer_ddp.module.to(device)`
        # — and NCCL has no backend for CPU tensors, so the broadcast
        # crashes. LoRA training has no mutable buffers (no BatchNorm,
        # no running stats), so disabling buffer broadcast is safe and
        # avoids the crash entirely.
        transformer_ddp = DDP(
            transformer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
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

    def set_adapter_both(self, pipeline, local_rank):
        """LoRA- and DDP-wrap both experts: ``{"high": bundle, "low": bundle}``, each bundle the
        4-tuple ``set_adapter`` returns (``set_adapter`` once per expert).

        ``config.train.lora_path`` is deliberately rejected here: it names a
        single adapter directory with no expert to attach it to. The
        alternating trainer resumes from the per-expert
        ``checkpoints/{high,low}/`` dirs instead.
        """
        if getattr(self.config.train, "lora_path", None):
            raise ValueError(
                "config.train.lora_path is not supported by set_adapter_both "
                "(it is ambiguous across the two experts). Resume from "
                "checkpoints/{high,low}/ instead."
            )
        prev_active = self._active_expert
        bundles = {}
        try:
            for expert in ("high", "low"):
                self.set_active_expert(expert)
                bundles[expert] = self.set_adapter(pipeline, local_rank)
                if is_main_process(self.rank):
                    self.logger.info(
                        f"LoRA + DDP wrapped {expert}-noise expert "
                        f"(pipeline.{self._trainable_attr})"
                    )
        finally:
            self._active_expert = prev_active
        self.expert_ddps = {e: b[0] for e, b in bundles.items()}
        return bundles

    # ------------------------------------------------------------------
    # Expert placement (GPU / CPU)
    # ------------------------------------------------------------------
    @staticmethod
    def expert_attr(expert):
        """pipeline attribute holding `expert`'s transformer."""
        return "transformer" if expert == "high" else "transformer_2"

    def move_expert(self, pipeline, expert, target):
        """Move one expert to ``target``. ``Module.to`` can leave weights behind after a PEFT round-trip,
        so every handle is moved and the parameters / buffers are then swept by hand."""
        attr = self.expert_attr(expert)
        expert_ddp = self.expert_ddps.get(expert)
        if expert_ddp is None:
            getattr(pipeline, attr).to(target)
            return
        for p in expert_ddp.module.parameters():   # grads would stay on the old device
            p.grad = None
        expert_ddp.module.to(target)
        getattr(pipeline, attr).to(target)
        target_dev = torch.device(target) if not isinstance(target, torch.device) else target
        n_p_moved = 0
        for p in expert_ddp.module.parameters():
            if p.device != target_dev:
                p.data = p.data.to(target_dev)
                n_p_moved += 1
        n_b_moved = 0
        for b in expert_ddp.module.buffers():
            if b.device != target_dev:
                b.data = b.data.to(target_dev)
                n_b_moved += 1
        if (n_p_moved + n_b_moved) > 0:
            self.logger.warning(
                f"_move_expert({expert}): brute-force moved {n_p_moved} params "
                f"+ {n_b_moved} buffers to {target_dev} that .to() missed."
            )

    def experts_to(self, pipeline, placement):
        """Place both experts: ``placement`` maps "high" / "low" -> device or "cpu"."""
        for expert in ("high", "low"):
            self.move_expert(pipeline, expert, placement[expert])

    @property
    def sequential_sampling(self):
        """One expert on GPU at a time while sampling (paged at the boundary)."""
        return (bool(getattr(self.config, "a14b_sequential_expert_sampling", True))
                and not bool(getattr(self.config, "a14b_both_experts_resident", False)))

    def _sampling_placement(self):
        """Every batch starts at high noise: the high expert alone (sequential) or both."""
        if self.sequential_sampling:
            self.experts_to(self.pipeline, {"high": self.device, "low": "cpu"})
            self._sampling_resident = "high"
        else:
            self.experts_to(self.pipeline, {"high": self.device, "low": self.device})
            self._sampling_resident = "both"

    def _swap_for_step(self, is_high_noise):
        """expert_swap_fn: page the expert the sampler is about to call onto GPU."""
        want = "high" if is_high_noise else "low"
        if self._sampling_resident in (want, "both"):
            return
        self.experts_to(self.pipeline, {e: (self.device if e == want else "cpu") for e in ("high", "low")})
        self._sampling_resident = want
        torch.cuda.empty_cache()

    # ---- Phase 1 / 2 placement (BaseTrainingUtils.sample_epoch) ----
    def sampling_to_device(self, pipeline):
        import gc
        self._sampling_placement()
        pipeline.vae.to(self.device)
        if pipeline.text_encoder is not None:
            pipeline.text_encoder.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        self._sampling_swap = self._swap_for_step if self.sequential_sampling else None
        pipeline.transformer.eval()
        pipeline.transformer_2.eval()

    def decode_offloaded(self, pipeline, latents_clean, last_batch):
        """Both experts leave the GPU for the decode; before another batch the sampling placement returns."""
        import gc
        self.experts_to(pipeline, {"high": "cpu", "low": "cpu"})
        self._sampling_resident = None
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            images = self.decode_latents(pipeline, latents_clean)
        if not last_batch:
            self._sampling_placement()
            torch.cuda.empty_cache()
        return images

    def foundation_to_cpu(self, pipeline):
        self.experts_to(pipeline, {"high": "cpu", "low": "cpu"})
        pipeline.vae.to("cpu")
        if pipeline.text_encoder is not None:
            pipeline.text_encoder.to("cpu")
        self._sampling_resident = None
        self._sampling_swap = None

    # ------------------------------------------------------------------
    # Text embedding plumbing (single UMT5 text encoder, max_seq_len=512)
    # ------------------------------------------------------------------
    def compute_text_embeddings(
        self, prompt, text_encoders, tokenizers, max_sequence_length=512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if self.pipeline is None:
            raise RuntimeError(
                "Wan2.2 pipeline is not initialized. Call get_pretrained_model first."
            )
        with torch.no_grad():
            prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=True,
                max_sequence_length=max_sequence_length,
                device=self.device,
            )
        return prompt_embeds.to(device=self.device)

    def get_negative_prompt_embeddings(self, text_encoders, tokenizers):
        if self.pipeline is None:
            raise RuntimeError("Wan2.2 pipeline is not initialized.")
        # config.negative_prompt (config/wan2_2_a14b_t2v.py), overridable with NEGATIVE_PROMPT; inert at CFG 1.
        neg_text = resolve_negative_prompt_text(getattr(self.config, "negative_prompt", ""))
        # Encode the NP as the PROMPT and take the FIRST return: the second return of encode_prompt is the
        # embedding of its own `negative_prompt` argument (default ""), which would discard the text.
        with torch.no_grad():
            neg_prompt_embed, _ = self.pipeline.encode_prompt(
                prompt=[neg_text],
                do_classifier_free_guidance=False,
                max_sequence_length=512,
                device=self.device,
            )
            if neg_text:
                empty_embed, _ = self.pipeline.encode_prompt(
                    prompt=[""], do_classifier_free_guidance=False,
                    max_sequence_length=512, device=self.device,
                )
                delta = (neg_prompt_embed - empty_embed).norm().item()
                print(f"[negative_prompt] non-empty NP in use ({len(neg_text)} chars) | L2 vs empty = {delta:.3f}")
                assert delta > 1.0, "negative prompt embed identical to empty — encoding bug"
        neg_prompt_embed = neg_prompt_embed.to(device=self.device)
        neg_pooled_prompt_embed = neg_prompt_embed.new_zeros((1, 0))
        return neg_prompt_embed, neg_pooled_prompt_embed

    def get_prompt_embeddings(self, prompts, text_encoders, tokenizers):
        prompt_embeds = self.compute_text_embeddings(prompts, text_encoders, tokenizers)
        pooled_prompt_embeds = prompt_embeds.new_zeros((len(prompts), 0))
        self._last_prompts = list(prompts)
        return prompt_embeds, pooled_prompt_embeds

    def expert_step_mask(self, timesteps_row):
        """Split one sampling schedule into the two experts' noise bands.

        ``timesteps_row`` is a 1-D tensor of the sampler timesteps in
        schedule order (i.e. ``sample_data["timesteps"][0]``, before any
        per-sample time permutation — every row is the same schedule at
        that point). Returns ``(high_mask, n_high)`` where ``high_mask[i]``
        is True when step ``i`` was handled by the high-noise expert
        (``timestep >= boundary_ratio * 1000``).

        """
        schedule = timesteps_row.to(torch.float32)
        num_train_t = float(
            getattr(self.pipeline.scheduler.config, "num_train_timesteps", 1000)
        )
        boundary_t = self.boundary_ratio * num_train_t
        high_mask = (schedule >= boundary_t)
        return high_mask, int(high_mask.sum().item())

    def build_sample_kwargs(
        self, prompt_embeds, pooled_prompt_embeds,
        sample_neg_prompt_embeds, sample_neg_pooled_prompt_embeds, batch_size,
    ):
        kwargs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": sample_neg_prompt_embeds[:batch_size],
            "num_inference_steps": self.config.sample.num_steps,
            "guidance_scale": self.config.sample.guidance_scale,
            "guidance_scale_2": getattr(
                self.config.sample, "guidance_scale_2", self.config.sample.guidance_scale,
            ),
            "output_type": "pt",
            "height": self.config.height,
            "width": self.config.width,
            "num_frames": self.config.frames,
            "noise_level": self.config.sample.noise_level,
            "deterministic": self.config.sample.deterministic,
            "solver": self.config.sample.solver,
            "boundary_ratio": self.boundary_ratio,
        }
        # T2V: no `image` / `condition` kwarg — the transformer consumes the
        # bare 16-channel latents.
        return kwargs

    def rollout(self, pipeline, **kwargs):
        kwargs.setdefault("num_frames", self.config.frames)
        kwargs.setdefault("boundary_ratio", self.boundary_ratio)
        # Same switch as the training loss and the ref anchor: the CFG uncond
        # must come from the same weights in the rollout as in the loss.
        kwargs.setdefault("np_base_velocity",
                          bool(getattr(self.config.sample, "np_base_velocity", True)))
        if "peft_model" not in kwargs:
            peft_model = getattr(self, "_peft_model", None)
            if peft_model is None and hasattr(pipeline.transformer, "disable_adapter"):
                peft_model = pipeline.transformer
            kwargs["peft_model"] = peft_model
        # Sequential expert sampling (set by sampling_to_device): the pipeline calls expert_swap_fn
        # before each step, so one expert at a time is on GPU.
        if self._sampling_swap is not None:
            kwargs.setdefault("expert_swap_fn", self._sampling_swap)
        images, all_latents, _log_probs, expert_mask = (
            patched_pipeline_with_logprob(pipeline, **kwargs)
        )
        return Rollout(images, all_latents, {"expert_mask": expert_mask})

    def decode_latents(self, pipeline, latents):
        latents = latents.to(dtype=pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(pipeline.vae.config.latents_std)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        decoded = [
            pipeline.vae.decode(latents[i : i + 1], return_dict=False)[0]
            for i in range(latents.shape[0])
        ]
        image = torch.cat(decoded, dim=0)
        image = pipeline.video_processor.postprocess_video(image, output_type="pt")
        return image

    # ------------------------------------------------------------------
    # Training velocity prediction: the trainer passes only the active expert's steps
    # (expert_step_mask); _forward_ctx checks it.
    # ------------------------------------------------------------------
    def _guidance_scale(self, kind):
        # Per-band CFG (official Wan2.2: a different guidance per expert): when the LOW-noise
        # expert is active the *_2 knob replaces the scalar (default None = single value).
        scale = super()._guidance_scale(kind)
        if self.active_expert == "low":
            g2 = {"train": getattr(self.config.train, "guidance_scale_2", None),
                  "sample": getattr(self.config.sample, "guidance_scale_2", None),
                  "ref": getattr(self.config.train, "ref_guidance_scale_2", None)}[kind]
            if g2 is not None:
                scale = float(g2)
        return scale

    def _cfg_options(self, kind):
        from src.models.guidance import cfg_options
        # sg-push NP follows np_base_velocity (the A14B path never honoured cfg_sg_push_adapter).
        return {**cfg_options(self.config, kind), "sg_np": "np_base"}

    def _forward_ctx(self, transformer_ddp, xt, samples, index):
        timesteps = samples["timesteps"][:, index].to(device=xt.device, dtype=torch.float32)
        # Every step must belong to the active expert's noise band.
        boundary_t = self.boundary_ratio * float(
            getattr(self.pipeline.scheduler.config, "num_train_timesteps", 1000))
        if self.active_expert == "high":
            ok, band, side = bool((timesteps >= boundary_t).all().item()), "high-noise", "low-noise"
        else:
            ok, band, side = bool((timesteps < boundary_t).all().item()), "low-noise", "high-noise"
        if not ok:
            raise RuntimeError(
                f"Wan2.2-T2V-A14B v_pred_fn (active expert {self.active_expert}, expecting {band}) "
                f"received a {side} timestep (min={float(timesteps.min()):.1f}, "
                f"max={float(timesteps.max()):.1f}, boundary_t={boundary_t:.1f}); the trajectory "
                "must be split with expert_step_mask.")
        return {"xt": xt, "timesteps": timesteps}

    def _branch(self, transformer_ddp, ctx, cond, embed_scale):
        return transformer_ddp(
            hidden_states=ctx["xt"], timestep=ctx["timesteps"], encoder_hidden_states=cond["embeds"],
            attention_kwargs=None, return_dict=False,
        )[0]

    # ------------------------------------------------------------------
    # Logging / eval — same shape as wan2_2_i2v.py
    # ------------------------------------------------------------------
    def save_sample(self, video_tensor, output_path):
        if torch.is_tensor(video_tensor):
            video = video_tensor.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            video = np.array(video_tensor)
        video = video.transpose(0, 2, 3, 1)
        frames = [(np.clip(f, 0.0, 1.0) * 255).astype(np.uint8) for f in video]
        imageio.mimsave(
            output_path, frames, fps=self.video_fps, codec="libx264", format="FFMPEG",
        )
        return output_path

    def log_train_samples(self, images, prompts, rewards_to_log, global_step, log_step, extra_log_dict=None):
        """Encode train-sample videos and fire a single bundled wandb.log call.

        ``extra_log_dict`` (reward/advantages/etc.) is merged into the same
        wandb.log so this section only touches step=global_step once — multiple
        same-step calls were being silently dropped by wandb.
        """
        if not is_main_process(self.rank):
            return
        videos_to_log = (
            [images[i] for i in range(images.shape[0])]
            if torch.is_tensor(images) else images
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            num_to_log = min(15, len(videos_to_log))
            for idx in range(num_to_log):
                video = videos_to_log[idx]
                if torch.is_tensor(video):
                    video = video.detach().to(dtype=torch.float32).cpu().numpy()
                elif not isinstance(video, np.ndarray):
                    video = np.array(video)
                video = video.transpose(0, 2, 3, 1)
                frames = [(np.clip(f, 0.0, 1.0) * 255).astype(np.uint8) for f in video]
                imageio.mimsave(
                    os.path.join(tmpdir, f"{idx}.mp4"),
                    frames, fps=self.video_fps, codec="libx264", format="FFMPEG",
                )
            merged = dict(extra_log_dict) if extra_log_dict else {}
            merged["videos"] = [
                self.wandb.Video(
                    os.path.join(tmpdir, f"{idx}.mp4"),
                    caption=f"{prompts[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                    format="mp4", fps=self.video_fps,
                )
                for idx in range(num_to_log)
            ]
            self.wandb.log(merged, step=log_step)

    def _log_eval_outputs(self, final_rewards, images, prompts, global_step):
        if not is_main_process(self.rank):
            return

        def _safe_mean(value):
            arr = np.asarray(value)
            valid = arr[arr != -10]
            return float(np.mean(valid)) if valid.size else -10.0

        videos_to_log = (
            [images[i] for i in range(images.shape[0])]
            if torch.is_tensor(images) else images
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            n = min(5, len(videos_to_log))
            for idx in range(n):
                video = videos_to_log[idx]
                if torch.is_tensor(video):
                    video = video.detach().to(dtype=torch.float32).cpu().numpy()
                elif not isinstance(video, np.ndarray):
                    video = np.array(video)
                video = video.transpose(0, 2, 3, 1)
                frames = [(np.clip(f, 0.0, 1.0) * 255).astype(np.uint8) for f in video]
                imageio.mimsave(
                    os.path.join(tmpdir, f"{idx}.mp4"),
                    frames, fps=self.video_fps, codec="libx264", format="FFMPEG",
                )
            self.wandb.log(
                {
                    "eval_videos": [
                        self.wandb.Video(
                            os.path.join(tmpdir, f"{idx}.mp4"),
                            caption=f"{prompts[idx]:.1000} | "
                            + " | ".join(
                                f"{k}: {final_rewards[k][idx]:.2f}"
                                for k in final_rewards
                                if final_rewards[k][idx] != -10
                            ),
                            format="mp4", fps=self.video_fps,
                        )
                        for idx in range(n)
                    ],
                    **{f"eval_reward_{k}": _safe_mean(v) for k, v in final_rewards.items()},
                },
                step=global_step,
            )

    def eval_fn(
        self, pipeline, test_dataloader, text_encoders, tokenizers,
        global_step, reward_fn, executor, mixed_precision_dtype,
        ema, transformer_trainable_parameters, tqdm_fn,
    ):
        if self.config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        pipeline.transformer.eval()
        pipeline.transformer_2.eval()

        neg_embeds, _ = self.get_negative_prompt_embeddings(text_encoders, tokenizers)
        sample_neg_embeds = neg_embeds.repeat(self.config.sample.test_batch_size, 1, 1)
        all_rewards = defaultdict(list)

        test_sampler = (
            DistributedSampler(
                test_dataloader.dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=False,
            )
            if self.world_size > 1
            else None
        )
        eval_loader = DataLoader(
            test_dataloader.dataset,
            batch_size=self.config.sample.test_batch_size,
            sampler=test_sampler,
            collate_fn=test_dataloader.collate_fn,
            num_workers=test_dataloader.num_workers,
        )

        for test_batch in tqdm_fn(
            eval_loader, desc="Eval: ",
            disable=not is_main_process(self.rank), position=0,
        ):
            prompts, prompt_metadata = test_batch
            embeds = self.compute_text_embeddings(prompts, text_encoders, tokenizers)
            cur_bs = len(embeds)
            cur_neg = sample_neg_embeds[:cur_bs]

            # T2V: no reference images.

            with torch_autocast(
                "cuda",
                enabled=(mixed_precision_dtype is not None),
                dtype=mixed_precision_dtype,
            ):
                with torch.no_grad():
                    images = self.rollout(
                        pipeline,
                        prompt_embeds=embeds,
                        negative_prompt_embeds=cur_neg,
                        num_inference_steps=self.config.sample.eval_num_steps,
                        guidance_scale=self.config.sample.guidance_scale,
                        guidance_scale_2=getattr(
                            self.config.sample, "guidance_scale_2",
                            self.config.sample.guidance_scale,
                        ),
                        output_type="pt",
                        height=self.config.height,
                        width=self.config.width,
                        num_frames=self.config.frames,
                        noise_level=self.config.sample.noise_level,
                        deterministic=True,
                        solver=self.config.sample.solver,
                        boundary_ratio=self.boundary_ratio,
                    ).media

            rewards_future = executor.submit(
                reward_fn, images, prompts, prompt_metadata, only_strict=False,
            )
            time.sleep(0)
            rewards, _ = rewards_future.result()

            for key, value in rewards.items():
                rewards_tensor = torch.as_tensor(value, device=self.device).float()
                gathered_value = gather_tensor_to_all(rewards_tensor, self.world_size)
                all_rewards[key].append(gathered_value.numpy())

        if is_main_process(self.rank):
            final_rewards = {k: np.concatenate(vl) for k, vl in all_rewards.items()}
            self._log_eval_outputs(final_rewards, images, prompts, global_step)

        if self.config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)

        if self.world_size > 1:
            dist.barrier()
