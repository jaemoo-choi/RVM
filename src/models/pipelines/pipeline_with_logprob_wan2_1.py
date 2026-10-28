"""WAN2.1 pipeline with log-probability tracking.

Patched version of WanPipeline.__call__ that returns all intermediate
latents and per-step log-probs for GRPO training.
Supports classifier-free guidance and all solvers (flow, dance, ddim, dpm1, dpm2).
"""

from typing import Callable, List, Optional, Union

import contextlib
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from .solver import run_sampling
from src.models.guidance import cfg



@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    noise_level: float = 0.7,
    deterministic: bool = False,
    solver: str = "flow",
    skip_decode: bool = False,
    peft_model=None,
    np_base_velocity: bool = True,
    velocity_callback: Optional[Callable] = None,
    time_fraction_start_idx: int = 0,
    peft_model_for_toggle=None,
    custom_sigmas=None,
):
    """Run WAN2.1 denoising with log-probability tracking.

    WAN2.1 uses a single text encoder. Prompt embeddings are expected to be
    pre-computed and passed directly (not raw text strings).

    Args:
        self: WanPipeline instance.
        prompt_embeds: Pre-computed text embeddings.
        negative_prompt_embeds: Pre-computed negative embeddings (for CFG).
        num_frames: Number of video frames to generate.
        noise_level: eta for stochastic solvers (ignored when deterministic=True).
        deterministic: If True, use deterministic sampling (eta=0).
        solver: One of "flow", "dance", "ddim", "dpm1", "dpm2".
        skip_decode: If True, skip VAE decode and return None for videos.
        peft_model: Optional PeftModel reference. When provided and CFG is
            active, uncond and cond are computed as two sequential forwards
            (batch = B each) through the currently active adapter, instead
            of one doubled-batch forward (batch = 2B). Reduces peak activation
            memory at the cost of an extra kernel launch.

    Returns:
        (videos, all_latents, all_log_probs): Decoded video tensors (or None if
        skip_decode), list of intermediate latents, and list of per-step
        log-probabilities.
    """
    self._guidance_scale = guidance_scale
    self._current_timestep = None
    self._interrupt = False

    # 1. Define call parameters
    batch_size = prompt_embeds.shape[0]
    device = self._execution_device

    # 2. Prepare embeddings for CFG
    if self.do_classifier_free_guidance and negative_prompt_embeds is not None:
        prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    else:
        prompt_embeds_combined = prompt_embeds

    # 3. Prepare 5D latent variables (B, C, T, H, W)
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
    )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=None,
    )
    self._num_timesteps = len(timesteps)
    sigmas = self.scheduler.sigmas.float()

    # Optional custom non-uniform discretization (config.sample.sigmas). WAN's
    # UniPCMultistepScheduler.set_timesteps does not accept a `sigmas=` arg, so
    # we override the schedule array directly: the dpm2/flow solvers consume this
    # `sigmas` array element-by-element (log-prob uses dt=σ[i+1]-σ[i], so
    # non-uniform spacing is exact), and scheduler.timesteps is updated so the
    # downstream noise-region reward split (train.py reads scheduler.timesteps)
    # routes by the custom per-step noise levels. custom_sigmas = the N per-step
    # sigmas (descending, no trailing 0); we append the 0.0 endpoint.
    if custom_sigmas is not None:
        cs = torch.tensor(list(custom_sigmas), dtype=torch.float32, device=device)
        sigmas = torch.cat([cs, cs.new_zeros(1)])
        self.scheduler.sigmas = sigmas
        self.scheduler.timesteps = (cs * 1000.0).to(device)
        self._num_timesteps = int(cs.numel())
        num_inference_steps = int(cs.numel())

    # 5. Define velocity prediction with CFG: src/models/guidance.py (kind="rollout"). Layout kept as
    #    it always was: with a peft_model, two sequential forwards (batch B each), the NP first --
    #    on the PRETRAINED base model when np_base_velocity (the default), matching the training
    #    loss and the reference anchor; NP_BASE_VELOCITY=0 runs both on the adapter. Without one,
    #    the doubled-batch fallback (adapter on both branches). Combined in fp32; without CFG the
    #    single forward is cast to the embeds dtype.
    layout = dict(doubled=peft_model is None, np_first=True,
                  dtype=torch.float32, single_dtype=prompt_embeds.dtype)
    uncond = negative_prompt_embeds if self.do_classifier_free_guidance else None

    def v_pred_fn(z, sigma):
        timesteps = torch.full(
            [z.shape[0]], sigma * 1000, device=z.device, dtype=torch.float32
        )

        def branch(embeds, embed_scale):
            return self.transformer(
                hidden_states=z,
                timestep=timesteps,
                encoder_hidden_states=embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]

        def pair(c, u):
            pred = self.transformer(
                hidden_states=torch.cat([z, z], dim=0),
                timestep=torch.cat([timesteps, timesteps], dim=0),
                encoder_hidden_states=prompt_embeds_combined,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            v_np, v_c = pred.chunk(2)
            return v_c, v_np

        noise_pred, noise_pred_uncond, noise_pred_text = cfg(
            branch, pair, self.transformer, prompt_embeds, uncond, guidance_scale, kind="rollout",
            np_base=np_base_velocity, layout=layout, return_parts=True,
        )
        if velocity_callback is not None:
            velocity_callback(sigma, z, noise_pred_uncond, noise_pred_text, noise_pred)
        return noise_pred

    # 6. Denoising loop. Forward the scheduler so the ``unipc`` solver in
    #    ``run_sampling`` can delegate to ``UniPCMultistepScheduler.step``
    #    (matches the WAN2.1 base model's trained sampler). Other solvers
    #    ignore the kwarg.
    latents, all_latents, all_log_probs = run_sampling(
        v_pred_fn, latents, sigmas, solver, deterministic, noise_level,
        time_fraction_start_idx=time_fraction_start_idx,
        peft_model_for_toggle=peft_model_for_toggle,
        scheduler=self.scheduler,
    )

    # 7. VAE decode (skipped when skip_decode=True to allow transformer offload)
    if not skip_decode:
        latents = latents.to(dtype=self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        # Decode per-sample to avoid PyTorch upsample_nearest3d INT_MAX element limit
        decoded = [self.vae.decode(latents[i : i + 1], return_dict=False)[0] for i in range(latents.shape[0])]
        image = torch.cat(decoded, dim=0)
        image = self.video_processor.postprocess_video(image, output_type=output_type)
    else:
        image = None

    self.maybe_free_model_hooks()

    return image, all_latents, all_log_probs
