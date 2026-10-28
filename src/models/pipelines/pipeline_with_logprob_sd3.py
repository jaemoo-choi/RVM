"""SD3 pipeline with log-probability tracking.

Patched version of StableDiffusion3Pipeline.__call__ that returns all
intermediate latents and per-step log-probs for GRPO training.
Supports classifier-free guidance and all solvers (flow, dance, ddim, dpm1, dpm2).
"""

from typing import Any, Dict, List, Optional, Union

import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from .solver import run_sampling
from src.models.guidance import cfg


@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    noise_level: float = 0.7,
    deterministic: bool = False,
    solver: str = "flow",
    skip_decode: bool = False,
    np_base_velocity: bool = False,
):
    """Run SD3 denoising with log-probability tracking.

    Args:
        self: StableDiffusion3Pipeline instance.
        noise_level: eta for stochastic solvers (ignored when deterministic=True).
        deterministic: If True, use deterministic sampling (eta=0).
        solver: One of "flow", "dance", "ddim", "dpm1", "dpm2".
        skip_decode: If True, skip VAE decode and return None for images.
        np_base_velocity: CFG only. If True, the uncond (negative-prompt) branch
            runs on the PRETRAINED base model (LoRA disabled) in a separate
            forward, matching ``SD3TrainingUtils.v_pred_fn`` under
            ``config.train.np_base_velocity``. Default False keeps the doubled
            batch through the active adapter (both branches on the "old" LoRA).

    Returns:
        (images, all_latents, all_log_probs): Decoded images (or None if
        skip_decode), list of intermediate latents, and list of per-step
        log-probabilities.
    """
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = None
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    joint_attn_kwargs = getattr(self, "joint_attention_kwargs", None)
    lora_scale = (
        joint_attn_kwargs.get("scale", None)
        if joint_attn_kwargs is not None
        else None
    )

    # 3. Encode prompts
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=None,
    )
    self._num_timesteps = len(timesteps)
    sigmas = self.scheduler.sigmas.float()

    # 6. Define velocity prediction with CFG
    # Timesteps are passed as float sigma*1000, exactly the value train.py stores
    # (pipeline.scheduler.timesteps) and the loss feeds back in. They were cast to
    # torch.long until 2026-10-18, which truncated the conditioning timestep and made
    # the rollout field differ from the loss-side field at theta == old (breaks the
    # exact SDE log-prob ratio of METHOD=flow_grpo). SD3 had never trained with CFG.
    # CFG combination: src/models/guidance.py (kind="rollout"). Layout kept as it always was: with
    # np_base_velocity two calls, NP first on the base model; otherwise one doubled
    # [NP; cond] batch on the active adapter; branches cast to the embeds dtype, then combined.
    layout = dict(doubled=not np_base_velocity, np_first=True,
                  dtype=prompt_embeds.dtype, single_dtype=prompt_embeds.dtype)
    if self.do_classifier_free_guidance:
        neg_embeds, pos_embeds = prompt_embeds.chunk(2)
        neg_pooled, pos_pooled = pooled_prompt_embeds.chunk(2)
        cond, uncond = (pos_embeds, pos_pooled), (neg_embeds, neg_pooled)
    else:
        cond, uncond = (prompt_embeds, pooled_prompt_embeds), None

    def v_pred_fn(z, sigma):
        joint_attn_kwargs = getattr(self, "joint_attention_kwargs", None)

        def _forward(x, embeds, pooled):
            t = torch.full([x.shape[0]], sigma * 1000, device=z.device, dtype=torch.float32)
            return self.transformer(
                hidden_states=x, timestep=t, encoder_hidden_states=embeds,
                pooled_projections=pooled, joint_attention_kwargs=joint_attn_kwargs,
                return_dict=False,
            )[0]

        def branch(c, embed_scale):
            return _forward(z, c[0], c[1])

        def pair(c, u):
            # the doubled batch was built at encode time as [NP; cond]
            v_np, v_c = _forward(torch.cat([z] * 2), prompt_embeds, pooled_prompt_embeds).chunk(2)
            return v_c, v_np

        return cfg(branch, pair, self.transformer, cond, uncond, guidance_scale, kind="rollout",
                   np_base=np_base_velocity, layout=layout)

    # 7. Denoising loop
    latents, all_latents, all_log_probs = run_sampling(
        v_pred_fn, latents, sigmas, solver, deterministic, noise_level
    )

    # 8. VAE decode (skipped when skip_decode=True to allow transformer offload)
    if not skip_decode:
        latents = latents.to(dtype=self.vae.dtype)
        scaling_factor = getattr(self.vae.config, "scaling_factor", None)
        shift_factor = getattr(self.vae.config, "shift_factor", None)
        if scaling_factor is not None and shift_factor is not None:
            latents = (latents / scaling_factor) + shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
    else:
        image = None

    self.maybe_free_model_hooks()

    return image, all_latents, all_log_probs
