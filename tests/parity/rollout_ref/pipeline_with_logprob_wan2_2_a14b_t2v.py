"""Wan2.2-T2V-A14B pipeline with log-probability tracking.

The A14B text-to-video checkpoint hosted at
``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` is loaded via ``diffusers.WanPipeline``.
It is the T2V sibling of ``pipeline_with_logprob_wan2_2_a14b_i2v.py`` and
shares the dual-expert structure, but drops all image conditioning:

1. **Dual expert ("A14B").** Two full 14B transformers are stacked in time —
   ``self.transformer`` handles timesteps ``>= boundary_ratio * 1000`` (high
   noise / early steps) and ``self.transformer_2`` handles the rest (low
   noise / late steps). ~28B params on disk, 14B active per forward.
   ``boundary_ratio`` comes from the pipeline config.
2. **No condition channels.** Unlike I2V (which concatenates 4 mask + 16
   VAE first-frame channels for ``in_channels≈36``), T2V feeds the bare
   ``latents`` to the transformer — ``in_channels == 16``. There is no
   ``condition`` tensor and no ``encoder_hidden_states_image``.

``expert_mask`` is returned alongside latents/log-probs as a diagnostic
record of which expert handled each step. Training does NOT consume it:
``Wan22A14BT2VTrainingUtils.v_pred_fn`` re-derives the dispatch from
``samples["timesteps"][:, index]`` so the decision stays correct under
train.py's per-sample time permutation (which would desync any
time-axis-aligned tensor not shuffled in lockstep). Only the expert selected
by ``TRAIN_EXPERT`` carries LoRA, so steps mapped to the frozen expert
contribute zero policy gradient (current vs old log-prob is identically 1).

Supports flow / dance / ddim / dpm1 / dpm2 / unipc solvers via the shared
``run_sampling`` helper.
"""

from typing import List, Optional, Union

import contextlib
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from .solver import run_sampling
from .lora_ctx import maybe_disable_lora



@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    guidance_scale_2: Optional[float] = None,
    np_base_velocity: bool = True,
    peft_model=None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    noise_level: float = 0.7,
    deterministic: bool = False,
    solver: str = "unipc",
    skip_decode: bool = False,
    boundary_ratio: Optional[float] = None,
    expert_swap_fn=None,
    **_unused,
):
    """Wan2.2-T2V-A14B denoising loop with log-probability tracking.

    Args:
        self: ``WanPipeline`` loaded from ``Wan-AI/Wan2.2-T2V-A14B-Diffusers``.
            Must have ``transformer_2`` populated and ``boundary_ratio`` set.
        prompt_embeds / negative_prompt_embeds: pre-computed UMT5 text embeds.
        guidance_scale: CFG weight for the high-noise stage.
        guidance_scale_2: CFG weight for the low-noise stage; defaults to
            ``guidance_scale`` when ``None``.
        boundary_ratio: Override the pipeline's ``boundary_ratio``.
        skip_decode: skip VAE decode and return ``None`` for the video tensor.
        expert_swap_fn: optional ``fn(is_high_noise: bool) -> None`` invoked
            immediately before each step's expert is used. Lets the caller keep
            only ONE 14B expert resident on GPU during sampling (~28 GB instead
            of ~56 GB). Because the schedule descends monotonically through
            ``boundary_ratio``, this fires exactly one high->low transition per
            call. ``None`` (the default) keeps the original behaviour, which
            assumes both experts are already on GPU.

    Returns:
        ``(videos, all_latents, all_log_probs, expert_mask)`` — ``expert_mask[i]``
        is ``True`` when step ``i`` was handled by ``self.transformer``
        (high-noise expert) and ``False`` for ``self.transformer_2``.
        Diagnostic only; training re-derives the dispatch per timestep.
    """
    self._guidance_scale = guidance_scale
    self._guidance_scale_2 = (
        guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
    )
    self._attention_kwargs = None
    self._current_timestep = None
    self._interrupt = False

    if boundary_ratio is None:
        boundary_ratio = getattr(self.config, "boundary_ratio", None)
    assert boundary_ratio is not None, (
        "Wan2.2-T2V-A14B requires `boundary_ratio`. The HF model_index.json "
        "sets it on `self.config`; pass it explicitly if loading the pipeline "
        "outside the standard from_pretrained path."
    )
    assert getattr(self, "transformer_2", None) is not None, (
        "Wan2.2-T2V-A14B requires both `transformer` (high-noise expert) and "
        "`transformer_2` (low-noise expert). Got transformer_2=None."
    )

    batch_size = prompt_embeds.shape[0]
    # _execution_device inspects pipeline components and can return CPU when
    # offloading leaves a mix of CPU/GPU components. Anchor to the transformer's
    # device — both experts are on GPU at this point per the train-loop phase
    # management in src/train_wan2_2_a14b.py.
    # With expert_swap_fn the inactive expert is parked on CPU, so probing
    # `self.transformer` alone can hand back CPU and silently allocate the
    # latents there. Take the first CUDA device among the components that are
    # guaranteed resident during sampling.
    device = None
    for _component in (self.transformer, self.transformer_2, self.vae):
        try:
            _d = next(_component.parameters()).device
        except (StopIteration, AttributeError):
            continue
        if _d.type == "cuda":
            device = _d
            break
        if device is None:
            device = _d
    if device is None:
        device = self._execution_device
    transformer_dtype = self.transformer.dtype

    # 1. Latents — pure noise, 16 channels, no conditioning.
    num_channels_latents = self.vae.config.z_dim
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

    # 2. Timestep schedule (flow-matching sigmas).
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=None,
    )
    self._num_timesteps = len(timesteps)
    sigmas = self.scheduler.sigmas.float()

    boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps
    have_neg = negative_prompt_embeds is not None

    # 3. Track which expert handled each step (diagnostic).
    expert_mask: List[bool] = []  # True == high-noise (transformer)

    def v_pred_fn(z, sigma):
        # Wan2.2 schedule encodes timesteps as sigma * num_train_timesteps.
        t_scalar = float(sigma) * float(self.scheduler.config.num_train_timesteps)
        is_high_noise = t_scalar >= boundary_timestep
        expert_mask.append(bool(is_high_noise))
        # Give the caller a chance to page the needed expert onto GPU (and the
        # other one off) before we touch it. No-op unless expert_swap_fn was
        # passed; see src/train_wan2_2.py.
        if expert_swap_fn is not None:
            expert_swap_fn(bool(is_high_noise))
        current_model = self.transformer if is_high_noise else self.transformer_2
        # CFG weight is per-stage; check the *active* stage's scale, not the
        # pipeline-level `do_classifier_free_guidance` (which only inspects the
        # high-noise guidance_scale and would skip CFG on low steps when only
        # guidance_scale_2 > 1).
        cur_g = guidance_scale if is_high_noise else self._guidance_scale_2

        latent_model_input = z.to(transformer_dtype)
        timestep = torch.full(
            [z.shape[0]], t_scalar, device=z.device, dtype=torch.float32
        )

        noise_pred = current_model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        if have_neg and cur_g > 1.0:
            # The CFG uncond runs on the PRETRAINED base model when
            # np_base_velocity (the default), matching the training loss and the
            # reference anchor. `current_model` is the expert actually being
            # called (high or low), so disable the adapter on that module rather
            # than on a separately-passed handle. NP_BASE_VELOCITY=0 keeps both
            # branches on the active adapter.
            with maybe_disable_lora(current_model, np_base_velocity):
                noise_uncond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            noise_pred = noise_uncond.to(torch.float32) + cur_g * (
                noise_pred.to(torch.float32) - noise_uncond.to(torch.float32)
            )
        else:
            noise_pred = noise_pred.to(torch.float32)
        return noise_pred

    # 4. Run the standard solver loop (flow / unipc / etc.).
    latents, all_latents, all_log_probs = run_sampling(
        v_pred_fn, latents, sigmas, solver, deterministic, noise_level,
        scheduler=self.scheduler,
    )

    # 5. VAE decode with Wan latents_mean/std normalisation.
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
        decoded = [
            self.vae.decode(latents[i : i + 1], return_dict=False)[0]
            for i in range(latents.shape[0])
        ]
        image_out = torch.cat(decoded, dim=0)
        image_out = self.video_processor.postprocess_video(
            image_out, output_type=output_type
        )
    else:
        image_out = None

    self.maybe_free_model_hooks()

    expert_mask_t = torch.tensor(expert_mask, dtype=torch.bool, device=device)
    return image_out, all_latents, all_log_probs, expert_mask_t
