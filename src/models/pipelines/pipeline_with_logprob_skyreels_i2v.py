"""SkyReels-I2V pipeline with log-probability tracking.

Patched version of HunyuanSkyreelsImageToVideoPipeline.__call__ that returns
all intermediate latents and per-step log-probs for GRPO training.

SkyReels-I2V takes a reference image as conditioning input:
1. The image is VAE-encoded into first_frame_latents (16 channels)
2. Temporal zero-padding extends it to match the video temporal dimension
3. At each denoising step, [noise_latent, first_frame_latents] are concatenated
   along the channel dimension (16+16=32 channels) before the transformer

Uses guidance embeddings (not CFG) with a fixed value of 1000.0 (DanceGRPO).
Supports all solvers (flow, dance, ddim, dpm1, dpm2).
"""

from typing import List, Optional, Union

import torch
import torchvision.transforms as T
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from .solver import run_sampling
from src.models.guidance import cfg


# Per-unit guidance-embedding multiplier: the transformer's `guidance` input is
# `guidance_scale * 1000` (diffusers guidance-distillation convention). The old
# hard-coded 1000.0 == guidance_scale 1.0; now guidance_scale actually scales it.
SKYREELS_GUIDANCE_VALUE = 1000.0
SKYREELS_VAE_SCALING_FACTOR = 0.476986


@torch.no_grad()
def pipeline_with_logprob(
    self,
    image=None,
    first_frame_latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: int = 53,
    num_inference_steps: int = 16,
    guidance_scale: float = 1.0,
    np_base_velocity: bool = True,
    peft_model=None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    noise_level: float = 0.7,
    deterministic: bool = False,
    solver: str = "dpm2",
    skip_decode: bool = False,
):
    """Run SkyReels-I2V denoising with log-probability tracking.

    Args:
        self: HunyuanVideoPipeline instance (xzyhku/SkyReels-V1-I2V).
        image: Reference image(s) for I2V conditioning (VAE-encoded on-the-fly).
        first_frame_latents: Pre-computed VAE-encoded reference image latents.
            Shape (B, 16, T_latent, H/8, W/8) with temporal zero-padding.
            If provided, `image` is ignored and no VAE encoding is needed.
        prompt_embeds: Pre-computed text embeddings.
        pooled_prompt_embeds: Pooled text embeddings.
        prompt_attention_mask: Attention mask for text embeddings.
        num_frames: Number of video frames to generate.
        noise_level: eta for stochastic solvers (ignored when deterministic=True).
        deterministic: If True, use deterministic sampling (eta=0).
        solver: One of "flow", "dance", "ddim", "dpm1", "dpm2".
        skip_decode: If True, skip VAE decode and return None for videos.

    Returns:
        (videos, all_latents, all_log_probs, first_frame_latents):
        Decoded video tensors (or None if skip_decode), list of intermediate
        latents, list of per-step log-probabilities, and the VAE-encoded
        reference image latents (needed for training v_pred_fn).
    """
    self._guidance_scale = guidance_scale
    self._current_timestep = None
    self._interrupt = False

    batch_size = prompt_embeds.shape[0]
    device = self._execution_device
    model_dtype = prompt_embeds.dtype

    # 1. Get first_frame_latents: use pre-computed if available, otherwise VAE-encode
    if first_frame_latents is not None:
        first_frame_latents = first_frame_latents.to(device=device, dtype=model_dtype)
    elif image is not None:
        first_frame_latents = _encode_reference_image(
            self, image, batch_size, num_frames, device, model_dtype, generator,
        )
    else:
        raise ValueError("SkyReels-I2V requires either `first_frame_latents` or `image`.")

    # 2. Prepare noise latents (16 channels, not 32 -- the other 16 come from image)
    # Note: we generate noise directly because the I2V pipeline's prepare_latents
    # expects `image` as the first arg and does its own VAE encoding + blending.
    # We handle image conditioning separately via _encode_reference_image.
    num_channels_latents = first_frame_latents.shape[1]  # 16
    vae_scale_factor_spatial = getattr(self.vae.config, "spatial_compression_ratio", 8)
    vae_scale_factor_temporal = getattr(self.vae.config, "temporal_compression_ratio", 4)
    latent_height = height // vae_scale_factor_spatial
    latent_width = width // vae_scale_factor_spatial
    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
    shape = (
        batch_size * num_images_per_prompt,
        num_channels_latents,
        num_latent_frames,
        latent_height,
        latent_width,
    )
    from diffusers.utils.torch_utils import randn_tensor
    latents = randn_tensor(shape, generator=generator, device=device, dtype=model_dtype)

    # 3. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=None,
    )
    self._num_timesteps = len(timesteps)
    sigmas = self.scheduler.sigmas.float()

    # 4. Define velocity prediction: concatenate [noise, first_frame] along channel dim
    #
    # TRUE CFG (added 2026-08-25, official SkyReels-V1 parity): the official
    # video_generate.py runs batch CFG (guidance_scale 6.0 + its negative
    # prompt) ON TOP of the guidance-distillation embedding, which it keeps at
    # embedded_guidance_scale=1.0. Historically this pipeline had NO uncond
    # branch — "cfg=5" runs (June 2026) only scaled the *embedding* to 5000,
    # a 5x out-of-distribution conditioning value with the negative embeds
    # silently ignored. Semantics now:
    #   guidance_scale == 1 or no neg embeds  -> single cond forward, embedding
    #       = guidance_scale*1000 (bit-identical to the historical behavior,
    #       so every training run is unaffected)
    #   guidance_scale > 1 with neg embeds    -> real CFG: [uncond; cond]
    #       batch, embedding pinned to 1.0*1000 for BOTH branches (official
    #       embedded_guidance_scale=1.0), uncond + s*(cond - uncond)
    do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None
    if do_true_cfg:
        cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        cfg_pooled = (
            torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            if (negative_pooled_prompt_embeds is not None and pooled_prompt_embeds is not None)
            else pooled_prompt_embeds
        )
        if prompt_attention_mask is not None and negative_prompt_attention_mask is not None:
            cfg_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )
        else:
            cfg_attention_mask = prompt_attention_mask

    def _forward(z_in, ff_in, embeds_in, pooled_in, mask_in, embed_gs, sigma):
        hidden_states = torch.cat([z_in, ff_in], dim=1)
        timesteps = torch.full(
            [z_in.shape[0]], sigma * 1000, device=z_in.device, dtype=torch.long
        )
        guidance_tensor = torch.tensor(
            [embed_gs * SKYREELS_GUIDANCE_VALUE] * z_in.shape[0],
            device=z_in.device,
            dtype=torch.bfloat16,
        )
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timesteps,
            encoder_hidden_states=embeds_in,
            encoder_attention_mask=mask_in,
            pooled_projections=pooled_in,
            guidance=guidance_tensor,
            return_dict=False,
        )[0].to(model_dtype)

    # CFG combination: src/models/guidance.py (kind="rollout"). Layout kept as it always was: with
    # np_base_velocity two sequential forwards (batch B each), conditional first, the NP on the
    # PRETRAINED base model, exactly as in the training loss and the reference anchor (the
    # doubled batch cannot give the two halves different adapter states); otherwise both branches
    # on the active adapter in one doubled [NP; cond] batch. _forward already casts to model_dtype.
    layout = dict(doubled=not np_base_velocity, np_first=False, dtype=None, single_dtype=None)
    cond = (prompt_embeds, pooled_prompt_embeds, prompt_attention_mask)
    uncond = None
    if do_true_cfg:
        _neg_mask = negative_prompt_attention_mask
        if _neg_mask is None and np_base_velocity:
            # HunyuanVideoTransformer3D calls .sum() on the mask, so None
            # crashes; full attention over the padded negative embeds
            # (same fallback as SkyReelsI2VTrainingUtils.v_pred_fn).
            _neg_mask = torch.ones(
                negative_prompt_embeds.shape[:2],
                device=negative_prompt_embeds.device, dtype=torch.long,
            )
        uncond = (negative_prompt_embeds, negative_pooled_prompt_embeds, _neg_mask)

    def v_pred_fn(z, sigma):
        # z: (B, 16, T, H, W) noise latent
        # first_frame_latents: (B, 16, T, H, W) VAE-encoded reference + zero padding
        def branch(c, embed_scale):
            # embed_scale: guidance_scale without true CFG (historical embedding scaling),
            # 1.0 for both CFG branches (official embedded_guidance_scale=1.0)
            return _forward(z, first_frame_latents, c[0], c[1], c[2], embed_scale, sigma)

        def pair(c, u):
            noise_pred = _forward(
                torch.cat([z, z], dim=0),
                torch.cat([first_frame_latents, first_frame_latents], dim=0),
                cfg_prompt_embeds, cfg_pooled, cfg_attention_mask, 1.0, sigma,
            )
            v_np, v_c = noise_pred.chunk(2, dim=0)
            return v_c, v_np

        return cfg(branch, pair, self.transformer, cond, uncond, guidance_scale, kind="rollout",
                   np_base=np_base_velocity, layout=layout)

    # 5. Denoising loop
    latents, all_latents, all_log_probs = run_sampling(
        v_pred_fn, latents, sigmas, solver, deterministic, noise_level
    )

    # 6. VAE decode (skipped when skip_decode=True to allow transformer offload)
    if not skip_decode:
        latents = latents.to(dtype=self.vae.dtype) / SKYREELS_VAE_SCALING_FACTOR
        decoded = [self.vae.decode(latents[i : i + 1], return_dict=False)[0] for i in range(latents.shape[0])]
        image_out = torch.cat(decoded, dim=0)
        image_out = self.video_processor.postprocess_video(
            image_out, output_type=output_type
        )
    else:
        image_out = None

    self.maybe_free_model_hooks()

    return image_out, all_latents, all_log_probs, first_frame_latents


def _encode_reference_image(pipeline, image, batch_size, num_frames, device, dtype, generator):
    """VAE-encode reference image(s) and prepare temporally-padded latents.

    Following DanceGRPO:
    1. Normalize image to [-1, 1]
    2. VAE encode -> (B, 16, 1, H/8, W/8)
    3. Multiply by scaling factor 0.476986
    4. Zero-pad temporal dim to match num_latent_frames

    Returns:
        first_frame_latents: (B, 16, T_latent, H/8, W/8)
    """
    from PIL import Image as PILImage

    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Handle different input types
    if isinstance(image, list):
        tensors = [preprocess(img).unsqueeze(0) for img in image]
        image_tensor = torch.cat(tensors, dim=0)
    elif isinstance(image, PILImage.Image):
        image_tensor = preprocess(image).unsqueeze(0)
        if batch_size > 1:
            image_tensor = image_tensor.repeat(batch_size, 1, 1, 1)
    elif torch.is_tensor(image):
        # Assume (B, C, H, W) in [0, 1], normalize to [-1, 1]
        image_tensor = image * 2.0 - 1.0
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    image_tensor = image_tensor.to(device=device, dtype=dtype)

    # Add temporal dimension: (B, C, H, W) -> (B, C, 1, H, W)
    image_tensor = image_tensor.unsqueeze(2)

    # VAE encode
    latent_dist = pipeline.vae.encode(image_tensor).latent_dist
    if generator is not None:
        first_frame_latents = latent_dist.sample(generator=generator)
    else:
        first_frame_latents = latent_dist.sample()
    first_frame_latents = first_frame_latents * SKYREELS_VAE_SCALING_FACTOR

    # Compute number of latent frames
    vae_temporal_factor = getattr(pipeline.vae.config, "temporal_compression_ratio", 4)
    num_latent_frames = (num_frames - 1) // vae_temporal_factor + 1

    # Zero-pad temporal dimension: (B, 16, 1, H/8, W/8) -> (B, 16, T_latent, H/8, W/8)
    if num_latent_frames > 1:
        padding = torch.zeros(
            first_frame_latents.shape[0],
            first_frame_latents.shape[1],
            num_latent_frames - 1,
            first_frame_latents.shape[3],
            first_frame_latents.shape[4],
            device=device,
            dtype=dtype,
        )
        first_frame_latents = torch.cat([first_frame_latents, padding], dim=2)

    return first_frame_latents
