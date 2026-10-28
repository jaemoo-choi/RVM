"""Phase [1], the sampling loop of an epoch: the model's rollout of every sample batch under the old adapter
(``model_utils`` provides sampling_to_device / sampling_inputs / build_sample_kwargs / rollout /
collect_sample_artifacts / decode_offloaded), collected into per-batch sample dicts."""
import os

import torch
from torch.cuda.amp import autocast as torch_autocast

from src.dataloader import DistributedKRepeatSampler


def sample_epoch(model_utils, pipeline, policies, train_iter, train_sampler, epoch, text_encoders, tokenizers,
                 neg_embeds, *, enable_amp, amp_dtype, progress):
    """Phase 1: ``config.sample.num_batches_per_epoch`` rollouts under the ``old`` adapter of every
    module in ``policies``. ``neg_embeds`` = (negative prompt embeds, pooled or None), repeated to the
    sample batch. Returns (per-batch sample dicts, last batch's media, last batch's prompts); the
    media wait on CPU in ``images_cpu`` for scoring."""

    model_utils.sampling_to_device(pipeline)
    n_batches = model_utils.config.sample.num_batches_per_epoch
    offload_during_decode = getattr(model_utils.config, "offload_transformer_during_decode", False)
    samples, last_images, last_prompts = [], None, None
    for i in progress(range(n_batches), f"Epoch {epoch}: sampling"):
        for m in policies:
            m.set_adapter("default")
        if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
            train_sampler.set_epoch(epoch * n_batches + i)
        prompts, prompt_metadata = next(train_iter)
        batch_inputs = model_utils.sampling_inputs(pipeline, prompts, prompt_metadata, text_encoders, tokenizers)
        prompt_embeds = batch_inputs["prompt_embeds"]
        pooled_prompt_embeds = batch_inputs["pooled_prompt_embeds"]

        for m in policies:
            m.set_adapter("old")
        with torch_autocast(enabled=enable_amp, dtype=amp_dtype):
            with torch.no_grad():
                sample_kwargs = model_utils.build_sample_kwargs(
                    prompt_embeds, pooled_prompt_embeds, neg_embeds[0], neg_embeds[1], len(prompts))
                if offload_during_decode:
                    sample_kwargs["skip_decode"] = True
                rollout = model_utils.rollout(pipeline, **sample_kwargs)
                images = rollout.media
        for m in policies:
            m.set_adapter("default")

        latents = rollout.trajectory
        rollout.latents = None            # the stacked copy is the only one kept
        latents_clean = latents[:, -1].clone()
        if model_utils.config.train.xt_from_latents:
            latents_trajectory = latents  # (B, num_steps+1, ...)
        else:
            del latents
        if offload_during_decode:
            images = model_utils.decode_offloaded(pipeline, latents_clean, last_batch=(i == n_batches - 1))

        timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(model_utils.device)
        images_cpu = images.detach().to("cpu", copy=True) if torch.is_tensor(images) else images
        last_images, last_prompts = images, prompts
        del images
        # Sampler-step index of each timestep, shuffled with `timesteps`.
        timestep_idx = (torch.arange(timesteps.shape[1], device=model_utils.device, dtype=torch.long)
                        .unsqueeze(0).expand(timesteps.shape[0], -1).contiguous())
        sample_data = {
            "prompt_ids": batch_inputs["prompt_ids"],
            "prompts_text": prompts,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "timesteps": timesteps,
            "timestep_idx": timestep_idx,
            "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
            "latents_clean": latents_clean,
            "images_cpu": images_cpu,
            "reward_prompts": prompts,
            "reward_prompt_metadata": prompt_metadata,
        }
        if model_utils.config.train.xt_from_latents:
            # Drop the final clean latent: latents[:, j] is the input at timesteps[:, j].
            traj = latents_trajectory[:, :-1]
            if os.getenv("XT_LATENTS_BF16", "0") == "1":
                traj = traj.to(torch.bfloat16)
            sample_data["latents"] = traj.contiguous()
            del latents_trajectory, traj
        model_utils.collect_sample_artifacts(sample_data, rollout)
        del rollout
        samples.append(sample_data)
    return samples, last_images, last_prompts
