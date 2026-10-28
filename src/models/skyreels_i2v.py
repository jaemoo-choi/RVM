"""Training utilities for SkyReels Image-to-Video (I2V).

Adapted from DanceGRPO's SkyReels-I2V implementation. The model
(xzyhku/SkyReels-V1-I2V) uses HunyuanVideoPipeline as its base class,
with I2V conditioning handled manually: reference image is VAE-encoded
and concatenated with noise along the channel dim.

Key differences from HunyuanVideo (T2V):
- Takes a reference image as conditioning input
- Reference image is VAE-encoded and concatenated with noise along channel dim
- Uses Flux pipeline to generate reference images from text prompts on-the-fly
  (following DanceGRPO's approach), or pre-generated cached images
- guidance value = 1000.0 (vs HunyuanVideo's 6018.0)
- Typically uses MQ (Motion Quality) reward from VideoAlign
"""

from collections import defaultdict
import os
import tempfile
import time

import imageio
import numpy as np
from PIL import Image
import contextlib
from src.models.pipelines.lora_ctx import maybe_disable_lora
import torch
import torch.distributed as dist
from torch import autocast as torch_autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from diffusers import HunyuanVideoPipeline, FluxPipeline
from peft import LoraConfig

from src.models.pipelines.pipeline_with_logprob_skyreels_i2v import pipeline_with_logprob as patched_pipeline_with_logprob
from src.utils.distributed import gather_tensor_to_all, is_main_process
from src.models.guidance import resolve_negative_prompt_text
from src.rollout.rollout import Rollout
from .base import BaseTrainingUtils


class SkyReelsI2VTrainingUtils(BaseTrainingUtils):
    def __init__(self, config, logger, wandb_module, device, rank, world_size):
        super().__init__(config, logger, wandb_module, device, rank, world_size)
        self.pipeline = None
        self.flux_pipeline = None

    @property
    def video_fps(self):
        return getattr(self.config, "video_fps", 24)

    @property
    def use_precomputed_embeddings(self):
        return getattr(self.config, "prompt_fn", "") in ("embedding", "embedding_image")

    def get_pretrained_model(self, mixed_precision_dtype, enable_amp):
        model_dtype = mixed_precision_dtype if enable_amp else torch.float32
        if self.use_precomputed_embeddings:
            # Load pipeline without text encoders to save GPU memory
            pipeline = HunyuanVideoPipeline.from_pretrained(
                self.config.pretrained.model, torch_dtype=model_dtype,
                text_encoder=None, text_encoder_2=None,
                tokenizer=None, tokenizer_2=None,
            )
        else:
            # Load the I2V pipeline (xzyhku/SkyReels-V1-I2V uses HunyuanVideoPipeline
            # as its base; I2V conditioning is handled manually via channel concat)
            pipeline = HunyuanVideoPipeline.from_pretrained(
                self.config.pretrained.model, torch_dtype=model_dtype,
            )
            pipeline.text_encoder.requires_grad_(False)
            pipeline.text_encoder_2.requires_grad_(False)

        pipeline.vae.requires_grad_(False)
        pipeline.transformer.requires_grad_(not self.config.use_lora)

        # VAE_TILING=1 — opt-in tiled VAE decode. At the DanceGRPO geometry
        # (400x640) the full decode fits in 80 GB, so this stays OFF by default
        # and no existing run changes behaviour. It is required at SkyReels'
        # OFFICIAL 544x960: the 3D conv decode of 53 frames then needs >79 GB
        # and OOMs on an H100. Tiles are blended by diffusers, but since this
        # path can in principle touch imaging quality, keep it explicit rather
        # than automatic so any run using it is identifiable from its config.
        if os.environ.get("VAE_TILING", "0") == "1":
            if hasattr(pipeline.vae, "enable_tiling"):
                pipeline.vae.enable_tiling()
                if is_main_process(self.rank):
                    print("[vae] tiling ENABLED (VAE_TILING=1)")
            else:
                raise RuntimeError("VAE_TILING=1 but this VAE has no enable_tiling()")

        self.pipeline = pipeline

        if self.use_precomputed_embeddings:
            text_encoders = [None, None]
            tokenizers = [None, None]
            # Load pre-computed negative embeddings from dataset
            embeddings_dir = os.path.join(self.config.dataset, "embeddings_train")
            self._neg_prompt_embeds = torch.load(
                os.path.join(embeddings_dir, "neg_prompt_embeds.pt"),
                map_location=self.device, weights_only=True,
            )
            self._neg_pooled_prompt_embeds = torch.load(
                os.path.join(embeddings_dir, "neg_pooled_prompt_embeds.pt"),
                map_location=self.device, weights_only=True,
            )
            if is_main_process(self.rank):
                self.logger.info("Using pre-computed embeddings; text encoders NOT loaded on GPU")
        else:
            text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
            tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

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
            pipeline.text_encoder_2.to(self.device, dtype=model_dtype)
        pipeline.transformer.to(self.device, dtype=model_dtype)

        # Load Flux pipeline for generating reference images from text
        # (skip if using pre-generated cached reference images or pre-computed embeddings)
        if not getattr(self.config, "use_cached_ref_images", False) and not self.use_precomputed_embeddings:
            ref_model = getattr(self.config, "reference_model", "black-forest-labs/FLUX.1-dev")
            if is_main_process(self.rank):
                self.logger.info(f"Loading Flux reference image generator from {ref_model}")
            self.flux_pipeline = FluxPipeline.from_pretrained(
                ref_model, torch_dtype=torch.bfloat16,
            )
            self.flux_pipeline.to(self.device)
        else:
            if is_main_process(self.rank):
                self.logger.info("Using pre-generated reference images, skipping Flux loading")

        return pipeline, text_encoders, tokenizers

    def get_lora_config(self):
        return LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=[
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ],
        )

    def generate_reference_image(self, prompt):
        """Generate a reference image from text using Flux (DanceGRPO pattern)."""
        with torch.no_grad():
            result = self.flux_pipeline(
                prompt,
                height=self.config.height,
                width=self.config.width,
                guidance_scale=3.5,
                num_inference_steps=30,
                max_sequence_length=512,
            )
        return result.images[0]

    def compute_text_embeddings(self, prompts, text_encoders, tokenizers, max_sequence_length=256):
        prompts = [prompts] if isinstance(prompts, str) else prompts
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized.")

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = (
                self.pipeline.encode_prompt(
                    prompt=prompts,
                    device=self.device,
                    max_sequence_length=max_sequence_length,
                )
            )

        return prompt_embeds.to(self.device), pooled_prompt_embeds.to(self.device), prompt_attention_mask.to(self.device)

    def get_negative_prompt_embeddings(self, text_encoders, tokenizers):
        if self.use_precomputed_embeddings:
            return self._neg_prompt_embeds.to(self.device), self._neg_pooled_prompt_embeds.to(self.device)

        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized.")

        # config.negative_prompt (config/skyreels_i2v.py), overridable with NEGATIVE_PROMPT; inert at CFG 1.
        neg_text = resolve_negative_prompt_text(getattr(self.config, "negative_prompt", ""))
        if neg_text:
            print(f"[negative_prompt] skyreels NP in use ({len(neg_text)} chars): {neg_text[:60]}...")
        with torch.no_grad():
            neg_embeds, neg_pooled, neg_mask = self.pipeline.encode_prompt(
                prompt=[neg_text],
                device=self.device,
                max_sequence_length=256,
            )

        # Stash the mask for build_sample_kwargs — the transformer needs the
        # REAL attention mask for the uncond branch when CFG is active.
        self._neg_prompt_attention_mask = neg_mask.to(self.device) if neg_mask is not None else None
        return neg_embeds.to(self.device), neg_pooled.to(self.device)

    def get_prompt_embeddings(self, prompts, text_encoders, tokenizers):
        if self.use_precomputed_embeddings:
            raise RuntimeError(
                "get_prompt_embeddings should not be called when using pre-computed embeddings. "
                "Embeddings come from the dataloader via prompt_metadata."
            )
        embeds, pooled, mask = self.compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=256,
        )
        self._last_prompt_attention_mask = mask
        self._last_prompts = prompts
        return embeds, pooled

    def extract_dataloader_metadata(self, prompts, prompt_metadata, text_encoders, tokenizers):
        if self.use_precomputed_embeddings:
            prompt_embeds = torch.stack(
                [m["prompt_embeds"] for m in prompt_metadata]
            ).to(self.device)
            pooled_list = [m["pooled_prompt_embeds"] for m in prompt_metadata]
            if pooled_list[0] is not None:
                pooled_prompt_embeds = torch.stack(pooled_list).to(self.device)
            else:
                pooled_prompt_embeds = None
            self._last_prompt_attention_mask = torch.stack(
                [m["prompt_attention_mask"] for m in prompt_metadata]
            ).to(self.device)
            self._last_prompts = list(prompts)
            out = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "prompt_ids": None,
            }
        else:
            out = super().extract_dataloader_metadata(
                prompts, prompt_metadata, text_encoders, tokenizers,
            )

        # Reference data: pre-computed VAE latents take priority over raw images.
        precomputed_latents = [
            m.get("first_frame_latents") for m in prompt_metadata
            if m.get("first_frame_latents") is not None
        ]
        if precomputed_latents:
            self._current_first_frame_latents = torch.stack(precomputed_latents).to(self.device)
            self._current_ref_images = None
        else:
            self._current_first_frame_latents = None
            ref_images = [m.get("ref_image") for m in prompt_metadata if m.get("ref_image")]
            self._current_ref_images = ref_images if ref_images else None
        return out

    def collect_sample_artifacts(self, sample_data, rollout=None):
        if rollout is not None and rollout.extras.get("first_frame_latents") is not None:
            sample_data["first_frame_latents"] = rollout.extras["first_frame_latents"]
        if hasattr(self, "_last_prompt_attention_mask"):
            sample_data["prompt_attention_mask"] = self._last_prompt_attention_mask

    def build_sample_kwargs(
        self,
        prompt_embeds,
        pooled_prompt_embeds,
        sample_neg_prompt_embeds,
        sample_neg_pooled_prompt_embeds,
        batch_size,
    ):
        # Generate reference images for this batch of prompts
        # We store them so the pipeline can use them
        kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": sample_neg_prompt_embeds[:batch_size],
            "negative_pooled_prompt_embeds": (
                sample_neg_pooled_prompt_embeds[:batch_size]
                if sample_neg_pooled_prompt_embeds is not None else None
            ),
            "negative_prompt_attention_mask": (
                self._neg_prompt_attention_mask.expand(batch_size, -1)
                if getattr(self, "_neg_prompt_attention_mask", None) is not None else None
            ),
            "num_inference_steps": self.config.sample.num_steps,
            "guidance_scale": self.config.sample.guidance_scale,
            "output_type": "pt",
            "height": self.config.height,
            "width": self.config.width,
            "num_frames": self.config.frames,
            "noise_level": self.config.sample.noise_level,
            "deterministic": self.config.sample.deterministic,
            "solver": self.config.sample.solver,
        }
        if hasattr(self, "_last_prompt_attention_mask"):
            kwargs["prompt_attention_mask"] = self._last_prompt_attention_mask
        # Reference data: pre-computed VAE latents > cached images > on-the-fly Flux
        if hasattr(self, "_current_first_frame_latents") and self._current_first_frame_latents is not None:
            kwargs["first_frame_latents"] = self._current_first_frame_latents
        elif hasattr(self, "_current_ref_images") and self._current_ref_images:
            kwargs["image"] = self._current_ref_images
        elif hasattr(self, "flux_pipeline"):
            ref_images = self._generate_ref_images_with_flux(self._last_prompts)
            kwargs["image"] = ref_images
        return kwargs

    def _generate_ref_images_with_flux(self, prompts):
        """Generate reference images on-the-fly using Flux (DanceGRPO approach).

        Generates one image per prompt. Flux is moved to GPU for generation
        and back to CPU afterward to save memory.
        """
        from PIL import Image

        self.flux_pipeline.to(self.device)
        ref_images = []
        for prompt in prompts:
            image = self.flux_pipeline(
                prompt,
                height=self.config.height,
                width=self.config.width,
                guidance_scale=3.5,
                num_inference_steps=30,
                max_sequence_length=512,
            ).images[0]
            ref_images.append(image)
        self.flux_pipeline.to("cpu")
        torch.cuda.empty_cache()
        return ref_images

    # ---- velocity hooks for BaseTrainingUtils.v_pred_fn (CFG lives in src/models/guidance.py) ----
    def _cfg_options(self, kind):
        from src.models.guidance import cfg_options
        # sg-push NP follows np_base_velocity (SkyReels never had a separate sg-push adapter).
        return {**cfg_options(self.config, kind), "sg_np": "np_base"}

    def _forward_ctx(self, transformer_ddp, xt, samples, index):
        # SkyReels-I2V input = cat([noise latents, first-frame latents]) on the channel dim.
        ff = samples.get("first_frame_latents", None) if isinstance(samples, dict) else None
        hidden = torch.cat([xt, ff], dim=1) if ff is not None else xt
        return {"xt": xt, "hidden": hidden, "timesteps": samples["timesteps"][:, index]}

    def _conds(self, samples, embeds, pooled_embeds, neg_embeds, neg_pooled_embeds):
        mask = samples.get("prompt_attention_mask", None) if isinstance(samples, dict) else None
        cond = {"embeds": embeds, "pooled": pooled_embeds, "mask": mask}
        if neg_embeds is None:
            return cond, None
        # The negative prompt has no stored mask: full attention over its padded length (an
        # explicit all-ones mask; HunyuanVideoTransformer3D crashes on None).
        neg_mask = torch.ones(neg_embeds.shape[:2], device=neg_embeds.device, dtype=torch.long)
        return cond, {"embeds": neg_embeds, "pooled": neg_pooled_embeds, "mask": neg_mask}

    def _branch(self, transformer_ddp, ctx, cond, embed_scale):
        # Guidance-distillation embedding = embed_scale * 1000 (the sampling pipeline's value):
        # the guidance scale itself without true CFG, 1.0 inside true CFG.
        guidance = torch.tensor([embed_scale * 1000.0] * ctx["xt"].shape[0],
                                device=ctx["xt"].device, dtype=torch.bfloat16)
        return transformer_ddp(
            hidden_states=ctx["hidden"], timestep=ctx["timesteps"], encoder_hidden_states=cond["embeds"],
            encoder_attention_mask=cond["mask"], pooled_projections=cond["pooled"], guidance=guidance,
            return_dict=False,
        )[0]

    def rollout(self, pipeline, **kwargs):
        kwargs.setdefault("num_frames", self.config.frames)
        # Same switch as the training loss and the ref anchor: the CFG uncond
        # must come from the same weights in the rollout as in the loss.
        kwargs.setdefault("np_base_velocity",
                          bool(getattr(self.config.sample, "np_base_velocity", True)))
        if "peft_model" not in kwargs:
            peft_model = getattr(self, "_peft_model", None)
            if peft_model is None and hasattr(pipeline.transformer, "disable_adapter"):
                peft_model = pipeline.transformer
            kwargs["peft_model"] = peft_model
        images, all_latents, _log_probs, first_frame_latents = (
            patched_pipeline_with_logprob(pipeline, **kwargs)
        )
        return Rollout(images, all_latents, {"first_frame_latents": first_frame_latents})

    def decode_latents(self, pipeline, latents):
        """Decode raw latents into video using SkyReels-I2V VAE."""
        from src.models.pipelines.pipeline_with_logprob_skyreels_i2v import SKYREELS_VAE_SCALING_FACTOR
        latents = latents.to(dtype=pipeline.vae.dtype) / SKYREELS_VAE_SCALING_FACTOR
        decoded = [pipeline.vae.decode(latents[i : i + 1], return_dict=False)[0] for i in range(latents.shape[0])]
        image = torch.cat(decoded, dim=0)
        image = pipeline.video_processor.postprocess_video(image, output_type="pt")
        return image

    def save_sample(self, video_tensor, output_path):
        if torch.is_tensor(video_tensor):
            video = video_tensor.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            video = np.array(video_tensor)
        video = video.transpose(0, 2, 3, 1)
        frames = [(np.clip(f, 0.0, 1.0) * 255).astype(np.uint8) for f in video]
        imageio.mimsave(output_path, frames, fps=self.video_fps, codec="libx264", format="FFMPEG")
        return output_path

    def log_train_samples(self, images, prompts, rewards_to_log, global_step, log_step):
        if not is_main_process(self.rank):
            return

        if torch.is_tensor(images):
            videos_to_log = [images[i] for i in range(images.shape[0])]
        else:
            videos_to_log = images

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
                imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames,
                                fps=self.video_fps, codec="libx264", format="FFMPEG")

            self.wandb.log(
                {
                    "videos": [
                        self.wandb.Video(
                            os.path.join(tmpdir, f"{idx}.mp4"),
                            caption=f"{prompts[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            format="mp4", fps=self.video_fps,
                        )
                        for idx in range(num_to_log)
                    ],
                },
                step=log_step,
            )

    def _log_eval_outputs(self, final_rewards, images, prompts, global_step):
        if not is_main_process(self.rank):
            return

        def _safe_reward_mean(value):
            arr = np.asarray(value)
            valid = arr[arr != -10]
            return float(np.mean(valid)) if valid.size else -10.0

        if torch.is_tensor(images):
            videos_to_log = [images[i] for i in range(images.shape[0])]
        else:
            videos_to_log = images

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
                imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames,
                                fps=self.video_fps, codec="libx264", format="FFMPEG")

            sampled_prompts = [prompts[i] for i in range(n)]
            sampled_rewards = [{k: final_rewards[k][i] for k in final_rewards} for i in range(n)]

            self.wandb.log(
                {
                    "eval_videos": [
                        self.wandb.Video(
                            os.path.join(tmpdir, f"{idx}.mp4"),
                            caption=f"{p:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in r.items() if v != -10),
                            format="mp4", fps=self.video_fps,
                        )
                        for idx, (p, r) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{k}": _safe_reward_mean(v) for k, v in final_rewards.items()},
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

        neg_embeds, _ = self.get_negative_prompt_embeddings(text_encoders, tokenizers)
        sample_neg_embeds = neg_embeds.repeat(self.config.sample.test_batch_size, 1, 1)
        all_rewards = defaultdict(list)

        test_sampler = (
            DistributedSampler(test_dataloader.dataset, num_replicas=self.world_size,
                               rank=self.rank, shuffle=False)
            if self.world_size > 1 else None
        )
        eval_loader = DataLoader(
            test_dataloader.dataset,
            batch_size=self.config.sample.test_batch_size,
            sampler=test_sampler,
            collate_fn=test_dataloader.collate_fn,
            num_workers=test_dataloader.num_workers,
        )

        for test_batch in tqdm_fn(eval_loader, desc="Eval: ",
                                   disable=not is_main_process(self.rank), position=0):
            prompts, prompt_metadata = test_batch
            embeds, pooled, mask = self.compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=256,
            )
            current_bs = len(embeds)
            cur_neg = sample_neg_embeds[:current_bs]

            # Get reference images: from metadata (cached) or generate on-the-fly
            ref_images = [m.get("ref_image") for m in prompt_metadata if m.get("ref_image")]
            if not ref_images:
                ref_images = [self.generate_reference_image(p) for p in prompts]

            with torch_autocast("cuda", enabled=(mixed_precision_dtype is not None),
                                dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images = self.rollout(
                        pipeline,
                        image=ref_images,
                        prompt_embeds=embeds,
                        pooled_prompt_embeds=pooled,
                        prompt_attention_mask=mask,
                        negative_prompt_embeds=cur_neg,
                        num_inference_steps=self.config.sample.eval_num_steps,
                        guidance_scale=self.config.sample.guidance_scale,
                        output_type="pt",
                        height=self.config.height,
                        width=self.config.width,
                        noise_level=self.config.sample.noise_level,
                        deterministic=True,
                        solver="flow",
                    ).media

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
            time.sleep(0)
            rewards, _ = rewards_future.result()

            for key, value in rewards.items():
                rewards_tensor = torch.as_tensor(value, device=self.device).float()
                gathered_value = gather_tensor_to_all(rewards_tensor, self.world_size)
                all_rewards[key].append(gathered_value.numpy())

        if is_main_process(self.rank):
            final_rewards = {key: np.concatenate(vl) for key, vl in all_rewards.items()}
            self._log_eval_outputs(final_rewards, images, prompts, global_step)

        if self.config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)

        if self.world_size > 1:
            dist.barrier()
