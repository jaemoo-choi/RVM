from collections import defaultdict
import os
import tempfile
import time

import imageio
import numpy as np
from src.models.pipelines.lora_ctx import maybe_disable_lora
import torch
import torch.distributed as dist
from torch import autocast as torch_autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from diffusers import WanPipeline
from peft import LoraConfig

from src.models.pipelines.pipeline_with_logprob_wan2_1 import pipeline_with_logprob as patched_pipeline_with_logprob
from src.utils.distributed import gather_tensor_to_all, is_main_process
from src.models.guidance import resolve_negative_prompt_text
from src.rollout.rollout import Rollout
from .base import BaseTrainingUtils


class Wan21TrainingUtils(BaseTrainingUtils):
    def __init__(self, config, logger, wandb_module, device, rank, world_size):
        super().__init__(config, logger, wandb_module, device, rank, world_size)
        self.pipeline = None
        self._peft_model = None  # set by set_adapter(); used for base-model CFG negative

    def set_adapter(self, pipeline, local_rank):
        result = super().set_adapter(pipeline, local_rank)
        # Store PeftModel reference so pipeline_with_logprob and v_pred_fn
        # can disable LoRA adapters for the CFG negative prediction.
        self._peft_model = result[0].module  # transformer_ddp.module
        return result

    def get_pretrained_model(self, mixed_precision_dtype, enable_amp):
        model_dtype = mixed_precision_dtype if enable_amp else torch.float32
        pipeline = WanPipeline.from_pretrained(self.config.pretrained.model, torch_dtype=model_dtype)
        pipeline.scheduler = self._maybe_override_flow_shift(pipeline.scheduler)
        pipeline.vae.requires_grad_(False)
        if pipeline.text_encoder is not None:
            pipeline.text_encoder.requires_grad_(False)
        pipeline.transformer.requires_grad_(not self.config.use_lora)
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

        text_encoder_dtype = model_dtype
        pipeline.to(self.device)
        pipeline.vae.to(self.device, dtype=model_dtype)
        if pipeline.text_encoder is not None:
            pipeline.text_encoder.to(self.device, dtype=text_encoder_dtype)
        pipeline.transformer.to(self.device, dtype=model_dtype)

        return pipeline, text_encoders, tokenizers

    def get_lora_config(self):
        # LoRA on every linear in self-attention (attn1: to_q/k/v/out),
        # cross-attention (attn2: to_q/k/v/out), and FFN (net.0.proj, net.2).
        # PEFT matches by name suffix so the strings cover both attention
        # blocks at once. r/alpha read from the config so 1.3B and 14B can
        # ship different sizes through the same utils class.
        r = getattr(self.config, "lora_r", 128)
        alpha = getattr(self.config, "lora_alpha", 64)
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "ffn.net.0.proj",
                "ffn.net.2",
            ],
        )

    def compute_text_embeddings(self, prompt, text_encoders, tokenizers, max_sequence_length=512):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if self.pipeline is None:
            raise RuntimeError("WAN pipeline is not initialized. Call get_pretrained_model first.")

        with torch.no_grad():
            prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=True,
                max_sequence_length=max_sequence_length,
            )

        return prompt_embeds.to(device=self.device)

    def get_negative_prompt_embeddings(self, text_encoders, tokenizers):
        if self.pipeline is None:
            raise RuntimeError("WAN pipeline is not initialized. Call get_pretrained_model first.")

        # config.negative_prompt (config/wan2_1.py), overridable with NEGATIVE_PROMPT; inert at CFG 1.
        neg_text = resolve_negative_prompt_text(getattr(self.config, "negative_prompt", ""))

        # Encode neg_text as the PROMPT and take the first return value.
        # Do NOT pass it as encode_prompt's `prompt` with do_cfg=True and grab
        # the second return — that second value is the embedding of
        # encode_prompt's own `negative_prompt` argument (default ""), so the
        # text would be encoded and silently DISCARDED. That exact bug shipped
        # 2026-08-23: banner printed, videos byte-identical to
        # the empty-NP run.
        with torch.no_grad():
            neg_prompt_embed, _ = self.pipeline.encode_prompt(
                prompt=[neg_text],
                do_classifier_free_guidance=False,
                max_sequence_length=512,
            )
            if neg_text:
                # Self-check in the job log: prove the embed differs from "".
                empty_embed, _ = self.pipeline.encode_prompt(
                    prompt=[""], do_classifier_free_guidance=False, max_sequence_length=512,
                )
                delta = (neg_prompt_embed - empty_embed).norm().item()
                print(
                    f"[negative_prompt] non-empty NP in use ({len(neg_text)} chars): "
                    f"{neg_text[:48]}... | L2(NP_embed - empty_embed) = {delta:.3f}"
                )
                assert delta > 1.0, "negative prompt embed identical to empty — encoding bug"

        neg_prompt_embed = neg_prompt_embed.to(device=self.device)
        neg_pooled_prompt_embed = neg_prompt_embed.new_zeros((1, 0))
        return neg_prompt_embed, neg_pooled_prompt_embed

    def get_prompt_embeddings(self, prompts, text_encoders, tokenizers):
        prompt_embeds = self.compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=512
        )
        pooled_prompt_embeds = prompt_embeds.new_zeros((len(prompts), 0))
        return prompt_embeds, pooled_prompt_embeds

    def build_sample_kwargs(
        self,
        prompt_embeds,
        pooled_prompt_embeds,
        sample_neg_prompt_embeds,
        sample_neg_pooled_prompt_embeds,
        batch_size,
    ):
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": sample_neg_prompt_embeds[:batch_size],
            "num_inference_steps": self.config.sample.num_steps,
            "guidance_scale": self.config.sample.guidance_scale,
            "output_type": "pt",
            "height": self.config.height,
            "width": self.config.width,
            "num_frames": self.config.frames,
            "noise_level": self.config.sample.noise_level,
            "deterministic": self.config.sample.deterministic,
            "solver": self.config.sample.solver,
            # Optional custom non-uniform discretization (config.sample.sigmas).
            # None -> shift-derived uniform schedule (default behavior).
            "custom_sigmas": list(getattr(self.config.sample, "sigmas", None))
            if getattr(self.config.sample, "sigmas", None) else None,
        }

    def _cfg_options(self, kind):
        from src.models.guidance import cfg_options
        options = cfg_options(self.config, kind)
        if options["sg_np"] == "base":
            # Wan2.1's "base" sg-push reuses the plain-CFG NP (the shared uncond, or the
            # np_base_velocity switch), not a forced base-model forward.
            options["sg_np"] = "np_base"
        return options

    def _compute_shared_uncond(self, transformer_ddp, xt, samples, j_idx,
                               neg_embeds, neg_pooled_embeds):
        """CFG uncond computed once per timestep and shared by the old / current / ref forwards
        (Wan's DiT has no dropout or BatchNorm, so the three would be identical).

        Only the base-model NP (train.np_base_velocity, the default) is shared: it is the same for
        every adapter. With the NP from the adapter (np_base_velocity=False) every forward computes
        its own NP after its conditional branch (src/models/guidance.py). A shared no-grad forward of
        the trainable adapter ran ahead of the gradient forward, and under autocast it cached
        grad-less bf16 casts of the LoRA weights that the gradient forward then reused; it also
        fed the trainable adapter's NP to the old forward (fix B3, so train.fix_shared_uncond_old
        no longer changes anything; src/loss/objectives.py still reads it).
        """
        np_base = bool(getattr(self.config.train, "np_base_velocity", True))
        max_g = max(self.config.sample.guidance_scale, self.config.train.guidance_scale,
                    self.config.train.ref_guidance_scale)
        if not np_base or max_g <= 1.0:
            return None
        with torch.no_grad(), maybe_disable_lora(transformer_ddp.module, True):
            return self._branch(transformer_ddp, {"xt": xt, "timesteps": samples["timesteps"][:, j_idx]},
                                {"embeds": neg_embeds}, 1.0)

    # ---- velocity hooks for BaseTrainingUtils.v_pred_fn (CFG lives in src/models/guidance.py) ----
    def _branch(self, transformer_ddp, ctx, cond, embed_scale):
        return transformer_ddp(
            hidden_states=ctx["xt"], timestep=ctx["timesteps"], encoder_hidden_states=cond["embeds"],
            attention_kwargs=None, return_dict=False,
        )[0]

    def _pair(self, transformer_ddp, ctx, cond, uncond):
        """One doubled batch [cond; NP] (gradient to both branches) -> (v_c, v_np)."""
        xt, ts = ctx["xt"], ctx["timesteps"]
        pred = transformer_ddp(
            hidden_states=torch.cat([xt, xt], dim=0), timestep=torch.cat([ts, ts], dim=0),
            encoder_hidden_states=torch.cat([cond["embeds"], uncond["embeds"]], dim=0),
            attention_kwargs=None, return_dict=False,
        )[0]
        v_c, v_np = pred.chunk(2)
        return v_c, v_np

    def rollout(self, pipeline, **kwargs):
        kwargs.setdefault("num_frames", self.config.frames)
        # Same switch as the training loss and the ref anchor: the CFG uncond
        # must come from the same weights in the rollout as in the loss.
        kwargs.setdefault("np_base_velocity",
                          bool(getattr(self.config.sample, "np_base_velocity", True)))
        # Pass PeftModel for base-model CFG negative. During training
        # self._peft_model is set by set_adapter(); during standalone eval
        # pipeline.transformer IS the PeftModel.
        if "peft_model" not in kwargs:
            peft_model = self._peft_model
            if peft_model is None and hasattr(pipeline.transformer, "disable_adapter"):
                peft_model = pipeline.transformer
            kwargs["peft_model"] = peft_model
        images, latents, _ = patched_pipeline_with_logprob(pipeline, **kwargs)
        return Rollout(images, latents)

    def decode_latents(self, pipeline, latents):
        """Decode raw latents into video using WAN2.1 VAE."""
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
        decoded = [pipeline.vae.decode(latents[i : i + 1], return_dict=False)[0] for i in range(latents.shape[0])]
        image = torch.cat(decoded, dim=0)
        image = pipeline.video_processor.postprocess_video(image, output_type="pt")
        return image

    @property
    def video_fps(self):
        return getattr(self.config, "video_fps", 16)

    def save_sample(self, video_tensor, output_path):
        """Save a single video sample to disk."""
        if torch.is_tensor(video_tensor):
            video = video_tensor.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            video = np.array(video_tensor)

        video = video.transpose(0, 2, 3, 1)
        frames = [(np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8) for frame in video]
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
                frames = [(np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8) for frame in video]
                imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames, fps=self.video_fps, codec="libx264", format="FFMPEG")

            self.wandb.log(
                {
                    "videos": [
                        self.wandb.Video(
                            os.path.join(tmpdir, f"{idx}.mp4"),
                            caption=f"{prompts[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            format="mp4",
                            fps=self.video_fps,
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
            if valid.size == 0:
                return -10.0
            return float(np.mean(valid))

        if torch.is_tensor(images):
            videos_to_log = [images[i] for i in range(images.shape[0])]
        else:
            videos_to_log = images

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(5, len(videos_to_log))
            for idx in range(num_samples_to_log):
                video = videos_to_log[idx]
                if torch.is_tensor(video):
                    video = video.detach().to(dtype=torch.float32).cpu().numpy()
                elif not isinstance(video, np.ndarray):
                    video = np.array(video)

                video = video.transpose(0, 2, 3, 1)
                frames = [(np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8) for frame in video]
                imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames, fps=self.video_fps, codec="libx264", format="FFMPEG")

            sampled_prompts_log = [prompts[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            self.wandb.log(
                {
                    "eval_videos": [
                        self.wandb.Video(
                            os.path.join(tmpdir, f"{idx}.mp4"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                            format="mp4",
                            fps=self.video_fps,
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": _safe_reward_mean(value) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    def eval_fn(
        self,
        pipeline,
        test_dataloader,
        text_encoders,
        tokenizers,
        global_step,
        reward_fn,
        executor,
        mixed_precision_dtype,
        ema,
        transformer_trainable_parameters,
        tqdm_fn,
    ):
        if self.config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        pipeline.transformer.eval()

        neg_prompt_embed, _ = self.get_negative_prompt_embeddings(text_encoders, tokenizers)
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.sample.test_batch_size, 1, 1)
        all_rewards = defaultdict(list)

        test_sampler = (
            DistributedSampler(test_dataloader.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
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
            eval_loader,
            desc="Eval: ",
            disable=not is_main_process(self.rank),
            position=0,
        ):
            prompts, prompt_metadata = test_batch
            prompt_embeds = self.compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=512
            )
            current_batch_size = len(prompt_embeds)
            if current_batch_size < len(sample_neg_prompt_embeds):
                current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            else:
                current_sample_neg_prompt_embeds = sample_neg_prompt_embeds

            with torch_autocast(
                "cuda",
                enabled=(mixed_precision_dtype is not None),
                dtype=mixed_precision_dtype,
            ):
                with torch.no_grad():
                    images = self.rollout(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=current_sample_neg_prompt_embeds,
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
            final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}
            self._log_eval_outputs(final_rewards, images, prompts, global_step)

        if self.config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)

        if self.world_size > 1:
            dist.barrier()
