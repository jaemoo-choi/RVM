"""src/engine/run.py's Run for a two-expert model (Wan2.2-A14B: high- / low-noise experts), used by
src/train_wan2_2.py. Each expert has its own adapters, optimizer, EMA and checkpoint folder
(checkpoints/{high,low}/); everything else is Run's. [4] (train_epoch) differs: per chunk of micro-batches
the high-noise expert takes one optimizer step on its band of the timesteps, then the low-noise one on
its band. Optional per-band rewards: ``config.reward_fn_{high,low}_noise`` re-aggregate the scored
components into two advantage tensors; the loss picks one per timestep with
``config.train.high_noise_threshold``.

Placement: the experts and the reward models never share the GPU.

    [1] sample   experts on the GPU (paged one at a time by the model utils), reward models on CPU
    [2] reward   experts on CPU, reward models on the GPU
    [4] train    the active expert (and its Adam state) on the GPU, the other on CPU
                 (config.a14b_both_experts_resident: both stay on the GPU)
"""
from __future__ import annotations

import gc
import os
import sys
import time

import torch

from src.ema import EMAModuleWrapper
from src.reward.rewards import finish_epoch_rewards, score_samples
from src.utils.distributed import broadcast_lora

from src.engine import checkpoint
from src.engine import wandb_run
from src.rollout import sample
from src.rollout.rollout import collate
from src.engine.run import Run, TrainPlan, Update, return_decay
from src.utils.batching import permute, time_view

EXPERTS = ("high", "low")     # optimization order within each chunk
# Timestep-axis keys (dim=1): sliced to each expert's band and permuted within it.
TIME_AXIS_KEYS = ("timesteps", "timestep_idx", "next_timesteps", "latents", "advantages")


def optimizer_state_to(optimizer, target):
    """An optimizer's state tensors to ``target`` (with its expert's weights: step() needs both on one device)."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(target)


class ExpertsRun(Run):
    """Run with ddps / params / old_params / optimizers / emas as dicts over EXPERTS."""

    experts = EXPERTS

    def __init__(self, config, **kwargs):
        self.both_resident = bool(getattr(config, "a14b_both_experts_resident", False))
        # page one expert at a time during sampling (one high -> low swap per batch)
        self.sequential_sampling = bool(getattr(config, "a14b_sequential_expert_sampling", True)) \
            and not self.both_resident
        self.resident = None          # the expert on the GPU while training
        self.train_videos = None      # (media, prompts, rewards) of the last sampled batch, sent with the epoch log
        self.ckpt_subdir = {e: os.path.join("checkpoints", e) for e in EXPERTS}
        super().__init__(config, **kwargs)

    # ---- setup ----
    def _marker_dir(self):
        return os.path.join(self.config.save_dir, self.ckpt_subdir["high"])

    def _ckpt_root(self, e):
        return os.path.join(self.config.save_dir, self.ckpt_subdir[e])

    def _set_adapters(self):
        bundles = self.model_utils.set_adapter_both(self.pipeline, self.local_rank)
        self.ddps, self.params, self.old_params, self.lora_configs = (
            {e: bundles[e][i] for e in EXPERTS} for i in range(4))

    def _enable_gradient_checkpointing(self):
        for e in EXPERTS:
            getattr(self.pipeline, self.model_utils.expert_attr(e)).enable_gradient_checkpointing()
        if self.is_main:
            self.logger.info("Gradient checkpointing enabled on pipeline.transformer (high) and "
                             "pipeline.transformer_2 (low)")

    def _build_optimizers(self):
        self.optimizers = {e: self._adamw(self.params[e]) for e in EXPERTS}

    def _log_reward_bands(self):
        config = self.config
        hi, lo = getattr(config, "reward_fn_high_noise", None), getattr(config, "reward_fn_low_noise", None)
        if hi is not None and lo is not None:
            fmt = lambda d: "{" + ", ".join(f"{k}:{float(v):g}" for k, v in sorted(d.items())) + "}"
            self.logger.info(
                "Noise-region reward ON (t>=%s -> high band). high=%s low=%s (%s)",
                getattr(config.train, "high_noise_threshold", 0.9), fmt(hi), fmt(lo),
                "IDENTICAL — set REWARD_FN_HIGH_NOISE / REWARD_FN_LOW_NOISE to differentiate"
                if fmt(hi) == fmt(lo) else "DIFFERENTIATED")
        else:
            self.logger.info("Noise-region reward OFF — both experts share config.reward_fn. "
                             "Set config.reward_fn_{high,low}_noise to differentiate them.")

    def _log_sizes(self):
        c, log = self.config, self.logger.info
        log("***** Running training (alternating dual-expert) *****")
        log(f"Training Method: {c.train.method}")
        log(f"Advantage: {c.train.adv}")
        log(f"  Num Epochs = {c.num_epochs}")
        log(f"  Sample batch size per device = {c.sample.train_batch_size}")
        log(f"  Train batch size per device = {c.train.batch_size}")
        log(f"  Gradient Accumulation steps = {c.train.gradient_accumulation_steps}")
        log("")
        log(f"  Total number of samples per epoch = {c.sample.train_batch_size * self.world_size * c.sample.num_batches_per_epoch}")
        log(f"  Total train batch size (w. parallel, distributed & accumulation) = "
            f"{c.train.batch_size * self.world_size * c.train.gradient_accumulation_steps}")
        log(f"  Number of inner epochs = {c.train.num_inner_epochs}")
        log(f"  alternating experts (order per chunk: {' -> '.join(EXPERTS)})")
        log(f"  both_experts_resident = {self.both_resident}")

    def _vram(self, tag):
        if self.rank == 0:
            try:
                free, total = torch.cuda.mem_get_info(self.device)
                self.logger.info(f"[vram {tag}] used={(total - free) / (1024 ** 3):.1f} GiB "
                                 f"pytorch_alloc={torch.cuda.memory_allocated(self.device) / (1024 ** 3):.1f} GiB")
            except Exception:
                pass

    def _load_reward_models(self):
        # the experts off the GPU so that the reward models load there; then the reward models to CPU
        self._vram("before_offload_for_reward_init")
        self.model_utils.foundation_to_cpu(self.pipeline)
        gc.collect()
        torch.cuda.empty_cache()
        self._vram("after_offload_for_reward_init")
        self.reward_fn = self._get_reward_fn()
        self._vram("after_reward_init_on_gpu")
        self._reward_to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        self._vram("after_reward_to_cpu")

    def _resume(self):
        config = self.config
        self.logger.info(f"Resuming from {config.resume_from} (per-expert subdirs)")
        for e in EXPERTS:
            checkpoint.load_adapters(self.ddps[e].module, self._ckpt_root(e), config, self.logger, prefix=f"[{e}] ")
            checkpoint.load_optimizer(self.optimizers[e], self._ckpt_root(e), self.logger, prefix=f"[{e}] ",
                             state_device=self.device)
        checkpoint.load_scaler(self.scaler, self._marker_dir(), self.enable_amp, self.logger)
        self.global_step, self.first_epoch = checkpoint.load_training_state(self._marker_dir(), self.device, self.logger)
        checkpoint.load_stat_tracker_history(self.prompt_history, self._marker_dir(), self.logger)

    def _build_emas(self):
        self.emas = {e: None for e in EXPERTS}
        if self.config.train.ema:
            for e in EXPERTS:
                self.emas[e] = EMAModuleWrapper(self.params[e], decay=0.9, update_step_interval=1, device=self.device)
                if self.config.resume_from:
                    checkpoint.load_ema(self.ddps[e].module, self.emas[e], self._ckpt_root(e), self.lora_configs[e],
                               self.logger, prefix=f"[{e}] ")

    def _zero_grad(self):
        for e in EXPERTS:
            self.optimizers[e].zero_grad(set_to_none=True)

    def _init_old(self):
        if not self.config.resume_from:
            for e in EXPERTS:
                for src, tgt in zip(self.params[e], self.old_params[e], strict=True):
                    tgt.data.copy_(src.detach().data)
                    assert src is not tgt
            self.logger.info("Initialized old policy from current policy (fresh training), both experts")
        else:
            self.logger.info("Using loaded old policy from checkpoint (not copying from current)")

    def _broadcast(self):
        for e in EXPERTS:
            broadcast_lora(self.ddps[e].module, self.world_size, self.device)
        self.logger.info("LoRA adapters of both experts broadcast from rank 0 (DDP gradient-sync fix)")

    def save(self, epoch_tag, permanent=False):
        """Both experts' rolling checkpoints <save_dir>/checkpoints/{high,low}/, or with ``permanent`` the
        snapshots <save_dir>/permanent/epoch_<epoch_tag>/checkpoints/{high,low}/."""
        for e in EXPERTS:
            ckpt_subdir = (os.path.join("permanent", f"epoch_{epoch_tag}", self.ckpt_subdir[e]) if permanent
                           else self.ckpt_subdir[e])
            checkpoint.save(self.logger, self.wandb, self.config.save_dir, self.ddps[e], self.global_step, self.rank,
                            self.emas[e], self.params[e], self.config, self.optimizers[e], self.scaler, epoch_tag,
                            self.prompt_history, ckpt_subdir=ckpt_subdir)
            self.model_utils.save_extra(self.config.save_dir, ckpt_subdir)

    # ---- the epoch ----
    def start_epoch(self, epoch):
        super().start_epoch(epoch)
        self.train_videos = None

    def sample(self, epoch):
        """[1] Checkpoint at the epoch's start, then the rollouts with both experts' old adapters."""
        config = self.config
        self._reward_to("cpu")
        for e in EXPERTS:                          # Adam states off the GPU while sampling
            optimizer_state_to(self.optimizers[e], "cpu")
        if self.is_main and not config.debug:
            if epoch % config.save_freq == 0:
                self.save(epoch)
            if epoch % max(1, config.num_epochs // 10) == 0:
                self.save(epoch, permanent=True)
        if self.is_main and epoch == self.first_epoch:
            self.logger.info("Phase 1 expert residency: " + ("sequential (one expert on GPU at a time)"
                                                             if self.sequential_sampling else "both experts on GPU"))
        start = time.time()
        torch.cuda.reset_peak_memory_stats(self.device)
        out = sample.sample_epoch(
            self.model_utils, self.pipeline, [self.ddps[e].module for e in EXPERTS], self.train_iter, self.train_sampler, epoch,
            self.text_encoders, self.tokenizers, self.sample_neg, enable_amp=self.enable_amp,
            amp_dtype=self.amp_dtype, progress=lambda it, desc: self.progress(it, desc))
        self.sampling_time = time.time() - start
        self.peak_sampling_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        if self.is_main:
            self.logger.info(f"Epoch {epoch}: Sampling took {self.sampling_time:.2f} seconds "
                             f"(peak VRAM: {self.peak_sampling_gb:.2f} GiB)")
        return out

    def score(self, epoch, rollouts, last_images, last_prompts):
        """[2] Experts off, reward models on; score, collate. The sampled videos are sent with the epoch
        log."""
        config, device = self.config, self.device
        self.model_utils.foundation_to_cpu(self.pipeline)
        self.resident = None
        gc.collect()
        torch.cuda.empty_cache()
        self._reward_to("gpu")
        torch.cuda.reset_peak_memory_stats(device)
        start = time.time()
        score_samples(self.reward_fn, rollouts, device=device, world_size=self.world_size, rank=self.rank,
                      logger=self.logger, progress=lambda it, desc: self.progress(it, desc))
        if self.is_main:
            self.logger.info(f"Epoch {epoch}: Reward took {time.time() - start:.2f} seconds "
                             f"(peak VRAM: {torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f} GiB)")
        if epoch % config.sample_log_freq == 0 and self.is_main:
            self.train_videos = (last_images, last_prompts, rollouts[-1]["rewards"]["avg"].detach().cpu())
        samples = finish_epoch_rewards(collate(rollouts, skip=("prompt_metadata",)), config,
                                       getattr(self.pipeline, "tokenizer", None))
        rollouts.clear()
        return samples

    def reward_log(self, epoch, samples):
        # every step of the sampling schedule is trained (each expert its band)
        self.num_train_timesteps = samples["timesteps"].shape[1]
        return super().reward_log(epoch, samples)

    def log_epoch(self, data):
        """The epoch's one W&B log, with the sampled videos when this epoch logs them."""
        if self.is_main:
            if self.train_videos is not None:
                self.model_utils.log_train_samples(*self.train_videos, self.global_step, wandb_run.wandb_step(self.global_step),
                                                   extra_log_dict=data)
            else:
                self.wandb.log(data, step=wandb_run.wandb_step(self.global_step))
        self.train_videos = None

    def place_for_training(self, samples):
        """[4] Reward models, VAE and text encoder off the GPU (samples too with cpu_offload_samples);
        both experts on it with a14b_both_experts_resident."""
        self._reward_to("cpu")
        self.pipeline.vae.to("cpu")
        if self.pipeline.text_encoder is not None:
            self.pipeline.text_encoder.to("cpu")
        self._samples_to_cpu_if_offloaded(samples)
        gc.collect()
        torch.cuda.empty_cache()
        if self.both_resident:
            self.model_utils.experts_to(self.pipeline, {e: self.device for e in EXPERTS})
            for e in EXPERTS:
                optimizer_state_to(self.optimizers[e], self.device)
            self.resident = "both"
        return samples

    def bands(self, samples, epoch, perm_device):
        """Each expert's band of the sampler steps: (bool masks over the steps, their sizes). Row 0 of
        timesteps (unpermuted) is the shared schedule."""
        high_mask, _ = self.model_utils.expert_step_mask(samples["timesteps"][0])
        high_mask = high_mask.to(perm_device)
        masks = {"high": high_mask, "low": ~high_mask}
        sizes = {e: int(masks[e].sum().item()) for e in EXPERTS}
        num_timesteps = samples["timesteps"].shape[1]
        for e in EXPERTS:
            if sizes[e] == 0:
                raise RuntimeError(
                    f"Wan2.2-T2V-A14B trajectory has zero {e}-noise steps (boundary_ratio="
                    f"{self.model_utils.boundary_ratio}, num_steps={num_timesteps}). The alternating trainer "
                    "needs both bands non-empty; check the scheduler shift / sigma curve.")
        if epoch == self.first_epoch and self.is_main:
            self.logger.info(f"Band split: high={sizes['high']} / low={sizes['low']} of {num_timesteps} steps "
                             f"(boundary_ratio={self.model_utils.boundary_ratio}); velocity target = x_0 "
                             "(latents_clean) for BOTH experts")
        return masks, sizes

    # ---- [4]: the base loop (Run.train_epoch) with these hooks ----
    def train_samples(self, samples):
        """in: the epoch's samples. out: the samples placed for training (place_for_training)."""
        return self.place_for_training(samples)

    def train_plan(self, epoch, samples):
        """in: the epoch's samples. out: TrainPlan with each expert's band of the sampler steps; one chunk of
        grad_accum micro-batches feeds one optimizer step per expert."""
        config = self.config
        N = samples["timesteps"].shape[0]
        offload = getattr(config, "cpu_offload_samples", False)
        perm_device = "cpu" if offload else self.device
        band_masks, n_bands = self.bands(samples, epoch, perm_device)
        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size
        grad_accum = config.train.gradient_accumulation_steps
        assert num_batches % grad_accum == 0, (
            f"num_batches ({num_batches}) must be divisible by gradient_accumulation_steps ({grad_accum}) "
            "so each chunk feeds exactly one high step and one low step.")
        num_chunks = num_batches // grad_accum
        if epoch == self.first_epoch and self.is_main:
            self.logger.info(f"Per epoch: {num_chunks} chunk(s) x {len(EXPERTS)} experts = "
                             f"{num_chunks * len(EXPERTS)} optimizer steps ({num_chunks} per expert), "
                             f"{grad_accum} micro-batches per chunk")
        return TrainPlan(N=N, num_batches=num_batches, batch_size=N // num_batches, grad_accum=grad_accum,
                         perm_device=perm_device, load_device=self.device if offload else None,
                         bands=band_masks, n_bands=n_bands)

    def shuffle(self, samples, plan):
        """in: samples, plan. out: {expert: samples} — one sample order for both experts; each expert's
        steps (its band) reshuffled within the band."""
        shared = permute(samples, torch.randperm(plan.N, device=plan.perm_device))
        views = {}
        for e in EXPERTS:
            perms_time = torch.stack([torch.randperm(plan.n_bands[e], device=plan.perm_device) for _ in range(plan.N)])
            views[e] = time_view(shared, perms_time, TIME_AXIS_KEYS, plan.bands[e])
        return views

    def updates(self, plan):
        """in: plan. out: [Update] — per chunk of grad_accum micro-batches, the high then the low expert,
        each over its band of the timesteps."""
        ga = plan.grad_accum
        return [Update(index=chunk * len(EXPERTS) + gi, ddp=self.ddps[e], optimizer=self.optimizers[e],
                       batches=range(chunk * ga, (chunk + 1) * ga), num_t=plan.n_bands[e],
                       accum_steps=ga * plan.n_bands[e], view=e, name=e, chunk=chunk)
                for chunk in range(plan.num_batches // ga) for gi, e in enumerate(EXPERTS)]

    def begin_update(self, upd):
        """in: the Update. out: None — its expert (and its Adam state) paged onto the GPU, made active."""
        self.activate(upd.name)

    def after_update(self, epoch, inner_epoch, plan, upd, norm, terms):
        """in: as Run.after_update. out: None — the expert's EMA step, the step's W&B log under
        "<expert>/", global_step + 1."""
        e = upd.name
        if self.emas[e] is not None:
            self.emas[e].step(self.params[e], self.global_step)
        self.probe(e, "ema:done")
        step_log = {f"{e}/{k}": v for k, v in self.step_logs(terms, upd).items()}
        self.log({f"{e}/grad_norm": norm.item(), "gradient_update_times": plan.steps_done, "epoch": epoch,
                  **step_log})
        self.global_step += 1
        self.model_utils.set_active_expert(None)

    def probe(self, expert, tag):
        """Per-rank stderr breadcrumbs, to find a rank stalling before a collective."""
        print(f"[straggler-probe] rank={self.rank} {expert} {tag} t={time.monotonic():.1f}", file=sys.stderr,
              flush=True)

    def activate(self, expert):
        """Before an expert's optimizer step: page it (and its Adam state) in, make it the active one."""
        self.probe(expert, "make_resident:begin")
        if not self.both_resident and self.resident != expert:
            for e in EXPERTS:            # free the outgoing expert before loading the incoming one
                if e != expert:
                    optimizer_state_to(self.optimizers[e], "cpu")
            self.model_utils.experts_to(self.pipeline, {e: (self.device if e == expert else "cpu") for e in EXPERTS})
            gc.collect()
            torch.cuda.empty_cache()
            optimizer_state_to(self.optimizers[expert], self.device)
            self.resident = expert
        self.probe(expert, "make_resident:done")
        self.model_utils.set_active_expert(expert)
        self.ddps[expert].train()

    def update_old(self):
        """[5] old = d·old + (1 − d)·current, both experts."""
        with torch.no_grad():
            decay = return_decay(self.global_step, self.config.decay_type)
            for e in EXPERTS:
                for src, tgt in zip(self.params[e], self.old_params[e], strict=True):
                    tgt.data.copy_(tgt.detach().data * decay + src.detach().clone().data * (1.0 - decay))
