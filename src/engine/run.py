"""The velocity pipeline's Run: every operation of the epoch skeleton (src/train.py:train). A pipeline
that differs subclasses Run and overrides the steps that differ (src/engine/run_experts.py: Wan2.2's two
experts); the skeleton stays the same.

    Run(config)           setup: distributed, W&B, models + adapters, optimizer, data, negative-prompt
                          embeddings, reward models, resume, EMA, old = current (fresh run), LoRA broadcast
    run.sample(epoch)     [1] the rolling / permanent checkpoint at the epoch's start, then the rollouts
    run.score(...)        [2] the reward models on the GPU, score, collate, the sample log
    run.signal(...)       [3] rewards -> per-sample advantages, the epoch's log
    run.train_epoch(...)  [4] for each micro-batch and timestep: loss (src/loss/step.py) -> backward;
                          every accum_steps backwards: optimizer step (run.backward / run.optimizer_step)
    run.update_old()      [5] old = d·old + (1 − d)·current
    run.log / log_epoch   W&B (rank 0)
    run.finish()          the final save

Device placement (VAE, reward models, transformer, samples on CPU / GPU) lives here too: the order of the
moves is the original trainer's, unchanged.
"""
from __future__ import annotations

import gc
import os
import time
from collections import defaultdict
from concurrent import futures
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
import tqdm as _tqdm
from torch.cuda.amp import GradScaler, autocast as torch_autocast

from src.dataloader import get_dataloaders
from src.ema import EMAModuleWrapper
from src.reward.rewards import finish_epoch_rewards, reward_epoch_log, score_samples
from src.models import build_model_utils
from src.reward import get_reward_fn
from src.utils.batching import micro_batch, permute, time_view
from src.utils.distributed import broadcast_lora, cleanup_distributed, is_main_process, setup_distributed, sync_grads
from src.utils.rng import set_seed

from src.advantage import advantages
from src.engine import checkpoint
from src.engine import wandb_run
from src.rollout import sample
from src.rollout.rollout import collate
from src.loss.objectives import objective_for
from src.loss.step import compute_loss

tqdm = partial(_tqdm.tqdm, dynamic_ncols=True)

# Sample entries with a timestep axis (dim 1), shuffled together in [4] (+ "latents" with train.xt_from_latents).
TIME_KEYS = ("timesteps", "timestep_idx", "next_timesteps", "advantages")


@dataclass
class TrainPlan:
    """One epoch's [4] layout (Run.train_plan; ExpertsRun adds its bands)."""
    N: int                      # samples on this rank
    num_batches: int            # micro-batches per inner epoch
    batch_size: int             # samples per micro-batch
    grad_accum: int             # micro-batches per optimizer step
    perm_device: object         # where the shuffle is drawn (cpu when samples are offloaded)
    load_device: object         # where a micro-batch is moved (None: samples already on the GPU)
    steps_done: int = 0         # optimizer steps taken this epoch
    bands: object = None        # Wan2.2: {expert: bool mask over the steps}
    n_bands: object = None      # Wan2.2: {expert: steps in its band}


@dataclass
class Update:
    """One optimizer step of [4]: which module, which micro-batches, how many timesteps."""
    index: int                  # position in the epoch's step order
    ddp: object                 # the DDP-wrapped module that takes the backward
    optimizer: object           # its optimizer
    batches: range              # micro-batch indices of this step
    num_t: int                  # timesteps per micro-batch (all T, or an expert's band)
    accum_steps: int            # backwards per step (len(batches) x num_t): the loss divisor
    view: object = None         # which shuffled view the micro-batches come from (Wan2.2: the expert)
    name: object = None         # None, or the expert ("high" / "low")
    chunk: int = 0              # Wan2.2: the chunk of micro-batches this step belongs to


class Run:
    """One training job's state. Attributes the epoch reads: config, rank, world_size, device, is_main,
    model_utils, objective, ddp (the DDP-wrapped PEFT transformer), optimizer, scaler, fp16,
    ema, global_step, first_epoch, prompt_history, num_train_timesteps, train_neg (the negative-prompt
    embeddings for the loss)."""

    def __init__(self, config, *, logger, wandb, entity):
        self.config, self.logger, self.wandb = config, logger, wandb
        self._init_distributed()
        self._init_wandb(entity)
        set_seed(config.seed, self.rank)  # per-rank seed, after W&B init
        self.model_utils = build_model_utils(
            getattr(config, "base_model", "sd3"), config=config, logger=logger, wandb_module=wandb,
            device=self.device, rank=self.rank, world_size=self.world_size)
        # The RL objective (src/loss/objectives.py): its loss terms and logs.
        self.objective = objective_for(config)

        # models
        self.pipeline, self.text_encoders, self.tokenizers = self.model_utils.get_pretrained_model(
            mixed_precision_dtype=self.amp_dtype, enable_amp=self.enable_amp)
        self._set_adapters()
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if getattr(config, "gradient_checkpointing", False):
            self._enable_gradient_checkpointing()
        self._build_optimizers()

        # data
        train_dataloader, _ = get_dataloaders(config=config, world_size=self.world_size, rank=self.rank,
                                              num_workers=0)
        self.train_sampler = train_dataloader.batch_sampler

        # negative-prompt embeddings
        neg, neg_pooled = self.model_utils.get_negative_prompt_embeddings(self.text_encoders, self.tokenizers)
        self.neg = (neg, neg_pooled)
        rep = lambda e, n, *dims: None if e is None else e.repeat(n, *dims)
        self.sample_neg = (rep(neg, config.sample.train_batch_size, 1, 1),
                           rep(neg_pooled, config.sample.train_batch_size, 1))
        self.train_neg = (rep(neg, config.train.batch_size, 1, 1), rep(neg_pooled, config.train.batch_size, 1))

        # The prompts trained so far (the trained_prompt_num log; saved in the checkpoint).
        self.prompt_history = advantages.PromptHistory()
        self._log_reward_bands()

        # rank-0 sample logging runs in the background
        self.sample_log_executor = futures.ThreadPoolExecutor(max_workers=1)
        self.sample_log_future = None
        self._log_sizes()

        # reward models
        self._load_reward_models()

        # resume, EMA (saved as lora_ema_policy; never used in training)
        self.first_epoch, self.global_step = 0, 0
        if checkpoint.resume_dir(config, config.save_dir, self._marker_dir()):
            self._resume()
        self._build_emas()

        self.num_train_timesteps = max(1, int(config.sample.num_steps * config.train.timestep_fraction))

        # Every rank finishes loading before rank 0 may overwrite the checkpoint.
        self.barrier()
        # Every wandb.log uses wandb_run.wandb_step(global_step): W&B drops logs at steps it already has.
        self.global_step = wandb_run.sync_global_step(self.global_step, wandb, self.rank, self.world_size, self.device,
                                              logger, resumed=bool(config.resume_from))
        logger.info("***** Running training *****")
        self.train_iter = iter(train_dataloader)
        self._zero_grad()

        # fresh run: old = current; then all ranks start from rank 0's adapters
        self._init_old()
        self._broadcast()

        self.pending_log = {}   # the previous epoch's timing log, sent with the next epoch's log
        # [3]'s advantage rule (src/advantage/advantages.py), with the objective's direction
        self.signal_spec = advantages.SignalSpec.from_adv(config)


    # ---- setup steps: distributed, W&B, adapters, resume, old (run_experts.py overrides some) ----
    def _init_distributed(self):
        config = self.config
        rank, world_size, local_rank = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
        setup_distributed(rank, local_rank, world_size)
        self.rank, self.world_size, self.local_rank = rank, world_size, local_rank
        self.is_main = is_main_process(rank)
        self.device = torch.device(f"cuda:{local_rank}")
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(config.mixed_precision)
        self.enable_amp = self.amp_dtype is not None
        self.fp16 = self.amp_dtype == torch.float16
        self.scaler = GradScaler(enabled=self.fp16)

    def _marker_dir(self):
        """Where the resume marker / W&B run id of a previous job live."""
        return os.path.join(self.config.save_dir, "checkpoints")

    def _init_wandb(self, entity):
        log_dir = self.config.save_dir
        wandb_run_id = wandb_run.read_wandb_run_id(log_dir, self._marker_dir())
        if self.is_main:
            os.makedirs(log_dir, exist_ok=True)
            wandb_run.start_wandb(self.wandb, self.config, log_dir, wandb_run_id, self.logger, entity)
            wandb_run.install_wandb_shutdown(self.wandb)
        self.logger.info(f"\n{self.config}")

    def _set_adapters(self):
        self.ddp, self.params, self.old_params, self.lora_config = self.model_utils.set_adapter(
            self.pipeline, self.local_rank)

    def _enable_gradient_checkpointing(self):
        self.pipeline.transformer.enable_gradient_checkpointing()
        if self.is_main:
            self.logger.info("Gradient checkpointing enabled on transformer")

    def _resume(self):
        config = self.config
        self.logger.info(f"Resuming from {config.resume_from} (new format with separate policies)")
        self.global_step, self.first_epoch = checkpoint.resume(
            self.ddp.module, self.optimizer, self.scaler, config, config.resume_from, self.logger,
            device=self.device, enable_amp=self.enable_amp)
        checkpoint.load_stat_tracker_history(self.prompt_history, self._marker_dir(), self.logger)

    def _init_old(self):
        if not self.config.resume_from:
            for src, tgt in zip(self.params, self.old_params, strict=True):
                tgt.data.copy_(src.detach().data)
                assert src is not tgt
            self.logger.info("Initialized old policy from current policy (fresh training)")
        else:
            self.logger.info("Using loaded old policy from checkpoint (not copying from current)")

    def _broadcast(self):
        broadcast_lora(self.ddp.module, self.world_size, self.device)
        self.logger.info("LoRA adapters broadcast from rank 0 (DDP gradient-sync fix)")

    def _log_sizes(self):
        c, log = self.config, self.logger.info
        samples_per_epoch = c.sample.train_batch_size * self.world_size * c.sample.num_batches_per_epoch
        total_train_batch_size = c.train.batch_size * self.world_size * c.train.gradient_accumulation_steps
        log("***** Running training *****")
        log(f"Training Method: {c.train.method}")
        log(f"Advantage: {c.train.adv}")
        log(f"  Num Epochs = {c.num_epochs}")
        log(f"  Sample batch size per device = {c.sample.train_batch_size}")
        log(f"  Train batch size per device = {c.train.batch_size}")
        log(f"  Gradient Accumulation steps = {c.train.gradient_accumulation_steps}")
        log("")
        log(f"  Total number of samples per epoch = {samples_per_epoch}")
        log(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        log(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
        log(f"  Number of inner epochs = {c.train.num_inner_epochs}")


    # ---- optimizers and EMA ----
    def _adamw(self, params):
        tr = self.config.train
        return torch.optim.AdamW(params, lr=tr.learning_rate, betas=(tr.adam_beta1, tr.adam_beta2),
                                 weight_decay=tr.adam_weight_decay, eps=tr.adam_epsilon)

    def _build_optimizers(self):
        self.optimizer = self._adamw(self.params)

    def _zero_grad(self):
        self.optimizer.zero_grad()

    def _build_emas(self):
        self.ema = None
        if self.config.train.ema:
            self.ema = EMAModuleWrapper(self.params, decay=0.9, update_step_interval=1, device=self.device)
            if self.config.resume_from:
                checkpoint.load_ema(self.ddp.module, self.ema, self.config.resume_from, self.lora_config, self.logger)


    # ---- reward models ----
    def _get_reward_fn(self):
        config = self.config
        return get_reward_fn(
            self.device, config.reward_fn,
            videoalign_dimension_weights=getattr(config, "videoalign_dimension_weights", None) or None,
            video_fps=getattr(config, "video_fps", None))

    def _load_reward_models(self):
        self.reward_fn = self._get_reward_fn()

    def _log_reward_bands(self):
        config = self.config
        if getattr(config, "reward_fn_high_noise", None) is not None and getattr(config, "reward_fn_low_noise", None) is not None:
            self.logger.info("Noise-region reward ON (t>=%s high): high=%s low=%s",
                             getattr(config.train, "high_noise_threshold", 0.9),
                             config.reward_fn_high_noise, config.reward_fn_low_noise)

    def _reward_to(self, where):
        if hasattr(self.reward_fn, f"to_{where}"):
            getattr(self.reward_fn, f"to_{where}")()


    # ---- device placement ----
    def _samples_to_cpu_if_offloaded(self, samples):
        if getattr(self.config, "cpu_offload_samples", False):
            for k, v in samples.items():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    samples[k] = v.cpu()
        return samples


    # ---- small helpers ----
    def autocast(self):
        return torch_autocast(enabled=self.enable_amp, dtype=self.amp_dtype)

    def progress(self, it=None, desc=None, **kw):
        return tqdm(it, desc=desc, disable=not self.is_main, **{"position": 0, **kw})

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()


    # ---- logging ----
    def log(self, data):
        """One W&B log at the current step (rank 0)."""
        if self.is_main:
            self.wandb.log(data, step=wandb_run.wandb_step(self.global_step))

    def log_epoch(self, data):
        """The epoch's one W&B log (rank 0)."""
        self.log(data)

    def reward_log(self, epoch, samples):
        """The epoch's reward log (+ the previous epoch's pending timing); sent with the signal log.
        in: epoch, samples. out: {key: value} (reward means over ranks, the pending timing log)."""
        log = {"epoch": epoch, **self.pending_log}
        self.pending_log = {}
        log.update(reward_epoch_log(samples, self.config, rank=self.rank, world_size=self.world_size,
                                    device=self.device, num_train_timesteps=self.num_train_timesteps))
        return log

    def timing(self, epoch, training_start):
        training_time = time.time() - training_start
        epoch_time = time.time() - self.epoch_start
        peak_training_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        if self.is_main:
            self.logger.info(f"Epoch {epoch}: Training took {training_time:.2f} seconds (peak VRAM: {peak_training_gb:.2f} GiB)")
            self.logger.info(f"Epoch {epoch}: Total epoch time (without eval) {epoch_time:.2f} seconds")
            self.pending_log = {
                "timing/sampling_time": self.sampling_time,
                "timing/training_time": training_time,
                "timing/epoch_time_without_eval": epoch_time,
                "memory/peak_sampling_gib": self.peak_sampling_gb,
                "memory/peak_training_gib": peak_training_gb,
            }


    # ---- the epoch, in order: [0] cache, [1] sample, [2] score, after training, [5] old, save ----
    def start_epoch(self, epoch):
        """in: epoch. out: None — the epoch clock, the data sampler's epoch, the model utils' epoch hook
        """
        self.epoch_start = time.time()
        if hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)
        self.model_utils.on_epoch_start(epoch)

    def sample(self, epoch):
        """[1] Checkpoint at the epoch's start (rolling every save_freq epochs, permanent every
        num_epochs // 10), then the rollouts with the old adapter (src/models/base.py:sample_epoch).
        in: epoch. out: (rollouts: [Rollout] one per sampling batch, last_images, last_prompts: the last
        batch's media and prompts for the sample log)."""
        config = self.config
        self.sampling_start = time.time()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._reward_to("cpu")
        if self.is_main and not config.debug:
            if epoch % config.save_freq == 0:
                self.save(epoch)
            if epoch % max(1, config.num_epochs // 10) == 0:
                self.save(epoch, permanent=True)
        return sample.sample_epoch(
            self.model_utils, self.pipeline, [self.ddp.module], self.train_iter, self.train_sampler, epoch, self.text_encoders,
            self.tokenizers, self.sample_neg, enable_amp=self.enable_amp, amp_dtype=self.amp_dtype,
            progress=lambda it, desc: self.progress(it, desc))

    def score(self, epoch, rollouts, last_images, last_prompts):
        """[2] Score the rollouts with the reward models (src/reward/rewards.py), collate them into one
        dict for the epoch, log sample images.
        in: epoch, the outputs of sample(). out: samples (dict of per-sample tensors + "rewards")."""
        config, device = self.config, self.device
        # reward models on, score, reward models off, transformer back on the GPU
        self.model_utils.foundation_to_cpu(self.pipeline)
        torch.cuda.empty_cache()
        self._reward_to("gpu")
        score_samples(self.reward_fn, rollouts, device=device, world_size=self.world_size, rank=self.rank,
                      logger=self.logger, progress=lambda it, desc: self.progress(it, desc))
        self._reward_to("cpu")
        torch.cuda.empty_cache()
        if hasattr(self.pipeline, "transformer"):
            self.pipeline.transformer.to(device)

        samples = finish_epoch_rewards(collate(rollouts), config, getattr(self.pipeline, "tokenizer", None))
        rollouts.clear()

        self.sampling_time = time.time() - self.sampling_start
        self.peak_sampling_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if self.is_main:
            self.logger.info(f"Epoch {epoch}: Sampling took {self.sampling_time:.2f} seconds "
                             f"(peak VRAM: {self.peak_sampling_gb:.2f} GiB)")

        # sample images (rank 0, async)
        if epoch % config.sample_log_freq == 0 and self.is_main:
            if self.sample_log_future is None or self.sample_log_future.done():
                num_log = len(last_images) if hasattr(last_images, '__len__') else last_images.shape[0]
                self.sample_log_future = self.sample_log_executor.submit(
                    self.model_utils.log_train_samples, last_images, last_prompts,
                    samples["rewards"]["avg"][-num_log:].cpu(), self.global_step, wandb_run.wandb_step(self.global_step))
            else:
                self.logger.warning("Skipping train sample logging at epoch %s because previous logging task "
                                    "is still running.", epoch)

        if getattr(config, "offload_transformer_during_decode", False):
            if next(self.pipeline.transformer.parameters()).device.type == "cpu":
                self.pipeline.transformer.to(device)
                torch.cuda.empty_cache()
        # only the transformer stays on the GPU (samples too, unless cpu_offload_samples)
        self.pipeline.vae.to("cpu")
        self._samples_to_cpu_if_offloaded(samples)
        gc.collect()
        torch.cuda.empty_cache()
        return samples

    # ---- [3] signal ----
    def signal(self, epoch, samples, epoch_log):
        """[3] rewards -> per-sample advantages.
        in:  samples (the scored epoch: rewards per sample), epoch_log (the reward log of reward_log)
        out: samples with "advantages" [N, T] (and the objective's signal entries); sends the epoch's W&B log."""
        samples, signal_log = advantages.compute_signals(
            samples, config=self.config, spec=self.signal_spec, epoch=epoch, prompt_history=self.prompt_history,
            num_train_timesteps=self.num_train_timesteps, rank=self.rank, world_size=self.world_size,
            device=self.device, logger=self.logger)
        self.log_epoch({**epoch_log, **signal_log})
        return samples

    # ---- [4] train: the fixed loop; the hooks below it are what a pipeline overrides ----
    def train_epoch(self, epoch, samples):
        """[4] gradient steps on the trained adapter(s).
        in:  samples (the epoch's samples with advantages)
        out: None (the trained weights, optimizer state, global_step, EMA change; per-step W&B logs)

        For each inner epoch: shuffle -> for each optimizer step (an Update): its micro-batches x
        timesteps: compute_loss -> backward; then optimizer_step and after_update. A pipeline changes
        the hooks (train_samples, train_plan, shuffle, updates, begin_update, after_update), not this loop."""
        objective, model_utils = self.objective, self.model_utils
        samples = self.train_samples(samples)
        training_start = time.time()
        torch.cuda.reset_peak_memory_stats(self.device)
        plan = self.train_plan(epoch, samples)
        for inner_epoch in range(self.config.train.num_inner_epochs):
            views = self.shuffle(samples, plan)
            updates = self.updates(plan)
            progress = self.progress(total=sum(len(u.batches) for u in updates),
                                     desc=f"Epoch {epoch}.{inner_epoch}: training")
            for upd in updates:
                self.begin_update(upd)
                terms = defaultdict(list)                # this optimizer step's loss terms
                for bi, b in enumerate(upd.batches):
                    batch = micro_batch(views[upd.view], b, plan.batch_size, plan.load_device)
                    desc = "Timestep" if upd.name is None else f"{upd.name} timestep"
                    for j in self.progress(range(upd.num_t), desc=desc, position=1, leave=False):
                        # loss: the model forwards + the objective's formula, reading the signal of [3]
                        with self.autocast():
                            loss, step_terms = compute_loss(model_utils, upd.ddp, batch, j, batch["prompt_embeds"],
                                                            batch["pooled_prompt_embeds"], *self.train_neg)
                        self.backward(loss, upd, last=(bi == len(upd.batches) - 1 and j == upd.num_t - 1))
                        for k, v in step_terms.items():
                            terms[k].append(v)
                    progress.update(1)
                norm = self.optimizer_step(upd)
                plan.steps_done += 1
                self.after_update(epoch, inner_epoch, plan, upd, norm, terms)
            progress.close()
        self.timing(epoch, training_start)
        self.barrier()

    def train_samples(self, samples):
        """in: the epoch's samples. out: the samples [4] trains on (Wan2.2: placed for its experts)."""
        return samples

    def train_plan(self, epoch, samples):
        """in: the epoch's samples. out: TrainPlan — the epoch's [4] sizes (micro-batch size, gradient
        accumulation) and where samples live."""
        config = self.config
        N = samples["timesteps"].shape[0]
        num_batches = N // config.train.batch_size
        offload = getattr(config, "cpu_offload_samples", False)
        self.ddp.train()
        return TrainPlan(N=N, num_batches=num_batches, batch_size=N // num_batches,
                         grad_accum=config.train.gradient_accumulation_steps,
                         perm_device="cpu" if offload else self.device, load_device=self.device if offload else None)

    def shuffle(self, samples, plan):
        """in: samples, plan. out: {view: samples} — the samples in a random order, each sample's timesteps
        in a random order (one view, key None)."""
        N, num_timesteps = samples["timesteps"].shape
        perm = torch.randperm(N, device=plan.perm_device)
        perms_time = torch.stack([torch.randperm(num_timesteps, device=plan.perm_device) for _ in range(N)])
        time_keys = list(TIME_KEYS) + (["latents"] if self.config.train.xt_from_latents else [])
        return {None: time_view(permute(samples, perm), perms_time, time_keys)}

    def updates(self, plan):
        """in: plan. out: [Update] — the epoch's optimizer steps in order: grad_accum micro-batches each,
        all T trained timesteps, on the policy (self.ddp / self.optimizer)."""
        T, ga = self.num_train_timesteps, plan.grad_accum
        return [Update(index=s, ddp=self.ddp, optimizer=self.optimizer,
                       batches=range(s * ga, min((s + 1) * ga, plan.num_batches)), num_t=T, accum_steps=ga * T)
                for s in range(-(-plan.num_batches // ga))]

    def begin_update(self, upd):
        """in: the Update about to run. out: None (Wan2.2: pages its expert onto the GPU)."""

    def backward(self, loss, upd, last):
        """in: one micro-batch x timestep loss, its Update, last (the step's last backward).
        out: None — the gradient of loss / accum_steps (an optimizer step averages its backwards) added
        to upd's module; DDP syncs on the step's last backward only."""
        scaled = loss / upd.accum_steps
        if self.fp16:
            scaled = self.scaler.scale(scaled)
        if last:
            scaled.backward()
        else:
            with upd.ddp.no_sync():
                scaled.backward()

    def optimizer_step(self, upd):
        """in: the Update. out: the gradient norm before clipping (tensor). Averages upd's gradients over
        the ranks (sync_grads), clips, steps upd.optimizer, zeroes the gradients."""
        self.probe(upd.name, "backwards:done")
        sync_grads(upd.optimizer, self.world_size)
        if self.fp16:
            self.scaler.unscale_(upd.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(upd.ddp.module.parameters(), self.config.train.max_grad_norm)
        self.probe(upd.name, "clip:done")
        if self.fp16:
            self.scaler.step(upd.optimizer)
        else:
            upd.optimizer.step()
        self.probe(upd.name, "optstep:done")
        if self.fp16:
            self.scaler.update()
        upd.optimizer.zero_grad(set_to_none=True)
        return norm

    def step_logs(self, terms, upd):
        """in: {term: [value per backward]} of one optimizer step, its Update. out: {term: float} — each the
        mean over the step's backwards, then over ranks."""
        keys = sorted(terms)
        means = torch.tensor([torch.mean(torch.stack(terms[k])).item() for k in keys], device=self.device)
        self.probe(upd.name, f"metrics-item:done nkeys={len(keys)}")
        dist.all_reduce(means, op=dist.ReduceOp.AVG)
        self.probe(upd.name, "allreduce:done")
        return {k: means[i].item() for i, k in enumerate(keys)}

    def after_update(self, epoch, inner_epoch, plan, upd, norm, terms):
        """in: the step's epoch / inner epoch, plan, Update, gradient norm, loss terms. out: None — the
        step's W&B log, global_step + 1, the EMA step."""
        self.log({"gradient_norm/grad_norm": norm.item(),
                  "gradient_update_times": plan.steps_done,
                  "epoch": epoch, **self.step_logs(terms, upd)})
        self.global_step += 1
        if self.ema is not None:
            self.ema.step(self.params, self.global_step)

    def probe(self, name, tag):
        """Per-rank breadcrumbs around the step's collectives (Wan2.2's straggler probe); off here."""

    def update_old(self):
        """[5] old = d·old + (1 − d)·current, d = return_decay(global_step, decay_type).
        in/out: none (the old adapter's weights change in place)."""
        with torch.no_grad():
            decay = return_decay(self.global_step, self.config.decay_type)
            for src, tgt in zip(self.params, self.old_params, strict=True):
                tgt.data.copy_(tgt.detach().data * decay + src.detach().clone().data * (1.0 - decay))

    def save(self, epoch_tag, permanent=False):
        """The rolling checkpoint <save_dir>/checkpoints/ (what a resume reads), or with ``permanent``
        the snapshot <save_dir>/permanent/epoch_<epoch_tag>/checkpoints/ (src/engine/checkpoint.py:save)."""
        ckpt_subdir = os.path.join("permanent", f"epoch_{epoch_tag}", "checkpoints") if permanent else "checkpoints"
        checkpoint.save(self.logger, self.wandb, self.config.save_dir, self.ddp, self.global_step, self.rank, self.ema,
                        self.params, self.config, self.optimizer, self.scaler, epoch_tag, self.prompt_history,
                        ckpt_subdir=ckpt_subdir)
        self.model_utils.save_extra(self.config.save_dir, ckpt_subdir)

    def finish(self):
        """The final checkpoint (rolling + permanent), W&B close, process-group teardown. in/out: none."""
        config = self.config
        if self.is_main and not config.debug:
            self.save(config.num_epochs)
            self.save(config.num_epochs, permanent=True)
            self.logger.info(f"Saved FINAL checkpoint (epoch={config.num_epochs}) after full training run")
        if self.is_main:
            if self.sample_log_future is not None:
                try:
                    self.sample_log_future.result(timeout=30)
                except Exception as e:
                    self.logger.warning(f"Final async sample logging task did not complete cleanly: {e}")
            if self.pending_log:
                self.wandb.log(self.pending_log)
            self.wandb.finish()
        self.sample_log_executor.shutdown(wait=False)
        cleanup_distributed()


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    elif decay_type == 3:
        flat = 10
        uprate = 0.01
        uphold = 0.5
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)
