"""How rewards become the learning signal of one epoch: per-prompt advantages.

ROLE: rewards -> the per-sample learning signal, once per epoch (no model forward, no gradients).
Everything that needs a whole prompt group (its mean / std) is done here, before the training phase
shuffles a group's samples across micro-batches and GPUs.
    in   this rank's scored samples: "rewards" {component: [n], "avg": [n]}, "prompts"
    out  the samples with "advantages" [n, T] (A at each sampler step, aligned with "timesteps"; with
         per-band rewards the noise band is already chosen per step), read by src/loss/step.py
         and the epoch log (advantage statistics).

The reward area (src/reward/rewards.py) leaves every rank with its scored samples (a ``rewards`` dict
and the ``prompts``); the sample cache keeps exactly that. compute_signals turns them into what the
loss consumes, in the training phase. Every method follows the same recipe; only the settings differ:

    1. center     d_i = R_i − mean_c(R)                       (c = the prompt group of sample i)
    2. scale      A_i = (s · d_i) / σ                         σ = prompt std + 1e-4 | global std + 1e-4 | const
    3. map        DiffusionNFT only: A -> clip(A, ±M) / M / 2 + 0.5, clipped to [0, 1]

    setting                    s          σ                         map
    adv=standard               scale      prompt std (global_std)   -
    adv=exact, beta != 0       beta       alpha                     -
    adv=exact, beta == 0       scale      1                         -
    adv=diffusionnft           1          prompt std (global_std)   ±M

SignalSpec.from_adv(config) gives the row; group_signal (src/advantage/utils.py) applies steps 1–3 (numpy
float64). Around the core, as separate optional steps: per-component whitening (train.whiten_rewards) and
the noise-region band rewards change the input.

    compute_signals           the trainers' one call per epoch (src/engine/run.py:Run.signal)
    SignalSpec, PromptHistory used by the engine
Everything else (the group math, prompt keys, whitening, a rank's rows, the log statistics) is
src/advantage/utils.py, imported only here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.advantage.utils import (add_advantage_stats, add_region_advantage_stats, finish_samples, group_signal,
                                prompt_key, rank_rows, region_rank_rows, whitened_reward)


# ---- the recipe ----

@dataclass(frozen=True)
class SignalSpec:
    """One row of the table in the module docstring."""
    scale: float                      # s
    sigma: str                        # "prompt_std" | "global_std" | "const"
    sigma_const: float = 1.0          # σ when sigma == "const" (alpha, or 1)
    nft_clip: Optional[float] = None  # DiffusionNFT: M

    @classmethod
    def from_adv(cls, config):
        """The row of train.adv (standard / exact / diffusionnft) and sample.global_std."""
        tr = config.train
        std = "global_std" if bool(getattr(config.sample, "global_std", False)) else "prompt_std"
        if tr.adv == "standard":
            return cls(scale=tr.scale, sigma=std)
        if tr.adv == "exact":
            return (cls(scale=tr.beta, sigma="const", sigma_const=tr.alpha) if tr.beta != 0
                    else cls(scale=tr.scale, sigma="const"))
        if tr.adv == "diffusionnft":
            return cls(scale=1.0, sigma=std, nft_clip=float(getattr(tr, "adv_clip_max", 5.0)))
        raise NotImplementedError(f"Advantage method {tr.adv} not implemented.")


class PromptHistory:
    """The prompts trained so far (the ``trained_prompt_num`` log; kept in the checkpoint as
    ``stat_tracker_history_prompts``)."""

    def __init__(self):
        self.history_prompts = set()

    def add(self, prompts):
        self.history_prompts.update(prompt_key(p) for p in np.unique(np.asarray(prompts)))


# ---- the trainers' epoch step ----


def compute_signals(samples, *, config, spec, epoch, prompt_history, num_train_timesteps, rank, world_size,
                    device, logger):
    """The epoch's learning signal from the scored samples of this rank (``rewards`` dict and
    ``prompts``, src/reward/rewards.py). Every rank calls it (it gathers):

        gather rewards and prompts (rank-major)
        -> A = group_signal(prompts, R, spec)     (+ the two noise bands)
        -> this rank's rows of A / band A -> the advantage statistics

    R is the combined reward "avg" (per-component whitened with train.whiten_rewards), repeated over
    the T trained timesteps. Runs in the training phase, after the sample cache, so an advantage
    hyperparameter changes without new rollouts. Returns (samples, log): every per-sample
    entry is a tensor (rewards -> the combined reward [n, T]; prompts dropped); ``log`` is filled on
    rank 0 (advantage statistics)."""
    from src.utils.distributed import is_main_process
    from src.utils.reward_gather import gather_epoch_rewards
    from src.utils.distributed import gather_prompts
    gathered = gather_epoch_rewards(samples, world_size=world_size, device=device,
                                    num_train_timesteps=num_train_timesteps)
    prompts_all = gather_prompts(samples.pop("prompts"), world_size)
    n = samples["timesteps"].shape[0]
    prompt_history.add(prompts_all)

    target = gathered["avg"]
    if getattr(config.train, "whiten_rewards", False):
        wsum = whitened_reward(gathered, config.reward_fn)
        target = np.repeat(wsum[:, None], num_train_timesteps, axis=1)
    advantages = group_signal(prompts_all, target, spec)            # [N, T]
    region_adv = None
    if "avg_low_noise" in gathered:
        bands = np.stack([gathered["avg_low_noise"][:, 0], gathered["avg_high_noise"][:, 0]], axis=1)
        region_adv = group_signal(prompts_all, bands, spec)          # [N, 2]: low, high
    log = {}

    # this rank's rows
    adv_rows = torch.from_numpy(rank_rows(advantages, world_size, n, rank)).to(device)
    band_rows = None
    if region_adv is not None:
        band_rows = {k: torch.from_numpy(v).to(device)
                     for k, v in region_rank_rows(region_adv, world_size, n, rank).items()}

    finish_samples(samples, adv_rows, band_rows, config, device=device)
    if is_main_process(rank):
        add_advantage_stats(log, advantages, config, logger)
        if region_adv is not None:
            add_region_advantage_stats(log, region_adv, gathered, epoch, logger)
    return samples, log
