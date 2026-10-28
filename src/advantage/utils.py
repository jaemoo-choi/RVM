"""Helpers of src/advantage/advantages.py (imported only there): the recipe's group math, prompt keys,
per-component whitening, a rank's rows of the gathered arrays, the advantage statistics of the epoch log,
and writing the signal into the samples. The entry points other modules call (SignalSpec, PromptHistory,
compute_signals) stay in advantages.py."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

# The two noise-band advantages (Wan2.2 per-band rewards, config.reward_fn_{high,low}_noise): the key each one
# gets in the samples, and its column in compute_signals' region_adv [N, 2] (0 = low-noise band, 1 = high-noise band).
REGION_ADVANTAGES = (("advantages_low_noise", 0), ("advantages_high_noise", 1))


def group_signal(groups, values, spec):
    """Steps 1–3 for every sample: ``groups`` [N] (prompt texts or ids), ``values`` [N] or [N, K] (a
    reward per column: the timesteps, or the two noise bands). Returns A, float64, shaped like
    ``values``."""
    groups = np.asarray(groups)
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    if spec.sigma == "global_std":
        global_sigma = np.std(values, axis=0, keepdims=True) + 1e-4
    for g in np.unique(groups):
        mask = groups == g
        rows = values[mask]
        d = rows - np.mean(rows, axis=0, keepdims=True)
        if spec.sigma == "prompt_std":
            sigma = np.std(rows, axis=0, keepdims=True) + 1e-4
        elif spec.sigma == "global_std":
            sigma = global_sigma
        else:
            sigma = spec.sigma_const
        a = spec.scale * d / sigma
        if spec.nft_clip is not None:
            m = spec.nft_clip
            a = np.clip(np.clip(a, -m, m) / m / 2.0 + 0.5, 0.0, 1.0)
        out[mask] = a
    return out


def prompt_key(prompt) -> str:
    """The stable key of a prompt in PromptHistory (saved in training_state.pt): 16 hex digits of
    blake2b over the UTF-8 text. Python's ``hash(str)`` used before is salted per process
    (PYTHONHASHSEED); checkpoint.load_stat_tracker_history discards such int-keyed histories."""
    return hashlib.blake2b(str(prompt).encode("utf-8"), digest_size=8).hexdigest()


def whitened_reward(gathered, reward_weights):
    """Per-component reward whitening (train.whiten_rewards): every component of ``reward_weights``
    (config.reward_fn) standardised over the gathered batch (unit std) BEFORE the weighted sum, so a
    component's raw scale no longer dominates the advantage (e.g. the bounded [0, 1]
    dynamic_degree_bump vs the wider-range HPS / VideoAlign terms) and the weights act as importance
    shares. Returns the per-sample weighted sum [N] (float64)."""
    wsum = None
    for name, w in reward_weights.items():
        if name not in gathered:
            raise KeyError(
                f"whiten_rewards: reward component '{name}' not in gathered "
                f"rewards ({sorted(gathered.keys())})"
            )
        comp = gathered[name].astype(np.float64)
        comp = comp.reshape(comp.shape[0], -1).mean(axis=1)  # [N] per-sample
        z = (comp - comp.mean()) / (comp.std() + 1e-8)
        term = float(w) * z
        wsum = term if wsum is None else wsum + term
    return wsum


def rank_rows(values_all, world_size, samples_per_rank, rank):
    """This rank's [samples_per_rank, -1] block of a gathered rank-major array [N, ...]."""
    return values_all.reshape(world_size, samples_per_rank, -1)[rank]


def region_rank_rows(region_adv, world_size, samples_per_rank, rank):
    """This rank's [n, 1] column of each band advantage, keyed like the batch (low, then high). A
    per-sample scalar: the trainers repeat it over the timesteps and src/loss/step.py picks the band
    per step by the actual noise level."""
    return {key: rank_rows(region_adv[:, col:col + 1], world_size, samples_per_rank, rank)
            for key, col in REGION_ADVANTAGES}


def add_advantage_stats(epoch_log, advantages, config, logger):
    """advantages_{mean,max,min,std,abs_mean} and the fraction at the clip bound (+ the job-log line)."""
    _adv_flat_np = np.asarray(advantages, dtype=np.float64).reshape(-1)
    _adv_clip_max = float(getattr(config.train, "adv_clip_max", 5.0))
    _adv_frac_saturated = (
        float(np.mean(np.abs(_adv_flat_np) >= _adv_clip_max))
        if _adv_clip_max > 0 else 0.0
    )
    logger.info(f"Advantages abs mean: {np.abs(_adv_flat_np).mean()}")
    epoch_log.update({
        "advantages_mean": float(_adv_flat_np.mean()),
        "advantages_max": float(_adv_flat_np.max()),
        "advantages_min": float(_adv_flat_np.min()),
        "advantages_std": float(_adv_flat_np.std()),
        "advantages_abs_mean": float(np.abs(_adv_flat_np).mean()),
        "advantages_frac_saturated": _adv_frac_saturated,
    })


def add_region_advantage_stats(epoch_log, region_adv, gathered_rewards_dict, epoch, logger):
    """advantages_{low,high}_noise_{mean,abs_mean,std} (src/train_wan2_2.py only) and the job-log
    line; with identical band weights the two are equal by construction, so a difference is the
    proof the split is live."""
    for _name, _col in (("low_noise", 0), ("high_noise", 1)):
        _r = np.asarray(region_adv[:, _col], dtype=np.float64)
        epoch_log[f"advantages_{_name}_mean"] = float(_r.mean())
        epoch_log[f"advantages_{_name}_abs_mean"] = float(np.abs(_r).mean())
        epoch_log[f"advantages_{_name}_std"] = float(_r.std())
    logger.info(
        "Epoch %d region rewards: high=%.4f low=%.4f | "
        "region advantages abs-mean: high=%.4f low=%.4f",
        epoch,
        float(gathered_rewards_dict["avg_high_noise"][:, 0].mean()),
        float(gathered_rewards_dict["avg_low_noise"][:, 0].mean()),
        float(np.abs(region_adv[:, 1]).mean()),
        float(np.abs(region_adv[:, 0]).mean()),
    )


def finish_samples(samples, adv_rows, band_rows, config, *, device):
    """Leave the samples in the signal stage's output form: the model inputs plus "advantages" [n, T]
    (T = every sampler step, aligned with "timesteps"; the noise band already chosen per step).
    Everything else the stage used is dropped."""
    timesteps = samples["timesteps"]
    n, T = timesteps.shape
    if band_rows is not None:   # the band of each sample's noise level at each step
        thr = float(getattr(config.train, "high_noise_threshold", 0.9))
        level = timesteps.to(device) / 1000.0
        adv = torch.where(level >= thr, band_rows["advantages_high_noise"][:, :1].to(device),
                          band_rows["advantages_low_noise"][:, :1].to(device))
    else:                       # one value per sample, the same at every step
        adv = adv_rows[:, :1].to(device).repeat(1, T)
    samples["advantages"] = adv
    for key in ("rewards", "advantages_low_noise", "advantages_high_noise"):
        samples.pop(key, None)
