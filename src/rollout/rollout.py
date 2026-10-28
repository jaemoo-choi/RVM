"""The sampling phase's records: what a model's rollout returns, and one epoch's samples.

    Rollout                     what every model's ``model_utils.rollout(pipeline, **kwargs)`` returns
    collate(samples, skip)      the per-batch sample dicts of an epoch -> one dict
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Rollout:
    """What ``model_utils.rollout(pipeline, **kwargs)`` returns, for every model.

    media    decoded images / videos [B, ...] (None with ``skip_decode``)
    latents  the sampler trajectory, a list of num_steps + 1 tensors [B, *latent];
             latents[-1] is the endpoint x0 (``trajectory`` stacks it to [B, S+1, *latent])
    extras   per-rollout conditioning the loss needs, added to the sample dict by
             ``model_utils.collect_sample_artifacts(sample_data, rollout)``:
             SkyReels ``first_frame_latents``, Wan2.2 ``expert_mask``
    """
    media: Optional[torch.Tensor]
    latents: list
    extras: dict = field(default_factory=dict)

    @property
    def trajectory(self) -> torch.Tensor:
        return torch.stack(self.latents, dim=1)


def collate(samples_data_list, skip=()):
    """One dict per epoch from the per-batch sample dicts: tensors concatenated on dim 0, lists
    (prompt texts) chained, dicts (rewards) per key, None kept; keys in ``skip`` dropped
    (src/train_wan2_2.py: prompt_metadata, only needed by the reward)."""
    def _collate_field(key, samples):
        first = samples[0][key]
        if first is None:
            return None
        if isinstance(first, list):
            # Non-tensor fields (e.g., prompts_text): concatenate lists
            result = []
            for s in samples:
                result.extend(s[key])
            return result
        if isinstance(first, dict):
            return {sk: torch.cat([s[key][sk] for s in samples], dim=0) for sk in first}
        return torch.cat([s[key] for s in samples], dim=0)

    return {k: _collate_field(k, samples_data_list) for k in samples_data_list[0].keys() if k not in skip}
