"""The one form every objective's loss is written in, and the record its formulas read. Pure tensor math.

    BoxTerm, total        weight · ‖y − sg[anchor + reward_step]‖² and the sum of a step's terms
    Inputs                what a formula reads (predictions, state, signal)
    ref_term              the β‖v_θ − sg[v_ref]‖² extra term
    feat_dims, bview, adaptive_norm   helpers

The formulas themselves live with their objectives (src/loss/objectives.py: each class's ``formula``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------------------------------
# The box form:  weight · ‖ y − sg[target] ‖² ,  target = anchor + reward_step
# ---------------------------------------------------------------------------------------------------
# * ``y``       the trainable quantity (v_θ, or a differentiable function of it: x̂₀ = x_t − t·v_θ,
#               NFT's v± mixtures);
# * ``target``  anchor + reward step, built under no_grad and detached here;
# * ``weight``  per-sample [B] coefficient (advantage, β, …);
# * ``norm``    optional per-sample normaliser applied inside the mean (the adaptive
#               1/sg(mean|x̂₀ − x₀|) of RVM / DiffusionNFT).
# A term's gradient is ``−weight · (target − y) · ∂y/∂θ / (D · norm)``: only weight × step reaches θ.

def feat_dims(x: torch.Tensor) -> tuple:
    return tuple(range(1, x.ndim))

def bview(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """[B] → [B, 1, …] broadcastable against ``like``."""
    return v.view(-1, *([1] * (like.ndim - 1)))


@dataclass
class BoxTerm:
    name: str
    y: torch.Tensor                       # [B, *feat], carries the gradient
    target: torch.Tensor                  # [B, *feat], detached in per_sample()
    weight: torch.Tensor | float = 1.0    # [B] or scalar
    norm: Optional[torch.Tensor] = None   # [B, 1, …] or None

    def per_sample(self) -> torch.Tensor:
        sq = (self.y - self.target.detach()) ** 2
        if self.norm is not None:
            sq = sq / self.norm
        return self.weight * sq.mean(dim=feat_dims(self.y))

    def loss(self) -> torch.Tensor:
        return self.per_sample().mean()


def total(terms) -> torch.Tensor:
    """Sum of the terms' batch means."""
    out = None
    for term in terms:
        value = term.loss()
        out = value if out is None else out + value
    return out


def adaptive_norm(y: torch.Tensor, target: torch.Tensor, floor: float = 1e-5) -> torch.Tensor:
    """sg(mean|y − target|) per sample, in float64 like compute_elbo's adaptive weight."""
    with torch.no_grad():
        return (y.double() - target.double()).abs().mean(dim=feat_dims(y), keepdim=True).clip(min=floor)


@dataclass
class Inputs:
    """Everything an objective may read; fields a method does not need may be None."""
    cfg: object                       # config.train
    v: torch.Tensor                   # v_θ (trainable)
    x_t: torch.Tensor
    x0: torch.Tensor
    t: torch.Tensor                   # [B] noise level in (0, 1]
    adv: torch.Tensor                 # [B] advantage (NFT: optimality r ∈ [0, 1])
    v_old: Optional[torch.Tensor] = None
    v_ref: Optional[torch.Tensor] = None
    elbo_gs: float = 1.0                    # G in RVM's adaptive weight (1 under sg-push)


def ref_term(name, inp: Inputs, coef: float) -> BoxTerm:
    """β‖v_θ − sg[v_ref]‖² (the 'extra' column)."""
    return BoxTerm(name, inp.v, inp.v_ref, weight=float(coef))
