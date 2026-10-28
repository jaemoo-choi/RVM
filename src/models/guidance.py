"""The only CFG / negative-prompt code.

A model contributes two forwards (no CFG inside them), both using whatever adapter is active:

    branch(x, timesteps, cond, embed_scale) -> v         one conditioning, batch B
    pair(x, timesteps, cond, uncond)        -> (v_c, v_np)  one doubled batch (gradient to both)

and ``cfg`` combines them for one forward kind (``train`` = trainable prediction, ``sample`` = the
old / rollout policy replayed in the loss, ``ref`` = frozen base, ``rollout`` = a sampling pipeline),
with the config.train switches of that kind (``cfg_options``):

| switch | value | gradient |
|---|---|---|
| scale ≤ 1 (or no NP)               | v_c                          | v_c |
| plain (default)                    | v_np + w (v_c − v_np)        | w·v_c, NP under no_grad |
| np_grad="grad" (DanceGRPO / npgrad)| v_np + w (v_c − v_np)        | w·v_c + (1−w)·v_np (NP grad kept on a p-fraction) |
| np_grad="swap" (npswap)            | v_np + w (v_c − v_np)        | per sample, prob q: NP only, else cond only |
| sg_push                            | v_c + (w−1)·sg(v_c − v_np)   | v_c at unit scale |
| kind="rollout" (sampling pipelines)| v_np + w (v_c − v_np)        | none (pipelines run under no_grad) |

NP source: ``np_base`` → base model (LoRA off), else the active adapter. For sg-push the NP is
``sg_np``: "current" / "old" adapter, "base" (forced base model) or "np_base" (follow ``np_base``,
reusing a shared uncond when the caller has one).

Rollout (``kind="rollout"``): the sampling pipelines (src/models/pipelines/pipeline_with_logprob_*.py)
call ``cfg`` with a ``layout`` dict that fixes how each pipeline always laid out its forwards — two calls (NP first or conditional first) or
one doubled batch through the active adapter, and the dtype each branch is cast to before the
combination — so every rollout is bit-identical to the former inline closures. The NP base switch
there is the layer-level ``lora_ctx.disable_lora`` on the module actually called (a pipeline holds
the inner transformer, and for Wan2.2 the active expert), as the closures did.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import torch

from src.models.pipelines.lora_ctx import maybe_disable_lora

from src.models.adapters import base, use


def resolve_negative_prompt_text(config_text):
    """The negative prompt for CFG: ``config_text`` (``config.negative_prompt``, set in each
    config/<model>.py), unless the NEGATIVE_PROMPT environment variable overrides it:

        unset / "official"   -> config_text
        "empty" / "none"     -> ""
        anything else        -> used verbatim

    The negative prompt only enters the velocity when the CFG scale is > 1."""
    raw = os.getenv("NEGATIVE_PROMPT")
    if raw is None or raw.strip().lower() == "official":
        return config_text
    if raw.strip().lower() in ("empty", "none"):
        return ""
    return raw


# How a sampling pipeline lays out its no-grad CFG forwards (kind="rollout" only):
#   doubled       one doubled batch through ``pair`` on the active adapter (the NP base switch does
#                 not apply: those pipelines only took this path with the NP on the adapter)
#   np_first      two calls: the NP forward before the conditional one (else conditional first)
#   dtype         each branch is cast to this before the combination (None: as the branch returns)
#   single_dtype  cast of the single conditional forward when CFG is off (None: as returned)
DEFAULT_LAYOUT = dict(doubled=False, np_first=True, dtype=torch.float32, single_dtype=None)


def _cast(v, dtype):
    return v if dtype is None else v.to(dtype)


def _rollout(branch, pair, module, cond, uncond, scale, np_base, true_cfg, layout, return_parts):
    w = float(scale)
    if w <= 1.0 or not true_cfg or uncond is None:
        v = _cast(branch(cond, embed_scale=w), layout["single_dtype"])
        return (v, None, None) if return_parts else v
    if layout["doubled"]:
        v_c, v_np = pair(cond, uncond)
    elif layout["np_first"]:
        with maybe_disable_lora(module, np_base):
            v_np = branch(uncond, embed_scale=1.0)
        v_c = branch(cond, embed_scale=1.0)
    else:
        v_c = branch(cond, embed_scale=1.0)
        with maybe_disable_lora(module, np_base):
            v_np = branch(uncond, embed_scale=1.0)
    a_np, a_c = _cast(v_np, layout["dtype"]), _cast(v_c, layout["dtype"])
    v = a_np + w * (a_c - a_np)
    return (v, v_np, v_c) if return_parts else v


def _bshape(x):
    return (-1,) + (1,) * (x.ndim - 1)


def cfg(branch: Callable, pair: Optional[Callable], module, cond, uncond, scale: float, *, kind: str,
        np_base: bool = True, np_grad: str = "none", np_grad_prob: float = 1.0, sg_push: bool = False,
        sg_np: str = "current", true_cfg: bool = True, cached_uncond: Optional[torch.Tensor] = None,
        layout: Optional[dict] = None, return_parts: bool = False):
    """Guided velocity for one forward kind. ``module`` = the PEFT model (adapter switching); for
    ``kind="rollout"`` the module whose LoRA layers the NP base switch disables.

    kind          "train" | "sample" | "ref" | "rollout"
    np_base       NP from the base model (LoRA off), else from the active adapter
    np_grad       "none" | "grad" | "swap" (train only); np_grad_prob: grad = fraction keeping the NP
                  gradient, swap = swap probability
    sg_push       train only; sg_np = "current" | "old" | "base" | "np_base"
    true_cfg      False: the model has no NP for this call (scale only enters the guidance embedding)
    ``layout`` / ``return_parts`` are rollout-only; ``return_parts`` gives ``(v, v_np, v_c)`` with the
    raw branch outputs (``(v, None, None)`` without CFG)."""
    if kind == "rollout":
        return _rollout(branch, pair, module, cond, uncond, scale, np_base, true_cfg,
                        layout or DEFAULT_LAYOUT, return_parts)
    w = float(scale)
    if w <= 1.0 or not true_cfg or uncond is None:
        return branch(cond, embed_scale=w)

    if kind == "train" and np_grad == "swap" and np_grad_prob > 0.0:
        v_c, v_np = pair(cond, uncond)
        v_c, v_np = v_c.to(torch.float32), v_np.to(torch.float32)
        swap = (torch.rand(v_np.shape[0], device=v_np.device) < np_grad_prob).view(_bshape(v_np))
        v_c = torch.where(swap, v_c.detach(), v_c)
        v_np = torch.where(swap, v_np, v_np.detach())
        return v_np + w * (v_c - v_np)
    if kind == "train" and np_grad == "grad":
        v_c, v_np = pair(cond, uncond)
        v_c, v_np = v_c.to(torch.float32), v_np.to(torch.float32)
        if np_grad_prob < 1.0:
            keep = torch.rand(v_np.shape[0], device=v_np.device) < np_grad_prob
            v_np = torch.where(keep.view(_bshape(v_np)), v_np, v_np.detach())
        return v_np + w * (v_c - v_np)

    sg = kind == "train" and sg_push
    # The conditional (gradient) forward runs FIRST: under autocast the no-grad NP forward would
    # otherwise cache grad-less casts of the fp32 LoRA weights, and the conditional forward that
    # reuses them would carry no gradient (caught by the GPU parity run, 2026-10-23).
    v_c = branch(cond, embed_scale=1.0).to(torch.float32)
    with torch.no_grad():
        if sg and sg_np == "current":
            v_np = branch(uncond, embed_scale=1.0)
        elif sg and sg_np == "old":
            # back to the caller's adapter afterwards
            with use(module, "old"):
                v_np = branch(uncond, embed_scale=1.0)
        elif cached_uncond is not None:
            v_np = cached_uncond
        elif kind == "ref":
            v_np = branch(uncond, embed_scale=1.0)       # adapters already disabled by the caller
        else:
            use_base = (sg and sg_np == "base") or np_base
            with base(module, use_base):
                v_np = branch(uncond, embed_scale=1.0)
    v_np = v_np.to(torch.float32)
    if sg:
        return v_c + (w - 1.0) * (v_c - v_np).detach()
    return v_np + w * (v_c - v_np)


def cfg_options(config, kind: str) -> dict:
    """The config.train switches of one forward kind, as ``cfg`` keywords (models override entries)."""
    tr = config.train
    np_grad, prob = "none", 1.0
    if kind == "train":
        q = float(getattr(tr, "cfg_swap_grad_prob", 0.0))
        if q > 0.0:
            np_grad, prob = "swap", q
        elif getattr(tr, "cfg_uncond_grad", False):
            np_grad, prob = "grad", float(getattr(tr, "cfg_uncond_grad_prob", 1.0))
    return dict(
        np_base=bool(getattr(tr, "np_base_velocity", True)),
        np_grad=np_grad, np_grad_prob=prob,
        sg_push=kind == "train" and bool(getattr(tr, "cfg_sg_push", False)),
        sg_np=str(getattr(tr, "cfg_sg_push_adapter", "current")),
        true_cfg=True,
    )
