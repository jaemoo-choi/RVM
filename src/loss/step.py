"""One loss step: one micro-batch at one timestep j -> the loss to backpropagate.

    state        the visited rollout state x_t (src/loss/states.py:rollout_state)
    predictions  v_θ on the trained adapter, and what the objective asks for: v_old ("old" adapter),
                 v_ref (frozen base); CFG / negative prompt: src/models/guidance.py
    signal       the per-sample advantage A at j, attached once per epoch by src/advantage/
    loss         the objective's terms (src/loss/objectives.py: each class's formula), summed; with the
                 step's detached logs

It never sees rewards or prompt groups (advantages.py's job) nor takes the gradient step (src/engine/run.py).
"""
from __future__ import annotations

import os
import torch

from src.models.adapters import base, policy, use
from src.loss.box import Inputs, total
from src.loss.objectives import objective_for
from src.loss.states import rollout_state


# The order of the random draws (condition drop, state noise, CFG swap / keep masks) and of the
# forwards (shared uncond, old, train, ref) is the one the runs were trained with
# (tests/parity/run_loss_parity.py). Fix (on by default): with NP from the adapter
# (np_base_velocity=False) the shared uncond was also fed to the OLD forward; it now computes its own
# (moot since stage A: a model shares only the base-model NP, ``_compute_shared_uncond`` returns None
# for an adapter NP).

def _drop_condition(embeds, pooled_embeds, neg_embeds, neg_pooled_embeds, p):
    """Per sample, replace the prompt embedding by the negative-prompt one with prob p."""
    drop = torch.rand(embeds.shape[0], device=embeds.device) < p

    def swap(x, nx):
        if x is None or nx is None:
            return x
        assert nx.shape == x.shape, f"cfg_drop_prob: negative {tuple(nx.shape)} != prompt {tuple(x.shape)}"
        return torch.where(drop.view(-1, *([1] * (x.ndim - 1))), nx.to(x.dtype), x)

    return swap(embeds, neg_embeds), swap(pooled_embeds, neg_pooled_embeds)


def _objective(utils):
    """The run's objective (src/loss/objectives.py), built once per model utils."""
    obj = getattr(utils, "_objective", None)
    if obj is None or obj.config is not utils.config:
        obj = utils._objective = objective_for(utils.config)
    return obj


def compute_loss(utils, ddp, batch, j_idx, embeds, pooled, neg, neg_pooled):
    """Loss and detached log terms for one visited state ``j_idx`` of one micro-batch (call under
    autocast). The run's objective (src/loss/objectives.py) says where the state is, which predictions
    its loss reads, and its terms and logs."""
    config = utils.config
    tr = config.train
    obj = _objective(utils)

    p_drop = float(getattr(tr, "cfg_drop_prob", 0.0))
    if p_drop > 0.0:
        embeds, pooled = _drop_condition(embeds, pooled, neg, neg_pooled, p_drop)

    # ---- state: the visited rollout state, or a re-noised one at the rollout's t (src/loss/states.py) ----
    x0 = batch["latents_clean"]
    xt, _, t, _ = rollout_state(batch, j_idx, getattr(tr, "xt_from_latents", False))

    sg_push = bool(getattr(tr, "cfg_sg_push", False))
    elbo_gs = 1.0 if sg_push else tr.guidance_scale
    module = ddp.module

    def predict(kind, cached=None):
        return utils.v_pred_fn(ddp, xt, batch, j_idx, embeds, pooled, neg, neg_pooled, kind,
                               cached_uncond=cached)

    # ---- predictions v_θ (train), v_old (sample, old adapter), v_ref (ref, frozen base) ----
    # v_old enters the loss of DiffusionNFT; for RVM it is only a diagnostic (aliased by the fast path,
    # skipped with SKIP_OLD_FORWARD=1).
    fast = (((sg_push and getattr(tr, "cfg_sg_push_adapter", "current") != "base")
             or getattr(tr, "cfg_uncond_grad", False))
            and float(tr.beta) == 0.0 and not obj.needs_old)
    if fast:
        # Only the trainable forward enters the loss (RVM, β = 0, NP not shared): old / ref alias it.
        policy(module)
        v = predict("train")
        v_old = v.detach()
        v_ref = v.detach()
    else:
        cached = utils._compute_shared_uncond(ddp, xt, batch, j_idx, neg, neg_pooled)
        np_base = bool(getattr(tr, "np_base_velocity", True))
        skip_old = not obj.needs_old and os.getenv("SKIP_OLD_FORWARD", "0") == "1"
        v_old = None
        if not skip_old:
            with use(module, "old"), torch.no_grad():
                v_old = predict("sample", cached)   # a shared uncond is only ever the base model's NP
        policy(module)
        v = predict("train", cached)
        if skip_old:
            v_old = v.detach()
        with torch.no_grad():
            with base(module):
                v_ref = predict("ref", cached if np_base else None)

    # ---- the learning signal (computed once per epoch by src/advantage/advantages.py) ----
    adv = batch["advantages"][:, j_idx]

    inp = Inputs(cfg=tr, v=v, x_t=xt, x0=x0, t=t, adv=adv, v_old=v_old, v_ref=v_ref, elbo_gs=elbo_gs)

    # ---- the objective's terms ----
    obj.prepare(inp, batch, j_idx)
    terms, _ = obj.terms(inp)
    loss = total(terms)

    # ---- logs ----
    with torch.no_grad():
        logs = obj.logs(inp, loss)
    return loss, logs
