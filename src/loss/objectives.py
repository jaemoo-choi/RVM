"""One class per RL objective (config.train.method). Each says, in one place:

    needs_old          whether its loss reads v_old (the "old" adapter's prediction); v_ref (frozen base)
                       is always available for the β‖v_θ − v_ref‖² term
    formula / terms()  its loss terms at one rollout state, in the box form of src/loss/box.py
    logs()             what one loss step logs (step_logs: loss, kl_div, x0_norm)

The model forwards and the gradient step are not here: src/loss/step.py:compute_loss runs the forwards,
src/engine/run.py the backward and the optimizer step.

| method (config.train.method) | weight | y | anchor | reward step | extra |
|---|---|---|---|---|---|
| rvm          | A/G, norm sg(mean|x̂₀−x₀|) | x̂₀(v_θ) | x₀ | 0 | β‖v_θ − v_ref‖² |
| diffusionnft | r·M/γ and (1−r)·M/γ, adaptive norms | x̂₀(v⁺), x̂₀(v⁻) | x₀ | 0 | β‖v_θ − v_ref‖² |

x̂₀(v) = x_t − t·v, so RVM's x₀-space box equals the velocity box A·t/(G·sg(mean|v − v_tgt|)) ·
‖v_θ − sg[v_tgt]‖². DiffusionNFT keeps its two implicit-policy branches
v⁺ = γv_θ + (1−γ)v_old, v⁻ = (1+γ)v_old − γv_θ with per-branch adaptive norms.
"""
from __future__ import annotations

import torch

from src.loss.box import BoxTerm, Inputs, adaptive_norm, bview, feat_dims, ref_term

# The method names (config.train.method).
METHODS = ("rvm", "diffusionnft")


def check_method(method):
    if method == "epg_noratio":
        raise ValueError("train.method 'epg_noratio' is now called 'rvm' (METHOD=rvm)")
    if method not in METHODS:
        raise ValueError(f"Unknown train.method {method!r}; kept methods: {sorted(METHODS)}")


# ---- what one loss step logs (every objective) ----

def step_logs(inp, loss):
    """loss (all terms), kl_div = ‖v_θ − v_ref‖² (distance from the frozen base; 0 when the fast path aliases
    v_ref to v_θ), x0_norm = mean x₀² of the generated samples (drops when outputs blur)."""
    feat = feat_dims(inp.x0)
    return {"loss": loss.detach(),
            "kl_div": ((inp.v.detach() - inp.v_ref) ** 2).mean(dim=feat).mean(),
            "x0_norm": torch.mean(inp.x0 ** 2)}


class Objective:
    """The defaults: an objective on the visited rollout state that reads v_old and v_ref."""
    needs_old = True

    def __init__(self, config):
        self.config = config
        self.tr = config.train

    def prepare(self, inp, batch, j_idx):
        """Fill what the formula reads besides the predictions (default: nothing)."""

    def terms(self, inp):
        raise NotImplementedError

    def logs(self, inp, loss):
        return step_logs(inp, loss)


class RVM(Objective):
    """RVM: weight A/G on ‖x̂₀(v_θ) − x₀‖² (adaptive norm) + β‖v_θ − v_ref‖².
    v_old only enters the diagnostics."""
    needs_old = False

    @staticmethod
    def formula(inp: Inputs):
        x0hat = inp.x_t - bview(inp.t, inp.x0) * inp.v
        norm = adaptive_norm(x0hat, inp.x0)
        policy = BoxTerm("policy", x0hat, inp.x0, weight=inp.adv / inp.elbo_gs, norm=norm)
        terms = [policy]
        beta = float(inp.cfg.beta)
        if beta != 0.0:
            terms.append(ref_term("ref", inp, beta))
        return terms, {}

    def terms(self, inp):
        return self.formula(inp)


class NFT(Objective):
    """DiffusionNFT: the two implicit-policy branches weighted by r and 1 − r (r: the NFT map of A)."""

    def prepare(self, inp, batch, j_idx):
        inp.adv = inp.adv.to(inp.v.dtype)       # r and 1 − r in v's dtype, like the reference code

    @staticmethod
    def formula(inp: Inputs):
        cfg = inp.cfg
        gamma = float(getattr(cfg, "diffusionnft_beta", 0.1))
        M = float(getattr(cfg, "adv_clip_max", 5.0))
        tt = bview(inp.t, inp.x0)
        v_plus = gamma * inp.v + (1.0 - gamma) * inp.v_old
        v_minus = (1.0 + gamma) * inp.v_old - gamma * inp.v
        y_plus = inp.x_t - tt * v_plus
        y_minus = inp.x_t - tt * v_minus
        r = inp.adv                     # in v's dtype, like the reference implementation
        # r and (1 - r) are formed in r's dtype, then lifted to float64 before scaling, so the weight
        # is not rounded in bf16 (the per-branch losses are float64 through the adaptive norm).
        w_pos = r.double() * M / gamma
        w_neg = (1.0 - r).double() * M / gamma
        terms = [
            BoxTerm("positive", y_plus, inp.x0, weight=w_pos, norm=adaptive_norm(y_plus, inp.x0)),
            BoxTerm("negative", y_minus, inp.x0, weight=w_neg, norm=adaptive_norm(y_minus, inp.x0)),
        ]
        if float(cfg.beta) != 0.0:
            terms.append(ref_term("ref", inp, cfg.beta))
        return terms, {}

    def terms(self, inp):
        return self.formula(inp)


OBJECTIVES = {"rvm": RVM, "diffusionnft": NFT}


def objective_for(config):
    """The objective of config.train.method."""
    method = config.train.method
    check_method(method)
    return OBJECTIVES[method](config)
