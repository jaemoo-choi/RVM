"""The noisy state x_t a loss step trains on.

    rollout_state(batch, j_idx, from_latents)  -> x_t, noise, t, t_exp
        the rollout's state at its own t: the visited trajectory state latents[:, j] with the noise
        it implies (XT_FROM_LATENTS), or x0 re-noised at t with fresh noise from the global RNG
"""
from __future__ import annotations

import torch


def rollout_state(batch, j_idx, from_latents):
    """State at the rollout's noise level t = timesteps[:, j_idx] / 1000 (t_exp: t viewed to x0).

    from_latents (train.xt_from_latents) and a stored trajectory: the visited state latents[:, j_idx]
    (cast to x0's dtype) and the noise it implies, (x_t − (1 − t) x0) / t. Otherwise x0 re-noised
    with ε ~ N(0, I) drawn from the global RNG (one draw per call, as the loss always did).
    """
    x0 = batch["latents_clean"]
    t = batch["timesteps"][:, j_idx] / 1000.0
    t_exp = t.view(-1, *([1] * (x0.ndim - 1)))
    if from_latents and "latents" in batch:
        xt = batch["latents"][:, j_idx].to(x0.dtype)
        noise = ((xt.float() - (1 - t_exp) * x0.float()) / t_exp.clamp(min=1e-6)).to(x0.dtype)
    else:
        noise = torch.randn_like(x0.float())
        xt = (1 - t_exp) * x0 + t_exp * noise
    return xt, noise, t, t_exp
