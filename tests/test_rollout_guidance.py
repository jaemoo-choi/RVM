"""Rollout CFG through src/models/guidance.py == the sampling pipelines' former inline closures (stage D).

The reference side is a verbatim copy of each pipeline at 17b0364f (tests/parity/rollout_ref/,
imported as a sibling module of src/models/pipelines so its relative imports resolve to the same
solver / lora_ctx); the new side is src/models/pipelines/pipeline_with_logprob_<model>.py. Both run
on stub pipelines around tiny bf16 transformers carrying real PEFT LoRA layers (so the NP base
switch matters), with the real ``run_sampling`` (flow SDE, seeded) and every CFG / NP mode the
pipeline supports. Must be identical: every velocity, latent and log-prob (torch.equal, same
dtype), and the transformer call log (order, module, batch, LoRA on/off, conditioning, embedding).

CPU only.
"""
import importlib.util
import itertools
import os
import sys
import types
import unittest

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
from peft.tuners.tuners_utils import BaseTunerLayer  # noqa: E402

REF_DIR = os.path.join(ROOT, "tests", "parity", "rollout_ref")
MODELS = ("sd3", "wan2_1", "skyreels_i2v", "wan2_2_a14b_t2v")


def _load(model, ref):
    if ref:
        name = f"src.models.pipelines._ref_pipeline_{model}"
        path = os.path.join(REF_DIR, f"pipeline_with_logprob_{model}.py")
    else:
        name = f"src.models.pipelines.pipeline_with_logprob_{model}"
        path = os.path.join(ROOT, "src", "models", "pipelines", f"pipeline_with_logprob_{model}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------
# tiny transformer with real LoRA layers
# ---------------------------------------------------------------------------------------------
class Tiny(nn.Module):
    def __init__(self, name, c_in, c_out, d_txt, seed, log):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.name, self.log = name, log
        self.proj = nn.Linear(c_in, c_out)
        self.txt = nn.Linear(d_txt, c_out)
        self.pool = nn.Linear(d_txt, c_out)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        self.c_in = c_in

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _lora_on(self):
        layers = [m for m in self.modules() if isinstance(m, BaseTunerLayer)]
        return all(not m.disable_adapters for m in layers)

    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections=None,
                joint_attention_kwargs=None, attention_kwargs=None, encoder_attention_mask=None,
                guidance=None, return_dict=False):
        dt = self.dtype
        x = hidden_states.to(dt)
        b = x.shape[0]
        self.log.append((
            self.name, b, self._lora_on(), float(encoder_hidden_states.float().sum()),
            None if guidance is None else float(guidance.float().sum()),
            None if encoder_attention_mask is None else int(encoder_attention_mask.sum()),
            str(timestep.dtype), float(timestep.float().sum()),
        ))
        xc = x.movedim(1, -1)                                        # channels last
        h = self.proj(xc)
        e = self.txt(encoder_hidden_states.to(dt).mean(1))           # (B, c_out)
        if pooled_projections is not None:
            e = e + self.pool(pooled_projections.to(dt))
        if encoder_attention_mask is not None:
            e = e * (1.0 + 0.01 * encoder_attention_mask.sum(1, keepdim=True).to(dt))
        if guidance is not None:
            e = e + (guidance.to(dt) / 1000.0)[:, None]
        e = e + (timestep.to(dt) / 1000.0)[:, None]
        shape = (b,) + (1,) * (h.ndim - 2) + (h.shape[-1],)
        h = torch.tanh(h + e.view(shape))
        return (h.movedim(-1, 1),)


def tiny(name, c_in, c_out, d_txt, seed, log):
    m = Tiny(name, c_in, c_out, d_txt, seed, log)
    torch.manual_seed(seed + 100)
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=4, target_modules=["proj", "txt"],
                                       init_lora_weights=False), m)
    # set after injection: PEFT reads a dict-like ``config`` when one is present
    m.config = types.SimpleNamespace(in_channels=c_in, guidance_embeds=False)
    return m.to(torch.bfloat16)


class Sched:
    def __init__(self, sigmas):
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * 1000
        self.config = types.SimpleNamespace(num_train_timesteps=1000)


def _retrieve(sched, n, device, sigmas=None, **_):
    return sched.timesteps, len(sched.timesteps)


SIGMAS = [1.0, 0.92, 0.75, 0.5, 0.25, 0.0]


class Stub:
    """The pipeline attributes the patched __call__s touch."""
    def __init__(self, transformer, transformer_2=None, latents=None):
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.scheduler = Sched(SIGMAS)
        self._latents = latents
        self._execution_device = torch.device("cpu")
        self.default_sample_size, self.vae_scale_factor = 8, 1
        self.config = types.SimpleNamespace(boundary_ratio=0.8)
        self.vae = types.SimpleNamespace(config=types.SimpleNamespace(
            spatial_compression_ratio=8, temporal_compression_ratio=4, z_dim=4))

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    def check_inputs(self, *a, **k):
        pass

    def encode_prompt(self, **kw):
        return (kw["prompt_embeds"], kw["negative_prompt_embeds"],
                kw["pooled_prompt_embeds"], kw["negative_pooled_prompt_embeds"])

    def prepare_latents(self, *a, **k):
        return self._latents.clone()

    def maybe_free_model_hooks(self):
        pass


# ---------------------------------------------------------------------------------------------
def _run(model, ref, case):
    mod = _load(model, ref)
    mod.retrieve_timesteps = _retrieve
    record = []
    real = mod.run_sampling

    def run_sampling(v_pred_fn, *a, **k):
        def wrapped(z, sigma):
            v = v_pred_fn(z, sigma)
            record.append(v)
            return v
        return real(wrapped, *a, **k)

    mod.run_sampling = run_sampling
    log, cb = [], []
    g = torch.Generator().manual_seed(1)
    B, D, L = 2, 6, 5
    bf = torch.bfloat16
    pos = torch.randn(B, L, D, generator=g).to(bf)
    neg = torch.randn(B, L, D, generator=g).to(bf)
    pos_p = torch.randn(B, D, generator=g).to(bf)
    neg_p = torch.randn(B, D, generator=g).to(bf)
    kw = dict(skip_decode=True, solver="flow", noise_level=0.7, deterministic=False)

    if model == "sd3":
        lat = torch.randn(B, 4, 3, 3, generator=g).to(bf)
        pipe = Stub(tiny("t", 4, 4, D, 0, log), latents=lat)
        kw.update(prompt_embeds=pos, negative_prompt_embeds=neg, pooled_prompt_embeds=pos_p,
                  negative_pooled_prompt_embeds=neg_p, height=3, width=3,
                  guidance_scale=case["gs"], np_base_velocity=case["np_base"])
    elif model == "wan2_1":
        lat = torch.randn(B, 4, 2, 3, 3, generator=g)
        pipe = Stub(tiny("t", 4, 4, D, 0, log), latents=lat)
        kw.update(prompt_embeds=pos, negative_prompt_embeds=neg if case["neg"] else None,
                  height=3, width=3, num_frames=5, guidance_scale=case["gs"],
                  np_base_velocity=case["np_base"],
                  peft_model=object() if case["peft"] else None,
                  velocity_callback=lambda *a: cb.append(a))
    elif model == "skyreels_i2v":
        pipe = Stub(tiny("t", 8, 4, D, 0, log))
        ff = torch.randn(B, 4, 2, 3, 3, generator=g).to(bf)
        mask = torch.ones(B, L, dtype=torch.long)
        mask[:, -1] = 0
        nmask = torch.ones(B, L, dtype=torch.long)
        nmask[:, -2:] = 0
        kw.update(first_frame_latents=ff, prompt_embeds=pos, pooled_prompt_embeds=pos_p,
                  negative_prompt_embeds=neg if case["neg"] else None,
                  negative_pooled_prompt_embeds=neg_p if case["neg"] else None,
                  prompt_attention_mask=mask,
                  negative_prompt_attention_mask=nmask if case["nmask"] else None,
                  height=24, width=24, num_frames=5, guidance_scale=case["gs"],
                  np_base_velocity=case["np_base"], generator=torch.Generator().manual_seed(7))
    else:
        lat = torch.randn(B, 4, 2, 3, 3, generator=g)
        pipe = Stub(tiny("high", 4, 4, D, 0, log), tiny("low", 4, 4, D, 1, log), latents=lat)
        kw.update(prompt_embeds=pos, negative_prompt_embeds=neg if case["neg"] else None,
                  height=3, width=3, num_frames=5, guidance_scale=case["gs"],
                  guidance_scale_2=case["gs2"], np_base_velocity=case["np_base"],
                  solver="flow")
    torch.manual_seed(1234)
    out = mod.pipeline_with_logprob(pipe, **kw)
    # every LoRA layer must be back on afterwards
    for t in (pipe.transformer, pipe.transformer_2):
        if t is not None:
            assert t._lora_on()
    return {"v": record, "latents": out[1], "log_probs": out[2], "log": log, "cb": cb,
            "extra": out[3] if len(out) > 3 else None}


def _cases(model):
    if model == "sd3":
        grid = dict(gs=[1.0, 3.0], np_base=[True, False])
    elif model == "wan2_1":
        grid = dict(gs=[1.0, 5.0], np_base=[True, False], peft=[True, False], neg=[True])
    elif model == "skyreels_i2v":
        grid = dict(gs=[1.0, 2.0, 6.0], np_base=[True, False], neg=[True, False],
                    nmask=[True, False])
    else:
        grid = dict(gs=[1.0, 4.0], gs2=[1.0, 3.0], np_base=[True, False], neg=[True, False])
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        case = dict(zip(keys, vals))
        if model == "skyreels_i2v" and not case["np_base"] and not case["nmask"]:
            continue    # doubled batch with a B-row mask for 2B rows: crashes in the old code too
        yield case


def _eq(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape \
            and torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(map(_eq, a, b))
    return a == b


class RolloutGuidanceParity(unittest.TestCase):
    def _check(self, model):
        n = 0
        for case in _cases(model):
            with self.subTest(model=model, **case):
                old, new = _run(model, True, case), _run(model, False, case)
                self.assertTrue(len(old["v"]) == len(SIGMAS) - 1)
                for k in ("log", "v", "latents", "log_probs", "cb", "extra"):
                    self.assertTrue(_eq(old[k], new[k]), f"{model} {case}: {k} differs")
                n += 1
        self.assertGreater(n, 0)

    def test_sd3(self):
        self._check("sd3")

    def test_wan2_1(self):
        self._check("wan2_1")

    def test_skyreels_i2v(self):
        self._check("skyreels_i2v")

    def test_wan2_2_a14b_t2v(self):
        self._check("wan2_2_a14b_t2v")

    def test_cfg_modes_exercised(self):
        """Sanity: the cases really switch paths (NP base vs adapter, CFG on/off differ)."""
        a = _run("sd3", False, dict(gs=3.0, np_base=True))
        b = _run("sd3", False, dict(gs=3.0, np_base=False))
        c = _run("sd3", False, dict(gs=1.0, np_base=True))
        self.assertFalse(_eq(a["v"], b["v"]))
        self.assertFalse(_eq(a["v"], c["v"]))
        self.assertEqual([e[1] for e in b["log"]][:1], [4])          # doubled batch
        self.assertEqual([e[2] for e in a["log"]][:2], [False, True])  # NP on base, first


if __name__ == "__main__":
    unittest.main()
