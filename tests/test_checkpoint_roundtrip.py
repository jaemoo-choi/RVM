"""src/engine/checkpoint.py: save -> resume round trip; folders written by the old code resume (stage B3).

A tiny PEFT LoRA model carries the trainers' adapters (default trainable, old frozen) and an AdamW. It is saved with the new checkpoint.save and with a verbatim copy of
the pre-B3 save_ckpt (src/utils/__init__.py @ 519d9df4): both must write the same files with the
same contents, and resuming either folder into a freshly built model with the new resume pieces must
restore the adapters (old from lora_old_policy/old, B1), the optimizer state, the EMA, the training
state and the stat-tracker history exactly. Log lines are checked against the old trainers' text.
"""
import json
import logging
import os
import tempfile
import types
import unittest

import numpy as np
import torch
from ml_collections import ConfigDict
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch.amp import GradScaler as _GradScaler

from src.engine import checkpoint
from src.engine import wandb_run
from src.advantage.advantages import PromptHistory
from src.advantage.utils import prompt_key

LOG = logging.getLogger("ckpt-test")


def GradScaler(enabled):
    """The trainers' torch.cuda.amp.GradScaler(enabled=...) without its deprecation warning."""
    return _GradScaler("cuda", enabled=enabled)


# ======================= reference: src/utils/__init__.py @ 519d9df4 =======================

def is_main_process(rank):
    return rank == 0


def _is_enospc(exc):
    import errno
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return True
    return "os error 28" in str(exc) or "No space left on device" in str(exc)


def old_save_ckpt(
    logger, wandb, save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler, epoch=None, stat_tracker=None, ckpt_subdir="checkpoints"
):
    if is_main_process(rank):
        import json
        import shutil
        import time
        save_root = os.path.join(save_dir, ckpt_subdir)
        os.makedirs(save_root, exist_ok=True)
        staging = os.path.join(save_root, ".staging")

        model_to_save = transformer_ddp.module

        def _write_all():
            shutil.rmtree(staging, ignore_errors=True)
            os.makedirs(staging, exist_ok=True)
            # current policy (default adapter) - without EMA
            model_to_save.set_adapter("default")
            model_to_save.save_pretrained(os.path.join(staging, "lora_current_policy"))
            # old policy adapter
            model_to_save.set_adapter("old")
            model_to_save.save_pretrained(os.path.join(staging, "lora_old_policy"))
            # EMA policy (if enabled)
            if config.train.ema and ema is not None:
                model_to_save.set_adapter("default")
                ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
                try:
                    model_to_save.save_pretrained(os.path.join(staging, "lora_ema_policy"))
                finally:
                    ema.copy_temp_to(transformer_trainable_parameters)
            model_to_save.set_adapter("default")
            # optimizer and scaler
            torch.save(optimizer.state_dict(), os.path.join(staging, "optimizer.pt"))
            if scaler is not None:
                torch.save(scaler.state_dict(), os.path.join(staging, "scaler.pt"))
            # global_step, epoch, stat_tracker state, and wandb run id for proper resumption
            training_state = {
                "global_step": global_step,
                "wandb_run_id": wandb.run.id if wandb.run else None,
            }
            if epoch is not None:
                training_state["epoch"] = epoch
            if stat_tracker is not None:
                training_state["stat_tracker_history_prompts"] = stat_tracker.history_prompts
            torch.save(training_state, os.path.join(staging, "training_state.pt"))
            # training config for evaluation
            with open(os.path.join(staging, "config.json"), "w") as f:
                json.dump(config.to_dict(), f, indent=2)

        retries = int(os.getenv("SAVE_ENOSPC_RETRIES", "15"))
        wait_s = float(os.getenv("SAVE_ENOSPC_WAIT_S", "60"))
        for attempt in range(retries + 1):
            try:
                _write_all()
                break
            except Exception as exc:
                model_to_save.set_adapter("default")
                shutil.rmtree(staging, ignore_errors=True)
                if not _is_enospc(exc) or attempt == retries:
                    logger.error(f"Checkpoint save to {save_root} failed ({exc!r}); the previous checkpoint is untouched")
                    raise
                logger.warning(f"Checkpoint save hit ENOSPC (attempt {attempt + 1}/{retries}); retrying in {wait_s:.0f}s: {exc}")
                time.sleep(wait_s)

        # Commit: move every staged file into place (renames only).
        for root, _dirs, files in os.walk(staging):
            rel = os.path.relpath(root, staging)
            dst_dir = save_root if rel == "." else os.path.join(save_root, rel)
            os.makedirs(dst_dir, exist_ok=True)
            for fn in files:
                os.replace(os.path.join(root, fn), os.path.join(dst_dir, fn))
        shutil.rmtree(staging, ignore_errors=True)
        logger.info(f"Saved all checkpoints to {save_root} at step {global_step} (staged, atomic commit)")


# ======================= a tiny model with the trainers' adapters =======================

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(6, 8)
        self.l2 = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


class FakeDDP:
    def __init__(self, module):
        self.module = module


class EMAStub:
    """EMAModuleWrapper's checkpoint surface (ema_parameters, copy_ema_to, copy_temp_to). src/ema.py
    stores the temporary copy with .cpu(), which on a CPU tensor is a view; here it is a real copy."""

    def __init__(self, params, scale):
        self.ema_parameters = [p.detach().clone() * scale for p in params]
        self.temp = None

    def copy_ema_to(self, params, store_temp=True):
        params = list(params)
        if store_temp:
            self.temp = [p.detach().clone() for p in params]
        for e, p in zip(self.ema_parameters, params):
            p.data.copy_(e)

    def copy_temp_to(self, params):
        for t, p in zip(self.temp, list(params)):
            p.data.copy_(t)
        self.temp = None


def trainable(model, adapter):
    model.set_adapter(adapter)
    out = [p for p in model.parameters() if p.requires_grad]
    model.set_adapter("default")
    return out


def build(seed):
    """Model with default / old adapters of distinct weights, its AdamW after
    two steps, an EMA, as src/models/base.py:set_adapter and src/train.py set them up."""
    torch.manual_seed(0)                      # same frozen base weights in every build
    lora = LoraConfig(r=2, lora_alpha=4, target_modules=["l1", "l2"], lora_dropout=0.0)
    model = get_peft_model(Tiny(), lora)
    model.add_adapter("old", lora)
    model.set_adapter("default")
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if "lora_" in name:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    params, old_params = trainable(model, "default"), trainable(model, "old")
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    x = torch.randn(5, 6, generator=g)
    model(x).pow(2).mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    ema = EMAStub(params, scale=0.5 + seed % 7 / 10)
    return types.SimpleNamespace(model=model, lora=lora, params=params, old=old_params,
                                 opt=opt, ema=ema)


def make_config(**train):
    return ConfigDict({"resume_from": "", "seed": 0, "name": "proj", "run_name": "run",
                       "train": {"ema": True, "lr": 1e-3, **train}})


def lora_weights(model, adapter):
    return {n: p.detach().clone() for n, p in model.named_parameters() if f".{adapter}." in n and "lora_" in n}


def same(test, a, b, path="x"):
    if isinstance(a, torch.Tensor):
        test.assertTrue(isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), path)
    elif isinstance(a, dict):
        test.assertEqual(sorted(map(str, a)), sorted(map(str, b)), path)
        for k in a:
            same(test, a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            same(test, x, y, f"{path}[{i}]")
    else:
        test.assertEqual(a, b, path)


def files(root):
    out = {}
    for d, _, names in os.walk(root):
        for n in names:
            out[os.path.relpath(os.path.join(d, n), root)] = os.path.join(d, n)
    return out


def read(path):
    if path.endswith(".safetensors"):
        return load_file(path)
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu", weights_only=False)
    with open(path) as f:                       # e.g. the model card PEFT writes
        return f.read()


WANDB = types.SimpleNamespace(run=types.SimpleNamespace(id="run-abc"))


def tracker():
    t = PromptHistory()
    t.history_prompts = {prompt_key("p0"), prompt_key("p1"), prompt_key("p2")}
    return t


def save(fn, src, save_dir, config, subdir="checkpoints", epoch=3):
    fn(LOG, WANDB, save_dir, FakeDDP(src.model), 7, 0, src.ema, src.params, config, src.opt,
       GradScaler(enabled=False), epoch, tracker(), ckpt_subdir=subdir)


# ======================= tests =======================

class SaveFormatTests(unittest.TestCase):
    def test_new_save_writes_what_the_old_save_wrote(self):
        src = build(1)
        config = make_config()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            save(old_save_ckpt, src, a, config)
            save(checkpoint.save, src, b, config)
            fa, fb = files(os.path.join(a, "checkpoints")), files(os.path.join(b, "checkpoints"))
            self.assertEqual(sorted(fa), sorted(fb))
            for rel in fa:
                same(self, read(fa[rel]), read(fb[rel]), rel)
            for d in ("lora_current_policy", "lora_old_policy", "lora_ema_policy"):
                self.assertIn(f"{d}/adapter_model.safetensors", fa)
                self.assertIn(f"{d}/old/adapter_model.safetensors", fa)
            self.assertFalse(os.path.exists(os.path.join(b, "checkpoints", ".staging")))
        self.assertIs(checkpoint.save_ckpt, checkpoint.save)


class ResumeTests(unittest.TestCase):
    def _resume(self, save_fn, **train):
        """Save ``src`` with ``save_fn``, resume into a fresh build like src/train.py does."""
        src = build(1)
        with tempfile.TemporaryDirectory() as d:
            save(save_fn, src, d, make_config())
            dst = build(2)
            tr = tracker()
            tr.history_prompts = set()
            config = make_config(**train)
            with self.assertLogs(LOG, level="INFO") as logs:
                self.assertTrue(checkpoint.resume_dir(config, d, os.path.join(d, "checkpoints")))
                ck = os.path.join(d, "checkpoints")
                self.assertEqual(config.resume_from, ck)
                step = checkpoint.resume(dst.model, dst.opt, GradScaler(enabled=False), config, ck, LOG,
                                device="cpu", enable_amp=False)
                checkpoint.load_ema(dst.model, dst.ema, ck, dst.lora, LOG)
                checkpoint.load_stat_tracker_history(tr, ck, LOG)
            self.assertEqual(wandb_run.read_wandb_run_id(d, ck), "run-abc")
        return src, dst, tr, step, [r.getMessage() for r in logs.records], ck

    def _check(self, src, dst, tr, step, old_from="old"):
        self.assertEqual(step, (7, 3))
        same(self, lora_weights(src.model, "default"), lora_weights(dst.model, "default"), "default")
        want_old = {n.replace(f".{old_from}.", ".old."): p for n, p in lora_weights(src.model, old_from).items()}
        same(self, want_old, lora_weights(dst.model, "old"), "old")
        same(self, src.opt.state_dict(), dst.opt.state_dict(), "optimizer")
        same(self, src.ema.ema_parameters, dst.ema.ema_parameters, "ema")
        self.assertEqual(tr.history_prompts, tracker().history_prompts)
        self.assertEqual(dst.model.active_adapter, "default")

    def test_roundtrip_new_save(self):
        src, dst, tr, step, msgs, ck = self._resume(checkpoint.save)
        self._check(src, dst, tr, step)
        self.assertEqual(msgs, [
            f"Loaded current policy from {ck}/lora_current_policy",
            f"Loaded old policy from {ck}/lora_old_policy/old",
            "Loaded optimizer state",
            "Loaded global_step: 7, epoch: 3",
            f"Loaded EMA policy from {ck}/lora_ema_policy",
            "Loaded stat_tracker with 3 historical prompts",
        ])

    def test_folder_written_by_old_save_resumes(self):
        src, dst, tr, step, _, _ = self._resume(old_save_ckpt)
        self._check(src, dst, tr, step)

    def test_fix_resume_old_off_loads_the_root(self):
        # B1 off: the root of lora_old_policy/ holds the "default" adapter (the pre-fix resume).
        src, dst, tr, step, msgs, ck = self._resume(checkpoint.save, fix_resume_old=False)
        self._check(src, dst, tr, step, old_from="default")
        self.assertIn(f"Loaded old policy from {ck}/lora_old_policy", msgs)

    def test_no_checkpoint_no_resume(self):
        with tempfile.TemporaryDirectory() as d:
            config = make_config()
            self.assertFalse(checkpoint.resume_dir(config, d, os.path.join(d, "checkpoints")))
            self.assertEqual(config.resume_from, "")
            self.assertIsNone(wandb_run.read_wandb_run_id(d, os.path.join(d, "checkpoints")))
            self.assertEqual(checkpoint.load_training_state(os.path.join(d, "checkpoints"), "cpu", LOG), (0, 0))

    def test_corrupt_optimizer_leaves_fresh_state(self):
        src = build(1)
        with tempfile.TemporaryDirectory() as d:
            save(checkpoint.save, src, d, make_config())
            ck = os.path.join(d, "checkpoints")
            opt_path = os.path.join(ck, "optimizer.pt")
            with open(opt_path, "rb") as f:
                blob = f.read()
            with open(opt_path, "wb") as f:          # a save cut by a full disk
                f.write(blob[: len(blob) // 2])
            for prefix, hint in (("", " Consider removing corrupted checkpoint."), ("[low] ", "")):
                dst = build(2)
                with self.assertLogs(LOG, level="WARNING") as logs:
                    checkpoint.load_optimizer(dst.opt, ck, LOG, prefix=prefix)
                msgs = [r.getMessage() for r in logs.records]
                self.assertTrue(msgs[0].startswith(f"{prefix}Failed to load optimizer state from {opt_path}: "))
                self.assertEqual(msgs[1], f"{prefix}Starting with fresh optimizer state.{hint}")


class PerExpertTests(unittest.TestCase):
    def test_wan2_2_layout(self):
        """src/train_wan2_2.py: checkpoints/{high,low}/, the same pieces per expert."""
        experts = {"high": build(1), "low": build(3)}
        with tempfile.TemporaryDirectory() as d:
            for e, src in experts.items():
                save(checkpoint.save, src, d, make_config(), subdir=os.path.join("checkpoints", e))
            roots = {e: os.path.join(d, "checkpoints", e) for e in experts}
            config = make_config()
            dst = {"high": build(5), "low": build(6)}
            with self.assertLogs(LOG, level="INFO") as logs:
                self.assertTrue(checkpoint.resume_dir(config, d, roots["high"]))
                for e in experts:
                    checkpoint.load_adapters(dst[e].model, roots[e], config, LOG, prefix=f"[{e}] ")
                    checkpoint.load_optimizer(dst[e].opt, roots[e], LOG, prefix=f"[{e}] ",
                                     state_device=torch.device("cpu"))
                checkpoint.load_scaler(GradScaler(enabled=False), roots["high"], False, LOG)
                step = checkpoint.load_training_state(roots["high"], "cpu", LOG)
                for e in experts:
                    checkpoint.load_ema(dst[e].model, dst[e].ema, roots[e], dst[e].lora, LOG, prefix=f"[{e}] ")
            self.assertEqual(config.resume_from, os.path.join(d, "checkpoints"))
            self.assertEqual(step, (7, 3))
            for e, src in experts.items():
                for adapter in ("default", "old"):
                    same(self, lora_weights(src.model, adapter), lora_weights(dst[e].model, adapter), f"{e}/{adapter}")
                same(self, src.opt.state_dict(), dst[e].opt.state_dict(), f"{e}/optimizer")
                same(self, src.ema.ema_parameters, dst[e].ema.ema_parameters, f"{e}/ema")
            msgs = [r.getMessage() for r in logs.records]
            self.assertEqual(msgs, [
                f"[high] Loaded current policy from {roots['high']}/lora_current_policy",
                f"[high] Loaded old policy from {roots['high']}/lora_old_policy/old",
                "[high] Loaded optimizer state (moved to cpu)",
                f"[low] Loaded current policy from {roots['low']}/lora_current_policy",
                f"[low] Loaded old policy from {roots['low']}/lora_old_policy/old",
                "[low] Loaded optimizer state (moved to cpu)",
                "Loaded global_step: 7, epoch: 3",
                f"[high] Loaded EMA policy from {roots['high']}/lora_ema_policy",
                f"[low] Loaded EMA policy from {roots['low']}/lora_ema_policy",
            ])

    def test_missing_ema_messages(self):
        src = build(1)
        with tempfile.TemporaryDirectory() as d:
            with self.assertLogs(LOG, level="INFO") as logs:
                checkpoint.load_ema(src.model, src.ema, d, src.lora, LOG)
                checkpoint.load_ema(src.model, src.ema, d, src.lora, LOG, prefix="[high] ")
        self.assertEqual([r.getMessage() for r in logs.records], [
            "No EMA checkpoint found, EMA will start from current policy weights",
            "[high] No EMA checkpoint found, EMA starts from current policy",
        ])


class WandbTests(unittest.TestCase):
    class FakeWandb:
        def __init__(self, fail_resume=False, step=0):
            self.calls, self.run, self.fail_resume, self.step = [], None, fail_resume, step

        def init(self, **kwargs):
            self.calls.append(kwargs)
            if "id" in kwargs and self.fail_resume:
                raise RuntimeError("no such run")
            self.run = types.SimpleNamespace(id=kwargs.get("id", "fresh-id"), step=self.step)

    def _kwargs(self, config, d):
        return dict(entity="ent", project="proj", name="run", config=config.to_dict(), dir=d)

    def test_resume_new_and_fallback(self):
        config = make_config()
        for run_id, fail, want_calls, want_id, want_msgs in (
            ("abc", False, [dict(id="abc", resume="allow")], "abc", ["Resumed wandb run with id: abc"]),
            (None, False, [{}], "fresh-id", ["Created new wandb run with id: fresh-id"]),
            ("abc", True, [dict(id="abc", resume="allow"), {}], "fresh-id",
             ["Failed to resume wandb run abc: no such run", "Creating new wandb run instead...",
              "Created new wandb run with id: fresh-id"]),
        ):
            with tempfile.TemporaryDirectory() as d:
                wb = self.FakeWandb(fail_resume=fail)
                with self.assertLogs(LOG, level="INFO") as logs:
                    wandb_run.start_wandb(wb, config, d, run_id, LOG, "ent")
                self.assertEqual(wb.calls, [{**self._kwargs(config, d), **c} for c in want_calls])
                self.assertEqual([r.getMessage() for r in logs.records], want_msgs)
                with open(os.path.join(d, "wandb_run_id.txt")) as f:
                    self.assertEqual(f.read(), want_id)
                self.assertEqual(wandb_run.read_wandb_run_id(d, os.path.join(d, "checkpoints")), want_id)

    def test_sync_global_step(self):
        wb = self.FakeWandb(step=12)
        wb.init()
        with self.assertLogs(LOG, level="INFO") as logs:
            self.assertEqual(wandb_run.sync_global_step(7, wb, 0, 1, "cpu", LOG), 12)
        self.assertEqual(logs.records[0].getMessage(),
                         "wandb counter (12) ahead of checkpoint global_step (7); advancing to match")
        self.assertEqual(wandb_run.sync_global_step(20, wb, 0, 1, "cpu", LOG), 20)
        self.assertEqual(wandb_run.sync_global_step(7, wb, 1, 1, "cpu", LOG), 7)     # only rank 0 reads wandb
        self.assertEqual(wandb_run.wandb_step(7), 7)                                  # no starting_step: no offset


class FakeWandbServer:
    """A W&B run as the 0.31.0 SDK + server treat it (checked live on 2026-10-24 in project
    refactor_check): one row per step; logs at the pending step merge into its row; a larger step
    commits the pending row; a log below the pending step or at/below a step the server already holds
    is dropped. wandb.init(resume=...) sets run.starting_step = last held step + 1 and leaves run.step
    (the local counter) at 0."""

    def __init__(self):
        self.rows = {}        # step -> (job, merged dict)
        self.dropped = []     # (job, step, data)
        self.logged = []      # (job, step) in call order
        self.run = None

    def init(self, job):
        held = max(self.rows) + 1 if self.rows else 0
        self.run = types.SimpleNamespace(starting_step=held, step=0, job=job, pending=None)

    def log(self, data, step=None, commit=None):
        run = self.run
        self.logged.append((run.job, step))
        if (self.rows and step <= max(self.rows)) or (run.pending is not None and step < run.pending[0]):
            self.dropped.append((run.job, step, dict(data)))
            return
        if run.pending is not None and step > run.pending[0]:
            self._commit()
        if run.pending is None:
            run.pending = (step, {})
        run.pending[1].update(data)

    def _commit(self):
        step, data = self.run.pending
        self.rows[step] = (self.run.job, data)
        self.run.pending = None

    def finish(self):
        if self.run.pending is not None:
            self._commit()
        self.run = None


class WandbStepOffsetTests(unittest.TestCase):
    """A job logs at global_step + an offset, so W&B never drops a log below the last step it holds."""

    def tearDown(self):
        wandb_run._WANDB_STEP_OFFSET = 0

    def test_offset_rules(self):
        wb = FakeWandbServer()
        wb.rows = {s: ("old", {}) for s in range(12)}
        wb.init("job")
        with self.assertLogs(LOG, level="INFO") as logs:
            self.assertEqual(wandb_run.sync_global_step(10, wb, 0, 1, "cpu", LOG), 10)
        self.assertEqual(wandb_run.wandb_step(10), 12)
        self.assertIn("logs at global_step + 2 (training global_step 10 unchanged)",
                      logs.records[0].getMessage())
        self.assertEqual(wandb_run.sync_global_step(15, wb, 0, 1, "cpu", LOG), 15)    # ahead: no offset
        self.assertEqual(wandb_run.wandb_step(15), 15)
        wandb_run.sync_global_step(10, wb, 0, 1, "cpu", LOG)
        # a job without a checkpoint still gets the offset; global_step untouched
        self.assertEqual(wandb_run.sync_global_step(10, wb, 0, 1, "cpu", LOG, resumed=False), 10)
        self.assertEqual(wandb_run.wandb_step(10), 12)
        fresh = FakeWandbServer(); fresh.init("new-run")                      # brand-new run: offset 0
        self.assertEqual(wandb_run.sync_global_step(0, fresh, 0, 1, "cpu", LOG, resumed=False), 0)
        self.assertEqual(wandb_run.wandb_step(0), 0)                                   # reset by every call
        wandb_run.sync_global_step(10, wb, 1, 1, "cpu", LOG)                          # non-zero rank: no offset
        self.assertEqual(wandb_run.wandb_step(10), 10)


class PromptHistoryTests(unittest.TestCase):
    """The stat tracker's prompt history (trained_prompt_num): stable keys, legacy histories
    discarded, loaded on every resume (auto-resume included)."""

    def test_prompt_key_is_stable_across_processes(self):
        import subprocess
        import sys
        code = ("import sys; sys.path.insert(0, %r); from src.advantage.utils import prompt_key; "
                "print(prompt_key('a cat on a skateboard'), hash('a cat on a skateboard'))"
                % os.getcwd())
        outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                               env={**os.environ, "PYTHONHASHSEED": seed}).stdout.split()
                for seed in ("1", "2")]
        self.assertEqual(outs[0][0], outs[1][0])                  # prompt_key: same in both processes
        self.assertEqual(outs[0][0], prompt_key("a cat on a skateboard"))
        self.assertNotEqual(outs[0][1], outs[1][1])               # hash(): salted per process

    def test_history_records_prompt_keys(self):
        t = PromptHistory()
        t.add(["a", "b", "a"])
        self.assertEqual(t.history_prompts, {prompt_key("a"), prompt_key("b")})

    def test_round_trip_and_legacy_int_history(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "training_state.pt")
            torch.save({"global_step": 3, "stat_tracker_history_prompts": tracker().history_prompts}, path)
            t = PromptHistory()
            with self.assertLogs(LOG, level="INFO") as logs:
                checkpoint.load_stat_tracker_history(t, d, LOG)
            self.assertEqual(t.history_prompts, tracker().history_prompts)
            self.assertEqual(logs.records[0].getMessage(), "Loaded stat_tracker with 3 historical prompts")

            torch.save({"global_step": 3, "stat_tracker_history_prompts": {hash("p0"), hash("p1")}}, path)
            t = PromptHistory()
            t.add(["x"])
            with self.assertLogs(LOG, level="WARNING") as logs:
                checkpoint.load_stat_tracker_history(t, d, LOG)
            self.assertEqual(t.history_prompts, {prompt_key("x")})   # this job's prompts kept, legacy dropped
            self.assertIn("Discarding the stat_tracker prompt history", logs.records[0].getMessage())
            self.assertIn("2 of 2 keys are per-process salted hash() ints", logs.records[0].getMessage())

            torch.save({"global_step": 3}, path)                       # no history saved: untouched
            checkpoint.load_stat_tracker_history(t, d, LOG)
            self.assertEqual(t.history_prompts, {prompt_key("x")})

    def test_trainers_load_history_inside_the_resume_block(self):
        """The load must run after checkpoint.resume_dir (which sets config.resume_from on an
        auto-resume): Run.__init__ calls _resume() only inside ``if checkpoint.resume_dir(...)``, and each run's
        _resume is the one place that loads the history."""
        import ast
        tree = ast.parse(open("src/engine/run.py").read())
        blocks = [n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
                  and getattr(n.test.func, "attr", None) == "resume_dir"]
        self.assertEqual(len(blocks), 1)
        inside = [getattr(c.func, "attr", None) for stmt in blocks[0].body for c in ast.walk(stmt)
                  if isinstance(c, ast.Call)]
        self.assertIn("_resume", inside)
        for path in ("src/engine/run.py", "src/engine/run_experts.py"):
            with self.subTest(run=path):
                tree = ast.parse(open(path).read())
                calls = [(f.name, n) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                         for n in ast.walk(f) if isinstance(n, ast.Call)
                         and getattr(n.func, "attr", None) == "load_stat_tracker_history"]
                self.assertEqual([name for name, _ in calls], ["_resume"])


if __name__ == "__main__":
    unittest.main()
