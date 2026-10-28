"""Checkpoint save / resume of the trainers. The wandb run is src/engine/wandb_run.py.

On-disk layout (unchanged, so existing run folders resume; Wan2.2 keeps one checkpoint per expert
in checkpoints/{high,low}/):

    <save_dir>/wandb_run_id.txt                     the wandb run every resubmit logs to
    <save_dir>/checkpoints/                         rolling checkpoint, what a resume reads
        lora_current_policy/                        PEFT save_pretrained of every adapter: "default"
                                                    at the root, the others in old/
        lora_old_policy/                            the same save; the old adapter is its old/ (B1)
        lora_ema_policy/                            "default" holding the EMA weights (train.ema)
        optimizer.pt  scaler.pt  training_state.pt  config.json
    <save_dir>/permanent/epoch_<N>/checkpoints/     snapshots, same layout

    save / save_ckpt          crash-safe write of one checkpoint (staging dir, then renames)
    resume                    a velocity trainer's resume: load_adapters, load_optimizer,
                              load_scaler, load_training_state (src/train_wan2_2.py calls the
                              pieces per expert); load_ema, load_stat_tracker_history
    resume_dir                whether the run resumes (sets config.resume_from)

The log lines are the ones
the trainers always printed (users grep them); a per-expert caller passes prefix="[high] " and gets
the lines of src/train_wan2_2.py.
"""
from __future__ import annotations

import os
import shutil
import zipfile

import torch


def _is_main(rank):
    return rank == 0


def _is_enospc(exc):
    """True for a 'no space left on device' failure (OSError errno 28, or the
    safetensors Rust error that only carries the text 'os error 28')."""
    import errno
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return True
    return "os error 28" in str(exc) or "No space left on device" in str(exc)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(
    logger, wandb, save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler, epoch=None, stat_tracker=None, ckpt_subdir="checkpoints"
):
    # ckpt_subdir selects where, under save_dir, the checkpoint is written.
    # Default "checkpoints" is the rolling checkpoint (overwritten each epoch,
    # used for resume). Pass e.g. "permanent/epoch_<N>/checkpoints" to write a
    # permanent epoch snapshot that is never overwritten.
    #
    # Crash-safe write: a shared storage allocation can hit 0 bytes
    # for minutes when other users write, and a save cut mid-file left a mixed /
    # truncated checkpoint that the next job could not resume from. Every file is
    # now first written into <save_root>/.staging; only after ALL of them succeed
    # are they moved into place with os.replace (a rename, needs no free space), so
    # the previous checkpoint stays intact until the new one is complete. A
    # no-space failure while staging is retried (SAVE_ENOSPC_RETRIES x
    # SAVE_ENOSPC_WAIT_S, default 15 x 60 s, inside the 30-min NCCL timeout).
    if _is_main(rank):
        import json
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


save_ckpt = save    # the name src/utils (and its callers) always used


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def resume_dir(config, log_dir, marker_dir):
    """True when the run resumes: config.resume_from is set, or ``marker_dir`` (the rolling
    checkpoint; Wan2.2: its high-expert dir) holds a saved current policy. Sets
    config.resume_from to <log_dir>/checkpoints."""
    marker = os.path.join(marker_dir, "lora_current_policy", "adapter_model.safetensors")
    if config.resume_from != "" or os.path.exists(marker):
        try:
            config.resume_from = os.path.join(log_dir, "checkpoints")
        except:  # noqa: E722 (a locked config keeps the resume_from it was given)
            assert config.resume_from
        return True
    return False


def old_policy_path(ckpt_dir, config):
    """Folder the "old" adapter is loaded from. PEFT's save_pretrained writes the adapter named
    "default" at the root of lora_old_policy/ and the real old adapter in its old/ subfolder; the
    root used to be loaded, resetting old <- current at every resume (B1, fixed 2026-10-23;
    train.fix_resume_old=False restores the old behaviour)."""
    path = os.path.join(ckpt_dir, "lora_old_policy")
    if (getattr(config.train, "fix_resume_old", True)
            and os.path.exists(os.path.join(path, "old", "adapter_model.safetensors"))):
        path = os.path.join(path, "old")
    return path


def load_adapters(module, ckpt_dir, config, logger, *, prefix=""):
    """Load the current ("default", trainable) and old (frozen; B1 rule) adapters."""
    lora_current_path = os.path.join(ckpt_dir, "lora_current_policy")
    if os.path.exists(lora_current_path):
        module.load_adapter(lora_current_path, adapter_name="default", is_trainable=True)
        logger.info(f"{prefix}Loaded current policy from {lora_current_path}")

    lora_old_path = old_policy_path(ckpt_dir, config)
    if os.path.exists(lora_old_path):
        module.load_adapter(lora_old_path, adapter_name="old", is_trainable=False)
        logger.info(f"{prefix}Loaded old policy from {lora_old_path}")


def load_optimizer(optimizer, ckpt_dir, logger, *, prefix="", state_device=None):
    """optimizer.pt -> ``optimizer``, read to CPU first (no OOM). ``state_device``: move the state
    tensors there afterwards (Wan2.2: the experts sit on CPU during its resume, but Adam needs the
    state on the parameters' GPU at step time). A corrupt file leaves a fresh state."""
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        try:
            # Load to CPU first to avoid OOM, optimizer will handle device placement
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            if state_device is None:
                logger.info(f"{prefix}Loaded optimizer state")
            else:
                for param, state in optimizer.state.items():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.device != state_device:
                            state[k] = v.to(state_device)
                logger.info(f"{prefix}Loaded optimizer state (moved to {state_device})")
        except (RuntimeError, EOFError, zipfile.BadZipFile) as e:
            logger.warning(f"{prefix}Failed to load optimizer state from {opt_path}: {e}")
            # src/train.py's wording; the per-expert trainer printed the first sentence only.
            logger.warning(f"{prefix}Starting with fresh optimizer state."
                           + ("" if prefix else " Consider removing corrupted checkpoint."))


def load_scaler(scaler, ckpt_dir, enable_amp, logger):
    """scaler.pt -> ``scaler`` (mixed precision only); a corrupt file leaves a fresh scaler."""
    scaler_path = os.path.join(ckpt_dir, "scaler.pt")
    if os.path.exists(scaler_path) and enable_amp:
        try:
            scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))
            logger.info("Loaded scaler state")
        except (RuntimeError, EOFError, zipfile.BadZipFile) as e:
            logger.warning(f"Failed to load scaler state from {scaler_path}: {e}")
            logger.warning("Starting with fresh scaler state")


def load_training_state(ckpt_dir, device, logger):
    """(global_step, epoch) from training_state.pt; (0, 0) without one. The wandb run id it also
    holds is read before wandb.init (read_wandb_run_id)."""
    global_step, first_epoch = 0, 0
    training_state_path = os.path.join(ckpt_dir, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location=device)
        global_step = training_state["global_step"]
        first_epoch = training_state.get("epoch", 0)
        logger.info(f"Loaded global_step: {global_step}, epoch: {first_epoch}")
    return global_step, first_epoch


def resume(module, optimizer, scaler, config, ckpt_dir, logger, *, device, enable_amp):
    """Resume a velocity trainer from ``ckpt_dir``: adapters (current, old), optimizer, scaler;
    returns (global_step, epoch) from training_state.pt ((0, 0) without one)."""
    load_adapters(module, ckpt_dir, config, logger)
    load_optimizer(optimizer, ckpt_dir, logger)
    load_scaler(scaler, ckpt_dir, enable_amp, logger)
    return load_training_state(ckpt_dir, device, logger)


def load_ema(module, ema, ckpt_dir, lora_config, logger, *, prefix=""):
    """lora_ema_policy/ -> ``ema.ema_parameters``, read through a temporary adapter. Without the
    folder the EMA starts from the current policy weights."""
    lora_ema_path = os.path.join(ckpt_dir, "lora_ema_policy")
    if os.path.exists(lora_ema_path):
        # Create a temporary adapter to load EMA weights
        module.add_adapter("temp_ema_load", lora_config)
        module.load_adapter(lora_ema_path, adapter_name="temp_ema_load", is_trainable=False)
        module.set_adapter("temp_ema_load")

        # Get the parameters from the temp_ema_load adapter
        temp_ema_params = [p for p in module.parameters() if p.requires_grad]

        # Copy EMA weights to the EMA wrapper's ema_parameters (correct attribute name)
        for ema_param, loaded_param in zip(ema.ema_parameters, temp_ema_params):
            ema_param.data.copy_(loaded_param.data)

        # Clean up: switch back to default adapter and delete temp adapter
        module.set_adapter("default")
        module.delete_adapter("temp_ema_load")

        logger.info(f"{prefix}Loaded EMA policy from {lora_ema_path}")
    else:
        # src/train.py's wording; the per-expert trainer printed the short one.
        logger.info(f"{prefix}No EMA checkpoint found, EMA starts from current policy" if prefix
                    else "No EMA checkpoint found, EMA will start from current policy weights")


def load_stat_tracker_history(stat_tracker, ckpt_dir, logger):
    """The stat tracker's prompt history from training_state.pt (feeds the trained_prompt_num log;
    the advantages never read it). Keys are src/advantage/utils.py:prompt_key strings; a history saved before
    2026-10-24 holds Python ``hash(str)`` ints, salted per process, so it cannot match a prompt of
    this process and is discarded (logged): trained_prompt_num restarts counting from this job."""
    training_state_path = os.path.join(ckpt_dir, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        if "stat_tracker_history_prompts" in training_state:
            history = training_state["stat_tracker_history_prompts"]
            n_salted = sum(1 for k in history if not isinstance(k, str))
            if n_salted:
                logger.warning(
                    f"Discarding the stat_tracker prompt history of {training_state_path}: {n_salted} of "
                    f"{len(history)} keys are per-process salted hash() ints (saved before the stable "
                    f"prompt_key), which match no prompt after a restart; trained_prompt_num restarts "
                    f"from this job's prompts")
                return
            stat_tracker.history_prompts = set(history)
            logger.info(f"Loaded stat_tracker with {len(stat_tracker.history_prompts)} historical prompts")
