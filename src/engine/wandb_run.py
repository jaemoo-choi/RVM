"""The wandb run of a trainer across resubmitted jobs.

    <save_dir>/wandb_run_id.txt   the wandb run every resubmit logs to

    read_wandb_run_id / start_wandb    the run id (save_dir file, else the checkpoint's) and wandb.init
    sync_global_step / wandb_step      a resumed job's global_step on every rank, and the per-job W&B step
                                       offset that keeps the logged steps monotonic
    install_wandb_shutdown             finish the run cleanly when a job exits (wall-time exit 86 included)
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _is_main(rank):
    return rank == 0


WANDB_RUN_ID_FILE = "wandb_run_id.txt"


def read_wandb_run_id(log_dir, ckpt_dir):
    """The run's wandb id: <log_dir>/wandb_run_id.txt, a lightweight file written right after
    wandb.init (so it persists even if the job dies before the first checkpoint), else the id in
    <ckpt_dir>/training_state.pt (older runs); None for a new run."""
    wandb_run_id = None
    wandb_run_id_file = os.path.join(log_dir, WANDB_RUN_ID_FILE)
    training_state_path = os.path.join(ckpt_dir, "training_state.pt")
    if os.path.exists(wandb_run_id_file):
        with open(wandb_run_id_file) as f:
            wandb_run_id = f.read().strip() or None
    elif os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        wandb_run_id = training_state.get("wandb_run_id")
    return wandb_run_id


def start_wandb(wandb, config, log_dir, wandb_run_id, logger, entity):
    """Rank 0: resume wandb run ``wandb_run_id`` (resume="allow"), or create a new run when there is
    none or the resume fails, then persist the id to <log_dir>/wandb_run_id.txt so later jobs
    (including those that fail before saving a checkpoint) log to the same page."""
    kwargs = dict(entity=entity, project=(getattr(config, "project", None) or config.name),
                  name=config.run_name, config=config.to_dict(), dir=log_dir)
    if wandb_run_id:
        # Resume the existing wandb run so all resubmits log to the same page.
        try:
            wandb.init(id=wandb_run_id, resume="allow", **kwargs)
            logger.info(f"Resumed wandb run with id: {wandb_run_id}")
        except Exception as e:
            logger.warning(f"Failed to resume wandb run {wandb_run_id}: {e}")
            logger.info("Creating new wandb run instead...")
            wandb.init(**kwargs)
            logger.info(f"Created new wandb run with id: {wandb.run.id}")
    else:
        # First run for this config — create a fresh wandb run.
        wandb.init(**kwargs)
        logger.info(f"Created new wandb run with id: {wandb.run.id}")
    with open(os.path.join(log_dir, WANDB_RUN_ID_FILE), "w") as f:
        f.write(wandb.run.id)


# The W&B log step of this job: global_step + an offset fixed once per job by sync_global_step.
_WANDB_STEP_OFFSET = 0


def wandb_step(global_step):
    """The ``step=`` of every wandb.log of the trainers: ``global_step`` plus this job's offset
    (sync_global_step). Training (return_decay, the EMA step, checkpoints) reads global_step only."""
    return int(global_step) + _WANDB_STEP_OFFSET


def sync_global_step(global_step, wandb, rank, world_size, device, logger, *, resumed=True):
    """The training global_step this job starts at, and its W&B log-step offset; both trainers call
    it once, before the epoch loop.

    global_step (unchanged behaviour): on resume (``resumed``) rank 0 advances it to wandb's local
    counter ``run.step`` when that is ahead and broadcasts it; a job that does not resume keeps its
    global_step (no collective). In wandb 0.31.0 ``run.step`` is 0 right after a resume, so this
    never moves global_step; it is kept as it was so training (return_decay, the EMA step) is
    unchanged.

    W&B step offset: W&B keeps one row per step and silently drops a log below the last step the
    server holds. A job that died mid-epoch logged past its checkpoint, so a resumed job logging at
    global_step would lose its first logs. Rank 0 (the only rank that logs) fixes
    offset = max(0, wandb.run.starting_step - global_step) for the job; ``starting_step`` is the
    step after the last one the server holds (0 for a new run). Every wandb.log goes through
    wandb_step(global_step), so W&B steps stay monotonic across jobs and nothing is persisted.
    """
    global _WANDB_STEP_OFFSET
    _WANDB_STEP_OFFSET = 0
    # The offset applies to every job, resumed or not (a job without a checkpoint may still find logged steps).
    if resumed and _is_main(rank):
        if wandb.run is not None:
            try:
                _wb_step = wandb.run.step  # next expected step
            except AttributeError:
                _wb_step = global_step
            if global_step < _wb_step:
                logger.info(
                    f"wandb counter ({_wb_step}) ahead of checkpoint "
                    f"global_step ({global_step}); advancing to match"
                )
                global_step = _wb_step
    if resumed and world_size > 1:
        _gs = torch.tensor([global_step], device=device, dtype=torch.long)
        dist.broadcast(_gs, src=0)
        global_step = int(_gs.item())
    if _is_main(rank) and wandb.run is not None:
        _server_next = int(getattr(wandb.run, "starting_step", 0) or 0)
        _WANDB_STEP_OFFSET = max(0, _server_next - int(global_step))
        if _WANDB_STEP_OFFSET:
            logger.info(
                f"wandb run holds steps up to {_server_next - 1}: this job logs at global_step + "
                f"{_WANDB_STEP_OFFSET} (training global_step {global_step} unchanged)"
            )
    return global_step


def install_wandb_shutdown(wandb):
    """Rank 0: on SIGTERM / SIGINT (SLURM wall, scancel, resubmit) finish the wandb run cleanly — it
    reopens as "running", not "crashed", on the next allocation — then re-raise the signal, so SLURM
    still records the job as killed by it (not an exit-0 COMPLETED)."""
    import signal

    def _wandb_clean_shutdown(signum, _frame):
        try:
            wandb.finish(exit_code=0, quiet=True)
        except Exception:
            pass
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _wandb_clean_shutdown)
        except Exception:
            pass
