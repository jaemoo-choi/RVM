"""Startup check of a training config, run by the trainers before any model is loaded.

A config that asks for something the code does not implement (an unknown objective, an ELBO / KL
variant, the dynamic-degree auxiliary loss, time_fraction < 1, single-expert Wan2.2-A14B training) is
rejected here with the field named, instead of training something else or failing minutes into the
job. A field the config does not carry passes.

    trainer   "velocity"  src/train.py
              "wan2_2"    src/train_wan2_2.py
"""
from __future__ import annotations

from src.loss.objectives import check_method

_NOT_IMPLEMENTED = "not implemented in this code"


def _fields(tr):
    return list(tr.keys()) if hasattr(tr, "keys") else list(vars(tr))


def _train_micro_batches(config):
    """Micro-batches per epoch and rank in the training phase: the samples this rank holds
    (sample.train_batch_size x sample.num_batches_per_epoch) // train.batch_size. None when the config
    lacks a field (the check is skipped)."""
    sample, tr = getattr(config, "sample", None), config.train
    try:
        per_rank = int(sample.train_batch_size) * int(sample.num_batches_per_epoch)
        batch_size = int(tr.batch_size)
    except (AttributeError, KeyError, TypeError):
        return None
    return per_rank // batch_size


def _partial_accum_error(config):
    """The trainer steps the optimizer every gradient_accumulation_steps micro-batches, so
    num_batches % gradient_accumulation_steps != 0 would leave gradients accumulated after the epoch's
    last step."""
    num_batches = _train_micro_batches(config)
    try:
        grad_accum = int(config.train.gradient_accumulation_steps)
    except (AttributeError, KeyError, TypeError):
        return None
    if num_batches is None or grad_accum <= 0 or num_batches % grad_accum == 0:
        return None
    return (f"train.gradient_accumulation_steps={grad_accum} does not divide the {num_batches} micro-batches "
            f"per epoch and rank (sample.train_batch_size x sample.num_batches_per_epoch // train.batch_size): "
            f"the {num_batches % grad_accum} leftover micro-batches' gradients would carry into the next "
            f"epoch's first optimizer step")


def validate_train_config(config, trainer: str, world_size: int = 1) -> None:
    """Raise ValueError listing every config field the trainers do not implement."""
    tr = config.train
    errors = []

    try:
        check_method(getattr(tr, "method", None))
    except ValueError as e:
        errors.append(str(e))

    elbo = getattr(tr, "elbo", "adaptive")
    if elbo != "adaptive":
        errors.append(f"train.elbo={elbo!r}: only the adaptive ELBO weighting is implemented (ELBO=adaptive)")

    kl_method = getattr(tr, "kl_method", "girsanov")
    if kl_method != "girsanov":
        errors.append(f"train.kl_method={kl_method!r}: only girsanov (the β‖v_θ − v_ref‖² term) is implemented")

    for key in sorted(k for k in _fields(tr) if k.startswith("dynamic_loss") and "lambda" in k):
        value = float(getattr(tr, key) or 0.0)
        if value != 0.0:
            errors.append(f"train.{key}={value}: the dynamic-degree auxiliary loss is {_NOT_IMPLEMENTED}")

    time_fraction = float(getattr(tr, "time_fraction", 1.0))
    if time_fraction != 1.0:
        errors.append(f"train.time_fraction={time_fraction}: training covers the whole trajectory "
                      f"(TIME_FRACTION=1); time_fraction < 1 is {_NOT_IMPLEMENTED}")

    if trainer == "wan2_2" and bool(getattr(tr, "whiten_rewards", False)):
        errors.append("train.whiten_rewards=True: src/train_wan2_2.py computes the per-prompt advantages of the "
                      "combined reward \"avg\" without per-component whitening and would silently ignore it; "
                      "whitening exists only in src/train.py")

    if trainer == "velocity":
        err = _partial_accum_error(config)
        if err:
            errors.append(err)

    base_model = getattr(config, "base_model", None)
    if base_model == "wan2_2_a14b_t2v" and trainer != "wan2_2":
        errors.append("base_model='wan2_2_a14b_t2v' trains only with src/train_wan2_2.py (both experts, "
                      "the *_def_alt* configs)")

    if errors:
        raise ValueError("training config rejected:\n  - " + "\n  - ".join(errors))
