"""Disable LoRA on any module, without needing the PeftModel wrapper.

``PeftModel.disable_adapter()`` only exists on the wrapper object, and most of
this repo never keeps one: ``get_peft_model`` mutates the transformer IN PLACE
(its attention/FFN submodules become ``LoraLinear``), and the wrapper is handed
to DDP while ``pipeline.transformer`` keeps pointing at the inner module. So the
sampling pipelines -- which call ``pipeline.transformer`` / ``transformer_2``
directly, and for Wan2.2 alternate between two experts -- have no wrapper to ask.

PEFT exposes the real switch one level down: every ``BaseTunerLayer`` has
``enable_adapters(bool)``. Toggling it on the layers inside whatever module is
about to be called gives the same effect as ``disable_adapter()`` and works for
the in-place case, for either A14B expert, and for a real PeftModel alike.

Used for the CFG negative-prompt (uncond) velocity: with
``NP_BASE_VELOCITY=1`` (the default) the uncond comes from the PRETRAINED base
model in the rollout pipeline, the training loss, the reference anchor and
evaluation, so the guided field that is trained is the one that is sampled.
"""

from __future__ import annotations

import contextlib


@contextlib.contextmanager
def disable_lora(module):
    """Temporarily disable every PEFT adapter layer inside ``module``.

    No-op (and no import cost) when the module holds no adapter layers, e.g. the
    frozen non-trainable expert, or an eval run loaded without LoRA.
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:  # peft absent -> nothing to disable
        yield
        return

    layers = [m for m in module.modules() if isinstance(m, BaseTunerLayer)]
    if not layers:
        yield
        return

    previously_disabled = [getattr(m, "_disable_adapters", False) for m in layers]
    try:
        for m in layers:
            m.enable_adapters(False)
        yield
    finally:
        for m, was_disabled in zip(layers, previously_disabled):
            m.enable_adapters(not was_disabled)


def maybe_disable_lora(module, disable: bool):
    """``disable_lora(module)`` when ``disable`` else a null context."""
    return disable_lora(module) if disable else contextlib.nullcontext()
