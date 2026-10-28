"""Which LoRA adapter a forward of a PEFT model runs on (moved from the old src/rl/loss.py).

    policy(module)             # the trainable adapter ("default"), left active for the backward
    with use(module, "old"):   # one block as another adapter (the "old" rollout policy)
    with base(module):         # one block on the frozen base model (adapters disabled)
"""
from __future__ import annotations

import contextlib

from src.models.pipelines.lora_ctx import disable_lora

#     policy(module)             # the trainable adapter ("default"), left active for the backward
#     with use(module, "old"):   # one block as another adapter (the "old" rollout policy)
#     with base(module):         # one block on the frozen base model (adapters disabled)
# ``use`` restores the adapter that was active before the block. PEFT's ``set_adapter`` also sets
# requires_grad (True for the named adapter only): ``policy`` is not scoped, so the trainable adapter
# stays active until the caller's backward, and no no-grad forward of the trainable adapter runs ahead
# of its gradient forward (autocast would cache grad-less casts).

def active(module):
    """Name of the active adapter (PeftModel keeps a str, tuner layers a one-element list)."""
    name = module.active_adapter
    if isinstance(name, (list, tuple)):
        return name[0] if len(name) == 1 else list(name)
    return name


def policy(module):
    """Make the trainable adapter ("default") active and leave it active."""
    module.set_adapter("default")


@contextlib.contextmanager
def use(module, role):
    """Run the block with adapter ``role`` active; the previous adapter is restored on exit."""
    prev = active(module)
    module.set_adapter(role)
    try:
        yield module
    finally:
        module.set_adapter(prev)


def base(module, enabled=True):
    """Run the block on the base model (every adapter layer disabled); a no-op if not ``enabled``.

    A PeftModel (and the test stubs) use their own ``disable_adapter()``, as the loss always did;
    any other module gets the layer-level switch of src/models/pipelines/lora_ctx.py. Both toggle
    the same LoRA layers and give the active adapter its requires_grad back on exit.
    """
    if not enabled:
        return contextlib.nullcontext()
    if hasattr(module, "disable_adapter"):
        return module.disable_adapter()
    return disable_lora(module)
