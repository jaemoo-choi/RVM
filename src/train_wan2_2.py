# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entrypoint for the two-expert (high-noise / low-noise MoE) model, Wan2.2-T2V-A14B:

    torchrun --nproc_per_node=N src/train_wan2_2.py --config config/wan2_2_a14b_t2v.py:<name>

Both experts are trained off ONE rollout, alternating at optimizer-step granularity:

    GRADIENT_STEP_PER_EPOCH=1  ->  high, low                  (2 updates)
    GRADIENT_STEP_PER_EPOCH=2  ->  high, low, high, low       (4 updates)

* Both experts carry a LoRA during sampling (both ``old`` adapters active): the rollout is the
  joint policy.
* Both experts use the same regression target (x0 = latents_clean, t = timesteps / 1000); the
  loss is src/loss/objectives.py.

The epoch is src/train.py's skeleton; the pipeline is src/engine/run_experts.py:ExpertsRun (setup per
expert, expert paging between CPU and GPU, [4] per expert and band, checkpoints/{high,low}/).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from absl import app  # noqa: E402


def main(argv):
    from src import train

    config = train.FLAGS.config
    assert getattr(config, "base_model", None) == "wan2_2_a14b_t2v", (
        "train_wan2_2.py is the Wan2.2-A14B-T2V entrypoint. "
        f"Got base_model={getattr(config, 'base_model', None)!r}; use train.py.")
    train.main(argv, trainer="wan2_2")


if __name__ == "__main__":
    from src import train  # noqa: F401  (defines the --config flag before absl parses argv)

    app.run(main)
