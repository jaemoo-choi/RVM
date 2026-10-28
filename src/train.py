# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The training skeleton every pipeline runs: RL post-training of a flow / diffusion model with LoRA
adapters. A pipeline is a Run class (its operations); the skeleton below is the same for all of them.

    [1] sample  a rollout per prompt with the old adapter
    [2] reward  score the rollouts (src/reward/rewards.py)
    [3] signal  rewards -> per-sample advantages / the objective's direction (src/advantage/advantages.py)
    [4] train   for each micro-batch and timestep: loss (src/loss/step.py) -> backward;
                every accum_steps backwards: optimizer step
    [5] update  old = d·old + (1 − d)·current; save

Pipelines (``main(trainer=...)``; each entrypoint picks one):

    velocity  src/train.py        Run          (src/engine/run.py)          sd3 / flux / wan2_1 / skyreels
    wan2_2    src/train_wan2_2.py ExpertsRun   (src/engine/run_experts.py)  Wan2.2-T2V-A14B, two experts
"""
import os
import sys

import wandb

# Launched as a script: put the repo root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import logging  # noqa: E402

from absl import app, flags  # noqa: E402
from ml_collections import config_flags  # noqa: E402

from config.paths import WANDB_ENTITY  # noqa: E402
from config.validate import validate_train_config  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def run_class(trainer):
    """The Run class of a pipeline."""
    if trainer == "wan2_2":
        from src.engine.run_experts import ExpertsRun
        return ExpertsRun
    from src.engine.run import Run
    return Run


def train(run):
    """The epoch skeleton."""
    for epoch in range(run.first_epoch, run.config.num_epochs):
        run.start_epoch(epoch)

        # ========== [1] Sample: rollouts with the old adapter ==========
        rollouts, last_images, last_prompts = run.sample(epoch)
        # ========== [2] Reward ==========
        samples = run.score(epoch, rollouts, last_images, last_prompts)
        del rollouts, last_images, last_prompts
        epoch_log = run.reward_log(epoch, samples)

        # ========== [3] Signal: rewards -> per-sample advantages ==========
        samples = run.signal(epoch, samples, epoch_log)

        # ========== [4] Train: gradient steps on the default adapter ==========
        run.train_epoch(epoch, samples)

        # ========== [5] Update old (checkpoints are saved at the next epoch's start) ==========
        run.update_old()

    run.finish()


def main(_, trainer="velocity"):
    config = FLAGS.config
    validate_train_config(config, trainer)
    run = run_class(trainer)(config, logger=logger, wandb=wandb, entity=WANDB_ENTITY)
    train(run)


if __name__ == "__main__":
    app.run(main)
