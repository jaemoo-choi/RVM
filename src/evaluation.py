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

import argparse
import datetime
import os
import json
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import logging
from ml_collections import ConfigDict

import torch.distributed as dist
from peft import PeftModel

# This module lives in src/ but is launched as a script (torchrun src/evaluation.py),
# so the repo root is not on sys.path by default. Add it so `config`, `src`, and
# `src.dataloader` resolve regardless of how the script is invoked.
import sys as _sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from src.reward import multi_score
# Model utils are imported lazily to avoid pulling in unnecessary dependencies
# (e.g., flash_attn via HunyuanVideoPipeline in the image_elbo environment)
from src.dataloader import get_testloader

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


# ======================== Helper Functions ========================
def infer_model_type_from_path(checkpoint_path):
    """Infer model type from checkpoint path structure."""
    path_lower = checkpoint_path.lower()
    if "/sd3/" in path_lower or "sd3" in os.path.basename(checkpoint_path).lower():
        return "sd3"
    elif "/flux/" in path_lower or "flux" in os.path.basename(checkpoint_path).lower():
        return "flux"
    elif "/wan2_2_a14b_t2v" in path_lower or "wan2_2_a14b_t2v" in os.path.basename(checkpoint_path).lower():
        return "wan2_2_a14b_t2v"
    elif "/wan2_1/" in path_lower or "/wan/" in path_lower or "wan" in os.path.basename(checkpoint_path).lower():
        return "wan2_1"
    elif "/skyreels/" in path_lower or "skyreels" in os.path.basename(checkpoint_path).lower():
        return "skyreels_i2v"
    return None


def load_training_config(checkpoint_path):
    """Load the training config from checkpoint directory."""
    # Try to find config.json in the checkpoint path
    config_paths = [
        os.path.join(checkpoint_path, "checkpoints", "config.json"),
        # Alternating Wan2.2-A14B runs (src/train_wan2_2.py) split the
        # checkpoint into checkpoints/{high,low}/ and write config.json into
        # EACH expert dir -- there is no flat checkpoints/config.json, so the
        # paths above all miss and eval used to die with FileNotFoundError
        # before it ever reached find_dual_expert_loras. The two files are
        # written from the same config object (verified byte-identical in
        # content on the ta1_alt / vlmta_alt / ta1_alt_diffusionnft runs), so
        # either one is authoritative; prefer high for determinism.
        os.path.join(checkpoint_path, "checkpoints", "high", "config.json"),
        os.path.join(checkpoint_path, "checkpoints", "low", "config.json"),
        os.path.join(checkpoint_path, "config.json"),
        os.path.join(os.path.dirname(checkpoint_path), "config.json"),  # Check parent dir
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Convert back to ConfigDict
            return ConfigDict(config_dict)
    
    raise FileNotFoundError(
        f"Training config not found in {checkpoint_path}. "
        "Please ensure the checkpoint was saved with the updated save_ckpt function."
    )


def create_eval_config(args, model_type, checkpoint_path):
    """Load training config and override with evaluation-specific settings."""
    # Load the full training config from checkpoint
    config = load_training_config(checkpoint_path)

    # Override dataset path to use the specified dataset
    config.dataset = f"dataset/{args.dataset}"
    # I2V models need image_prompt so the test loader provides reference images.
    if args.dataset in ["geneval", "geneval_unseen_objects"]:
        config.prompt_fn = "geneval"
    elif model_type == "skyreels_i2v":
        config.prompt_fn = "image_prompt"
    else:
        config.prompt_fn = "text"
    
    # Override sample settings for evaluation if provided
    if args.num_inference_steps is not None:
        config.sample.num_steps = args.num_inference_steps
    # Custom non-uniform discretization: comma-separated descending sigmas (no
    # trailing 0). Used to sample a subset of the trained schedule, e.g. take
    # every k-th of a 16-step model's sigmas to integrate with 4/8 of the exact
    # trained timesteps (not a freshly rebuilt N-step schedule). Consumed via
    # config.sample.sigmas -> custom_sigmas in the pipeline. Sets num_steps to
    # match so time_fraction / region splits stay consistent.
    if getattr(args, "sigmas", None):
        _sig = [float(s) for s in args.sigmas.split(",") if s.strip()]
        config.sample.sigmas = _sig
        config.sample.num_steps = len(_sig)
    if args.guidance_scale is not None:
        config.sample.guidance_scale = args.guidance_scale
    if args.eval_batch_size is not None:
        config.sample.test_batch_size = args.eval_batch_size
    # Frame-count override: evaluate at a different frame count than the
    # training config.json records (e.g. base model at 81 frames vs trained
    # 53). Must satisfy the VAE temporal constraint 4k+1 (81 = 4*20+1).
    if getattr(args, "num_frames", None) is not None:
        config.frames = args.num_frames
    # SDE sampler override: when --sde_eta N is set, switch the eval sampler to
    # Euler-flow SDE with η=N (matches TaRoS / leaderboard convention).
    # Without it, sampling uses each checkpoint's training-faithful settings.
    if getattr(args, "sde_eta", None) is not None:
        config.sample.solver = "flow"
        config.sample.deterministic = False
        config.sample.noise_level = float(args.sde_eta)
    
    # Set LoRA path from checkpoint. --lora_policy picks the adapter:
    #   current (DEFAULT since 2026-10-01, user: "eval = non-EMA by default")
    #           -> checkpoints/lora_current_policy, the non-EMA policy that actually
    #           sampled/trained (training never uses the EMA weights; EMA is only
    #           tracked and written out at checkpoint time).
    #   ema     -> checkpoints/lora_ema_policy (what every eval before 2026-10-01 used).
    # Skipped when --no_lora is set: caller wants the un-tuned base model.
    if getattr(args, "no_lora", False):
        config.train.lora_path = None
    else:
        prefer_ema = getattr(args, "lora_policy", "current") == "ema"
        lora_path = find_lora_checkpoint(checkpoint_path, prefer_ema=prefer_ema)
        if lora_path is None:
            raise FileNotFoundError(f"LoRA checkpoint not found in: {checkpoint_path}")
        config.train.lora_path = lora_path

    return config


def get_model_utils(model_type, config, device, rank, world_size):
    """Get the appropriate model utility class."""
    # Use dummy logger and wandb for evaluation
    class DummyLogger:
        def info(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
    
    class DummyWandb:
        run = None
        def log(self, *args, **kwargs):
            pass
        def Image(self, *args, **kwargs):
            pass
    
    logger = DummyLogger()
    wandb_module = DummyWandb()
    
    from src.models import build_model_utils
    return build_model_utils(model_type, config=config, logger=logger, wandb_module=wandb_module,
                             device=device, rank=rank, world_size=world_size)


def find_dual_expert_loras(checkpoint_path, prefer_ema=True):
    """Locate BOTH experts' adapters for an alternating Wan2.2-A14B run.

    ``src/train_wan2_2.py`` trains the high- and low-noise experts
    together and writes them to ``checkpoints/high/`` and ``checkpoints/low/``. Returns
    ``{"high": path, "low": path}`` when both are present, else ``None``.
    """
    found = {}
    for expert in ("high", "low"):
        root = os.path.join(checkpoint_path, "checkpoints", expert)
        candidates = ["lora_current_policy"]
        if prefer_ema:
            candidates.insert(0, "lora_ema_policy")
        for name in candidates:
            path = os.path.join(root, name)
            if os.path.exists(path):
                found[expert] = path
                break
    return found if len(found) == 2 else None


def find_lora_checkpoint(checkpoint_path, prefer_ema=True):
    """Find LoRA checkpoint in the given path."""
    possible_paths = [
        os.path.join(checkpoint_path, "checkpoints", "lora_ema_policy" if prefer_ema else "lora_current_policy"),
        os.path.join(checkpoint_path, "checkpoints", "lora_current_policy"),
        os.path.join(checkpoint_path, "lora"),
    ]
    # Alternating A14B layout: no flat checkpoints/lora_*_policy exists, so
    # fall back to the high-noise expert's adapter. That keeps
    # `config.train.lora_path` non-None (create_eval_config raises otherwise);
    # the low-noise adapter is attached separately in the main eval flow.
    dual = find_dual_expert_loras(checkpoint_path, prefer_ema=prefer_ema)
    if dual is not None:
        possible_paths.append(dual["high"])

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def setup_distributed(rank, world_size, device_id=None):
    """Initializes the distributed process group."""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    timeout_minutes = int(os.getenv("DIST_TIMEOUT_MINUTES", "180"))
    dist_timeout = datetime.timedelta(minutes=timeout_minutes)
    
    # Initialize with device_id to avoid race conditions
    if device_id is not None:
        device = torch.device(f"cuda:{device_id}")
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=dist_timeout,
        )
    else:
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            timeout=dist_timeout,
        )


def cleanup_distributed():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


def is_main_process(rank):
    """Checks if the current process is the main one (rank 0)."""
    return rank == 0


# ======================== VBench-Official Helper ========================
# Map each --prompt_source to the file under dataset/vbench/. All three are
# 946-line rewrites of the same VBench prompt suite, paired line-by-line with
# VBench_full_info.json. wanx_extended_{qwen14b,qwen3b} both go through Wan2.1's
# `LM_EN_SYS_PROMPT` + QwenPromptExpander; only the rewriter model size differs.
VBENCH_EXTENDED_PROMPT_FILES = {
    "wanx_extended_qwen14b": "dataset/vbench/wanx_extended_qwen14b.txt",
    "wanx_extended_qwen3b": "dataset/vbench/wanx_extended_qwen3b.txt",
    "self_forcing_extended": "dataset/vbench/self_forcing_extended.txt",
}


def _run_vbench_official(
    args,
    rank,
    world_size,
    device,
    model_type,
    model_utils,
    eval_config,
    pipeline,
    text_encoders,
    tokenizers,
    mixed_precision_dtype,
    enable_amp,
    time_fraction_start_idx,
):
    """Generate videos for the official VBench prompt suite, then score with
    `mode="vbench_standard"` so all 16 dims are populated and Quality / Semantic
    / Total are directly comparable to the public VBench leaderboard / TaRoS Tab 1.

    Sampling is training-faithful: cfg, time_fraction, num_steps, solver come
    from the training config — only the prompt source and naming change.
    """
    import vbench as vbench_pkg

    is_video_model = model_type in ("wan2_1", "wan2_2_a14b_t2v", "skyreels_i2v")
    if not is_video_model:
        raise ValueError(
            f"--vbench_official is for video models only. Got model_type={model_type}."
        )

    # ---- Locate VBench_full_info.json ----
    full_info_path = args.vbench_full_info_path or os.path.join(
        os.path.dirname(vbench_pkg.__file__), "VBench_full_info.json"
    )
    if not os.path.exists(full_info_path):
        raise FileNotFoundError(f"VBench_full_info.json not found at {full_info_path}")
    with open(full_info_path) as f:
        full_info = json.load(f)

    # ---- Load long generation prompts based on --prompt_source ----
    # `gen_prompts[i]` is what we feed to Wan2.1.
    # `file_prompts[i]` is what we use as the mp4 filename so VBench's
    # mode="vbench_standard" can match it back to a JSON entry via dim mapping.
    n_full = len(full_info)
    file_prompts = [e["prompt_en"] for e in full_info]
    if args.prompt_source in VBENCH_EXTENDED_PROMPT_FILES:
        aug_path = args.prompt_aug_file or VBENCH_EXTENDED_PROMPT_FILES[args.prompt_source]
        if not os.path.isabs(aug_path):
            aug_path = os.path.join(os.getcwd(), aug_path)
        if not os.path.exists(aug_path):
            raise FileNotFoundError(
                f"--prompt_source={args.prompt_source} needs the prompt file at "
                f"{aug_path}."
            )
        with open(aug_path) as f:
            gen_prompts = [line.rstrip("\n") for line in f if line.strip()]
        if len(gen_prompts) != n_full:
            raise ValueError(
                f"Extended prompt file {aug_path} has {len(gen_prompts)} non-empty lines, "
                f"but VBench_full_info.json has {n_full} entries. Pairing would be "
                f"misaligned — refusing to proceed."
            )
        if is_main_process(rank):
            print(
                f"[vbench_official] prompt_source={args.prompt_source}: generating with "
                f"long prompts from {aug_path}, saving filenames as the short prompts "
                f"from {os.path.basename(full_info_path)}."
            )
    elif args.prompt_source == "short":
        gen_prompts = list(file_prompts)
        if is_main_process(rank):
            print(
                "[vbench_official] prompt_source=short: using JSON short prompts for "
                "both generation and filename (out-of-distribution for Wan2.1; ablation only)."
            )
    else:
        raise ValueError(f"unknown --prompt_source={args.prompt_source}")

    # Optional shard slicing — for splitting the 4,730-video full run across
    # multiple sbatch jobs.
    start_idx = max(0, int(args.start_prompt_idx))
    end_idx = int(args.end_prompt_idx) if args.end_prompt_idx is not None else n_full
    end_idx = min(end_idx, n_full)
    if start_idx >= end_idx:
        raise ValueError(f"empty prompt slice [{start_idx}, {end_idx})")

    n_per_prompt = max(1, int(args.num_videos_per_prompt))

    # Optional video_idx filter — parallel-shard at the (prompt × video_idx)
    # granularity by submitting one job per requested video_idx. Lets us
    # generate `-1.mp4`, `-2.mp4`, `-3.mp4`, `-4.mp4` in parallel after a
    # previous N=1 run already populated `-0.mp4`.
    video_idx_filter = None
    if getattr(args, "video_idx_filter", None):
        video_idx_filter = {
            int(x) for x in args.video_idx_filter.split(",") if x.strip() != ""
        }
        if is_main_process(rank):
            print(f"[vbench_official] video_idx_filter active: {sorted(video_idx_filter)}")

    # ---- Build flat task list ((prompt_idx, video_idx, gen_text, file_text)) ----
    tasks = []
    for absolute_idx in range(start_idx, end_idx):
        gen_text = gen_prompts[absolute_idx]
        file_text = file_prompts[absolute_idx]
        for vi in range(n_per_prompt):
            if video_idx_filter is not None and vi not in video_idx_filter:
                continue
            tasks.append((absolute_idx, vi, gen_text, file_text))

    # Distribute across ranks deterministically
    rank_tasks = tasks[rank::world_size]

    # ---- Output directories ----
    videos_dir = os.path.join(args.output_dir, "videos")
    if is_main_process(rank):
        os.makedirs(videos_dir, exist_ok=True)
    dist.barrier()

    # ---- Generation phase ----
    if not args.skip_generation:
        # Negative-prompt embeddings (mirrors legacy path)
        sample_neg_prompt_embeds = None
        sample_neg_pooled_prompt_embeds = None
        neg_prompt_embed, neg_pooled_prompt_embed = model_utils.get_negative_prompt_embeddings(
            text_encoders, tokenizers,
        )
        bs = max(int(eval_config.sample.test_batch_size), 1)
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(bs, 1, 1)
        if neg_pooled_prompt_embed is not None:
            sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(bs, 1)

        if is_main_process(rank):
            print(
                f"[vbench_official] {len(tasks)} total videos to generate "
                f"({end_idx - start_idx} prompts × {n_per_prompt}); "
                f"this rank: {len(rank_tasks)}"
            )
            print(
                f"[vbench_official] sampler: num_steps={eval_config.sample.num_steps} "
                f"guidance_scale={eval_config.sample.guidance_scale} "
                f"solver={eval_config.sample.solver} "
                f"time_fraction_start_idx={time_fraction_start_idx}"
            )

        # Iterate batches
        from tqdm import tqdm
        for batch_start in tqdm(
            range(0, len(rank_tasks), bs),
            desc=f"vbench_official gen (rank {rank})",
            disable=not is_main_process(rank),
        ):
            batch_tasks = rank_tasks[batch_start:batch_start + bs]

            # Per-task skip: when extending N (e.g. 1 → 5 videos/prompt) on an
            # already-populated videos_dir, we want to keep the existing
            # `<prompt>-<i>.mp4` files and only generate the missing video_idx
            # values. The previous all-or-nothing batch skip would regenerate
            # existing files whenever any task in the batch was missing.
            batch_tasks = [
                t for t in batch_tasks
                if not os.path.exists(os.path.join(videos_dir, f"{t[3]}-{t[1]}.mp4"))
            ]
            if not batch_tasks:
                continue
            batch_size = len(batch_tasks)
            # `gen_text` is fed to Wan2.1 (long, in-distribution).
            # `file_text` becomes the mp4 filename (short, matches VBench JSON).
            prompts = [t[2] for t in batch_tasks]  # gen_text

            # Per-element seeded generators — different noise per (prompt, video_idx).
            generators = [
                torch.Generator(device=device).manual_seed(
                    int(t[0]) * 100003 + int(t[1]) * 991 + int(getattr(eval_config, "seed", 0))
                )
                for t in batch_tasks
            ]

            with torch.amp.autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = model_utils.get_prompt_embeddings(
                        prompts, text_encoders, tokenizers,
                    )

                    sample_kwargs = model_utils.build_sample_kwargs(
                        prompt_embeds,
                        pooled_prompt_embeds,
                        sample_neg_prompt_embeds,
                        sample_neg_pooled_prompt_embeds,
                        batch_size,
                    )
                    # noise_level normally forced to 0 for ODE-deterministic eval;
                    # honor non-zero values when SDE override is in effect.
                    if getattr(args, "sde_eta", None) is None:
                        sample_kwargs["noise_level"] = 0
                    sample_kwargs["generator"] = generators

                    _toggle_lora = (
                        time_fraction_start_idx > 0
                        and hasattr(pipeline.transformer, "disable_adapter_layers")
                    )
                    if _toggle_lora:
                        pipeline.transformer.disable_adapter_layers()
                        sample_kwargs["time_fraction_start_idx"] = time_fraction_start_idx
                        sample_kwargs["peft_model_for_toggle"] = pipeline.transformer
                    try:
                        images = model_utils.rollout(pipeline, **sample_kwargs).media
                    finally:
                        if _toggle_lora and hasattr(pipeline.transformer, "enable_adapter_layers"):
                            pipeline.transformer.enable_adapter_layers()

            # Save each video with the SHORT prompt filename so VBench's
            # mode="vbench_standard" can find it via VBench_full_info.json's
            # dim mapping. The long prompt was used to drive generation only.
            for i, (_, video_idx, _gen_text, file_text) in enumerate(batch_tasks):
                output_path = os.path.join(videos_dir, f"{file_text}-{video_idx}.mp4")
                model_utils.save_sample(images[i], output_path)

            del images
            torch.cuda.empty_cache()

    # All ranks must finish generating before scoring starts.
    dist.barrier()

    # ---- Scoring phase ----
    if args.skip_scoring:
        if is_main_process(rank):
            print("[vbench_official] --skip_scoring set; videos written, scoring skipped.")
        return

    from src.reward.vbench_scorer import ALL_DIMENSIONS as VBENCH_ALL_DIMENSIONS

    # Run all 16 dims by default. The GRiT-backed semantic dims
    # (object_class/multiple_objects/color/spatial_relationship) use vbench's
    # vendored `third_party/grit_model` (needs detectron2, which IS installed)
    # and `scene` uses the vendored `third_party/tag2Text` (NOT the `ram`
    # package) — so all 16 score with the weights already cached under
    # ~/.cache/vbench. A genuine per-dim failure is caught below and that dim is
    # dropped rather than aborting the whole run.
    #
    # Opt-out: DROP_DETECTION_DIMS=1 (or legacy RUN_DETECTION_DIMS=0) drops the
    # 5 detection/scene dims; ONLY_DETECTION_DIMS=1 runs ONLY those 5 (for
    # incremental rescoring of videos that already have the other 11 dims).
    _DETECTION_DIMS = {
        "object_class", "multiple_objects", "color", "spatial_relationship",
        "scene",
    }
    _drop_detection = (
        os.environ.get("DROP_DETECTION_DIMS", "0") == "1"
        or os.environ.get("RUN_DETECTION_DIMS", "1") == "0"
    )
    if _drop_detection:
        DROP_DIMS_MISSING_DEPS = set(_DETECTION_DIMS)
    else:
        DROP_DIMS_MISSING_DEPS = set()
    if os.environ.get("ONLY_DETECTION_DIMS", "0") == "1":
        DROP_DIMS_MISSING_DEPS = set(VBENCH_ALL_DIMENSIONS) - _DETECTION_DIMS
    # When evaluating a partial prompt range (e.g. 100 prompts for a CFG
    # ablation), some VBench dims have ZERO entries in our slice. Their
    # `compute_<dim>` then divides by an empty `video_results` list →
    # ZeroDivisionError. Drop dims with no prompts in our range.
    sliced_entries = full_info[start_idx:end_idx]
    dims_with_data = set()
    for entry in sliced_entries:
        for d in entry.get("dimension", []):
            dims_with_data.add(d)
    score_dims = [
        d for d in VBENCH_ALL_DIMENSIONS
        if d not in DROP_DIMS_MISSING_DEPS and d in dims_with_data
    ]
    # --semantic_only: keep just the VBench semantic dimensions. Combined with
    # the default DROP_DIMS_MISSING_DEPS (which removes the 5 detection-backed
    # semantic dims) this leaves the 4 dep-free semantic dims: temporal_style,
    # overall_consistency, human_action, appearance_style.
    if getattr(args, "semantic_only", False):
        from src.reward.vbench_scorer import SEMANTIC_DIMS as VBENCH_SEMANTIC_DIMS
        score_dims = [d for d in score_dims if d in VBENCH_SEMANTIC_DIMS]
    dropped_no_data = [
        d for d in VBENCH_ALL_DIMENSIONS
        if d not in DROP_DIMS_MISSING_DEPS and d not in dims_with_data
    ]
    if dropped_no_data and is_main_process(rank):
        print(
            f"[vbench_official] dropped {len(dropped_no_data)} dims with no prompts "
            f"in [{start_idx}, {end_idx}): {dropped_no_data}"
        )

    vbench_output = os.path.join(args.output_dir, "vbench_eval")
    if is_main_process(rank):
        os.makedirs(vbench_output, exist_ok=True)
    dist.barrier()

    if is_main_process(rank):
        print(
            f"[vbench_official] running vbench_standard on "
            f"{len(os.listdir(videos_dir))} mp4 files in {videos_dir}"
        )
        print(f"[vbench_official] scoring dims ({len(score_dims)}): {score_dims}")
        print(f"[vbench_official] dropped dims (deps missing): {sorted(DROP_DIMS_MISSING_DEPS)}")

    # Apply the single-process patch when the prompt range is small enough
    # that VBench's `distribute_list_to_rank` would shard a per-dim subset
    # to fewer than `world_size` videos per rank → ZeroDivisionError. The
    # full 946-prompt suite has enough density per dim that the native
    # distributed path works; partial-range runs (e.g. 100 prompts to test
    # CFG ablations) need each rank to process the full list independently.
    n_prompts = end_idx - start_idx
    use_single_process = n_prompts < 300
    eval_name = "leaderboard_eval"

    from src.reward.vbench_scorer import _vbench_single_process

    # Pass a torch.device (NOT str): vbench's GRiT path does `device.type`
    # (object_class/multiple_objects/color/spatial_relationship), which throws
    # `'str' object has no attribute 'type'` on a string device. The only
    # string-equality device check in vbench (motion_smoothness `== 'cuda'`) is
    # already False for "cuda:N" and behaves identically, so torch.device is safe.
    vbench_runner = vbench_pkg.VBench(
        device=device,
        full_info_dir=full_info_path,
        output_path=vbench_output,
    )
    if is_main_process(rank):
        print(
            f"[vbench_official] scoring with single_process="
            f"{use_single_process} (n_prompts={n_prompts})"
        )
    def _evaluate(name, dims):
        if use_single_process:
            with _vbench_single_process():
                vbench_runner.evaluate(
                    videos_path=videos_dir, name=name, dimension_list=dims,
                    mode="vbench_standard", local=True,
                )
        else:
            vbench_runner.evaluate(
                videos_path=videos_dir, name=name, dimension_list=dims,
                mode="vbench_standard", local=True,
            )

    # Fast path: one bulk evaluate loads each submodule once. If it raises
    # (e.g. one dim's model genuinely can't load), fall back to a per-dim loop
    # so a single broken dim doesn't sink the other 15. Per-dim results are
    # merged into one `{eval_name}_eval_results.json` for the parser below.
    try:
        _evaluate(eval_name, score_dims)
    except Exception as bulk_err:  # noqa: BLE001
        if is_main_process(rank):
            print(f"[vbench_official] bulk evaluate failed ({bulk_err}); "
                  f"retrying per-dim and dropping any that fail.")
        merged = {}
        ok_dims = []
        for d in score_dims:
            per_name = f"{eval_name}_{d}"
            try:
                _evaluate(per_name, [d])
                ok_dims.append(d)
                if is_main_process(rank):
                    pj = os.path.join(vbench_output, f"{per_name}_eval_results.json")
                    if os.path.exists(pj):
                        with open(pj) as f:
                            merged.update(json.load(f))
            except Exception as dim_err:  # noqa: BLE001
                if is_main_process(rank):
                    print(f"[vbench_official]   dim={d} failed: {dim_err}. Dropping.")
            if dist.is_initialized():
                dist.barrier()
        if is_main_process(rank):
            print(f"[vbench_official] per-dim fallback scored {len(ok_dims)}/"
                  f"{len(score_dims)} dims: {ok_dims}")
            with open(os.path.join(vbench_output, f"{eval_name}_eval_results.json"), "w") as f:
                json.dump(merged, f, indent=2)

    dist.barrier()

    # Rank 0 parses results into a leaderboard-form aggregate.
    if is_main_process(rank):
        results_json = os.path.join(vbench_output, f"{eval_name}_eval_results.json")
        if not os.path.exists(results_json):
            raise FileNotFoundError(
                f"Expected VBench result file at {results_json} but it's missing."
            )
        with open(results_json) as f:
            results = json.load(f)

        # Official VBench leaderboard aggregation: per-dim min-max normalize,
        # dynamic_degree weight 0.5, Total = (4*Quality + 1*Semantic)/5.
        from src.reward.vbench_aggregate import leaderboard_scores_t2v

        per_dim_avg = {}
        for dim, payload in results.items():
            # entry shape: [avg_score, [{"video_path": ..., "video_results": ...}, ...]]
            if isinstance(payload, (list, tuple)) and len(payload) >= 1:
                try:
                    per_dim_avg[dim] = float(payload[0])
                except (TypeError, ValueError):
                    pass

        lb = leaderboard_scores_t2v(per_dim_avg)
        quality, semantic, total = lb["Quality"], lb["Semantic"], lb["Total"]

        # Multiply by 100 to match the public-leaderboard / TaRoS Tab 1 form.
        # Raw per-dim are reported un-normalized (leaderboard convention); the
        # Quality/Semantic/Total use the normalized aggregation above.
        scaled = {f"{d}_x100": v * 100.0 for d, v in per_dim_avg.items()}
        if quality is not None:
            scaled["Quality_x100"] = quality * 100.0
        if semantic is not None:
            scaled["Semantic_x100"] = semantic * 100.0
        if total is not None:
            scaled["Total_x100"] = total * 100.0

        agg = {
            "raw_per_dim_avg": per_dim_avg,
            "normalized_per_dim": lb["normalized_per_dim"],
            "leaderboard_scale_x100": scaled,
            "scored_dims": lb["scored_dims"],
            "missing_dims": lb["missing_dims"],
            "num_videos_per_prompt": n_per_prompt,
            "prompt_slice": [start_idx, end_idx],
            "videos_dir": videos_dir,
            "vbench_results_json": results_json,
        }
        agg_path = os.path.join(args.output_dir, "average_scores_vbench_official.json")
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)

        print("\n--- VBench official (×100, leaderboard form) ---")
        for k in sorted(scaled):
            print(f"  {k:<40}: {scaled[k]:.2f}")
        print(f"Aggregate saved to {agg_path}")

    dist.barrier()


def _summarize_diversity(output_dir, k, n_limit, seed0_rows):
    """Merge the diversity embedding / score shards and compute diversity.json (rank 0)."""
    import glob as _glob
    from src.reward.diversity import diversity_summary, mean_se, CLIP_ID, DINO_ID

    emb = {}  # (sid, seed) -> (clip, dino)
    for fp in sorted(_glob.glob(os.path.join(output_dir, ".diversity_emb", "*.npz"))):
        z = np.load(fp)
        seed = int(z["seed"])
        for i, sid in enumerate(z["sids"].tolist()):
            # setdefault: a DistributedSampler pad duplicate keeps the lowest rank (as the row merge does)
            emb.setdefault((int(sid), seed), (z["clip"][i], z["dino"][i]))
    scores = {}  # (sid, seed) -> {reward: value}
    for r in seed0_rows:
        scores[(int(r["sample_id"]), 0)] = r["scores"]
    for fp in _glob.glob(os.path.join(output_dir, ".diversity_shard_rank*.jsonl")):
        with open(fp) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scores[(int(rec["sample_id"]), int(rec["seed"]))] = rec["scores"]

    cand = sorted({sid for sid, _ in emb if n_limit is None or sid < int(n_limit)})
    sids = [s for s in cand if all((s, j) in emb for j in range(k))]
    missing = [s for s in cand if s not in sids]
    if missing:
        print(f"[diversity] WARNING: {len(missing)} sample ids lack some seed embeddings "
              f"(e.g. resumed seed-0 rows from a run without --diversity_seeds); excluded: {missing[:10]}")
    if not sids:
        raise RuntimeError("[diversity] no sample has all K seed embeddings")
    clip = np.stack([np.stack([emb[(s, j)][0] for j in range(k)]) for s in sids])
    dino = np.stack([np.stack([emb[(s, j)][1] for j in range(k)]) for s in sids])
    np.savez(os.path.join(output_dir, "diversity_emb.npz"), sample_ids=np.asarray(sids), clip=clip, dino=dino)
    res = diversity_summary(clip, dino)
    res["sample_ids"] = sids

    # rewards over ALL K images of the diversity prompts (seed 0 + extra seeds)
    names = sorted({n for s in sids for j in range(k) for n, v in scores.get((s, j), {}).items()
                    if isinstance(v, (int, float))})
    rew = {}
    for n in names:
        mat = np.array([[float(scores.get((s, j), {}).get(n, np.nan)) for j in range(k)] for s in sids])
        mat[mat == -10.0] = np.nan  # same -10 sentinel filter as average_scores
        ok = ~np.isnan(mat).all(1)
        if not ok.any():
            continue
        m = mat[ok]
        rew[n] = {"mean": float(np.nanmean(m)), "se": mean_se(np.nanmean(m, 1))["se"],
                  "per_image_std_within_prompt": float(np.nanmean(np.nanstd(m, 1))),
                  "per_seed_mean": [float(np.nanmean(m[:, j])) for j in range(k)]}
    res["reward_all_seeds"] = rew
    res["protocol"] = {
        "seeds": list(range(k)),
        "seed0": "the normal single-image eval sample, torch.Generator(device).manual_seed(sample_id)",
        "seed_j>=1": "torch.Generator(device).manual_seed(j * 1000003 + sample_id), one per sample (seed 0 = the main pass, same rule with j = 0)",
        "prompts": "all" if n_limit is None else f"sample_id < {int(n_limit)}",
        "embeddings": {"clip": f"{CLIP_ID} get_image_features (L2-normalised)",
                       "dino": f"{DINO_ID} pooler_output (CLS, layernormed; L2-normalised)"},
        "se": "within-prompt: over prompts; cross-prompt: over seeds; rewards: over per-prompt means",
    }
    w, x = res["within_prompt"], res["cross_prompt"]
    res["summary_line"] = (
        f"K={k} x {len(sids)} prompts | DINOv2 pair-cos {w['dino_pair_cos']['mean']:.3f}+-{w['dino_pair_cos']['se']:.3f}"
        f" | CLIP pair-cos {w['clip_pair_cos']['mean']:.3f}+-{w['clip_pair_cos']['se']:.3f}"
        f" | CLIP Vendi {w['clip_vendi']['mean']:.2f}+-{w['clip_vendi']['se']:.2f} (/{k})"
        f" | DINOv2 Vendi {w['dino_vendi']['mean']:.2f}+-{w['dino_vendi']['se']:.2f} (/{k})"
        + (f" | cross-prompt CLIP Vendi {x['x_clip_vendi']['mean']:.2f} (/{len(sids)})" if x else ""))
    return res

# ======================== Main Evaluation Function ========================
def main(args):
    # CLI validation
    if not args.vbench_official and args.dataset is None:
        raise ValueError("--dataset is required unless --vbench_official is set.")

    # --- Distributed Setup ---
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Enable blocking mode for better error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError(f"Rank {rank}: CUDA is not available!")
    
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise RuntimeError(
            f"Rank {rank} (local_rank {local_rank}): "
            f"Requested GPU {local_rank} but only {num_gpus} GPUs are visible. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
        )
    
    # Stagger device initialization slightly to avoid race condition
    # Each rank waits a tiny bit based on its local_rank
    import time
    time.sleep(local_rank * 0.1)  # 100ms per rank - more conservative
    
    # Print diagnostic info for each rank
    print(f"Rank {rank} (local_rank {local_rank}): "
          f"Attempting to claim GPU {local_rank} out of {num_gpus} visible GPUs", flush=True)
    
    # Set CUDA device BEFORE any CUDA operations or distributed init
    device = torch.device(f"cuda:{local_rank}")
    
    # Try to initialize CUDA context explicitly first
    try:
        # Force CUDA initialization on this specific device
        with torch.cuda.device(local_rank):
            # Create a tiny tensor to initialize the CUDA context
            _ = torch.zeros(1, device=device)
        torch.cuda.set_device(device)
        print(f"Rank {rank} (local_rank {local_rank}): Successfully set device to {device}", flush=True)
    except Exception as e:
        print(f"Rank {rank} (local_rank {local_rank}): FAILED to set device {device}", flush=True)
        print(f"Exception type: {type(e).__name__}", flush=True)
        print(f"Exception message: {str(e)}", flush=True)
        # Try to get more GPU info
        try:
            print(f"GPU {local_rank} name: {torch.cuda.get_device_name(local_rank)}", flush=True)
            print(f"GPU {local_rank} capability: {torch.cuda.get_device_capability(local_rank)}", flush=True)
        except:
            pass
        raise
    
    # Small delay to ensure device is fully initialized
    time.sleep(0.1)
    
    # Initialize distributed WITH device_id to prevent race conditions
    setup_distributed(rank, world_size, device_id=local_rank)
    
    # CRITICAL: Wait for ALL ranks to set their devices before any CUDA operations
    dist.barrier()

    # --- Determine Model Type ---
    if args.model_type:
        model_type = args.model_type
    else:
        # Trust config.base_model from training config first (handles run dirs
        # whose paths don't contain the model name, e.g. wan_videoalign_corrected_vgr).
        model_type = None
        try:
            _probe_cfg = load_training_config(args.checkpoint_path)
            model_type = _probe_cfg.get("base_model") if hasattr(_probe_cfg, "get") else getattr(_probe_cfg, "base_model", None)
        except FileNotFoundError:
            pass
        if not model_type:
            model_type = infer_model_type_from_path(args.checkpoint_path)
        if model_type is None:
            raise ValueError(
                f"Could not infer model type from checkpoint path: {args.checkpoint_path}. "
                "Please specify --model_type explicitly."
            )
        if is_main_process(rank):
            print(f"Inferred model type: {model_type}")

    # Load training config from checkpoint directory
    eval_config = create_eval_config(args, model_type, args.checkpoint_path)

    # Use mixed precision from training config (or override with auto-detection)
    if eval_config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif eval_config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    else:
        mixed_precision_dtype = None

    enable_amp = mixed_precision_dtype is not None

    if is_main_process(rank):
        print(f"Loaded training config from checkpoint")
        print(f"Running distributed evaluation with {world_size} GPUs.")
        if enable_amp:
            precision_str = "fp16" if mixed_precision_dtype == torch.float16 else "bf16"
            print(f"Using mixed precision: {precision_str}")
        os.makedirs(args.output_dir, exist_ok=True)
        if args.save_images:
            os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    
    # Wait for rank 0 to create directories before other ranks proceed
    dist.barrier()

    results_filepath = os.path.join(args.output_dir, f"evaluation_results_{args.dataset}.jsonl")

    # Get model utility class
    model_utils = get_model_utils(model_type, eval_config, device, rank, world_size)

    # --- Load Model and Pipeline ---
    if is_main_process(rank):
        print(f"Loading {model_type} pipeline...")
        if args.no_lora:
            print("[--no_lora] Skipping LoRA load — evaluating un-tuned base model.")
        else:
            print(f"Loading LoRA from: {eval_config.train.lora_path}  "
                  f"(--lora_policy {getattr(args, 'lora_policy', 'current')})")

    # Apply TF32 settings from config
    if eval_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load pretrained model using model-specific utility
    pipeline, text_encoders, tokenizers = model_utils.get_pretrained_model(
        mixed_precision_dtype, enable_amp
    )

    # Load LoRA weights using config path (skipped when --no_lora is set).
    #
    # Wan2.2-A14B: both experts' adapters (checkpoints/{high,low}/); every other model: one
    # adapter on `transformer`.
    _is_a14b = eval_config.base_model == "wan2_2_a14b_t2v"
    _dual_loras = (
        find_dual_expert_loras(
            args.checkpoint_path,
            prefer_ema=(getattr(args, "lora_policy", "current") == "ema"),
        ) if _is_a14b else None
    )

    if _dual_loras is not None:
        _lora_targets = {
            "transformer": _dual_loras["high"],
            "transformer_2": _dual_loras["low"],
        }
    elif _is_a14b and not args.no_lora:
        raise FileNotFoundError(
            f"{args.checkpoint_path} has no checkpoints/{{high,low}}/lora_current_policy: "
            "a Wan2.2-A14B evaluation needs both experts' adapters."
        )
    elif _is_a14b:
        _lora_targets = {"transformer": None, "transformer_2": None}   # --no_lora: base experts
    else:
        _lora_targets = {"transformer": eval_config.train.lora_path}

    if not args.no_lora:
        for _attr, _path in _lora_targets.items():
            setattr(
                pipeline,
                _attr,
                PeftModel.from_pretrained(
                    getattr(pipeline, _attr), _path, is_trainable=False,
                ),
            )
            if is_main_process(rank):
                print(f"[lora] pipeline.{_attr} <- {_path}")
    for _attr in _lora_targets:
        getattr(pipeline, _attr).eval()
        getattr(pipeline, _attr).to(device)

    # ---- Compute time_fraction_start_idx (needed for both eval modes) ----
    _tf = float(getattr(eval_config.train, "time_fraction", 1.0))
    assert 0.0 < _tf <= 1.0, f"time_fraction must be in (0, 1], got {_tf}"
    _ns = int(eval_config.sample.num_steps)
    time_fraction_start_idx = _ns - max(1, int(_ns * _tf))

    # ---- VBench-official leaderboard-comparable mode ----
    if args.vbench_official:
        _run_vbench_official(
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
            model_type=model_type,
            model_utils=model_utils,
            eval_config=eval_config,
            pipeline=pipeline,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            mixed_precision_dtype=mixed_precision_dtype,
            enable_amp=enable_amp,
            time_fraction_start_idx=time_fraction_start_idx,
        )
        cleanup_distributed()
        return

    # --- Load Dataset and Dataloader ---
    if is_main_process(rank):
        print(f"Loading dataset from: {eval_config.dataset}")

    # Use get_testloader from dataloader.py to create test dataloader
    test_dataloader = get_testloader(eval_config, world_size, rank, num_workers=0)
    
    # Determine if this is a video model
    is_video_model = model_type in ("wan2_1", "wan2_2_a14b_t2v", "skyreels_i2v")

    # Parse reward scorers from command-line arguments
    if args.reward_scorers:
        # Format: "scorer1:weight1,scorer2:weight2,..."
        all_reward_scorers = {}
        for scorer_spec in args.reward_scorers.split(","):
            parts = scorer_spec.strip().split(":")
            scorer_name = parts[0].strip()
            weight = float(parts[1]) if len(parts) > 1 else 1.0
            all_reward_scorers[scorer_name] = weight
    else:
        # Default reward scorers based on model type and dataset
        default_image_scorers = {
            "geneval": ["geneval", "pickscore", "aesthetic", "clipscore", "hpsv2", "imagereward"],
            "geneval_unseen_objects": ["geneval", "pickscore", "aesthetic", "clipscore", "hpsv2", "imagereward"],
            "ocr": ["ocr", "pickscore", "aesthetic", "clipscore", "hpsv2", "imagereward"],
            "pickscore": ["imagereward", "pickscore", "aesthetic", "clipscore", "hpsv2"],
            "drawbench": ["imagereward", "pickscore", "aesthetic", "clipscore", "hpsv2"],
        }
        # Video models: full VBench 16-dim (Quality/Semantic/Total breakdown
        # surfaces via per-dim keys) + VideoAlign (one scorer call yields
        # VQ/MQ/TA/Overall via the dim-dict path in MultiScorer).
        default_video_scorers = ["vbench_full", "videoalign"]
        if is_video_model:
            scorers = default_video_scorers
        else:
            scorers = default_image_scorers.get(args.dataset, ["pickscore", "aesthetic", "clipscore"])
        all_reward_scorers = {scorer: 1.0 for scorer in scorers}
    
    if is_main_process(rank):
        print(f"Using reward scorers: {list(all_reward_scorers.keys())}")
        print(f"Evaluation batch size: {eval_config.sample.test_batch_size}")

    # --- Instantiate Reward Models ---
    if is_main_process(rank):
        print("Initializing reward models...")
    scoring_fn = multi_score(
        device,
        all_reward_scorers,
        videoalign_dimension_weights=getattr(eval_config, "videoalign_dimension_weights", None),
        video_fps=getattr(eval_config, "video_fps", None),
    )

    # --- Negative prompt embeddings (match train.py sampling path) ---
    # Needed so build_sample_kwargs can slice per-batch negative embeds.
    sample_neg_prompt_embeds = None
    sample_neg_pooled_prompt_embeds = None
    if model_type != "sd3":
        neg_prompt_embed, neg_pooled_prompt_embed = model_utils.get_negative_prompt_embeddings(
            text_encoders, tokenizers,
        )
        neg_repeat = max(int(eval_config.sample.test_batch_size), 1)
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(neg_repeat, 1, 1)
        if neg_pooled_prompt_embed is not None:
            sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(neg_repeat, 1)

    # SD3 checkpoints TRAINED with CFG (train.guidance_scale > 1) and a frozen base-model
    # NP (train.np_base_velocity) must be sampled with that same guided field: the stock
    # diffusers pipeline below runs both CFG branches through the LoRA. Those runs go through
    # the patched pipeline (deterministic Euler flow ODE == the stock FlowMatchEuler sampler)
    # with the uncond on the base model and the training-time "" NP at 128 tokens. Every
    # cfg-1-trained SD3 checkpoint keeps the stock path, so earlier SD3 numbers stay comparable.
    sd3_base_np_eval = (
        model_type == "sd3"
        and float(eval_config.sample.guidance_scale) > 1.0
        and float(getattr(eval_config.train, "guidance_scale", 1.0)) > 1.0
        and bool(getattr(eval_config.train, "np_base_velocity", True))
    )
    if sd3_base_np_eval:
        sd3_neg_embeds, sd3_neg_pooled = model_utils.get_negative_prompt_embeddings(
            text_encoders, tokenizers,
        )
        print("[sd3 eval] CFG-trained checkpoint with base-model NP: patched pipeline, "
              f"np_base_velocity=True, guidance={eval_config.sample.guidance_scale}")

    # --- Within-prompt diversity (opt-in, --diversity_seeds K >= 2; src/reward/diversity.py) ---
    # Seed 0 IS the normal single-image eval (main loop below: torch.Generator(sample_id) per sample),
    # so its images / rows / average_scores are bit-identical to a run without the flag. Seeds
    # 1..K-1 run afterwards for the first --diversity_prompts sample ids, one
    # torch.Generator(seed * 1000003 + sample_id) per sample (batch- and world-size invariant).
    div_k = int(getattr(args, "diversity_seeds", 0) or 0)
    div_on = div_k >= 2
    div_n = getattr(args, "diversity_prompts", None)
    div_emb_dir = os.path.join(args.output_dir, ".diversity_emb")
    div_embedder = None
    if div_k == 1 or div_k < 0:
        raise ValueError(f"--diversity_seeds must be 0 (off) or >= 2, got {div_k}")
    if div_on:
        if model_type not in ("sd3", "flux"):
            raise ValueError(f"--diversity_seeds supports the image models (sd3, flux), got {model_type}")
        from src.reward.diversity import DiversityEmbedder, to_pil as _div_to_pil
        # fork_rng: loading the embedders must not move the CPU / CUDA RNG the seed-0 pass reads.
        with torch.random.fork_rng(devices=[device.index]):
            div_embedder = DiversityEmbedder(device)
        if is_main_process(rank):
            os.makedirs(div_emb_dir, exist_ok=True)
            print(f"[diversity] K={div_k} seeds, prompts: "
                  f"{'all' if div_n is None else f'sample_id < {div_n}'}")
        dist.barrier()

    def _div_wanted(sid):
        return div_on and (div_n is None or sid < int(div_n))

    def _div_embed_save(images, idx, sids, seed):
        """Embed images[idx] (CLIP + DINOv2) and persist them before the score rows are written."""
        if not idx:
            return
        with torch.random.fork_rng(devices=[device.index]):
            sel = images[idx] if torch.is_tensor(images) else torch.stack([images[i] for i in idx])
            ce, de = div_embedder(_div_to_pil(sel))
        fn = f"r{rank:03d}{_range_tag}_s{seed}_{sids[0]:05d}_{time.time_ns()}.npz"
        np.savez(os.path.join(div_emb_dir, fn), sids=np.asarray(sids, dtype=np.int64),
                 seed=np.int64(seed), clip=ce, dino=de)

    # --- Evaluation Loop ---
    results_this_rank = []
    sample_counter = 0  # Track sample indices for this rank

    # Mirror train.py:297-306,432-457 — when time_fraction < 1, the early
    # (1 - tf) of steps must run with the LoRA disabled, then re-enabled at
    # `time_fraction_start_idx` inside solver.run_sampling().
    time_fraction = float(getattr(eval_config.train, "time_fraction", 1.0))
    assert 0.0 < time_fraction <= 1.0, f"time_fraction must be in (0, 1], got {time_fraction}"
    _num_steps = int(eval_config.sample.num_steps)
    _active_region = max(1, int(_num_steps * time_fraction))
    time_fraction_start_idx = _num_steps - _active_region

    if is_main_process(rank):
        # Log configuration being used
        if is_video_model:
            print(f"Generating videos: {eval_config.height}x{eval_config.width}, {eval_config.frames} frames")
        else:
            print(f"Generating images: {eval_config.resolution}x{eval_config.resolution}")
        print(f"Inference steps: {eval_config.sample.num_steps}, Guidance scale: {eval_config.sample.guidance_scale}")
        print(f"time_fraction={time_fraction} -> start_idx={time_fraction_start_idx}, active_region={_active_region}")

    # ---------- Resume support: per-rank shard files ----------
    # Each rank appends one JSON line per completed sample to its own shard.
    # On startup, we scan our rank's shard to find already-done sample_ids
    # (keyed by metadata["global_idx"] when available, else the local
    # sample_counter for legacy datasets). Skipping done samples lets a
    # resubmitted job pick up where the previous walltime cut it off.
    # Range sharding (generic path; 2026-08-25): --start_prompt_idx /
    # --end_prompt_idx restrict this JOB to global_idx in [start, end), so
    # several jobs can generate DISJOINT slices of one suite concurrently into
    # the same output_dir. Each job writes range-suffixed shard files (no
    # cross-job append races), and resume reads EVERY shard file present so
    # progress from serial full-range jobs and other slices is honored.
    _gs = max(0, int(getattr(args, "start_prompt_idx", 0) or 0))
    _ge = getattr(args, "end_prompt_idx", None)
    _range_active = _gs > 0 or _ge is not None
    _range_tag = f"_{_gs}_{_ge if _ge is not None else 'end'}" if _range_active else ""
    shard_path = os.path.join(
        args.output_dir, f".eval_shard_rank{rank:03d}{_range_tag}.jsonl"
    )
    completed_ids = set()
    import glob as _glob
    for _sp in _glob.glob(os.path.join(args.output_dir, ".eval_shard_rank*.jsonl")):
        with open(_sp, "r") as _shf:
            for _line in _shf:
                try:
                    _rec = json.loads(_line)
                    completed_ids.add(int(_rec["sample_id"]))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue
    if completed_ids:
        print(f"[resume rank {rank}] {len(completed_ids)} sample_ids already done (all shards)")
    if _range_active:
        print(f"[range rank {rank}] restricting to global_idx in [{_gs}, {_ge})")
    shard_file = open(shard_path, "a", buffering=1)  # line-buffered

    # One batch of samples through the model's eval sampler, one seeded torch.Generator per sample
    # (seed 0 = the main pass, seeds >= 1 = --diversity_seeds).
    def _generate_batch(prompts, metadata, current_batch_size, generators=None):
        _gen_kw = {} if generators is None else {"generator": generators}
        with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
            with torch.no_grad():
                # SD3 evaluation uses plain inference path to reduce VRAM usage
                # (pipeline_with_logprob keeps extra per-step buffers for training).
                if sd3_base_np_eval:
                    p_embeds, p_pooled = model_utils.get_prompt_embeddings(prompts, text_encoders, tokenizers)
                    n = len(prompts)
                    images = model_utils.rollout(
                        pipeline,
                        prompt_embeds=p_embeds,
                        pooled_prompt_embeds=p_pooled,
                        negative_prompt_embeds=sd3_neg_embeds.repeat(n, 1, 1),
                        negative_pooled_prompt_embeds=sd3_neg_pooled.repeat(n, 1),
                        num_inference_steps=eval_config.sample.num_steps,
                        guidance_scale=eval_config.sample.guidance_scale,
                        output_type="pt",
                        height=eval_config.resolution,
                        width=eval_config.resolution,
                        deterministic=True,
                        solver="flow",
                        np_base_velocity=True,
                        **_gen_kw,
                    ).media
                elif model_type == "sd3":
                    generated = pipeline(
                        prompts,
                        num_inference_steps=eval_config.sample.num_steps,
                        guidance_scale=eval_config.sample.guidance_scale,
                        output_type="pt",
                        height=eval_config.resolution,
                        width=eval_config.resolution,
                        **_gen_kw,
                    )
                    images = generated[0] if isinstance(generated, tuple) else generated.images
                else:
                    # Mirror train.py sampling path so I2V reference images,
                    # attention masks, and precomputed embeddings are honored.
                    use_precomputed = eval_config.prompt_fn in ("embedding", "embedding_image")
                    if use_precomputed:
                        prompt_embeds = torch.stack(
                            [m["prompt_embeds"] for m in metadata]
                        ).to(device)
                        pooled_list = [m.get("pooled_prompt_embeds") for m in metadata]
                        if pooled_list and pooled_list[0] is not None:
                            pooled_prompt_embeds = torch.stack(pooled_list).to(device)
                        else:
                            pooled_prompt_embeds = None
                        prompt_attention_mask = torch.stack(
                            [m["prompt_attention_mask"] for m in metadata]
                        ).to(device)
                        model_utils._last_prompt_attention_mask = prompt_attention_mask
                    else:
                        prompt_embeds, pooled_prompt_embeds = model_utils.get_prompt_embeddings(
                            prompts, text_encoders, tokenizers,
                        )

                    # I2V: propagate reference images / precomputed first-frame latents.
                    if model_type == "skyreels_i2v":
                        precomputed_latents = [
                            m.get("first_frame_latents") for m in metadata
                            if m and m.get("first_frame_latents") is not None
                        ]
                        if precomputed_latents:
                            model_utils._current_first_frame_latents = torch.stack(
                                precomputed_latents
                            ).to(device)
                            model_utils._current_ref_images = None
                        else:
                            ref_images = [
                                m.get("ref_image") for m in metadata
                                if m and m.get("ref_image") is not None
                            ]
                            if not ref_images:
                                raise RuntimeError(
                                    f"{model_type} eval needs reference images. "
                                    f"Expected ImagePromptDataset to populate "
                                    f"metadata['ref_image'] from {eval_config.dataset}/ref_images_test/."
                                )
                            model_utils._current_ref_images = ref_images
                            model_utils._current_first_frame_latents = None

                    sample_kwargs = model_utils.build_sample_kwargs(
                        prompt_embeds,
                        pooled_prompt_embeds,
                        sample_neg_prompt_embeds,
                        sample_neg_pooled_prompt_embeds,
                        current_batch_size,
                    )
                    sample_kwargs["noise_level"] = 0
                    sample_kwargs.update(_gen_kw)
                    # Apply training-time `time_fraction` policy: disable LoRA
                    # for the early (1 - tf) fraction of steps; solver re-enables
                    # it at `time_fraction_start_idx`.
                    _toggle_lora = (
                        time_fraction_start_idx > 0
                        and hasattr(pipeline.transformer, "disable_adapter_layers")
                    )
                    if _toggle_lora:
                        pipeline.transformer.disable_adapter_layers()
                        sample_kwargs["time_fraction_start_idx"] = time_fraction_start_idx
                        sample_kwargs["peft_model_for_toggle"] = pipeline.transformer
                    try:
                        images = model_utils.rollout(pipeline, **sample_kwargs).media
                    finally:
                        if _toggle_lora and hasattr(pipeline.transformer, "enable_adapter_layers"):
                            pipeline.transformer.enable_adapter_layers()
        return images

    for batch in tqdm(test_dataloader, desc=f"Evaluating (Rank {rank})", disable=not is_main_process(rank)):
        prompts, metadata = batch
        current_batch_size = len(prompts)

        # Resolve global sample_ids for resume bookkeeping. Fall back to the
        # rank-local counter when the dataset doesn't carry an index (legacy
        # datasets like TextPromptDataset; resume won't help those but the
        # eval still runs).
        batch_sample_ids = [
            int(m["global_idx"]) if (m and "global_idx" in m) else (sample_counter + i)
            for i, m in enumerate(metadata)
        ]
        # Skip batches where every sample is already completed or out of this
        # job's [start, end) range slice.
        def _mine(sid):
            if sid in completed_ids:
                return False
            if _range_active:
                if sid < _gs or (_ge is not None and sid >= _ge):
                    return False
            return True
        if not any(_mine(sid) for sid in batch_sample_ids):
            sample_counter += current_batch_size
            continue

        # Seed 0 pinned per sample: torch.Generator(0 * 1000003 + sample_id), the same rule as
        # the --diversity_seeds images j >= 1, so the image eval is reproducible run to run and independent of
        # batch size / GPU count (it was unseeded: model/scorer loading reseeded the global RNG randomly).
        images = _generate_batch(prompts, metadata, current_batch_size,
                                 generators=[torch.Generator(device=device).manual_seed(sid) for sid in batch_sample_ids])

        all_scores, _ = scoring_fn(images, prompts, metadata, only_strict=False)
        print(all_scores)

        if div_on:  # seed-0 embeddings of this batch's new diversity samples
            _di = [i for i, sid in enumerate(batch_sample_ids)
                   if sid not in completed_ids and _div_wanted(sid)]
            _div_embed_save(images, _di, [batch_sample_ids[i] for i in _di], 0)

        for i in range(current_batch_size):
            sample_idx = batch_sample_ids[i]
            # Resumed batch can have a mix of done/not-done samples; skip the
            # already-done ones so we don't double-write to the shard.
            if sample_idx in completed_ids:
                continue
            raw_meta = metadata[i] if metadata else {}
            # Strip non-JSON-serializable entries (PIL images, tensors) left in
            # metadata by ImagePromptDataset / EmbeddingPromptDataset.
            serializable_meta = {
                k: v for k, v in (raw_meta or {}).items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
            result_item = {
                "sample_id": sample_idx,
                "prompt": prompts[i],
                "metadata": serializable_meta,
                "scores": {},
            }

            if args.save_images:
                # Use utility class method for model-specific saving (jpg for images, mp4 for video)
                extension = ".mp4" if is_video_model else ".jpg"
                output_path = os.path.join(args.output_dir, "images", f"{sample_idx:05d}{extension}")
                saved_path = model_utils.save_sample(images[i], output_path)
                result_item["video_path" if is_video_model else "image_path"] = saved_path

            for score_name, score_values in all_scores.items():
                if isinstance(score_values, torch.Tensor):
                    result_item["scores"][score_name] = score_values[i].detach().cpu().item()
                else:
                    result_item["scores"][score_name] = float(score_values[i])

            results_this_rank.append(result_item)
            # Persist incrementally so a walltime kill doesn't lose this row.
            shard_file.write(json.dumps(result_item) + "\n")
            completed_ids.add(sample_idx)

        sample_counter += current_batch_size
        del images, all_scores
        torch.cuda.empty_cache()

    shard_file.close()

    # --- Diversity: seeds 1..K-1 (after the seed-0 pass) ---
    if div_on:
        div_shard_path = os.path.join(args.output_dir, f".diversity_shard_rank{rank:03d}{_range_tag}.jsonl")
        div_done = set()
        for _sp in _glob.glob(os.path.join(args.output_dir, ".diversity_shard_rank*.jsonl")):
            with open(_sp) as _shf:
                for _line in _shf:
                    try:
                        _rec = json.loads(_line)
                        div_done.add((int(_rec["sample_id"]), int(_rec["seed"])))
                    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                        continue
        div_file = open(div_shard_path, "a", buffering=1)
        div_counter = 0
        for batch in tqdm(test_dataloader, desc=f"Diversity seeds (Rank {rank})",
                          disable=not is_main_process(rank)):
            prompts_all, metadata_all = batch
            sids_all = [
                int(m["global_idx"]) if (m and "global_idx" in m) else (div_counter + i)
                for i, m in enumerate(metadata_all)
            ]
            div_counter += len(prompts_all)
            for seed in range(1, div_k):
                keep = [i for i, sid in enumerate(sids_all)
                        if _div_wanted(sid) and (sid, seed) not in div_done
                        and not (_range_active and (sid < _gs or (_ge is not None and sid >= _ge)))]
                if not keep:
                    continue
                prompts = [prompts_all[i] for i in keep]
                metadata = [metadata_all[i] for i in keep]
                sids = [sids_all[i] for i in keep]
                generators = [torch.Generator(device=device).manual_seed(seed * 1000003 + sid) for sid in sids]
                images = _generate_batch(prompts, metadata, len(keep), generators=generators)
                all_scores, _ = scoring_fn(images, prompts, metadata, only_strict=False)
                _div_embed_save(images, list(range(len(keep))), sids, seed)
                for i, sid in enumerate(sids):
                    row = {"sample_id": sid, "seed": seed, "prompt": prompts[i], "scores": {}}
                    if args.save_images:
                        row["image_path"] = model_utils.save_sample(
                            images[i], os.path.join(args.output_dir, "images", f"{sid:05d}_s{seed}.jpg"))
                    for score_name, score_values in all_scores.items():
                        if isinstance(score_values, torch.Tensor):
                            row["scores"][score_name] = score_values[i].detach().cpu().item()
                        else:
                            row["scores"][score_name] = float(score_values[i])
                    div_file.write(json.dumps(row) + "\n")
                    div_done.add((sid, seed))
                del images, all_scores
                torch.cuda.empty_cache()
        div_file.close()

    # --- Gather and Save Results ---
    # all_gather_object is a synchronization collective; we use it as a
    # barrier so rank 0 only reads the shard files after every rank has
    # closed its shard handle. The actual merge happens from disk so that
    # a resumed run includes rows persisted by previous (killed) runs in
    # addition to this run's in-memory results.

    all_gathered_results = [None] * world_size
    dist.all_gather_object(all_gathered_results, [])  # sync only

    if is_main_process(rank):
        # Read every rank's shard from disk — includes any rows persisted
        # by earlier (walltime-killed) runs alongside this run's rows.
        flat_results = []
        seen_ids = set()
        for r in range(world_size):
            shard_p = os.path.join(args.output_dir, f".eval_shard_rank{r:03d}.jsonl")
            if not os.path.exists(shard_p):
                continue
            with open(shard_p, "r") as _f:
                for _line in _f:
                    try:
                        rec = json.loads(_line)
                    except json.JSONDecodeError:
                        continue
                    sid = rec.get("sample_id")
                    if sid is None or sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    flat_results.append(rec)

        flat_results.sort(key=lambda x: x["sample_id"])

        with open(results_filepath, "w") as f_out:
            for result_item in flat_results:
                f_out.write(json.dumps(result_item) + "\n")

        print(f"\nEvaluation finished. All {len(flat_results)} results saved to {results_filepath}")

        # Loud guard against silent truncation: the merge dedups on sample_id,
        # so any dataset whose metadata lacks a unique global_idx collapses to
        # one rank's worth of rows and the averages below are computed over a
        # fraction of the benchmark. This is a correctness bug, not a warning to
        # live with -- see TextPromptDataset.__getitem__.
        try:
            expected = len(test_dataloader.dataset)
        except (AttributeError, TypeError):
            expected = None
        if expected and len(flat_results) < expected:
            print(
                f"WARNING: merged {len(flat_results)} rows but the eval set has "
                f"{expected} prompts -- {expected - len(flat_results)} missing. "
                f"Scores below average only the rows present. Check that the "
                f"dataset emits a unique metadata['global_idx'] per sample."
            )

        all_scores_agg = defaultdict(list)

        for result in flat_results:
            for score_name, score_value in result["scores"].items():
                if isinstance(score_value, (int, float)):
                    all_scores_agg[score_name].append(score_value)

        average_scores = {
            name: np.mean(list(filter(lambda score: score != -10.0, scores))) for name, scores in all_scores_agg.items()
        }

        print("\n--- Average Scores ---")
        if not average_scores:
            print("No scores were found to average.")
        else:
            for name, avg_score in sorted(average_scores.items()):
                print(f"{name:<20}: {avg_score:.4f}")
        print("----------------------")

        avg_scores_filepath = os.path.join(args.output_dir, f"average_scores_{args.dataset}.json")
        with open(avg_scores_filepath, "w") as f_avg:
            json.dump(average_scores, f_avg, indent=4)
        print(f"Average scores also saved to {avg_scores_filepath}")

        if div_on:
            div_res = _summarize_diversity(args.output_dir, div_k, div_n, flat_results)
            with open(os.path.join(args.output_dir, "diversity.json"), "w") as f_div:
                json.dump(div_res, f_div, indent=1)
            # short summary in the eval's results JSON (flat float keys; seed-0 reward means untouched)
            for m, v in div_res["within_prompt"].items():
                average_scores[f"diversity/{m}"] = v["mean"]
            for m, v in div_res["cross_prompt"].items():
                average_scores[f"diversity/{m}"] = v["mean"]
            average_scores["diversity/n_prompts"] = div_res["n_prompts"]
            average_scores["diversity/n_seeds"] = div_res["n_seeds"]
            with open(avg_scores_filepath, "w") as f_avg:
                json.dump(average_scores, f_avg, indent=4)
            print(f"[diversity] {div_res['summary_line']}")
            print(f"[diversity] saved to {os.path.join(args.output_dir, 'diversity.json')}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion model in a distributed manner.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the LoRA checkpoint directory (e.g., './logs/sd3/run_name' or './logs/sd3/run_name/checkpoints'). "
             "Model type will be auto-detected from the path structure if not specified. "
             "The training config will be loaded from this directory.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["sd3", "flux", "wan2_1", "wan2_2_a14b_t2v", "skyreels_i2v"],
        help="Type of the base model. "
             "If not specified, will auto-detect from checkpoint_path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset type (e.g., 'geneval', 'ocr', 'pickscore', 'drawbench'). "
             "Will be loaded from 'dataset/{dataset}' directory. Required unless "
             "--vbench_official is set, in which case the official VBench prompt suite "
             "is used instead.",
    )
    parser.add_argument(
        "--reward_scorers",
        type=str,
        default=None,
        help="Comma-separated list of reward scorers with optional weights. "
             "Format: 'scorer1:weight1,scorer2:weight2,...' or just 'scorer1,scorer2,...' (weights default to 1.0). "
             "Example: 'pickscore:1.0,aesthetic:0.5,clipscore:1.0' or 'pickscore,aesthetic,clipscore'. "
             "If not specified, uses dataset-specific defaults.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=14,
        help="Batch size for evaluation. If not specified, uses the value from the training config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_output",
        help="Directory to save evaluation results and generated images.",
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=None, 
        help="Number of inference steps for the diffusion pipeline. If not specified, uses the value from training config.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale. If not specified, uses the value from training config.",
    )
    parser.add_argument(
        "--sigmas",
        type=str,
        default=None,
        help="Comma-separated descending sigmas (no trailing 0) for custom "
             "non-uniform discretization, e.g. a subset of the trained "
             "schedule. Overrides --num_inference_steps; num_steps is set to "
             "len(sigmas).",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save generated images (jpg for SD3/Flux) or videos (mp4 for WAN) to the output directory."
    )
    parser.add_argument(
        "--vbench_official",
        action="store_true",
        help="Switch to VBench leaderboard-comparable protocol: load the official "
             "946-prompt suite from VBench_full_info.json, generate N videos per prompt "
             "with the model's training-time sampler, save mp4s under "
             "<output_dir>/videos/<prompt>-<i>.mp4, then score with mode='vbench_standard' "
             "for all 16 dimensions. Skips per-batch reward scoring; --reward_scorers / "
             "--dataset are ignored when this flag is set.",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Skip LoRA loading — evaluate the un-tuned base model. Use this to score the "
             "baseline (pretrained) model. --checkpoint_path is still used to read a "
             "training config.json (for model name, frames, resolution, etc.); the LoRA "
             "weights inside that dir are NOT loaded.",
    )
    parser.add_argument(
        "--sde_eta",
        type=float,
        default=None,
        help="If set, override the eval sampler to Euler-flow SDE with eta=this value "
             "(matches TaRoS / leaderboard convention: solver='flow', deterministic=False, "
             "noise_level=eta). Use 0.5 for the TaRoS Wan2.1 inference protocol. Without "
             "this flag, sampling uses each checkpoint's training-faithful sampler.",
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Videos per prompt for --vbench_official. Default 1 (sanity check, ~6 GPU-h "
             "for Wan2.1-1.3B). Set to 5 for full VBench leaderboard convention "
             "(~32 GPU-h, may need multi-job sharding).",
    )
    parser.add_argument(
        "--vbench_full_info_path",
        type=str,
        default=None,
        help="Override path to VBench_full_info.json. Defaults to the file shipped with "
             "the installed vbench package.",
    )
    parser.add_argument(
        "--prompt_source",
        type=str,
        default="wanx_extended_qwen14b",
        choices=["wanx_extended_qwen14b", "wanx_extended_qwen3b",
                 "self_forcing_extended", "short"],
        help="Source of generation prompts for --vbench_official. "
             "wanx_extended_qwen14b (default): VBench prompts extended via Wan2.1's "
             "LM_EN_SYS_PROMPT + Qwen2.5-14B-Instruct (matches our training pipeline "
             "in scripts/dataset/extend_prompts_wan.sh). "
             "wanx_extended_qwen3b: same template but with Qwen2.5-3B-Instruct — "
             "byte-identical to Wan-Video/Wan2.1's official "
             "all_dimension_aug_wanx_seed42.txt; use this for leaderboard / TaRoS "
             "byte-comparable runs. "
             "self_forcing_extended: Self-Forcing's independent rewrite from "
             "dataset/vbench/self_forcing_extended.txt — stylistically distinct second "
             "source for the same 946 VBench prompts. "
             "short: use VBench_full_info.json's short prompts directly (poor Wan2.1 "
             "input distribution; ablation only).",
    )
    parser.add_argument(
        "--prompt_aug_file",
        type=str,
        default=None,
        help="Optional override path to the augmented prompt file. When unset, "
             "--prompt_source resolves to dataset/vbench/<prompt_source>.txt. Must have "
             "one prompt per line in the same order as VBench_full_info.json (946 lines).",
    )
    parser.add_argument(
        "--start_prompt_idx",
        type=int,
        default=0,
        help="For sharding --vbench_official across multiple sbatch jobs: only generate "
             "videos for prompts in [start_prompt_idx, end_prompt_idx). Default 0 = no shard.",
    )
    parser.add_argument(
        "--end_prompt_idx",
        type=int,
        default=None,
        help="See --start_prompt_idx. Default None = use all prompts.",
    )
    parser.add_argument(
        "--video_idx_filter",
        type=str,
        default=None,
        help="Comma-separated list of video_idx values to keep (e.g. '1,2,3,4'). "
             "When set, only generate videos with vi in this set. Combine with "
             "--num_videos_per_prompt N to shard parallel jobs at the video_idx "
             "granularity — e.g. submit one job per vi to fill in `-1`..`-4` after "
             "a prior N=1 run already produced `-0.mp4`. Default None = no filter.",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="With --vbench_official: skip video generation and only run the VBench "
             "scoring pass on whatever mp4s are already in <output_dir>/videos/. "
             "Use after multiple shard jobs have populated the directory.",
    )
    parser.add_argument(
        "--skip_scoring",
        action="store_true",
        help="With --vbench_official: only generate videos, don't run VBench scoring. "
             "Use for shard jobs that just need to populate <output_dir>/videos/.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Override the frame count from the training config.json (e.g. 81 to "
             "evaluate a model trained at 53 frames). Must satisfy the VAE temporal "
             "constraint 4k+1. No-op for image models.",
    )
    parser.add_argument(
        "--lora_policy",
        type=str,
        default="current",
        choices=["current", "ema"],
        help="Which saved adapter to evaluate. 'current' (DEFAULT since 2026-10-01) = "
             "checkpoints/lora_current_policy, the non-EMA policy training actually "
             "samples with; 'ema' = checkpoints/lora_ema_policy (every eval before "
             "2026-10-01 used this). Falls back to lora_current_policy when the "
             "requested adapter is missing.",
    )
    parser.add_argument(
        "--diversity_seeds",
        type=int,
        default=0,
        help="Image models (sd3, flux): within-prompt diversity. 0 (default) = off, behaviour unchanged. "
             "K >= 2 generates K images per prompt: seed 0 is the normal eval image (torch.Generator(sample_id)), seeds "
             "1..K-1 use torch.Generator(seed*1000003 + sample_id). Every image is reward-scored; "
             "writes diversity.json (DINOv2-L / CLIP-L pairwise cosine distance + Vendi per prompt, "
             "cross-prompt per seed) and diversity/* keys in average_scores_<dataset>.json. "
             "Extra seeds are saved as images/<id>_s<j>.jpg.",
    )
    parser.add_argument(
        "--diversity_prompts",
        type=int,
        default=None,
        help="With --diversity_seeds: only sample_ids < N get the extra seeds / diversity metrics "
             "(default: all prompts). The seed-0 eval always covers the whole dataset.",
    )
    parser.add_argument(
        "--semantic_only",
        action="store_true",
        help="With --vbench_official: score only the VBench semantic dimensions. With "
             "the default deps (no RUN_DETECTION_DIMS) this is the 4 dep-free semantic "
             "dims: temporal_style, overall_consistency, human_action, appearance_style. "
             "Quality/Total aggregates are omitted; only Semantic_x100 is reported.",
    )

    args = parser.parse_args()
    main(args)
