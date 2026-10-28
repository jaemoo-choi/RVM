"""VBench reward scorer for video quality evaluation.

Wraps the VBench library (github.com/Vchitect/VBench) to evaluate specific
quality dimensions as training rewards.

VBench is designed for offline evaluation (file-based I/O, JSON output).
This scorer bridges it for online use by:
1. Writing generated videos to temp .mp4 files
2. Calling VBench.evaluate() with lightweight dimensions
3. Parsing the per-video scores from saved JSON results

Only lightweight dimensions suitable for online training are used by default:
- temporal_flickering: pixel MAE between consecutive frames (no model needed)
- subject_consistency: DINO ViT-B/16 cosine similarity across frames
- aesthetic_quality: CLIP ViT-L/14 + linear aesthetic head
- imaging_quality: MUSIQ model from pyiqa
- overall_consistency: ViCLIP video-text similarity

Heavy dimensions (object_class, human_action, scene, etc.) are excluded
as they require GRiT/UMT/Tag2Text and are too slow for online use.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import typing
from collections import OrderedDict
from pathlib import Path

import imageio
import numpy as np
import torch

# VBench's motion_smoothness/dynamic_degree call `torch.load` without
# `weights_only=False`. On torch >= 2.6 this trips the new safe-globals guard
# for `typing.OrderedDict` / `collections.OrderedDict`. Whitelist them so
# legitimate VBench checkpoints load instead of erroring out.
try:
    torch.serialization.add_safe_globals([OrderedDict, typing.OrderedDict])
except Exception:
    pass


@contextlib.contextmanager
def _vbench_single_process():
    """Force VBench's internal helpers to behave as single-process.

    VBench uses `torch.distributed` to shard videos across ranks via
    `distribute_list_to_rank` and gather via `gather_list_of_dict`. When the
    outer training/eval script has already initialised a process group with
    world_size > 1 and we feed VBench a per-rank batch (often just 1 video),
    VBench shards that 1 video to one rank and the other ranks process zero
    videos — triggering `ZeroDivisionError` (`sim / cnt` etc.).

    Several VBench modules import `get_rank` / `get_world_size` directly via
    `from .distributed import get_rank`, which keeps a stale reference even
    after we patch the source module. We therefore swap the binding on every
    submodule that imported them too — notably `vbench` itself (the save-only-
    on-rank-0 guard in evaluate()) and each per-dimension module that calls
    `distribute_list_to_rank` or `get_world_size` to decide gathering.
    """
    try:
        import vbench  # noqa: F401
        from vbench import distributed as vbench_dist
    except Exception:
        yield
        return

    import sys

    fake_ws = lambda: 1
    fake_rk = lambda: 0

    targets = [vbench_dist]
    for mod_name, mod in list(sys.modules.items()):
        if not isinstance(mod_name, str) or not mod_name.startswith("vbench"):
            continue
        if mod is vbench_dist:
            continue
        if mod is None:
            continue
        targets.append(mod)

    saved: list[tuple[object, str, object]] = []
    for mod in targets:
        for attr_name, fake in (("get_world_size", fake_ws), ("get_rank", fake_rk)):
            if hasattr(mod, attr_name):
                saved.append((mod, attr_name, getattr(mod, attr_name)))
                setattr(mod, attr_name, fake)
    try:
        yield
    finally:
        for mod, attr_name, orig in saved:
            setattr(mod, attr_name, orig)


# Dimensions suitable for online training (lightweight models or no model)
ONLINE_DIMENSIONS = [
    "temporal_flickering",
    "subject_consistency",
    "aesthetic_quality",
    "imaging_quality",
]

# All VBench dimensions (for offline evaluation only)
ALL_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "appearance_style",
]

# Standard VBench groupings used to derive Quality / Semantic / Total scores.
# Source of truth is src/reward/vbench_aggregate.py (vendored from the official
# VBench scripts/constant.py). QUALITY_DIMS / SEMANTIC_DIMS are the underscore
# dim-name sets; the actual aggregation (min-max normalize, dynamic_degree 0.5
# weight, Total = (4*Q + 1*S)/5) lives in leaderboard_scores_t2v.
from .vbench_aggregate import (  # noqa: E402
    QUALITY_DIMS,
    SEMANTIC_DIMS,
    leaderboard_scores_t2v,
)

# DEPRECATED: the old plain-mean weighting (Total = 0.55*Q + 0.45*S over
# un-normalized dim means) was wrong. Kept as the correct 4:1 split (0.8/0.2)
# only so external imports of these names don't break; new code must use
# leaderboard_scores_t2v, which also applies the per-dim min-max normalization
# these scalars cannot express.
QUALITY_TOTAL_WEIGHT = 0.8
SEMANTIC_TOTAL_WEIGHT = 0.2

# VBench rejects these dimensions in mode="custom_input" (see vbench/__init__.py
# `check_dimension_requires_extra_info`). They depend on extra metadata that
# only the standard VBench prompt set carries. Filter them out before calling
# evaluate() so we don't trip the assert and lose the whole batch.
CUSTOM_INPUT_UNSUPPORTED_DIMS = {
    "object_class",
    "multiple_objects",
    "scene",
    "appearance_style",
    "color",
    "spatial_relationship",
}

# Some VBench dims save raw per-video scores on a non-[0,1] scale even though
# the leaderboard average is normalized. We normalize per-video to keep the
# Quality / Semantic / Total aggregations comparable across dims.
PER_VIDEO_SCALE = {
    "imaging_quality": 1.0 / 100.0,  # MUSIQ raw output is 0..100
}


class VBenchScorer(torch.nn.Module):
    """VBench reward scorer for video generation.

    Expected input: videos as numpy array [B, T, H, W, C] (uint8).
    Prompts: list of strings.

    Args:
        device: CUDA device.
        dimensions: list of VBench dimension names to evaluate.
            Defaults to ONLINE_DIMENSIONS (lightweight only).
        video_fps: FPS for saving temporary video files.
        full_info_dir: path to VBench_full_info.json (auto-detected if None).
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        dimensions: list[str] | None = None,
        video_fps: int = 8,
        full_info_dir: str | None = None,
    ):
        super().__init__()
        # Store a torch.device (not str): vbench's GRiT path does `device.type`,
        # which throws on a string device.
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.video_fps = video_fps
        self.dimensions = dimensions or ONLINE_DIMENSIONS
        self._vbench = None
        self._import_error = None
        self._warned_once = False

        # Try to initialize VBench
        try:
            from vbench import VBench
            self._vbench_cls = VBench
        except ImportError as e:
            self._import_error = str(e)
            self._vbench_cls = None
            return

        # Locate full_info_dir
        if full_info_dir is None:
            # Try common locations
            candidates = [
                os.environ.get("VBENCH_FULL_INFO_DIR", ""),
            ]
            for c in candidates:
                if c and os.path.exists(os.path.join(c, "VBench_full_info.json")):
                    full_info_dir = c
                    break

            # Try to find it from the vbench package installation
            if full_info_dir is None:
                try:
                    import vbench as vbench_pkg
                    pkg_dir = os.path.dirname(vbench_pkg.__file__)
                    if os.path.exists(os.path.join(pkg_dir, "VBench_full_info.json")):
                        full_info_dir = pkg_dir
                except Exception:
                    pass

        self._full_info_dir = full_info_dir

    def _get_vbench(self, output_path: str):
        """Lazily create VBench instance."""
        if self._vbench_cls is None:
            return None

        kwargs = {"device": self.device, "output_path": output_path}
        if self._full_info_dir:
            kwargs["full_info_dir"] = self._full_info_dir

        try:
            return self._vbench_cls(**kwargs)
        except Exception as e:
            if not self._warned_once:
                print(f"[VBenchScorer] Failed to create VBench instance: {e}")
            return None

    @staticmethod
    def _parse_vbench_results(
        output_path: str, name: str, dimensions: list[str], num_videos: int
    ) -> dict[str, list[float]]:
        """Parse per-video scores from VBench's `<name>_eval_results.json`.

        VBench writes a single `<output_path>/<name>_eval_results.json` per
        evaluate() call, containing `{dim: [avg, [{video_path, video_results,
        ...}, ...]]}`. We pull the per-video scalar from `video_results` and
        align videos by their `sample_<idx>.mp4` filename so distributed
        ordering can't desync the index.
        """
        results: dict[str, list[float]] = {}
        results_path = os.path.join(output_path, f"{name}_eval_results.json")
        if not os.path.exists(results_path):
            return results

        try:
            with open(results_path, "r") as f:
                data = json.load(f)
        except Exception:
            return results

        for dim in dimensions:
            if dim not in data:
                continue
            entry = data[dim]
            # entry is [avg, [{"video_path": ..., "video_results": <scalar or dict>}, ...]]
            if not (isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], list)):
                continue
            video_records = entry[1]
            scale = PER_VIDEO_SCALE.get(dim, 1.0)

            scores = [0.0] * num_videos
            for record in video_records:
                if not isinstance(record, dict):
                    continue
                vp = record.get("video_path", "")
                idx = None
                # Filename format from this scorer: sample_<idx>.mp4
                base = os.path.basename(str(vp))
                if base.startswith("sample_"):
                    try:
                        idx = int(base[len("sample_"):].split(".")[0])
                    except Exception:
                        idx = None
                if idx is None or not (0 <= idx < num_videos):
                    continue
                val = record.get("video_results", 0.0)
                if isinstance(val, dict):
                    val = float(np.mean(list(val.values())))
                try:
                    scores[idx] = float(val) * scale
                except (TypeError, ValueError):
                    scores[idx] = 0.0
            results[dim] = scores

        return results

    @staticmethod
    def _fallback_temporal_consistency(videos: np.ndarray) -> list[float]:
        """Compute a lightweight proxy score in [0, 1] when VBench is unavailable."""
        scores: list[float] = []
        vids = videos.astype(np.float32) / 255.0
        for vid in vids:
            if vid.shape[0] < 2:
                scores.append(0.5)
                continue
            frame_delta = np.abs(vid[1:] - vid[:-1]).mean()
            score = float(np.clip(1.0 - (2.0 * frame_delta), 0.0, 1.0))
            scores.append(score)
        return scores

    @staticmethod
    def _aggregate_quality_semantic_total(
        dim_scores: dict[str, list[float]], batch_size: int
    ) -> dict[str, list[float]]:
        """Compute VBench Quality / Semantic / Total per video from per-dim scores.

        Uses the official leaderboard math (per-dim min-max normalization,
        dynamic_degree 0.5 weight, Total = (4*Q + 1*S)/5) via
        leaderboard_scores_t2v applied per video. Because that aggregation is
        affine in the per-dim scores, normalizing per-video then averaging
        downstream gives the same Quality/Semantic/Total as the official
        normalize-the-dataset-average procedure.

        Only whichever dims were actually evaluated contribute; if a group has
        no evaluated dims it is omitted and Total falls back to the other group.
        """
        out: dict[str, list[float]] = {"Quality": [], "Semantic": [], "Total": []}
        have_q = any(d in QUALITY_DIMS for d in dim_scores)
        have_s = any(d in SEMANTIC_DIMS for d in dim_scores)

        for i in range(batch_size):
            agg = leaderboard_scores_t2v(
                {d: dim_scores[d][i] for d in dim_scores}
            )
            out["Quality"].append(agg["Quality"] if agg["Quality"] is not None else 0.0)
            out["Semantic"].append(agg["Semantic"] if agg["Semantic"] is not None else 0.0)
            out["Total"].append(agg["Total"] if agg["Total"] is not None else 0.0)

        if not have_q:
            del out["Quality"]
        if not have_s:
            del out["Semantic"]
        return out

    @torch.no_grad()
    def __call__(
        self, videos: np.ndarray, prompts: list[str], metadata
    ) -> tuple[list[float], dict[str, list[float]]]:
        del metadata

        if not isinstance(videos, np.ndarray):
            raise ValueError(f"Expected numpy.ndarray for videos, got {type(videos)}")
        if videos.ndim != 5:
            raise ValueError(f"Expected videos shape [B, T, H, W, C], got {videos.shape}")
        if videos.dtype != np.uint8:
            videos = np.clip(videos, 0, 255).astype(np.uint8)

        batch_size = len(videos)

        if self._vbench_cls is None:
            if not self._warned_once:
                self._warned_once = True
                print(
                    f"[VBenchScorer] VBench not available ({self._import_error}). "
                    "Falling back to temporal-consistency proxy."
                )
            return self._fallback_temporal_consistency(videos), {}

        with tempfile.TemporaryDirectory(prefix="vbench_") as tmpdir:
            # Save videos as mp4 files
            video_dir = os.path.join(tmpdir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            output_path = os.path.join(tmpdir, "output")
            os.makedirs(output_path, exist_ok=True)

            prompt_map = {}
            for idx, (video, prompt) in enumerate(zip(videos, prompts)):
                video_path = os.path.join(video_dir, f"sample_{idx:04d}.mp4")
                try:
                    imageio.mimsave(
                        video_path, list(video),
                        fps=self.video_fps, codec="libx264", format="FFMPEG",
                    )
                    prompt_map[video_path] = prompt
                except Exception as e:
                    if not self._warned_once:
                        print(f"[VBenchScorer] Video export failed: {e}")
                    return self._fallback_temporal_consistency(videos), {}

            # Run VBench evaluation per-dimension so that one missing/broken
            # submodule (e.g., GRiT for object_class) doesn't sink the whole
            # eval. Failed dimensions are dropped; we report whatever worked.
            vbench = self._get_vbench(output_path)
            if vbench is None:
                if not self._warned_once:
                    self._warned_once = True
                    print("[VBenchScorer] Could not initialize VBench. Using fallback.")
                return self._fallback_temporal_consistency(videos), {}

            # Filter out dims VBench can't handle in custom_input mode so the
            # bulk evaluate() doesn't trip the assert and abort the batch.
            supported = [d for d in self.dimensions if d not in CUSTOM_INPUT_UNSUPPORTED_DIMS]
            unsupported_skipped = [d for d in self.dimensions if d in CUSTOM_INPUT_UNSUPPORTED_DIMS]

            dim_scores: dict[str, list[float]] = {}
            failed_dims: list[str] = list(unsupported_skipped)

            # Fast path: one bulk evaluate() loads each submodule once and
            # iterates the videos for all supported dims.
            bulk_name = "reward_eval"
            bulk_ok = False
            if supported:
                try:
                    with _vbench_single_process():
                        vbench.evaluate(
                            videos_path=video_dir,
                            name=bulk_name,
                            prompt_list=prompt_map,
                            dimension_list=supported,
                            mode="custom_input",
                            local=True,
                        )
                    bulk_ok = True
                except Exception as e:
                    if not self._warned_once:
                        print(f"[VBenchScorer] bulk evaluate failed ({e}); falling back to per-dim.")

            if bulk_ok:
                parsed = self._parse_vbench_results(output_path, bulk_name, supported, batch_size)
                for dim in supported:
                    if dim in parsed and any(parsed[dim]):
                        dim_scores[dim] = parsed[dim]
                    else:
                        failed_dims.append(dim)
            else:
                # Per-dim slow fallback: one dim at a time, isolating failures.
                for dim in supported:
                    eval_name = f"reward_eval_{dim}"
                    try:
                        with _vbench_single_process():
                            vbench.evaluate(
                                videos_path=video_dir,
                                name=eval_name,
                                prompt_list=prompt_map,
                                dimension_list=[dim],
                                mode="custom_input",
                                local=True,
                            )
                    except Exception as e:
                        failed_dims.append(dim)
                        if not self._warned_once:
                            print(f"[VBenchScorer] dim={dim} failed: {e}. Skipping.")
                        continue

                    parsed = self._parse_vbench_results(output_path, eval_name, [dim], batch_size)
                    if dim in parsed and any(parsed[dim]):
                        dim_scores[dim] = parsed[dim]
                    else:
                        failed_dims.append(dim)

            if failed_dims and not self._warned_once:
                self._warned_once = True
                print(f"[VBenchScorer] Skipped dims with no parsed scores: {failed_dims}")

            if not dim_scores:
                if not self._warned_once:
                    self._warned_once = True
                    print("[VBenchScorer] No VBench dimensions succeeded. Using fallback.")
                return self._fallback_temporal_consistency(videos), {}

            # Average across evaluated dimensions per video (single scalar reward).
            stacked = np.stack([np.array(dim_scores[d]) for d in dim_scores], axis=0)
            all_scores = stacked.mean(axis=0)

            # Build the dim-dict: each evaluated dimension + Quality/Semantic/Total.
            dim_dict: dict[str, list[float]] = {dim: list(scores) for dim, scores in dim_scores.items()}
            dim_dict.update(self._aggregate_quality_semantic_total(dim_scores, batch_size))

            return all_scores.tolist(), dim_dict
