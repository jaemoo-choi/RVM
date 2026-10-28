from PIL import Image
import io
import numpy as np
import torch
import gc
import logging
import os
import sys
import json
import socket
import subprocess
import tempfile
import time
import atexit
from collections import defaultdict


logger = logging.getLogger(__name__)


# Default location of the pip overlay that shadows the training env's
# transformers 4.46 with 4.53.2 + qwen_vl_utils for the Qwen2.5-VL judges.
# The sbatch scripts set VLM_OVERLAY_PYTHONPATH; this is the fallback.
from config.paths import VS2_OVERLAY as _DEFAULT_VLM_OVERLAY  # noqa: E402


def _is_cuda_oom(error):
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _build_score_function(score_name, factory, device):
    if "device" not in factory.__code__.co_varnames:
        return factory()

    target_device = torch.device(device) if not isinstance(device, torch.device) else device

    try:
        return factory(target_device)
    except Exception as error:
        if target_device.type != "cuda" or not _is_cuda_oom(error):
            raise

        logger.warning(
            "CUDA OOM while initializing reward scorer '%s' on %s; retrying on CPU.",
            score_name,
            target_device,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return factory(torch.device("cpu"))


def _to_bchw_float01(images):
    """Convert image/video batch to B,C,H,W float in [0, 1].

    For videos, the first frame is selected.
    Supports torch/numpy in channel-first or channel-last layouts.
    """
    if isinstance(images, torch.Tensor):
        x = images.detach().cpu()
        if x.ndim == 5:
            # B,T,C,H,W or B,T,H,W,C
            if x.shape[2] == 3:
                x = x[:, 0]
            elif x.shape[-1] == 3:
                x = x[:, 0].permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 5D tensor shape: {tuple(x.shape)}")
        elif x.ndim == 4:
            # B,C,H,W or B,H,W,C
            if x.shape[1] == 3:
                pass
            elif x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 4D tensor shape: {tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported tensor rank for images/videos: {x.ndim}")

        x = x.float()
        if x.numel() > 0 and x.max().item() > 1.0:
            x = x / 255.0
        return x.clamp(0, 1)

    if isinstance(images, np.ndarray):
        x = images
        if x.ndim == 5:
            # B,T,C,H,W or B,T,H,W,C
            if x.shape[2] == 3:
                x = x[:, 0]
            elif x.shape[-1] == 3:
                x = x[:, 0].transpose(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 5D ndarray shape: {x.shape}")
        elif x.ndim == 4:
            # B,C,H,W or B,H,W,C
            if x.shape[1] == 3:
                pass
            elif x.shape[-1] == 3:
                x = x.transpose(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 4D ndarray shape: {x.shape}")
        else:
            raise ValueError(f"Unsupported ndarray rank for images/videos: {x.ndim}")

        x = torch.from_numpy(x).float()
        if x.numel() > 0 and x.max().item() > 1.0:
            x = x / 255.0
        return x.clamp(0, 1)

    raise ValueError(f"Unsupported input type for image conversion: {type(images)}")


def _to_bhwc_uint8(images):
    bchw = _to_bchw_float01(images)
    arr = (bchw * 255).round().to(torch.uint8).numpy()
    return arr.transpose(0, 2, 3, 1)


def _to_bthwc_uint8(images):
    """Convert image/video batch to B,T,H,W,C uint8.

    For image batches, T=1.
    """
    if isinstance(images, torch.Tensor):
        x = images.detach().cpu()
        if x.ndim == 5:
            # B,T,C,H,W or B,T,H,W,C
            if x.shape[2] == 3:
                pass
            elif x.shape[-1] == 3:
                x = x.permute(0, 1, 4, 2, 3)
            else:
                raise ValueError(f"Unsupported 5D tensor shape for videos: {tuple(x.shape)}")
        elif x.ndim == 4:
            # B,C,H,W or B,H,W,C
            if x.shape[1] == 3:
                x = x.unsqueeze(1)
            elif x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2).unsqueeze(1)
            else:
                raise ValueError(f"Unsupported 4D tensor shape for images/videos: {tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported tensor rank for videos: {x.ndim}")

        x = x.float()
        if x.numel() > 0 and x.max().item() <= 1.0:
            x = x * 255.0
        x = x.clamp(0, 255).round().to(torch.uint8)
        return x.permute(0, 1, 3, 4, 2).numpy()

    if isinstance(images, np.ndarray):
        x = images
        if x.ndim == 5:
            # B,T,C,H,W or B,T,H,W,C
            if x.shape[2] == 3:
                x = x.transpose(0, 1, 3, 4, 2)
            elif x.shape[-1] == 3:
                pass
            else:
                raise ValueError(f"Unsupported 5D ndarray shape for videos: {x.shape}")
        elif x.ndim == 4:
            # B,C,H,W or B,H,W,C
            if x.shape[1] == 3:
                x = x.transpose(0, 2, 3, 1)
            elif x.shape[-1] == 3:
                pass
            else:
                raise ValueError(f"Unsupported 4D ndarray shape for images/videos: {x.shape}")
            x = np.expand_dims(x, axis=1)
        else:
            raise ValueError(f"Unsupported ndarray rank for videos: {x.ndim}")

        if x.dtype != np.uint8:
            if np.issubdtype(x.dtype, np.floating) and x.size > 0 and np.max(x) <= 1.0:
                x = x * 255.0
            x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    raise ValueError(f"Unsupported input type for video conversion: {type(images)}")


# ---------------------------------------------------------------------------
# Qwen2.5-VL judge rewards (VideoScore2, UnifiedReward-2.0) via overlay worker
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _OverlayVLMProxy(torch.nn.Module):
    """In-process handle for a Qwen2.5-VL judge running in an overlay subprocess.

    VideoScore2 / UnifiedReward-2.0 need transformers >= 4.49, which conflicts
    with the trainer's pinned 4.46. We therefore run the actual model in a child
    process launched with the ``vs2_overlay`` PYTHONPATH (transformers 4.53.2),
    and talk to it over a unix socket (see ``vlm_reward_worker.py``). This proxy:

    - writes the generated frames to temp mp4s **in the trainer process** (imageio
      lives in the 4.46 env) and sends the paths to the worker;
    - is an ``nn.Module`` with no params so ``MultiScorer._get_scorer_modules``
      picks it up; its ``.to()`` is overridden to forward offload/reload to the
      worker, mirroring the in-process VideoAlign offload around the train step.
    """

    def __init__(self, device, model_key, fps=16, axes=None, connect_timeout=900.0, rpc_timeout=7200.0):
        super().__init__()
        self.model_key = model_key
        self.axes = tuple(axes) if axes else None
        self._device_str = str(device)
        self.fps = int(fps)
        self._rpc_timeout = rpc_timeout
        self._tmpdir = tempfile.mkdtemp(prefix=f"vlm_{model_key}_")
        self._sock_path = os.path.join(self._tmpdir, "worker.sock")
        self._proc = None
        self._conn = None
        self._spawn()
        self._connect(connect_timeout)
        atexit.register(self.close)

    # -- lifecycle ---------------------------------------------------------
    def _spawn(self):
        overlay = os.environ.get("VLM_OVERLAY_PYTHONPATH", _DEFAULT_VLM_OVERLAY)
        if not os.path.isdir(overlay):
            raise RuntimeError(
                f"VLM overlay PYTHONPATH not found: {overlay}. Set VLM_OVERLAY_PYTHONPATH "
                f"to the transformers-4.53.2 + qwen_vl_utils overlay dir."
            )
        env = os.environ.copy()
        # Overlay first so its transformers/qwen_vl_utils shadow the 4.46 env;
        # repo root kept so the child can `import src.reward...`.
        env["PYTHONPATH"] = os.pathsep.join([overlay, _REPO_ROOT, env.get("PYTHONPATH", "")])
        worker = os.path.join(_REPO_ROOT, "src", "reward", "vlm_reward_worker.py")
        cmd = [
            sys.executable, "-u", worker,
            "--model", self.model_key,
            "--device", self._device_str,
            "--socket", self._sock_path,
            "--repo", _REPO_ROOT,
        ]
        if self.axes:
            cmd += ["--axes", ",".join(self.axes)]
        logger.info("Spawning %s overlay worker: %s", self.model_key, " ".join(cmd))
        self._proc = subprocess.Popen(cmd, env=env)

    def _connect(self, timeout):
        deadline = time.monotonic() + timeout
        last_err = None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"{self.model_key} worker exited (code {self._proc.returncode}) before "
                    f"binding its socket; check stderr above."
                )
            if os.path.exists(self._sock_path):
                try:
                    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    conn.connect(self._sock_path)
                    conn.settimeout(self._rpc_timeout)
                    self._conn = conn
                    logger.info("Connected to %s worker", self.model_key)
                    return
                except OSError as exc:  # socket file exists but not listening yet
                    last_err = exc
            time.sleep(1.0)
        raise RuntimeError(
            f"Timed out after {timeout:.0f}s connecting to {self.model_key} worker "
            f"(last error: {last_err})."
        )

    def _rpc(self, obj):
        self._conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))
        chunks = []
        while True:
            b = self._conn.recv(65536)
            if not b:
                raise RuntimeError(f"{self.model_key} worker closed the connection unexpectedly.")
            chunks.append(b.decode("utf-8"))
            if "\n" in chunks[-1]:
                break
        resp = json.loads("".join(chunks))
        if not resp.get("ok"):
            raise RuntimeError(f"{self.model_key} worker error: {resp.get('error')}")
        return resp

    def close(self):
        if self._conn is not None:
            try:
                self._rpc({"cmd": "shutdown"})
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()

    # -- offload hooks (called by MultiScorer.to_cpu/to_gpu) ---------------
    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        dev_str = str(device) if device is not None else ""
        if self._conn is not None:
            self._rpc({"cmd": "to_cpu" if dev_str.startswith("cpu") else "to_gpu"})
        return self

    # -- scoring -----------------------------------------------------------
    def forward(self, images, prompts, metadata):
        import imageio

        videos = _to_bthwc_uint8(images)  # [B, T, H, W, C] uint8
        paths = []
        for idx, video in enumerate(videos):
            out_path = os.path.join(self._tmpdir, f"sample_{idx}.mp4")
            imageio.mimsave(out_path, list(video), fps=self.fps, codec="libx264", format="FFMPEG")
            paths.append(out_path)
        resp = self._rpc({"cmd": "score", "video_paths": paths, "prompts": list(prompts)})
        for p in paths:  # don't let temp mp4s accumulate across epochs
            try:
                os.unlink(p)
            except OSError:
                pass
        return resp["scores"], {}

    __call__ = forward


def videoscore2_score(device, fps=16):
    """VideoScore2 (TIGER-Lab) judge reward: mean of VQ/TA/PC `exp`, in [0, 1]."""
    return _OverlayVLMProxy(device, "videoscore2", fps=fps)


def unifiedreward2_score(device, fps=16):
    """UnifiedReward-2.0 (CodeGoat24) judge reward: mean of AL/PH/ST `exp`, in [0, 1]."""
    return _OverlayVLMProxy(device, "unifiedreward2", fps=fps)


def spectra_reward_score(device, fps=16):
    """SpectraReward: training-free MLLM reward, mean log p(prompt | video).

    "Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for
    Text-to-Image Generation" (arXiv:2607.11886). A FROZEN pretrained MLLM
    teacher-forces the generation prompt conditioned on the sample; the reward
    is the mean per-token log-likelihood, so no reward model is trained and no
    preference data is needed. Unlike the judge rewards it is NOT in [0, 1]:
    values are negative log-probs (typically ~[-3, 0]), higher = the prompt
    reads back better. Weight accordingly (GRPO normalises within-group, so the
    group STD is what matters).
    """
    return _OverlayVLMProxy(device, "spectra", fps=fps)


def semantic_v2a_score(device, fps=15):
    """Prompt-engineered semantic reward `v2a_absence`: rating, in [0, 1].

    A Qwen2.5-VL judge counts how many of the prompt's STATED things are missing
    from or contradicted by the clip, answers one digit 1-5, and the reward is the
    first-token expectation mapped to [0, 1]. Template (the actual reward) lives at
    `scripts/eval/semantic_prompts/v2a_absence.txt`.

    Measured on the 12-set / 4,377-video benchmark it is the ONLY component with
    positive skill on both axes (Quality +0.135 / Semantic +0.278); std 0.138, so
    weight it ~7x a unit-std component. `fps=15` writes the temp mp4 at the
    generation frame rate the benchmark used; the judge's own frame sampling is
    the template's `# fps: 2.0`.
    """
    return _OverlayVLMProxy(device, "semantic_v2a", fps=fps)


def semantic_v4c_score(device, fps=15):
    """Prompt-engineered semantic reward `v4c_short_yesno`: yes/no, in [0, 1].

    Same judge, stricter question: is EVERY stated thing present and correct
    (No if even one is missing), with an explicit instruction to ignore anything
    the prompt never states. Reward is p(Yes)/(p(Yes)+p(No)) at the first token.
    Template: `scripts/eval/semantic_prompts/v4c_short_yesno.txt`.

    Best component on human_action (+0.658) and the best Semantic average
    (+0.291); std 0.069, so it needs ~15x a unit-std component. NOTE: benchmarked
    against the short VBench sentence via `{short_prompt}`; training data has one
    prompt per sample, so it sees the extended caption instead (see
    semantic_vlm_scorer.py).
    """
    return _OverlayVLMProxy(device, "semantic_v4c", fps=fps)


def videoscore2_ta_score(device, fps=16):
    """VideoScore2 text-to-video-alignment axis ONLY (`TA` exp, in [0, 1]).

    Per-axis variant added 2026-08-24 for the A14B alternating VLM-TA reward
    mix: the mean-of-3 `videoscore2` dilutes the alignment signal with VQ/PC.
    Empirical spread on the 104-clip human-eval set: std 0.163 — weight it
    ~4-5x to match videoalign_ta's effective magnitude (std 0.75 at weight 1).
    """
    return _OverlayVLMProxy(device, "videoscore2", fps=fps, axes=("TA",))


def unifiedreward2_al_score(device, fps=16):
    """UnifiedReward-2.0 Alignment axis ONLY (`AL` exp, in [0, 1]).

    Runner-up judge in the VLM-TA mix. Very compressed spread (std 0.044 on
    the human-eval set) — a light second opinion, not a primary signal.
    """
    return _OverlayVLMProxy(device, "unifiedreward2", fps=fps, axes=("AL",))


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        images = _to_bhwc_uint8(images)
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn


def aesthetic_score(device):
    from .aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = (_to_bchw_float01(images) * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from .clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        images = _to_bchw_float01(images)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def hpsv3_score(device):
    from .hpsv3_scorer import HPSv3Scorer

    from .hpsv3_scorer import get_shared_hpsv3_scorer
    scorer = get_shared_hpsv3_scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = _to_bchw_float01(images)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def video_hpsv3_score(device, num_sample_frames=4):
    """HPSv3 for video: uniformly samples frames and averages per-frame scores."""
    from .hpsv3_scorer import HPSv3Scorer

    from .hpsv3_scorer import get_shared_hpsv3_scorer
    scorer = get_shared_hpsv3_scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            x = images.detach().cpu().float()
        else:
            x = torch.from_numpy(images).float()

        if x.numel() > 0 and x.max().item() > 1.0:
            x = x / 255.0
        x = x.clamp(0, 1)

        if x.ndim == 5:
            if x.shape[-1] == 3:
                x = x.permute(0, 1, 4, 2, 3)
            scores = scorer.score_video(x, prompts, num_sample_frames=num_sample_frames)
        else:
            if x.ndim == 4 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            scores = scorer(x, prompts)

        return scores, {}

    return _fn


def _normalize_video_tensor(images):
    """Helper: turn the reward dispatcher's image arg into a [B,T,C,H,W] tensor in [0,1]."""
    if isinstance(images, torch.Tensor):
        x = images.detach().cpu().float()
    else:
        x = torch.from_numpy(images).float()
    if x.numel() > 0 and x.max().item() > 1.0:
        x = x / 255.0
    x = x.clamp(0, 1)
    if x.ndim == 5 and x.shape[-1] == 3:
        x = x.permute(0, 1, 4, 2, 3).contiguous()
    return x


def video_hpsv3_general_score(device, num_sample_frames=4):
    """HPSv3-general: per-frame mean score with a fixed visual-quality prompt
    (``"A high-quality image"``), ignoring the video caption. Pure visual
    quality signal — frame-averaged like ``video_hpsv3_score`` but with the
    prompt swapped.
    """
    from .hpsv3_scorer import HPSv3Scorer

    from .hpsv3_scorer import get_shared_hpsv3_scorer
    scorer = get_shared_hpsv3_scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        x = _normalize_video_tensor(images)
        if x.ndim != 5:
            raise ValueError(
                f"video_hpsv3_general expects [B,T,C,H,W], got shape {tuple(x.shape)}"
            )
        scores = scorer.score_video_general(x, num_sample_frames=num_sample_frames)
        return scores, {}

    return _fn


def video_hpsv3_percentile_score(device, num_sample_frames=4, top_k_pct=0.3):
    """HPSv3-percentile: text-video alignment using the video caption, but
    aggregated over the top ``top_k_pct`` (default 30%) of per-frame scores
    rather than the mean. Robust to temporal content drift dragging the
    average down.
    """
    from .hpsv3_scorer import HPSv3Scorer

    from .hpsv3_scorer import get_shared_hpsv3_scorer
    scorer = get_shared_hpsv3_scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        x = _normalize_video_tensor(images)
        if x.ndim != 5:
            raise ValueError(
                f"video_hpsv3_percentile expects [B,T,C,H,W], got shape {tuple(x.shape)}"
            )
        scores = scorer.score_video_percentile(
            x, prompts, num_sample_frames=num_sample_frames, top_k_pct=top_k_pct,
        )
        return scores, {}

    return _fn


def video_dino_ia_score(device, num_sample_frames=4):
    """DINOv2 image-alignment for I2V (Q-Save IA proxy).

    Computes weighted cosine similarity between the per-sample input image
    (read from ``metadata[i]['ref_image']``, populated by ``ImagePromptDataset``)
    and a small set of generated video frames. See ``src/reward/dino_ia_scorer.py``
    for the formula.
    """
    from .dino_ia_scorer import DinoIAScorer

    scorer = DinoIAScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if not metadata:
            raise ValueError("dino_ia requires per-sample metadata with 'ref_image'.")
        ref_images = []
        for m in metadata:
            ref = m.get("ref_image") if isinstance(m, dict) else None
            if ref is None:
                raise ValueError("dino_ia: each metadata entry must carry a 'ref_image'.")
            ref_images.append(ref)

        if isinstance(images, torch.Tensor):
            x = images.detach().cpu().float()
        else:
            x = torch.from_numpy(images).float()

        if x.numel() > 0 and x.max().item() > 1.5:
            x = x / 255.0
        x = x.clamp(0, 1)

        if x.ndim != 5:
            raise ValueError(f"dino_ia expects a 5-D video tensor, got shape {tuple(x.shape)}")
        if x.shape[-1] == 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        T = x.shape[1]

        if T >= 53:
            frame_ids = [0, 17, 35, 52]
        else:
            n = max(1, min(num_sample_frames, T))
            frame_ids = sorted(set(torch.linspace(0, T - 1, n).round().long().tolist()))

        scores = scorer.score_video(x, ref_images, frame_ids=frame_ids)
        return scores, {}

    return _fn


def hpsv3_quality_drift_score(device):
    """HPSv3 quality drift: penalizes quality degradation relative to first frame.

    Scores frames with the fixed prompt "A high-quality, ultra-detailed and
    well-structured image." and returns -|quality(frame) - quality(first_frame)|.
    """
    from .hpsv3_scorer import HPSv3Scorer

    from .hpsv3_scorer import get_shared_hpsv3_scorer
    scorer = get_shared_hpsv3_scorer(dtype=torch.float32, device=device)
    quality_prompt = "A high-quality, ultra-detailed and well-structured image."

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            x = images.detach().cpu().float()
        else:
            x = torch.from_numpy(images).float()

        if x.numel() > 0 and x.max().item() > 1.0:
            x = x / 255.0
        x = x.clamp(0, 1)

        # Ensure [B, T, C, H, W]
        if x.ndim == 5 and x.shape[-1] == 3:
            x = x.permute(0, 1, 4, 2, 3)

        B, T, C, H, W = x.shape

        # Score first frame
        first_frames = x[:, 0]  # [B, C, H, W]
        first_quality = scorer(first_frames, [quality_prompt] * B)  # [B]

        # Score sampled frames (every 4th frame, matching original)
        sample_indices = list(range(3, T, 4))  # frames at index 3, 7, 11, ...
        if not sample_indices:
            sample_indices = [T - 1]

        drift_scores = torch.zeros(B, device=scorer.device)
        for fi in sample_indices:
            frame_batch = x[:, fi]  # [B, C, H, W]
            frame_quality = scorer(frame_batch, [quality_prompt] * B)  # [B]
            drift_scores += -1.0 * torch.abs(frame_quality - first_quality)

        drift_scores /= len(sample_indices)
        return drift_scores, {}

    return _fn


def hpsv2_score(device):
    from .hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = _to_bchw_float01(images)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def video_hpsv2_score(device, num_sample_frames=4):
    """HPSv2 for video: uniformly samples frames and averages per-frame scores.

    Following BranchGRPO's pattern of scoring individual frames with HPSv2
    and aggregating, rather than only using the first frame.
    """
    from .hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            x = images.detach().cpu().float()
        else:
            x = torch.from_numpy(images).float()

        # Normalise to [0, 1]
        if x.numel() > 0 and x.max().item() > 1.0:
            x = x / 255.0
        x = x.clamp(0, 1)

        if x.ndim == 5:
            # B,T,C,H,W or B,T,H,W,C
            if x.shape[-1] == 3:
                x = x.permute(0, 1, 4, 2, 3)
            scores = scorer.score_video(x, prompts, num_sample_frames=num_sample_frames)
        else:
            # Fallback: image input → same as hpsv2_score
            if x.ndim == 4 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            scores = scorer(x, prompts)

        return scores, {}

    return _fn


def pickscore_score(device):
    from .pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = _to_bhwc_uint8(images)
        images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def imagereward_score(device):
    from .imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = _to_bhwc_uint8(images)
        images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def geneval_score(device):
    from .gen_eval import load_geneval

    batch_size = 64
    compute_geneval = load_geneval(device)

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        images = _to_bhwc_uint8(images)
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            pil_images = [Image.fromarray(image) for image in image_batch]

            data = {
                "images": pil_images,
                "metadatas": list(metadata_batched),
                "only_strict": only_strict,
            }
            scores, rewards, strict_rewards, group_rewards, group_strict_rewards = compute_geneval(**data)

            all_scores += scores
            all_rewards += rewards
            all_strict_rewards += strict_rewards
            all_group_strict_rewards.append(group_strict_rewards)
            all_group_rewards.append(group_rewards)
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn


def ocr_score(device):
    from .ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        images = _to_bhwc_uint8(images)
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def video_ocr_score(device):
    from .ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        images = _to_bthwc_uint8(images)
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def vbench_score(device, video_fps=None):
    from .vbench_scorer import VBenchScorer

    kwargs = {"device": device}
    if video_fps is not None:
        kwargs["video_fps"] = int(video_fps)
    scorer = VBenchScorer(**kwargs)

    def _fn(images, prompts, metadata):
        images = _to_bthwc_uint8(images)
        scores, dim_scores = scorer(images, prompts, metadata)
        return scores, dim_scores

    return _fn


def vbench_full_score(device, video_fps=None):
    """Full 16-dimension VBench evaluation (offline use only — slow).

    Matches the TaRoS evaluation protocol: VBench Quality, Semantic, Total.
    Requires full VBench installation with all model checkpoints.
    """
    from .vbench_scorer import VBenchScorer, ALL_DIMENSIONS

    kwargs = {"device": device, "dimensions": ALL_DIMENSIONS}
    if video_fps is not None:
        kwargs["video_fps"] = int(video_fps)
    scorer = VBenchScorer(**kwargs)

    def _fn(images, prompts, metadata):
        images = _to_bthwc_uint8(images)
        scores, dim_scores = scorer(images, prompts, metadata)
        return scores, dim_scores

    return _fn


def videoalign_score(device, video_fps=16, reward_key="Overall", eval_num_frames=None, dimension_weights=None):
    from .videoalign_scorer import VideoAlignScorer

    scorer = VideoAlignScorer(
        device=device,
        video_fps=video_fps,
        reward_key=reward_key,
        eval_num_frames=eval_num_frames,
        dimension_weights=dimension_weights,
    )

    def _fn(images, prompts, metadata):
        images = _to_bthwc_uint8(images)
        scores, dim_scores = scorer(images, prompts, metadata)
        return scores, dim_scores

    return _fn


def _videoalign_single_dim_score(device, dim, video_fps=8):
    """Build a closure that scores a single VideoAlign dimension via the
    shared backbone. The underlying ``VideoAlignScorer`` always returns all
    four dims; this wrapper picks ``dim`` and exposes the rest as detail.
    """
    from .videoalign_scorer import get_shared_videoalign_scorer

    scorer = get_shared_videoalign_scorer(device, video_fps=video_fps)

    def _fn(images, prompts, metadata):
        images = _to_bthwc_uint8(images)
        _, dim_scores = scorer(images, prompts, metadata)
        # dim_scores: {"VQ": [...], "MQ": [...], "TA": [...], "Overall": [...]}
        # Return only the picked dim — suppress the full dict so the aggregator
        # doesn't log redundant cross-keys like reward_videoalign_mq_TA when
        # videoalign_ta is already registered as its own scorer.
        return dim_scores[dim], {}

    return _fn


def videoalign_vq_score(device):
    """VideoAlign Visual Quality only (used by HunyuanVideo in DanceGRPO)."""
    return _videoalign_single_dim_score(device, "VQ")


def videoalign_mq_score(device):
    """VideoAlign Motion Quality only (used by SkyReels-I2V in DanceGRPO)."""
    return _videoalign_single_dim_score(device, "MQ")


def videoalign_ta_score(device):
    """VideoAlign Text Alignment only."""
    return _videoalign_single_dim_score(device, "TA")


def videogen_reward_score(device):
    """VideoGen-Reward (VideoReward from KlingAI/VideoAlign).

    Overall score = VQ + MQ + TA. Uses Qwen2-VL-2B backbone with
    specialized reward heads. Weights from HuggingFace: KwaiVGI/VideoReward.
    """
    return videoalign_score(device, video_fps=8, reward_key="Overall")


def dynamic_degree_score(device, num_pairs=8, pass_rate=0.25):
    """VBench-style binary dynamic_degree reward: 1.0 if ≥`pass_rate` of
    evaluated pairs exceed the per-pair threshold `6 * min(H,W)/256`, else 0.0.

    Same scale as VBench's leaderboard score (0/1 per video, averaged across
    a dataset). Comparable in magnitude to VideoAlign-MQ's z-scored output
    (~[-2, +2]) once weighted into the recipe. Includes the raw mean flow
    magnitude in the metadata dict for diagnostic logging.
    """
    from .dynamic_degree_scorer import get_shared_dynamic_degree_scorer
    scorer = get_shared_dynamic_degree_scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        x = _normalize_video_tensor(images)
        if x.ndim != 5:
            raise ValueError(
                f"dynamic_degree expects [B,T,C,H,W], got shape {tuple(x.shape)}"
            )
        binary = scorer.score_video_vbench_binary(
            x, num_pairs=num_pairs, pass_rate=pass_rate,
        )
        # Also return the underlying continuous score for diagnostics.
        with torch.no_grad():
            mean_flow = scorer.score_video_no_grad(x, num_pairs=num_pairs)
        return binary, {"raw_flow": mean_flow}

    return _fn


def dynamic_degree_continuous_score(device, num_pairs=8):
    """Continuous flow-magnitude reward (mean of per-pair top-5%). Use only
    if you want a continuous gradient signal — not a 1:1 match with VBench's
    leaderboard scoring.
    """
    from .dynamic_degree_scorer import get_shared_dynamic_degree_scorer
    scorer = get_shared_dynamic_degree_scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        x = _normalize_video_tensor(images)
        if x.ndim != 5:
            raise ValueError(
                f"dynamic_degree_continuous expects [B,T,C,H,W], got shape {tuple(x.shape)}"
            )
        return scorer.score_video_no_grad(x, num_pairs=num_pairs), {}

    return _fn


def dynamic_degree_smooth_score(device, num_pairs=8, tau_scale=1.0):
    """Smooth (clipped-linear) dynamic_degree reward in [0, 1].

    Per-pair soft pass ``clamp(m_k / (tau_scale * tau), 0, 1)`` averaged over
    pairs (see ``DynamicDegreeScorer.score_video_smooth``). Bounded [0, 1] like
    the binary `dynamic_degree`, so it is a drop-in at the same weight, but
    continuous — gives nonzero GRPO advantage even when a prompt's K rollouts
    are all weakly-moving. Opt in by listing ``"dynamic_degree_smooth"`` in a
    config ``reward_fn``; the binary ``"dynamic_degree"`` is unchanged.

    ``tau_scale`` (saturation point) is fixed at 1.0: full credit exactly at the
    VBench motion threshold. >1 would demand more motion, <1 less.
    """
    tau_scale = float(tau_scale)

    from .dynamic_degree_scorer import get_shared_dynamic_degree_scorer
    scorer = get_shared_dynamic_degree_scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        x = _normalize_video_tensor(images)
        if x.ndim != 5:
            raise ValueError(
                f"dynamic_degree_smooth expects [B,T,C,H,W], got shape {tuple(x.shape)}"
            )
        smooth = scorer.score_video_smooth(
            x, num_pairs=num_pairs, tau_scale=tau_scale,
        )
        # Same diagnostic as the binary path: raw mean top-5% flow magnitude.
        with torch.no_grad():
            mean_flow = scorer.score_video_no_grad(x, num_pairs=num_pairs)
        return smooth, {"raw_flow": mean_flow}

    return _fn


def dynamic_degree_smooth2_score(device, num_pairs=8, tau_scale=1.0, pass_rate=0.25):
    """Smooth dynamic_degree v2 reward in (0, 1).

    Blends the clipped-linear motion-quantity term (``r_clamp``, identical to
    ``dynamic_degree_smooth``) with a smooth sigmoid relaxation of VBench's
    ≥``pass_rate``-of-pairs-moving binary gate (``r_soft_binary``), weighted
    ``0.2 * r_clamp + 0.8 * r_soft_binary`` (see
    ``DynamicDegreeScorer.score_video_smooth2``). Like the binary
    `dynamic_degree`, it is bounded so it drops in at the same weight, but it is
    continuous AND rewards getting *enough* pairs over threshold (not just more
    flow on a few). Opt in by listing ``"dynamic_degree_smooth2"`` in a config
    ``reward_fn``; the binary ``"dynamic_degree"`` and ``"dynamic_degree_smooth"``
    are unchanged.

    ``tau_scale`` (saturation point) is fixed at 1.0: full credit exactly at the
    VBench motion threshold. >1 demands more motion, <1 less.
    """
    tau_scale = float(tau_scale)

    from .dynamic_degree_scorer import get_shared_dynamic_degree_scorer
    scorer = get_shared_dynamic_degree_scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        x = _normalize_video_tensor(images)
        if x.ndim != 5:
            raise ValueError(
                f"dynamic_degree_smooth2 expects [B,T,C,H,W], got shape {tuple(x.shape)}"
            )
        smooth2 = scorer.score_video_smooth2(
            x, num_pairs=num_pairs, tau_scale=tau_scale, pass_rate=pass_rate,
        )
        # Same diagnostic as the binary path: raw mean top-5% flow magnitude.
        with torch.no_grad():
            mean_flow = scorer.score_video_no_grad(x, num_pairs=num_pairs)
        return smooth2, {"raw_flow": mean_flow}

    return _fn


def dynamic_degree_smooth3_score(
    device,
    num_pairs=8,
    tau_scale=1.0,
    bg_pct=0.5,
    bg_cap_scale=1.0,
    bg_max_scale=3.0,
    bg_penalty=1.0,
):
    """Smooth dynamic_degree v3 reward in [0, 1] — background-aware.

    Rewards the top-5% (subject/scene) motion exactly like
    ``dynamic_degree_smooth`` (``clamp(m / (tau_scale * tau), 0, 1)``) but
    multiplies it by a smooth penalty on *too-fast background / global motion*,
    proxied by the per-pixel flow ``bg_pct`` quantile (default median). A fast
    subject on a still background keeps full credit; a fast camera pan (which
    moves nearly all pixels) is smoothly penalized once its background flow
    exceeds ``bg_cap_scale * tau`` and is fully penalized at ``bg_max_scale *
    tau``. Camera motion below the cap is free — see
    ``DynamicDegreeScorer.score_video_smooth3``.

    Bounded [0, 1] like the binary `dynamic_degree`, so it drops in at the same
    weight. Opt in by listing ``"dynamic_degree_smooth3"`` in a config
    ``reward_fn``; the binary `dynamic_degree` and the other smooth variants are
    unchanged.
    """
    tau_scale = float(tau_scale)
    bg_pct = float(bg_pct)
    bg_cap_scale = float(bg_cap_scale)
    bg_max_scale = float(bg_max_scale)
    bg_penalty = float(bg_penalty)

    from .dynamic_degree_scorer import get_shared_dynamic_degree_scorer
    scorer = get_shared_dynamic_degree_scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        x = _normalize_video_tensor(images)
        if x.ndim != 5:
            raise ValueError(
                f"dynamic_degree_smooth3 expects [B,T,C,H,W], got shape {tuple(x.shape)}"
            )
        smooth3 = scorer.score_video_smooth3(
            x,
            num_pairs=num_pairs,
            tau_scale=tau_scale,
            bg_pct=bg_pct,
            bg_cap_scale=bg_cap_scale,
            bg_max_scale=bg_max_scale,
            bg_penalty=bg_penalty,
        )
        # Diagnostics: raw mean top-5% flow magnitude (same as other variants)
        # plus the per-video background-flow proxy that drives the penalty.
        with torch.no_grad():
            mean_flow = scorer.score_video_no_grad(x, num_pairs=num_pairs)
            _, bg_pairs = scorer._score_per_pair_stats(x, num_pairs, bg_pct)
            bg_flow = bg_pairs.mean(dim=1)
        return smooth3, {"raw_flow": mean_flow, "bg_flow": bg_flow}

    return _fn


def dynamic_degree_bump_score(device, num_pairs=8, lo=None, c=None, hi=None, bg_pct=0.5):
    """Non-negative "bump" motion reward in [0, 1] — peaked on foreground motion.

    ``fg = (top-5% flow) - (median bg flow)`` = localized motion *above* the background
    (same stats as ``dynamic_degree_smooth3``: ``raw_flow`` and ``bg_flow``). The reward is a
    trapezoid window on ``fg``::

        R = clamp((fg - lo)/(c - lo), 0, 1) * clamp((hi - fg)/(hi - c), 0, 1)

    so it is 0 for too-little motion (static, ``fg < lo``), peaks at 1 at the real-motion level
    (``fg = c``), and falls to 0 for excess motion (fake / background churn, ``fg > hi``). Bounded
    [0, 1], non-negative, and **self-limiting** — "more flow" past ``c`` *lowers* the reward, so the
    background-churn hack earns ~0. Designed from a human-labeled set: defaults ``lo=3, c=5, hi=8``. Tune via env
    ``DD_BUMP_LO`` / ``DD_BUMP_C`` / ``DD_BUMP_HI``. Opt in with ``"dynamic_degree_bump"``.
    """
    import os

    lo = float(os.getenv("DD_BUMP_LO", "3.0")) if lo is None else float(lo)
    c = float(os.getenv("DD_BUMP_C", "5.0")) if c is None else float(c)
    hi = float(os.getenv("DD_BUMP_HI", "8.0")) if hi is None else float(hi)
    bg_pct = float(bg_pct)

    from .dynamic_degree_scorer import get_shared_dynamic_degree_scorer
    scorer = get_shared_dynamic_degree_scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        x = _normalize_video_tensor(images)
        if x.ndim != 5:
            raise ValueError(
                f"dynamic_degree_bump expects [B,T,C,H,W], got shape {tuple(x.shape)}"
            )
        with torch.no_grad():
            m, g = scorer._score_per_pair_stats(x, num_pairs, bg_pct)  # [B, num_pairs]
            raw_flow = m.mean(dim=1)
            bg_flow = g.mean(dim=1)
            fg = raw_flow - bg_flow
            rise = ((fg - lo) / (c - lo)).clamp(0.0, 1.0)
            fall = ((hi - fg) / (hi - c)).clamp(0.0, 1.0)
            bump = (rise * fall).clamp(0.0, 1.0)
        return bump, {"raw_flow": raw_flow, "bg_flow": bg_flow, "fg": fg}

    return _fn


def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        images = _to_bhwc_uint8(images)

        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]
        return score, {}

    return _fn


class MultiScorer:
    """Callable multi-reward scorer with GPU memory management.

    Supports offloading reward models to CPU during the training phase
    (when only the transformer is needed) and reloading to GPU for scoring.

    Usage:
        reward_fn = multi_score(device, score_dict)
        scores = reward_fn(images, prompts, metadata)  # normal scoring
        reward_fn.to_cpu()   # offload to CPU before training
        reward_fn.to_gpu()   # reload to GPU before next sampling
    """

    def __init__(self, device, score_dict, videoalign_dimension_weights=None, video_fps=None):
        score_functions = {
            "ocr": ocr_score,
            "video_ocr": video_ocr_score,
            "vbench": vbench_score,
            "vbench_full": vbench_full_score,
            "videoalign": videoalign_score,
            "videoalign_vq": videoalign_vq_score,
            "videoalign_mq": videoalign_mq_score,
            "videoalign_ta": videoalign_ta_score,
            "videoreward": videoalign_score,
            "videogen_reward": videogen_reward_score,
            "imagereward": imagereward_score,
            "pickscore": pickscore_score,
            "aesthetic": aesthetic_score,
            "jpeg_compressibility": jpeg_compressibility,
            "unifiedreward": unifiedreward_score_sglang,
            "spectra": spectra_reward_score,
            "semantic_v2a": semantic_v2a_score,
            "semantic_v4c": semantic_v4c_score,
            "videoscore2": videoscore2_score,
            "videoscore2_ta": videoscore2_ta_score,
            "unifiedreward2_al": unifiedreward2_al_score,
            "unifiedreward2": unifiedreward2_score,
            "geneval": geneval_score,
            "clipscore": clip_score,
            "hpsv2": hpsv2_score,
            "video_hpsv2": video_hpsv2_score,
            "hpsv3": hpsv3_score,
            "video_hpsv3": video_hpsv3_score,
            "video_hpsv3_general": video_hpsv3_general_score,
            "video_hpsv3_percentile": video_hpsv3_percentile_score,
            "hpsv3_quality_drift": hpsv3_quality_drift_score,
            "dynamic_degree": dynamic_degree_score,
            "dynamic_degree_continuous": dynamic_degree_continuous_score,
            "dynamic_degree_smooth": dynamic_degree_smooth_score,
            "dynamic_degree_smooth2": dynamic_degree_smooth2_score,
            "dynamic_degree_smooth3": dynamic_degree_smooth3_score,
            "dynamic_degree_bump": dynamic_degree_bump_score,
            "dino_ia": video_dino_ia_score,
        }
        self.device = device
        self.score_dict = score_dict
        self.score_fns = {}
        videoalign_keys = {"videoalign", "videoreward", "videogen_reward"}
        videoalign_single_dim_keys = {"videoalign_vq", "videoalign_mq", "videoalign_ta"}
        vbench_keys = {"vbench", "vbench_full"}
        for score_name, weight in score_dict.items():
            if score_name in videoalign_keys and videoalign_dimension_weights:
                kwargs = {"dimension_weights": dict(videoalign_dimension_weights)}
                # Match mp4 write fps to the model's real generation fps so
                # Qwen2-VL sees videos at the correct playback speed. Falls
                # back to videoalign_score's own default when not provided.
                if video_fps is not None:
                    kwargs["video_fps"] = int(video_fps)
                self.score_fns[score_name] = videoalign_score(device, **kwargs)
            elif score_name in videoalign_single_dim_keys and video_fps is not None:
                # Single-dim variants (videoalign_mq/ta/vq) write the temp mp4 via
                # the shared backbone's saved video_fps. Default in their factory
                # is 8 — but if the run actually generates at e.g. 15 fps, Qwen2-VL
                # would otherwise see motion at half real speed and mis-score MQ.
                # Mirror the videoalign_keys path: thread the config's video_fps
                # in here too so the shared scorer encodes at the real playback fps.
                dim = score_name.split("_", 1)[1].upper()  # "vq" -> "VQ" etc.
                self.score_fns[score_name] = _videoalign_single_dim_score(
                    device, dim, video_fps=int(video_fps),
                )
            elif score_name in vbench_keys and video_fps is not None:
                # VBench re-encodes our numpy frames to a temp mp4. Without
                # this, it defaults to fps=8 — wrong playback speed for
                # motion-dependent dims (motion_smoothness, dynamic_degree,
                # temporal_*) when generation produced 15 fps.
                self.score_fns[score_name] = score_functions[score_name](
                    device, video_fps=int(video_fps)
                )
            else:
                self.score_fns[score_name] = _build_score_function(
                    score_name, score_functions[score_name], device
                )

    def __call__(self, images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in self.score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = self.score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = self.score_fns[score_name](images, prompts, metadata)
                if isinstance(rewards, dict) and rewards:
                    for dim_key, dim_vals in rewards.items():
                        score_details[f"{score_name}_{dim_key}"] = dim_vals
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    def _get_scorer_modules(self):
        """Recursively collect every nn.Module reachable under each score_fn.

        score_fns come in two shapes (and a hybrid we have to handle):
        - Closures over an nn.Module (e.g. video_hpsv3_score, video_dino_ia_score):
          the module sits in the closure cells.
        - Class instances with __call__ that hold an nn.Module via attribute
          (e.g. videoalign_score). videoalign in particular is an nn.Module
          ITSELF that wraps a *non*-nn.Module helper (VideoVLMRewardInference)
          which in turn holds the actual ~14 GB Qwen2VLRewardModelBT as a plain
          attribute — `nn.Module.to()` will NOT recurse into it because the
          helper is not registered as a child module.

        So we cannot stop descending when we hit an nn.Module: there may be
        another nn.Module hidden inside its `__dict__` via a plain Python
        attribute (e.g. self.inferencer.model). Walk through everything,
        deduplicating with an id-set to avoid cycles. Skip Tensor leaves —
        they have no useful children and pull in tons of internal state.
        """
        modules = []
        seen = set()

        def _walk(obj):
            if obj is None:
                return
            oid = id(obj)
            if oid in seen:
                return
            seen.add(oid)
            # Tensor / Parameter leaves: no useful structure, skip.
            if isinstance(obj, torch.Tensor):
                return
            if isinstance(obj, torch.nn.Module):
                modules.append(obj)
                # IMPORTANT: keep descending. nn.Module.to() only walks
                # _modules / _parameters / _buffers; it misses non-Module
                # attributes that themselves contain nn.Modules.
            # Closure cells (for `def`-style score_fns).
            closure = getattr(obj, "__closure__", None)
            if closure:
                for cell in closure:
                    try:
                        _walk(cell.cell_contents)
                    except ValueError:
                        pass  # empty cell
            # Instance attributes (for class-style score_fns / wrappers).
            inst_dict = getattr(obj, "__dict__", None)
            if isinstance(inst_dict, dict):
                for v in inst_dict.values():
                    _walk(v)

        for score_fn in self.score_fns.values():
            _walk(score_fn)
        return modules

    def to_cpu(self):
        """Offload all reward model weights to CPU to free GPU memory."""
        for module in self._get_scorer_modules():
            module.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def to_gpu(self):
        """Reload all reward model weights to GPU for scoring."""
        for module in self._get_scorer_modules():
            module.to(self.device)


def multi_score(device, score_dict, videoalign_dimension_weights=None, video_fps=None):
    return MultiScorer(
        device,
        score_dict,
        videoalign_dimension_weights=videoalign_dimension_weights,
        video_fps=video_fps,
    )


def score_samples(reward_fn, samples, *, device, world_size, rank, logger, progress):
    """Phase 2: score every per-batch sample dict of the epoch in place (``sample["rewards"]``, float
    tensors on ``device``); the media go to ``device`` for scoring and are dropped with the reward
    prompts. Logs the per-rank scoring time spread (rank 0)."""
    import time
    from src.utils.distributed import gather_tensor_to_all
    waits = []
    for item in progress(samples, "Scoring rewards"):
        t0 = time.perf_counter()
        media = item.pop("images_cpu").to(device, non_blocking=True)
        rewards, _ = reward_fn(media, item.pop("reward_prompts"), item.pop("reward_prompt_metadata"),
                               only_strict=True)
        waits.append(time.perf_counter() - t0)
        item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
        del media
    if waits:
        # A rank much slower than its peers warns before the following all-gather would time out.
        local = torch.tensor([float(sum(waits)), float(max(waits))], device=device, dtype=torch.float32)
        stats = gather_tensor_to_all(local, world_size).reshape(world_size, 2)
        if rank == 0:
            totals, maxes = stats[:, 0].tolist(), stats[:, 1].tolist()
            logger.info("[reward latency] per-rank total(s)=[%s] batch-max(s)=[%s] spread=%.1fs",
                        ", ".join(f"{t:.1f}" for t in totals), ", ".join(f"{m:.1f}" for m in maxes),
                        max(totals) - min(totals))


# ---------------------------------------------------------------------------
# The reward area of an epoch (the trainers' Phase 2): score -> finish -> log. Everything here is
# per sample and method-free; the per-prompt advantages are computed from these
# rewards later, in the training phase (src/advantage/advantages.py:compute_signals).
# ---------------------------------------------------------------------------

def region_rewards(rewards, low_weights, high_weights):
    """Noise-region rewards: the scored components [n] re-aggregated under the low- and high-noise
    weight dicts (config.reward_fn_{low,high}_noise; no reward-model compute). Returns
    {"avg_low_noise", "avg_high_noise"} (in that order), each [n]."""
    scored = dict(rewards)
    out = {}
    for key, weights in (("avg_low_noise", low_weights), ("avg_high_noise", high_weights)):
        combined = None
        for name, w in weights.items():
            if name not in scored:
                raise KeyError(
                    f"Region reward component '{name}' is not scored. "
                    f"Add it to config.reward_fn (scored: "
                    f"{sorted(scored.keys())})."
                )
            term = scored[name] * float(w)
            combined = term if combined is None else combined + term
        scored[key] = out[key] = combined
    return out


def region_weights(config):
    """(low-noise weights, high-noise weights) when both config.reward_fn_{low,high}_noise are set,
    else None."""
    low = getattr(config, "reward_fn_low_noise", None)
    high = getattr(config, "reward_fn_high_noise", None)
    return (low, high) if low is not None and high is not None else None


def finish_epoch_rewards(samples, config, tokenizer):
    """The collated epoch after scoring -> what the sample cache keeps: ``samples["prompts"]``, the
    prompt each sample was scored for (the decoded ``prompt_ids`` when the model tokenizes, else the
    raw text; the advantages group by it), and the noise-region rewards when configured. In place."""
    prompt_ids = samples.pop("prompt_ids", None)
    texts = samples.pop("prompts_text", None)
    if prompt_ids is not None:
        samples["prompts"] = tokenizer.batch_decode(prompt_ids.cpu().numpy(), skip_special_tokens=True)
    else:
        samples["prompts"] = list(texts)
    region = region_weights(config)
    if region is not None:
        samples["rewards"].update(region_rewards(samples["rewards"], *region))
    return samples


def mean_of_top(prompts, values, top_percentage):
    """Mean over prompts (sorted) of each prompt's mean reward at or above its (100 − top)-th
    percentile (all its rewards for top = 100); ``values`` [N, ...]."""
    prompts = np.asarray(prompts)
    values = np.asarray(values, dtype=np.float64)
    means = []
    for p in np.unique(prompts):
        rows = values[prompts == p]
        if top_percentage == 100:
            means.append(np.mean(rows))
            continue
        top = rows[rows >= np.percentile(rows, 100 - top_percentage)]
        if top.size > 0:
            means.append(np.mean(top))
    return np.mean(means) if means else 0.0


def reward_epoch_log(samples, config, *, rank, world_size, device, num_train_timesteps):
    """The epoch's reward log (every rank calls it: it gathers; the dict is filled on rank 0): reward/<k> and
    reward/<k>_std of every reward, reward/zero_std_ratio and reward/std_mean over the prompt groups,
    reward/mean_top_{100,75,50,25,10} of the combined reward "avg" (mean_of_top over the per-timestep
    "avg", as the logs always were). {} for a sample cache written before the signals
    refactor (its log comes from compute_signals)."""
    from src.utils.reward_gather import calculate_zero_std_ratio, gather_epoch_rewards
    from src.utils.distributed import gather_prompts
    if not isinstance(samples.get("rewards"), dict):
        return {}
    gathered = gather_epoch_rewards(samples, world_size=world_size, device=device,
                                    num_train_timesteps=num_train_timesteps)
    prompts_all = gather_prompts(samples["prompts"], world_size)
    if rank != 0:
        return {}
    log = {}
    for key, value in gathered.items():
        log[f"reward/{key}"] = value.mean()
        if "_accuracy" not in key:
            log[f"reward/{key}_std"] = value.std()
    zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all, gathered)
    log["reward/zero_std_ratio"] = zero_std_ratio
    log["reward/std_mean"] = reward_std_mean
    for top in (100, 75, 50, 25, 10):
        log[f"reward/mean_top_{top}"] = mean_of_top(prompts_all, gathered["avg"], top)
    return log


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "test_cases/nasa.jpg",
    ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])
    prompts = [
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {"unifiedreward": 1.0}
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
