"""Overlay-env worker process for the Qwen2.5-VL judge rewards.

VideoScore2 (``TIGER-Lab/VideoScore2``) and UnifiedReward-2.0
(``CodeGoat24/UnifiedReward-2.0-qwen-7b``) are Qwen2.5-VL-7B checkpoints that
need ``transformers >= 4.49`` + ``qwen_vl_utils``. The shared training env is
pinned to ``transformers 4.46`` for the Wan2.2 diffusion stack, so these models
cannot be imported in the trainer process. This script is the other half of the
split: the trainer (4.46) spawns it with the ``vs2_overlay`` PYTHONPATH prepended
(transformers 4.53.2 shadows 4.46), it loads ONE scorer, and answers requests
over a unix-domain socket.

Protocol — newline-delimited JSON, one request -> one response:
    {"cmd": "ping"}                                  -> {"ok": true}
    {"cmd": "to_cpu"}                                -> {"ok": true}   # offload model to CPU
    {"cmd": "to_gpu"}                                -> {"ok": true}   # reload model to GPU
    {"cmd": "score", "video_paths": [...],           -> {"ok": true, "scores": [float,...]}
                     "prompts": [...]}
    {"cmd": "shutdown"}                              -> {"ok": true}   # then exit

``score`` returns, per video, the mean of the model's per-axis ``exp`` scalars
(VideoScore2: VQ/TA/PC; UnifiedReward-2.0: AL/PH/ST), normalised to [0, 1] by
dividing the 1-5 scale by 5. Axes the model failed to emit are skipped; a video
with no parseable axis scores 0.0.

Launched by ``src/reward/rewards.py:_OverlayVLMProxy``; not run by hand. For a
standalone smoke test see the ``__main__`` of that proxy / the launch in
``scripts/wan2_2/wan2_2_a14b_videoscore2.sh``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="[vlm_reward_worker %(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("vlm_reward_worker")

# Axes whose `exp` scalar we average, per model.
_MODEL_AXES = {
    "videoscore2": ("VQ", "TA", "PC"),
    "unifiedreward2": ("AL", "PH", "ST"),
    # SpectraReward has a single scalar (mean prompt log-likelihood), not a set
    # of 1-5 judged axes; the pseudo-axis keeps the CLI/registry uniform.
    "spectra": ("SPEC",),
    # Prompt-engineered semantic judges: one scalar already in [0, 1] (see
    # semantic_vlm_scorer.py). Pseudo-axis, same reason as spectra.
    "semantic_v2a": ("SEM",),
    "semantic_v4c": ("SEM",),
}


def _build_scorer(model_key: str, device: str):
    """Instantiate the requested scorer under the overlay transformers."""
    if model_key == "videoscore2":
        from src.reward.videoscore2_scorer import VideoScore2Scorer

        return VideoScore2Scorer(device=device)
    if model_key == "unifiedreward2":
        from src.reward.unifiedreward2_scorer import UnifiedReward2Scorer

        return UnifiedReward2Scorer(device=device)
    if model_key == "spectra":
        from src.reward.spectra_reward_scorer import SpectraRewardScorer

        return SpectraRewardScorer(device=device)
    if model_key in ("semantic_v2a", "semantic_v4c"):
        from src.reward.semantic_vlm_scorer import SemanticVLMScorer

        # The reward IS the template; model_key selects which one.
        return SemanticVLMScorer(template=model_key, device=device)
    raise ValueError(f"unknown model key: {model_key!r}")


def _scalar_from_result(result: dict, axes) -> float:
    """Mean of the available per-axis `exp` scalars, normalised 1-5 -> [0, 1].

    SpectraReward is already a scalar (mean prompt log-likelihood, typically in
    [-3, 0]) and is passed through unscaled: GRPO normalises advantages within a
    group, so only the reward WEIGHT needs to account for its magnitude.
    """
    if "spectra" in result:
        return float(result["spectra"])
    # Prompt-engineered semantic judges already return [0, 1]; no 1-5 rescale.
    if "semantic" in result:
        return float(result["semantic"])
    vals = []
    for axis in axes:
        dim = result.get(axis)
        exp = getattr(dim, "exp", None) if dim is not None else None
        if exp is not None:
            vals.append(float(exp))
    if not vals:
        return 0.0
    return (sum(vals) / len(vals)) / 5.0


def _recv_line(conn: socket.socket) -> str | None:
    """Read one newline-terminated message (requests are small)."""
    chunks = []
    while True:
        b = conn.recv(65536)
        if not b:
            return "".join(chunks) if chunks else None
        chunks.append(b.decode("utf-8"))
        if "\n" in chunks[-1]:
            return "".join(chunks)


def _handle(req: dict, scorer, device: str, axes) -> dict:
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True}
    if cmd == "to_cpu":
        scorer.model.to("cpu")
        import torch

        torch.cuda.empty_cache()
        return {"ok": True}
    if cmd == "to_gpu":
        scorer.model.to(device)
        return {"ok": True}
    if cmd == "score":
        paths = req["video_paths"]
        prompts = req.get("prompts") or [""] * len(paths)
        scores = []
        for path, prompt in zip(paths, prompts):
            result = scorer.score_video(path, prompt)
            scores.append(_scalar_from_result(result, axes))
        return {"ok": True, "scores": scores}
    raise ValueError(f"unknown cmd: {cmd!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(_MODEL_AXES))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--socket", required=True, help="unix socket path to bind")
    ap.add_argument("--repo", required=True, help="repo root for `import src...`")
    ap.add_argument("--axes", default="", help="comma-separated axis subset (e.g. 'TA'); empty = all model axes")
    args = ap.parse_args()

    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)

    axes = _MODEL_AXES[args.model]
    if args.axes.strip():
        req = tuple(a.strip().upper() for a in args.axes.split(",") if a.strip())
        bad = [a for a in req if a not in axes]
        if bad:
            raise SystemExit(f"--axes {bad} not in {args.model} axes {axes}")
        axes = req
    import transformers

    logger.info("model=%s device=%s transformers=%s", args.model, args.device, transformers.__version__)
    scorer = _build_scorer(args.model, args.device)
    logger.info("scorer ready")

    # Bind the socket only after the model is loaded, so the parent's connect()
    # succeeding is proof the worker is ready to serve.
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(1)
    logger.info("listening on %s", args.socket)

    try:
        while True:
            conn, _ = srv.accept()
            try:
                while True:
                    line = _recv_line(conn)
                    if line is None:
                        break  # peer closed; wait for a new connection
                    try:
                        req = json.loads(line)
                        if req.get("cmd") == "shutdown":
                            conn.sendall((json.dumps({"ok": True}) + "\n").encode("utf-8"))
                            return
                        resp = _handle(req, scorer, args.device, axes)
                    except Exception as exc:  # report, keep the worker alive
                        logger.error("request failed: %s\n%s", exc, traceback.format_exc())
                        resp = {"ok": False, "error": str(exc)}
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            finally:
                conn.close()
    finally:
        srv.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    main()
