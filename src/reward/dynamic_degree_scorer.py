"""Dynamic-degree reward / loss scorer.

Mirrors VBench's `dynamic_degree` evaluation metric (RAFT-based optical flow,
top-5% pixel magnitude per frame pair) but exposes both a no-grad reward
path AND a grad-enabled differentiable-loss path. Aggregation: mean of
per-pair top-5% flow magnitude across N consecutive frame pairs.

VBench reference: vbench/dynamic_degree.py:42-93.
"""
import os
from typing import Optional

import torch
import torch.nn as nn
from easydict import EasyDict as edict


def _resolve_raft_ckpt() -> str:
    """Match VBench's checkpoint discovery: ${VBENCH_CACHE_DIR or ~/.cache/vbench}/raft_model/models/raft-things.pth.

    VBench downloads on first use (vbench/utils.py:282-290). We don't
    auto-download here — fail loudly if missing; user runs vbench's setup
    or downloads manually from
    ``https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip``.
    """
    from config.paths import VBENCH_CACHE_DIR
    return os.path.join(VBENCH_CACHE_DIR, "raft_model", "models", "raft-things.pth")


# Per-process singleton (mirrors get_shared_hpsv3_scorer / get_shared_videoalign_scorer).
_DYNAMIC_DEGREE_SCORER_CACHE: dict[tuple, "DynamicDegreeScorer"] = {}


def get_shared_dynamic_degree_scorer(dtype, device) -> "DynamicDegreeScorer":
    key = (dtype, str(device))
    scorer = _DYNAMIC_DEGREE_SCORER_CACHE.get(key)
    if scorer is None:
        scorer = DynamicDegreeScorer(dtype, device)
        _DYNAMIC_DEGREE_SCORER_CACHE[key] = scorer
    return scorer


class DynamicDegreeScorer(nn.Module):
    """RAFT-based dynamic-degree scorer for both reward (no_grad) and
    differentiable-loss (with_grad) paths.

    Score per video: mean of `max_rad` across N uniformly-sampled frame
    pairs, where `max_rad = mean(top 5% of per-pixel flow magnitude)`.
    Same per-pair formula as VBench's `DynamicDegree.get_score`
    (vbench/dynamic_degree.py:42-56) but aggregated as a smooth mean
    instead of VBench's threshold-and-count binary.
    """

    def __init__(self, dtype, device, ckpt_path: Optional[str] = None):
        super().__init__()
        self.dtype = dtype
        self.device = device

        from vbench.third_party.RAFT.core.raft import RAFT

        if ckpt_path is None:
            ckpt_path = _resolve_raft_ckpt()
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"RAFT checkpoint not found at {ckpt_path}. "
                "Run VBench's dynamic_degree once to auto-download, or fetch "
                "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip "
                "and unzip into ~/.cache/vbench/raft_model/."
            )

        args = edict({
            "model": ckpt_path,
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        })
        self.model = RAFT(args)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Strip `module.` prefix if checkpoint was saved from DataParallel.
        new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal: per-pair flow magnitude → top-5% mean
    # ------------------------------------------------------------------

    def _per_pair_max_rad(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Compute flow between two frame batches and return the top-5%
        mean per sample.

        image1, image2: [B, 3, H, W] float in [0, 255].
        Returns: [B] float — same as VBench's `get_score` line 42-56.
        """
        from vbench.third_party.RAFT.core.utils_core.utils import InputPadder

        padder = InputPadder(image1.shape)
        image1_p, image2_p = padder.pad(image1, image2)
        # iters=20 matches VBench (dynamic_degree.py:77).
        _, flow_up = self.model(image1_p, image2_p, iters=20, test_mode=True)

        # flow_up: [B, 2, H, W]
        u = flow_up[:, 0]
        v = flow_up[:, 1]
        rad = torch.sqrt(u * u + v * v)  # [B, H, W]
        b, h, w = rad.shape
        flat = rad.reshape(b, h * w)
        cut = max(1, int(h * w * 0.05))
        topk = flat.topk(cut, dim=1, largest=True, sorted=False).values  # [B, cut]
        return topk.mean(dim=1)  # [B]

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_video_no_grad(
        self,
        video_frames: torch.Tensor,
        num_pairs: int = 8,
    ) -> torch.Tensor:
        """Continuous mean of per-pair max_rad. video_frames: [B,T,C,H,W]."""
        return self._score_per_pair(video_frames, num_pairs).mean(dim=1)

    def score_video_with_grad(
        self,
        video_frames: torch.Tensor,
        num_pairs: int = 8,
    ) -> torch.Tensor:
        """Differentiable-loss path. Keeps autograd graph through RAFT."""
        return self._score_per_pair(video_frames, num_pairs).mean(dim=1)

    @torch.no_grad()
    def score_video_vbench_binary(
        self,
        video_frames: torch.Tensor,
        num_pairs: int = 8,
        pass_rate: float = 0.25,
    ) -> torch.Tensor:
        """VBench-style binary dynamic_degree: 1.0 if ≥`pass_rate` of evaluated
        pairs exceed the per-pair threshold `6 * min(H,W)/256`, else 0.0.

        Faithful to VBench's spec (vbench/dynamic_degree.py:84-93): the
        per-pair threshold is identical, and the 25% pass-rate rule is the
        natural normalization of VBench's `count_num = round(4*T/16) = T/4`
        applied to whatever subset of pairs we evaluate.
        """
        per_pair = self._score_per_pair(video_frames, num_pairs)  # [B, num_pairs]
        h, w = video_frames.shape[-2], video_frames.shape[-1]
        thres = self.vbench_threshold(h, w)
        pass_frac = (per_pair > thres).to(per_pair.dtype).mean(dim=1)
        return (pass_frac >= pass_rate).to(per_pair.dtype)

    @torch.no_grad()
    def score_video_smooth(
        self,
        video_frames: torch.Tensor,
        num_pairs: int = 8,
        tau_scale: float = 1.0,
    ) -> torch.Tensor:
        """Smooth (clipped-linear) dynamic_degree: continuous [0, 1] relaxation
        of the binary indicator.

        Per-pair soft pass `smoothpass_k = clamp(m_k / (tau_scale * tau), 0, 1)`
        where `tau = vbench_threshold(H, W)` and `m_k` is the per-pair top-5%
        mean flow magnitude; video reward = mean over the `num_pairs` pairs.

        `tau_scale` (default 1.0) sets the saturation point: 1.0 reaches full
        credit exactly at the VBench threshold, >1 requires more motion, <1
        less. Bounded to [0, 1] so the saturation caps motion-quantity
        reward-hacking (unlike the unbounded mean-flow continuous score).
        """
        tau_scale = float(tau_scale)
        if tau_scale <= 0.0:
            raise ValueError(f"tau_scale must be > 0, got {tau_scale}")
        per_pair = self._score_per_pair(video_frames, num_pairs)  # [B, num_pairs]
        h, w = video_frames.shape[-2], video_frames.shape[-1]
        tau = self.vbench_threshold(h, w) * tau_scale
        smooth = (per_pair / tau).clamp(0.0, 1.0)
        return smooth.mean(dim=1)  # [B] in [0, 1]

    @torch.no_grad()
    def score_video_smooth2(
        self,
        video_frames: torch.Tensor,
        num_pairs: int = 8,
        tau_scale: float = 1.0,
        pass_rate: float = 0.25,
    ) -> torch.Tensor:
        """Smooth dynamic_degree v2: blend the clipped-linear motion-quantity
        term with a smooth (sigmoid) relaxation of VBench's
        ``≥pass_rate``-of-pairs-moving binary gate.

        With ``m`` = per-pair top-5% mean flow magnitude and
        ``tau = vbench_threshold(H, W) * tau_scale``::

            r_clamp       = clamp(m / tau, 0, 1).mean()            # motion quantity, [0,1]
            moving_prob   = sigmoid(6 * (m / tau - 1))            # per-pair P(moving)
            motion_ratio  = moving_prob.mean()                    # soft fraction of moving pairs
            r_soft_binary = sigmoid(8 * (motion_ratio - pass_rate))  # smooth "≥pass_rate move"
            r_dynamic     = 0.2 * r_clamp + 0.8 * r_soft_binary

        The 0.8 weight on the soft-binary term mirrors that VBench's metric is
        gated by the pass-rate rule (``score_video_vbench_binary``), while the
        0.2 ``r_clamp`` term keeps a smooth gradient on raw motion magnitude
        below the threshold. Unlike ``score_video_smooth`` (pure ``r_clamp``),
        v2 also rewards getting *enough* pairs over the threshold, not just more
        flow on a few. All means are over the ``num_pairs`` dimension; returns
        ``[B]`` in (0, 1).
        """
        tau_scale = float(tau_scale)
        if tau_scale <= 0.0:
            raise ValueError(f"tau_scale must be > 0, got {tau_scale}")
        per_pair = self._score_per_pair(video_frames, num_pairs)  # [B, num_pairs] = m
        h, w = video_frames.shape[-2], video_frames.shape[-1]
        tau = self.vbench_threshold(h, w) * tau_scale
        ratio = per_pair / tau                                    # [B, num_pairs]
        r_clamp = ratio.clamp(0.0, 1.0).mean(dim=1)              # [B]
        moving_prob = torch.sigmoid(6.0 * (ratio - 1.0))         # [B, num_pairs]
        motion_ratio = moving_prob.mean(dim=1)                   # [B]
        r_soft_binary = torch.sigmoid(8.0 * (motion_ratio - pass_rate))  # [B]
        return 0.2 * r_clamp + 0.8 * r_soft_binary              # [B] in (0, 1)

    @torch.no_grad()
    def score_video_smooth3(
        self,
        video_frames: torch.Tensor,
        num_pairs: int = 8,
        tau_scale: float = 1.0,
        bg_pct: float = 0.5,
        bg_cap_scale: float = 1.0,
        bg_max_scale: float = 3.0,
        bg_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Smooth dynamic_degree v3: clipped-linear motion-quantity reward
        (identical to ``score_video_smooth``) multiplied by a smooth penalty
        that ramps in only once the *background / global* motion gets too fast.

        Rationale: the top-5% mean flow ``m`` rewards fast pixels regardless of
        whether they belong to a moving subject or to a fast-panning background.
        A fast subject on a still background is good (high ``m``, low background
        flow); a fast camera pan moves nearly every pixel (high ``m`` AND high
        background flow). We proxy background / global motion by the per-pixel
        flow ``bg_pct`` quantile (default 0.5 = median), which a small fast
        subject cannot inflate. Camera motion is left free below the cap and is
        smoothly penalized only once it exceeds it.

        With ``m`` = per-pair top-5% mean flow, ``g`` = per-pair ``bg_pct``
        quantile of per-pixel flow, ``tau = vbench_threshold(H, W)``::

            r_motion = clamp(m / (tau_scale * tau), 0, 1)               # [0,1]
            lo, hi   = bg_cap_scale * tau, bg_max_scale * tau
            bg_over  = clamp((g - lo) / (hi - lo), 0, 1)                # 0 below cap, 1 past hi
            r_pair   = r_motion * (1 - bg_penalty * bg_over)

        video reward = mean over the ``num_pairs`` pairs, in ``[0, 1]``.

        Knobs (all in units of the resolution-aware VBench threshold ``tau``):
        - ``tau_scale``: motion saturation point, like ``score_video_smooth``.
        - ``bg_pct``: percentile used as the background/global-motion proxy.
        - ``bg_cap_scale``: background flow allowed (penalty-free) up to this.
        - ``bg_max_scale``: background flow at which the penalty is maximal.
        - ``bg_penalty`` in [0, 1]: max fraction of motion reward removed when
          the background is moving at/above ``bg_max_scale * tau``.
        """
        tau_scale = float(tau_scale)
        bg_pct = float(bg_pct)
        bg_cap_scale = float(bg_cap_scale)
        bg_max_scale = float(bg_max_scale)
        bg_penalty = float(bg_penalty)
        if tau_scale <= 0.0:
            raise ValueError(f"tau_scale must be > 0, got {tau_scale}")
        if not 0.0 < bg_pct < 1.0:
            raise ValueError(f"bg_pct must be in (0, 1), got {bg_pct}")
        if bg_max_scale <= bg_cap_scale:
            raise ValueError(
                f"bg_max_scale ({bg_max_scale}) must be > bg_cap_scale ({bg_cap_scale})"
            )
        if not 0.0 <= bg_penalty <= 1.0:
            raise ValueError(f"bg_penalty must be in [0, 1], got {bg_penalty}")

        per_pair_m, per_pair_bg = self._score_per_pair_stats(
            video_frames, num_pairs, bg_pct
        )  # each [B, num_pairs]
        h, w = video_frames.shape[-2], video_frames.shape[-1]
        tau = self.vbench_threshold(h, w)
        r_motion = (per_pair_m / (tau_scale * tau)).clamp(0.0, 1.0)
        lo = bg_cap_scale * tau
        hi = bg_max_scale * tau
        bg_over = ((per_pair_bg - lo) / (hi - lo)).clamp(0.0, 1.0)
        per_pair = r_motion * (1.0 - bg_penalty * bg_over)
        return per_pair.clamp(0.0, 1.0).mean(dim=1)  # [B] in [0, 1]

    # ------------------------------------------------------------------

    def _per_pair_stats(
        self, image1: torch.Tensor, image2: torch.Tensor, bg_pct: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like ``_per_pair_max_rad`` but returns ``(m, g)`` from a single RAFT
        pass: ``m`` = top-5% mean flow magnitude (motion quantity), ``g`` =
        per-pixel flow ``bg_pct`` quantile (background/global-motion proxy).

        image1, image2: [B, 3, H, W] float in [0, 255]. Returns two [B] tensors.
        """
        from vbench.third_party.RAFT.core.utils_core.utils import InputPadder

        padder = InputPadder(image1.shape)
        image1_p, image2_p = padder.pad(image1, image2)
        _, flow_up = self.model(image1_p, image2_p, iters=20, test_mode=True)

        u = flow_up[:, 0]
        v = flow_up[:, 1]
        rad = torch.sqrt(u * u + v * v)  # [B, H, W]
        b, h, w = rad.shape
        flat = rad.reshape(b, h * w)
        cut = max(1, int(h * w * 0.05))
        m = flat.topk(cut, dim=1, largest=True, sorted=False).values.mean(dim=1)  # [B]
        g = flat.quantile(bg_pct, dim=1)  # [B]
        return m, g

    def _score_per_pair_stats(
        self, video_frames: torch.Tensor, num_pairs: int, bg_pct: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pair ``(m, g)`` stats, each shape [B, num_pairs]. Shares the
        normalization/pair-sampling logic with ``_score_per_pair``."""
        if video_frames.ndim != 5:
            raise ValueError(
                f"DynamicDegreeScorer expects [B, T, C, H, W], got {tuple(video_frames.shape)}"
            )
        b, t, c, h, w = video_frames.shape
        if c != 3:
            raise ValueError(f"expected C=3, got C={c}")

        x = video_frames.to(self.device)
        x_max = x.detach().amax().item() if x.numel() > 0 else 1.0
        if x_max <= 1.0001:
            x = x.clamp(0, 1) * 255.0
        else:
            x = x.clamp(0, 255)

        n = max(1, min(num_pairs, t - 1))
        idx = torch.linspace(0, t - 2, n).round().long().tolist()

        m_pairs, g_pairs = [], []
        for i in idx:
            f1 = x[:, i].contiguous()
            f2 = x[:, i + 1].contiguous()
            m, g = self._per_pair_stats(f1, f2, bg_pct)
            m_pairs.append(m)
            g_pairs.append(g)
        m_out = torch.stack(m_pairs, dim=0).t().contiguous()  # [B, n_pairs]
        g_out = torch.stack(g_pairs, dim=0).t().contiguous()  # [B, n_pairs]
        return m_out, g_out

    def _score_per_pair(self, video_frames: torch.Tensor, num_pairs: int) -> torch.Tensor:
        """Compute per-pair max_rad, return shape [B, num_pairs]."""
        if video_frames.ndim != 5:
            raise ValueError(
                f"DynamicDegreeScorer expects [B, T, C, H, W], got {tuple(video_frames.shape)}"
            )
        b, t, c, h, w = video_frames.shape
        if c != 3:
            raise ValueError(f"expected C=3, got C={c}")

        # Normalize to [0, 255] (RAFT operates on uint8-range floats per
        # VBench's get_frames at vbench/dynamic_degree.py:96-113).
        x = video_frames.to(self.device)
        x_max = x.detach().amax().item() if x.numel() > 0 else 1.0
        if x_max <= 1.0001:
            x = x.clamp(0, 1) * 255.0
        else:
            x = x.clamp(0, 255)

        # Sample (i, i+1) pair indices uniformly across the T-1 available
        # consecutive pairs. Min 1 pair, max T-1.
        n = max(1, min(num_pairs, t - 1))
        idx = torch.linspace(0, t - 2, n).round().long().tolist()

        per_pair = []
        for i in idx:
            f1 = x[:, i].contiguous()      # [B, 3, H, W]
            f2 = x[:, i + 1].contiguous()
            per_pair.append(self._per_pair_max_rad(f1, f2))
        # [n_pairs, B] -> [B, n_pairs]
        return torch.stack(per_pair, dim=0).t().contiguous()

    @staticmethod
    def vbench_threshold(height: int, width: int) -> float:
        """VBench's per-pair threshold (dynamic_degree.py:60-61): 6 * scale/256
        where scale = min(H, W). At 480x832 this is 11.25 px/frame.
        """
        scale = min(height, width)
        return 6.0 * (scale / 256.0)
