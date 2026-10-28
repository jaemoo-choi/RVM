"""HPSv3 (Human Preference Score v3) reward scorer.

Uses the hpsv3 package to score images/videos against text prompts.
For videos, uniformly samples frames and averages per-frame scores.
"""

import torch
import torch.nn as nn
import numpy as np


# Per-process cache so multiple reward keys (video_hpsv3_general,
# video_hpsv3_percentile, hpsv3_quality_drift) sharing the same backbone end up
# with ONE HPSv3 instance. The model weights are ~17 GB; loading two copies on
# every rank is what tipped the host RAM into cgroup-OOM during Phase 1
# offloading.
_HPSV3_SCORER_CACHE: dict[tuple, "HPSv3Scorer"] = {}


def get_shared_hpsv3_scorer(dtype, device):
    """Return the per-(dtype, device) shared HPSv3Scorer instance."""
    key = (dtype, str(device))
    scorer = _HPSV3_SCORER_CACHE.get(key)
    if scorer is None:
        scorer = HPSv3Scorer(dtype, device)
        _HPSV3_SCORER_CACHE[key] = scorer
    return scorer


class HPSv3Scorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        from hpsv3 import HPSv3RewardInferencer
        from config.paths import HF_HOME
        import os
        ckpt = os.path.join(HF_HOME, "hub", "models--MizzenAI--HPSv3", "snapshots", "main", "HPSv3.safetensors")
        if os.path.exists(ckpt):
            self.inferencer = HPSv3RewardInferencer(checkpoint_path=ckpt, device=device)
        else:
            self.inferencer = HPSv3RewardInferencer(device=device)
        self.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        """Score a batch of images.

        Args:
            images: tensor of shape [B, C, H, W] float in [0, 1].
            prompts: list of strings, length B.

        Returns:
            Tensor of shape [B] with HPSv3 rewards.
        """
        # hpsv3==1.0.0 exposes ``HPSv3RewardInferencer.reward(image_paths,
        # prompts)`` only — image_paths goes through Qwen-VL's
        # ``process_vision_info``, which accepts file paths, URLs, or PIL
        # Images. Convert our BCHW float[0,1] tensor to a list of PIL.Image.
        from PIL import Image

        x = images.detach().to(dtype=torch.float32).cpu()
        if x.numel() > 0 and x.max().item() <= 1.0001:
            x = x.clamp(0, 1).mul(255).round().to(torch.uint8)
        else:
            x = x.clamp(0, 255).to(torch.uint8)
        x = x.permute(0, 2, 3, 1).contiguous().numpy()  # [B, H, W, C] uint8
        pil_imgs = [Image.fromarray(arr, mode="RGB") for arr in x]

        rewards = self.inferencer.reward(pil_imgs, list(prompts))
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, device=self.device, dtype=self.dtype)
        rewards = rewards.to(device=self.device, dtype=self.dtype)
        # hpsv3 ``reward`` returns shape ``[B, 2]`` where column 0 is mu (mean
        # reward) and column 1 is sigma (uncertainty). Per the package's own
        # example (``hpsv3.inference``: "miu and sigma. we select miu as the
        # final output"), take column 0.
        if rewards.ndim > 1 and rewards.shape[-1] >= 1:
            rewards = rewards[:, 0]
        return rewards.contiguous()

    @torch.no_grad()
    def score_video(self, video_frames, prompts, num_sample_frames=4):
        """Score a batch of videos by averaging HPSv3 over sampled frames.

        Uniformly samples frames from each video and averages the per-frame scores.

        Args:
            video_frames: tensor of shape [B, T, C, H, W] float in [0, 1].
            prompts: list of strings, length B.
            num_sample_frames: number of frames to uniformly sample per video.

        Returns:
            Tensor of shape [B] with averaged HPSv3 scores.
        """
        B, T, C, H, W = video_frames.shape
        n_frames = min(num_sample_frames, T)
        indices = torch.linspace(0, T - 1, n_frames).round().long()

        all_scores = torch.zeros(B, device=self.device)
        for fi in indices:
            frame_batch = video_frames[:, fi]  # [B, C, H, W]
            scores = self(frame_batch, prompts)
            all_scores += scores
        all_scores /= n_frames
        return all_scores

    @torch.no_grad()
    def score_video_general(self, video_frames, num_sample_frames=4,
                            prompt="A high-quality image"):
        """HPSv3-general: visual quality only, ignoring the video caption.

        Scores every sampled frame against a fixed prompt (default
        ``"A high-quality image"``) and returns the mean over frames.
        """
        B, T, C, H, W = video_frames.shape
        n_frames = min(num_sample_frames, T)
        indices = torch.linspace(0, T - 1, n_frames).round().long()

        prompts = [prompt] * B
        all_scores = torch.zeros(B, device=self.device)
        for fi in indices:
            frame_batch = video_frames[:, fi]
            scores = self(frame_batch, prompts)
            all_scores += scores
        all_scores /= n_frames
        return all_scores

    @torch.no_grad()
    def score_video_percentile(self, video_frames, prompts,
                               num_sample_frames=4, top_k_pct=0.3):
        """HPSv3-percentile: text-video alignment via top-30% frame mean.

        Scores every sampled frame against the actual video caption, then
        per video keeps only the top ``top_k_pct`` highest frame scores and
        averages those. Mitigates the impact of low rewards from temporary
        content drift across frames.
        """
        B, T, C, H, W = video_frames.shape
        n_frames = min(num_sample_frames, T)
        indices = torch.linspace(0, T - 1, n_frames).round().long()

        # [n_frames, B] then transpose → [B, n_frames]
        per_frame = torch.zeros(n_frames, B, device=self.device)
        for i, fi in enumerate(indices):
            per_frame[i] = self(video_frames[:, fi], prompts)
        per_frame = per_frame.t().contiguous()

        k = max(1, int(round(n_frames * top_k_pct)))
        topk = per_frame.topk(k, dim=1).values  # [B, k]
        return topk.mean(dim=1)
