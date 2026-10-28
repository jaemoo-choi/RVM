"""DINOv2 image-alignment reward scorer.

Computes cosine similarity between an input/reference image and selected
generated video frames using DINOv2 [CLS] embeddings. Used as a substitute
for the unreleased Q-Save IA dimension in the I2V context.

Formula:
    R_IA = w_first * cos(phi(I), phi(f_0))
         + w_avg   * mean({cos(phi(I), phi(f_j)) for j in frame_ids})

with default w_first=0.7, w_avg=0.3, frame_ids=[0, 17, 35, 52] for T=53.
"""

from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class DinoIAScorer(nn.Module):
    """DINOv2 cosine-similarity scorer for input image vs generated frames.

    Embeds reference image and selected frames via DINOv2 [CLS] tokens,
    L2-normalises, and combines per the TAGRPO-recipe weighted formula.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, dtype, device, model_name: str = "facebook/dinov2-large", img_size: int = 224):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.model_name = model_name
        self.img_size = img_size

        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name).to(device=device, dtype=dtype)
        mean = torch.tensor(self.IMAGENET_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD, dtype=dtype, device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eval()

    def _to_bchw_float01(self, x: Union[torch.Tensor, "Image.Image", Sequence]) -> torch.Tensor:
        """Coerce input to BCHW float tensor in [0, 1] on CPU.

        Accepts:
          - torch.Tensor [B, C, H, W] or [B, H, W, C], float [0,1] or uint8.
          - List[PIL.Image] (heterogeneous sizes OK) -> resized to (img_size,
            img_size) and stacked. The fixed resize is needed because
            ``torch.stack`` requires uniform shapes; DINOv2 also bilinear-
            resizes inputs to img_size internally, so this is dim-consistent.
          - Single PIL.Image -> 1xCximg_sizeximg_size.
        """
        if isinstance(x, Image.Image):
            x = [x]
        if isinstance(x, (list, tuple)) and len(x) and isinstance(x[0], Image.Image):
            from torchvision.transforms.functional import to_tensor
            target = (self.img_size, self.img_size)
            return torch.stack([
                to_tensor(img.convert("RGB").resize((target[1], target[0]),
                                                    Image.BILINEAR))
                for img in x
            ]).clamp(0, 1)

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.detach().float()
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.numel() > 0 and x.max().item() > 1.5:
            x = x / 255.0
        return x.clamp(0, 1)

    def _preprocess(self, x_bchw: torch.Tensor) -> torch.Tensor:
        x = x_bchw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear",
                          align_corners=False, antialias=True)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _embed(self, x_bchw: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised [CLS] embeddings of shape [B, D]."""
        pixel_values = self._preprocess(x_bchw)
        outputs = self.model(pixel_values=pixel_values)
        emb = outputs.last_hidden_state[:, 0]
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def score_video(
        self,
        video_frames,
        ref_images: Union[Sequence[Image.Image], torch.Tensor],
        frame_ids: Optional[Iterable[int]] = None,
        first_frame_weight: float = 0.7,
        sampled_avg_weight: float = 0.3,
    ) -> torch.Tensor:
        """Compute R_IA per the TAGRPO-recipe formula.

        Args:
            video_frames: tensor or array, [B, T, C, H, W] or [B, T, H, W, C],
                          float [0,1] or uint8 (auto-detected).
            ref_images: list of PIL.Image (length B) or BCHW tensor [B, C, H, W].
            frame_ids: indices into T to sample. Defaults to ``[0, T//3, 2T//3, T-1]``
                       which yields ``[0, 17, 35, 52]`` for T=53.
            first_frame_weight: weight on cos(phi(I), phi(f_0)).
            sampled_avg_weight: weight on mean of cos(phi(I), phi(f_j)) over frame_ids.

        Returns:
            Tensor [B] with the weighted IA score per video.
        """
        x = video_frames
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        x = x.detach().cpu().float()
        if x.numel() > 0 and x.max().item() > 1.5:
            x = x / 255.0
        x = x.clamp(0, 1)
        if x.ndim == 5 and x.shape[-1] == 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if x.ndim != 5:
            raise ValueError(f"score_video expects 5-D video tensor, got shape {tuple(x.shape)}")
        B, T, C, H, W = x.shape

        if frame_ids is None:
            frame_ids = [0, T // 3, (2 * T) // 3, T - 1]
        frame_ids = sorted({max(0, min(T - 1, int(i))) for i in frame_ids})
        if not frame_ids:
            frame_ids = [0]

        ref_bchw = self._to_bchw_float01(ref_images)
        if ref_bchw.shape[0] != B:
            raise ValueError(
                f"ref_images batch ({ref_bchw.shape[0]}) does not match video batch ({B})"
            )
        ref_emb = self._embed(ref_bchw)  # [B, D]

        unique_ids = sorted({0, *frame_ids})
        frame_embs = {fi: self._embed(x[:, fi]) for fi in unique_ids}

        first_sim = (ref_emb * frame_embs[0]).sum(dim=-1)  # [B]
        sampled_sims = torch.stack(
            [(ref_emb * frame_embs[fi]).sum(dim=-1) for fi in frame_ids], dim=0
        ).mean(dim=0)  # [B]

        ia = first_frame_weight * first_sim + sampled_avg_weight * sampled_sims
        return ia.contiguous()
