"""Within-prompt / cross-prompt image diversity for T2I evaluation (src/evaluation.py --diversity_seeds K).

The metric definitions and embedding backbones:
  - CLIP ViT-L/14 image embeddings (`get_image_features`, the clipscore backbone; loaded through
    src/reward/clip_scorer.ClipScorer), L2-normalised;
  - DINOv2-L CLS embeddings (`facebook/dinov2-large` `pooler_output`, layer-normed CLS), L2-normalised;
  - pair_cos = mean pairwise cosine distance over the upper triangle; vendi = exp(entropy of the eigenvalues of
    K/n), K the cosine-similarity kernel.
Per prompt the metrics are computed over its K seeds and averaged over prompts (SE over prompts); cross-prompt
diversity is computed per seed across all prompts and averaged over seeds (SE over seeds).

Weights are read from HF_HOME (config/paths); the repo default is HF offline.
"""
import numpy as np

CLIP_ID = "openai/clip-vit-large-patch14"
DINO_ID = "facebook/dinov2-large"
WITHIN_METRICS = ("dino_pair_cos", "clip_pair_cos", "clip_vendi", "dino_vendi")


# ------------------------------------------------------------------ metrics (numpy only)
def vendi(x):
    """x: (n, d) L2-normalised rows. Vendi = exp(entropy of eigenvalues of K/n), K = cosine-similarity kernel."""
    x = np.asarray(x)
    k = (x @ x.T) / x.shape[0]
    w = np.linalg.eigvalsh(k.astype(np.float64))
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def pair_cos(x):
    """Mean pairwise cosine distance (1 - cos) over all unordered pairs of the L2-normalised rows of x."""
    x = np.asarray(x)
    n = x.shape[0]
    s = x @ x.T
    iu = np.triu_indices(n, 1)
    return float((1.0 - s[iu]).mean())


def _se(v):
    v = np.asarray(v, dtype=np.float64)
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def mean_se(v):
    return {"mean": float(np.mean(v)), "se": _se(v), "n": int(len(v))}


def within_prompt(clip, dino):
    """clip / dino: (K, d) normalised embeddings of one prompt's K images -> the four within-prompt metrics."""
    return {"dino_pair_cos": pair_cos(dino), "clip_pair_cos": pair_cos(clip),
            "clip_vendi": vendi(clip), "dino_vendi": vendi(dino)}


def diversity_summary(clip, dino):
    """clip / dino: (P, K, d) arrays (prompt, seed, dim) of normalised embeddings.

    Returns {"per_prompt": {metric: [P values]}, "within_prompt": {metric: mean/se over prompts},
    "cross_prompt": {x_<emb>_{pair_cos,vendi}: mean/se over the K seeds}}.
    """
    clip, dino = np.asarray(clip), np.asarray(dino)
    assert clip.shape[:2] == dino.shape[:2], (clip.shape, dino.shape)
    P, K = clip.shape[:2]
    per = {m: [] for m in WITHIN_METRICS}
    for p in range(P):
        for m, v in within_prompt(clip[p], dino[p]).items():
            per[m].append(v)
    cross = {}
    if P >= 2:
        for nm, e in (("clip", clip), ("dino", dino)):
            pc = [pair_cos(e[:, j]) for j in range(K)]
            vv = [vendi(e[:, j]) for j in range(K)]
            cross[f"x_{nm}_pair_cos"] = mean_se(pc)
            cross[f"x_{nm}_vendi"] = mean_se(vv)
    return {"n_prompts": int(P), "n_seeds": int(K), "per_prompt": per,
            "within_prompt": {m: mean_se(v) for m, v in per.items()}, "cross_prompt": cross}


# ------------------------------------------------------------------ embeddings
def to_pil(images):
    """(B, C, H, W) float tensor in [0, 1] -> list of PIL RGB images, quantised exactly as the side script's PNGs
    ((x.clamp(0, 1) * 255).round() uint8)."""
    from PIL import Image
    arr = (images.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


class DiversityEmbedder:
    """CLIP ViT-L/14 + DINOv2-L image embedder (fp32, eval mode). `clip_scorer` may be an existing
    src.reward.clip_scorer.ClipScorer to share its weights; otherwise one is built (same loading)."""

    def __init__(self, device, clip_scorer=None):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from config.paths import HF_HOME  # noqa: F401  (HF cache = HF_HOME; resolved by transformers via env)

        if clip_scorer is None:
            from src.reward.clip_scorer import ClipScorer
            clip_scorer = ClipScorer(device)
        self.device = device
        self.clip = clip_scorer.model.eval()
        self.clip_proc = clip_scorer.processor
        self.dino = AutoModel.from_pretrained(DINO_ID).to(device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(DINO_ID)
        self._torch = torch

    def __call__(self, pil_images):
        """list of PIL images -> (clip (B, 768), dino (B, 1024)) L2-normalised float32 numpy arrays."""
        torch = self._torch
        with torch.no_grad():
            ce = self.clip.get_image_features(
                **self.clip_proc(images=pil_images, return_tensors="pt").to(self.device)).float()
            de = self.dino(**self.dino_proc(images=pil_images, return_tensors="pt").to(self.device)).pooler_output.float()
        ce = torch.nn.functional.normalize(ce, dim=-1).cpu().numpy()
        de = torch.nn.functional.normalize(de, dim=-1).cpu().numpy()
        return ce, de
