"""Prompt-engineered semantic reward: a Qwen2.5-VL judge driven by a text template.

The reward is the TEMPLATE, not this file. `scripts/eval/semantic_prompts/*.txt`
holds the judge instructions; this module only turns one of them into a scalar.
The two templates promoted to training rewards by the 12-set / 4,377-video
benchmark are:

    semantic_v2a -> v2a_absence.txt      (rating: how many stated things are missing)
    semantic_v4c -> v4c_short_yesno.txt  (yesno:  is EVERYTHING stated present)

Scoring reads the FIRST-token logits only -- no generation:
    rating  expectation over {"1".."5"}, mapped to [0, 1] as (E - 1) / 4
    yesno   p("Yes") / (p("Yes") + p("No"))
Both stay continuous, so a rollout group of K samples rarely ties and the GRPO
advantage keeps its ordering. Greedy decoding would collapse to a few integers.

Needs transformers >= 4.49 for Qwen2.5-VL, which conflicts with the trainer's
pinned 4.46 -- so this runs in the overlay subprocess (`vlm_reward_worker.py`,
`PYTHONPATH=.../vs2_overlay`), never in the trainer process. The trainer talks to
it through `_OverlayVLMProxy` in `rewards.py`.

PROMPT PLACEHOLDERS. The benchmark distinguished `{prompt}` (the extended caption
the model was actually conditioned on) from `{short_prompt}` (the original VBench
sentence the dimension labels were built from). Training datasets carry ONE prompt
per sample -- `dataset/vidprom_filtered_extended/train.txt` is a flat list of
extended captions -- so both placeholders receive that same string here. v4c was
benchmarked against the short sentence; under training it sees the extended one,
which makes it behave more like v2a than the benchmark implies; re-measure on the pilot's own rollouts.
"""

from __future__ import annotations

import logging
import os
import re

import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
_PROMPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts", "eval", "semantic_prompts",
)

# Reward key -> template file. Keep in sync with `_MODEL_AXES` in vlm_reward_worker.py.
TEMPLATES = {
    "semantic_v2a": "v2a_absence.txt",
    "semantic_v4c": "v4c_short_yesno.txt",
}


def parse_template(path: str) -> dict:
    """Plain-text template -> {mode, fps, video_kwargs, system, user}.

    Byte-for-byte the same parser as `scripts/eval/score_semantic_vlm.py`, so a
    template that was benchmarked there scores identically here.
    """
    raw = open(path).read()
    meta = {"mode": "rating", "fps": 2.0, "system": "", "user": "", "video_kwargs": {}}
    for m in re.finditer(r"^#\s*(\w+)\s*:\s*(.+)$", raw, flags=re.M):
        k, v = m.group(1).lower(), m.group(2).strip()
        if k == "mode":
            meta[k] = v
        elif k == "fps":
            meta[k] = float(v)
        elif k in ("max_pixels", "total_pixels", "min_pixels", "nframes"):
            # Passed straight to qwen_vl_utils: a 53-frame 480x832 clip written at
            # fps 15 is ~7x the frames of the fps-2 default, so without a pixel
            # budget the visual-token count explodes.
            meta["video_kwargs"][k] = int(float(v))
    section = None
    buf = {"system": [], "user": []}
    for line in raw.splitlines():
        m = re.match(r"^===\s*(system|user)\s*===\s*$", line.strip(), flags=re.I)
        if m:
            section = m.group(1).lower()
            continue
        if section:
            buf[section].append(line)
    meta["system"] = "\n".join(buf["system"]).strip()
    meta["user"] = "\n".join(buf["user"]).strip()
    if not meta["user"]:
        raise ValueError(f"{path}: no '=== user ===' section")
    if meta["mode"] not in ("rating", "yesno"):
        raise ValueError(f"{path}: mode must be 'rating' or 'yesno', got {meta['mode']}")
    return meta


def resolve_template(name_or_path: str) -> str:
    """Accept a reward key (`semantic_v2a`), a bare filename, or a full path."""
    if name_or_path in TEMPLATES:
        return os.path.join(_PROMPT_DIR, TEMPLATES[name_or_path])
    if os.path.isabs(name_or_path) or os.path.sep in name_or_path:
        return name_or_path
    return os.path.join(_PROMPT_DIR, name_or_path)


class SemanticVLMScorer:
    """Qwen2.5-VL judge for ONE template. Interface matches the other overlay scorers."""

    def __init__(self, template: str, device: str = "cuda",
                 model_id: str = DEFAULT_MODEL, dtype=torch.bfloat16):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.template_path = resolve_template(template)
        self.tpl = parse_template(self.template_path)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        tok = self.processor.tokenizer
        # Scores read logits[:, -1, :]. Under the default RIGHT padding that slot
        # is a pad token for every sequence but the longest, so a batch would be
        # garbage for all but one item. Left padding makes -1 the real last token.
        tok.padding_side = "left"

        def ids_for(words):
            out = []
            for w in words:
                got = None
                for c in (w, " " + w):
                    t = tok.encode(c, add_special_tokens=False)
                    if len(t) == 1:
                        got = t[0]
                        break
                if got is None:
                    raise ValueError(f"no single-token id for {w!r}")
                out.append(got)
            return out

        self.rating_ids = ids_for(["1", "2", "3", "4", "5"])
        self.yes_id, self.no_id = ids_for(["Yes", "No"])
        logger.info(
            "SemanticVLMScorer ready: template=%s mode=%s fps=%s video_kwargs=%s",
            os.path.basename(self.template_path), self.tpl["mode"],
            self.tpl["fps"], self.tpl["video_kwargs"],
        )

    def _fill(self, text: str, prompt: str) -> str:
        # Training data carries one prompt per sample; both placeholders get it.
        return text.replace("{prompt}", prompt).replace("{short_prompt}", prompt)

    def score_batch(self, video_paths, prompts) -> list[float]:
        from qwen_vl_utils import process_vision_info

        tpl = self.tpl
        texts, all_imgs, all_vids = [], [], []
        for path, prompt in zip(video_paths, prompts):
            vid = {"type": "video", "video": path, "fps": tpl["fps"]}
            vid.update(tpl["video_kwargs"])
            messages = []
            if tpl["system"]:
                messages.append(
                    {"role": "system",
                     "content": [{"type": "text", "text": self._fill(tpl["system"], prompt)}]}
                )
            messages.append(
                {"role": "user",
                 "content": [vid, {"type": "text", "text": self._fill(tpl["user"], prompt)}]}
            )
            texts.append(
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            imgs, vids = process_vision_info(messages)
            if imgs:
                all_imgs.extend(imgs)
            if vids:
                all_vids.extend(vids)
        inputs = self.processor(
            text=texts, images=all_imgs or None, videos=all_vids or None,
            return_tensors="pt", padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :].float()
        if tpl["mode"] == "yesno":
            p = torch.softmax(logits[:, [self.yes_id, self.no_id]], dim=-1)
            return [float(x) for x in p[:, 0]]
        p = torch.softmax(logits[:, self.rating_ids], dim=-1)
        exp = (p * torch.arange(1, 6, device=p.device, dtype=p.dtype)).sum(dim=-1)
        return [float((e - 1.0) / 4.0) for e in exp]

    def score_video(self, video_path: str, prompt: str = "") -> dict:
        """One clip -> {"semantic": float in [0, 1]}. Shape the worker expects."""
        return {"semantic": self.score_batch([video_path], [prompt])[0]}
