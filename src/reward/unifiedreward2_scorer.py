"""UnifiedReward-2.0 scorer — VLM-as-judge video reward (Qwen2.5-VL based).

Wraps CodeGoat24/UnifiedReward-2.0-qwen-7b, a standard Qwen2.5-VL checkpoint, so
it loads through the same transformers path / pip overlay as VideoScore2 (no
sglang/vLLM server needed). Uses UnifiedReward-2.0's official pointwise video
prompt (inference_qwen/UnifiedReward-2.0-inference/point_score_APS_video_generation.py):
16 uniformly-sampled frames, three 1-5 axes —

    Alignment Score (AL) -- match between video and caption
    Physics Score   (PH) -- gravity / motion / collisions make physical sense
    Style Score     (ST) -- visual appeal regardless of caption

PH is the on-target axis for the real-vs-fake motion question. Returns the same
hard / soft / exp scalars as the VS2 scorer (exp = logit expectation over 1-5,
recommended for ranking metrics).
"""

from __future__ import annotations

import logging
import re

import torch

from .videoscore2_scorer import DimScore

logger = logging.getLogger(__name__)

MODEL_NAME = "CodeGoat24/UnifiedReward-2.0-qwen-7b"
NUM_FRAMES = 16

# Official UnifiedReward-2.0 pointwise video-generation prompt.
USER_PROMPT_TEMPLATE = (
    "You are presented with a generated video and its associated text caption. "
    "Your task is to analyze the video across multiple dimensions in relation to "
    "the caption. Specifically:\n"
    "Provide overall assessments for the video along the following axes (each rated from 1 to 5):\n"
    "- Alignment Score: How well the video matches the caption in terms of content.\n"
    "- Physics Score: How well the gravity, movements, collisions, and interactions make physical sense.\n"
    "- Style Score: How visually appealing the video looks, regardless of caption accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "Alignment Score (1-5): X\n"
    "Physics Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\n"
    "Text Caption: [{t2v_prompt}]"
)

# UnifiedReward-2.0 emits FLOAT scores, e.g. "Alignment Score (1-5): 2.49".
# Capture the number after the label's colon (the "(1-5)" in between has no colon,
# so [^\n:]* skips it). Values outside [1,5] are model garbage -> dropped.
_LABELS = {
    "AL": r"Alignment Score[^\n:]*:\s*([0-9]+(?:\.[0-9]+)?)",
    "PH": r"Physics Score[^\n:]*:\s*([0-9]+(?:\.[0-9]+)?)",
    "ST": r"Style Score[^\n:]*:\s*([0-9]+(?:\.[0-9]+)?)",
}


class UnifiedReward2Scorer:
    def __init__(self, device: str = "cuda", dtype=torch.bfloat16, num_frames: int = NUM_FRAMES,
                 max_new_tokens: int = 1024):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.num_frames = num_frames
        self.max_new_tokens = max_new_tokens

        logger.info("Loading %s ...", MODEL_NAME)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=dtype
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.tokenizer = self.processor.tokenizer

    @torch.no_grad()
    def score_video(self, video_path: str, t2v_prompt: str) -> dict:
        from qwen_vl_utils import process_vision_info

        user_prompt = USER_PROMPT_TEMPLATE.format(t2v_prompt=(t2v_prompt or "").strip())
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path, "nframes": self.num_frames},
            {"type": "text", "text": user_prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        # No no_repeat_ngram_size: the three score lines intentionally repeat
        # "Score (1-5):", and blocking that ngram corrupts the output format.
        gen = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
        )
        in_len = inputs["input_ids"].shape[1]
        buf = self.tokenizer.decode(gen[0, in_len:], skip_special_tokens=True)

        # Restrict to the final answer if a <think> preamble is present.
        sec_off = buf.rfind("</think>")
        section = buf[sec_off:] if sec_off != -1 else buf

        result = {"raw": buf}
        for dim, label_re in _LABELS.items():
            val = None
            for m in re.finditer(label_re, section, re.IGNORECASE):
                v = float(m.group(1))
                if 1.0 <= v <= 5.0:       # drop out-of-range model garbage
                    val = v               # keep the last valid match (final answer)
            # store the parsed float as the continuous score (hard = rounded)
            result[dim] = DimScore(round(val) if val is not None else None, val, val)
        return result
