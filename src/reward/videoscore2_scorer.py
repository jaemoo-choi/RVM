"""VideoScore2 scorer — VLM-as-judge video quality scoring.

Wraps TIGER-Lab/VideoScore2 (Qwen2.5-VL-7B fine-tuned with SFT->GRPO on the
VideoFeedback2 dataset). The model watches frames sampled at `fps` plus the
text prompt and emits a chain-of-thought review ending in three 1-5 scores:

    visual quality (VQ)  -- clarity, smoothness, artifacts
    text-to-video alignment (TA)  -- fidelity to the prompt
    physical/common-sense consistency (PC)  -- naturalness / physics plausibility

For each dimension we return three flavours of scalar:
  - hard: the integer 1-5 the model decoded (argmax token)
  - soft: the official `ll_based_soft_score_normed` (best_score * normalized_prob)
  - exp : the probability-weighted expectation  E[s] = sum_s s * p(s) / sum_s p(s)

`exp` is the recommended scalar for ranking metrics (Spearman / AUC): it is
monotonic in the model's belief and breaks the ties that integer scores create.

This module deliberately needs transformers >= 4.49 (Qwen2.5-VL) which is newer
than the shared training env. Run it under the pip overlay env (see
human_eval/run_videoscore2.py) so the training stack is untouched.

Reference: VideoScore2 (arXiv:2509.22799); model card TIGER-Lab/VideoScore2.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "TIGER-Lab/VideoScore2"

# Official evaluation prompt (model card). The clip's own t2v prompt is spliced
# in, and we pin the output format so the three scores are reliably parseable.
USER_PROMPT_TEMPLATE = (
    "You are an expert for evaluating AI-generated videos from three dimensions:\n"
    "(1) visual quality – clarity, smoothness, artifacts;\n"
    "(2) text-to-video alignment – fidelity to the prompt;\n"
    "(3) physical/common-sense consistency – naturalness and physics plausibility.\n\n"
    'Video prompt: {t2v_prompt}\n\n'
    "First reason briefly about each dimension, then output the final line exactly "
    "in this format:\n"
    "visual quality: <v_score>; text-to-video alignment: <t_score>, "
    "physical/common-sense consistency: <p_score>"
)

# VS2 emits a <think>...</think> CoT then a final answer that always lists the
# three scores in order, keyed by "(1)/(2)/(3)", e.g.
#   (1) visual quality – clarity, smoothness, artifacts: 3
#   (2) text-to-video alignment – fidelity to the prompt: 4
#   (3) physical/common-sense consistency – naturalness and physics plausibility: 3
# but the wording drifts at higher fps ("(1) Visual Quality: 3", "(2) Text-to-Video
# Match: 4", "(3) Physical Consistency:3"). Keying on the number is wording-robust.
# Each dim: ordered patterns tried in turn. The "(1)/(2)/(3)" marker is the clean
# case; the dimension word is a fallback for higher-fps drift where the model drops
# a marker or merges dims onto one line (e.g. "(2) ...alignment: 4, Physical Cons: 4").
_LABELS = {
    "VQ": [r"\(\s*1\s*\)", r"visual quality"],
    "TA": [r"\(\s*2\s*\)", r"text-to-video\s*(?:alignment|match)"],
    "PC": [r"\(\s*3\s*\)", r"physical"],
}


@dataclass
class DimScore:
    hard: Optional[int]
    soft: Optional[float]
    exp: Optional[float]


def _digit_token_ids(tokenizer) -> dict[int, list[int]]:
    """Map score value 1..5 -> candidate single-token ids.

    Qwen may render the score as a bare digit ("3") or space-prefixed (" 3")
    depending on context; we collect both single-token forms and pool their
    probability mass so the soft/exp scores are read from the right token.
    """
    out = {}
    for s in range(1, 6):
        ids = set()
        for variant in (str(s), " " + str(s)):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        if ids:
            out[s] = sorted(ids)
    return out


def _score_from_logits(logits: torch.Tensor, digit_ids: dict[int, list[int]]):
    """Return (hard_argmax, soft_official, expectation) over digits 1..5."""
    probs_full = torch.softmax(logits.float(), dim=-1)
    vals, probs = [], []
    for s, tids in digit_ids.items():
        vals.append(s)
        probs.append(float(sum(probs_full[t].item() for t in tids)))
    if not probs:
        return None, None, None
    total = sum(probs)
    if total <= 0:
        return None, None, None
    best_i = int(np.argmax(probs))
    best_score = vals[best_i]
    soft = round(best_score * (probs[best_i] / total), 4)              # official
    exp = round(sum(v * p for v, p in zip(vals, probs)) / total, 4)    # E[s]
    return best_score, soft, exp


class VideoScore2Scorer:
    def __init__(self, device: str = "cuda", dtype=torch.bfloat16, fps: float = 2.0,
                 max_new_tokens: int = 1024):
        from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.fps = fps
        self.max_new_tokens = max_new_tokens

        # VS2 is a plain Qwen2.5-VL checkpoint (no auto_map); load the class directly
        # so we don't depend on AutoModelForVision2Seq's mapping across versions.
        logger.info("Loading %s ...", MODEL_NAME)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=dtype
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, use_fast=False
        )
        self.digit_ids = _digit_token_ids(self.tokenizer)

    @torch.no_grad()
    def score_video(self, video_path: str, t2v_prompt: str) -> dict:
        from qwen_vl_utils import process_vision_info

        user_prompt = USER_PROMPT_TEMPLATE.format(t2v_prompt=(t2v_prompt or "").strip())
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": self.fps},
            {"type": "text", "text": user_prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            fps=self.fps, padding=True, return_tensors="pt",
        ).to(self.device)

        # Greedy + scores: greedy makes the decoded digit == argmax so `hard`
        # matches the logits we read, and the whole run is reproducible.
        # no_repeat_ngram_size breaks the greedy repetition loops that the model
        # can fall into with larger frame contexts (higher fps), where it would
        # otherwise spin on "</score></multi-dimensional analysis>..." until the
        # token budget runs out and never emit the final score line. Has no effect
        # on normal CoT (the three score lines share no 4-grams).
        gen = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
            do_sample=False, no_repeat_ngram_size=4,
            output_scores=True, return_dict_in_generate=True,
        )
        in_len = inputs["input_ids"].shape[1]
        gen_ids = gen.sequences[0, in_len:].tolist()
        step_scores = gen.scores  # tuple len == len(gen_ids); [vocab] each (batch dim 0)

        # Reconstruct per-token text to map char positions -> generated-token index.
        toks = [self.tokenizer.decode([t]) for t in gen_ids]
        starts, buf = [], ""
        for tk in toks:
            starts.append(len(buf))
            buf += tk

        # The final answer (after the last </think>) lists the three scores in a
        # fixed order keyed by "(1)/(2)/(3)", but the dimension *wording* varies
        # with fps (e.g. "visual quality - clarity...: 3" vs "Visual Quality: 3",
        # "text-to-video alignment: 4" vs "Text-to-Video Match: 4"). So we key on
        # the numbered marker + the first ": <digit>" on that line, not the words.
        sec_off = buf.rfind("</think>")
        sec_off = sec_off + len("</think>") if sec_off != -1 else 0
        section = buf[sec_off:]

        def char_to_token(cpos: int):
            for i, s in enumerate(starts):
                if s <= cpos < s + len(toks[i]):
                    return i
            return None

        def locate(patterns):
            # Try each pattern in order; within a pattern take the LAST match whose
            # line has ":<digit>" (the final-answer line, not a CoT mention).
            for pat in patterns:
                for m in reversed(list(re.finditer(pat, section, re.IGNORECASE))):
                    eol = section.find("\n", m.end())
                    seg = section[m.end():eol if eol != -1 else len(section)]
                    md = re.search(r":\s*([1-5])", seg)
                    if md:
                        return char_to_token(sec_off + m.end() + md.start(1))
            return None

        result = {"raw": buf}
        for dim, label_re in _LABELS.items():
            ti = locate(label_re)
            if ti is None or ti >= len(step_scores):
                result[dim] = DimScore(None, None, None)
                continue
            hard, soft, exp = _score_from_logits(step_scores[ti][0], self.digit_ids)
            result[dim] = DimScore(hard, soft, exp)
        return result
