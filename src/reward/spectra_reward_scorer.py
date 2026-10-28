"""SpectraReward — training-free MLLM reward via image/video-conditioned prompt
log-likelihood ("read the prompt back from the generation").

Reference: "Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for
Text-to-Image Generation" (arXiv:2607.11886), project page
https://huangrh99.github.io/SpectraReward/, code https://github.com/huangrh99/AlphaGRPO.

    R_M(x, y) = 1/(T-1) * sum_t log p_M(x_{t+1} | x_{<=t}, y)

with `x` the tokenized generation prompt, `y` the generated sample, and `M` a
FROZEN pretrained MLLM (no reward training, no preference data). Per the paper,
end-of-turn / EOS tokens are excluded from the average so template tokens do not
dilute the signal; the mean (not sum) keeps prompts of different lengths
comparable.

Instantiation here: the sample goes in a user turn with a short captioning
instruction, and the generation prompt is teacher-forced as the ASSISTANT turn
content. The scored span is exactly the assistant text: we tokenize the chat
prefix (ending at "<|im_start|>assistant\\n") and the prefix+prompt string with
the SAME vision inputs and score only the tokens beyond the prefix, so no
template or EOS token can enter the average.

Video extension (ours): frames are sampled at `fps` and fed as a video turn —
the reward then measures how well the prompt reads back from the whole clip.
The per-token log-probs are also returned (the paper's "semantic spectrum"),
which localizes which parts of the prompt the sample fails to support.

Needs transformers >= 4.49 for Qwen2.5-VL — run under the pip overlay
(`PYTHONPATH=${VS2_OVERLAY}`, config/paths.sh), never by upgrading the
shared training env.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Backbone. The paper sweeps Gemma3 / InternVL3.5 / Qwen3-VL at 8B-235B and
# notes ~30B unlocks most of the benefit; we use the 7B Qwen2.5-VL already
# cached in HF_HOME (the same base family our VS2/UR2 judges are built on) so
# the reward runs fully offline. Backbone size is a documented caveat, not a
# method change: the estimator is backbone-agnostic.
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Measured scale (1100 clips, 2026-08-31) — needed to weight this against the
# [0,1] judge rewards. WITHIN-GROUP std is what matters: GRPO normalises the
# advantage inside a same-prompt group, so the between-prompt spread (3.1x
# larger, and the reason raw scores are not comparable across prompts) never
# reaches the loss.
#   extended ~100-token prompts: mean -1.71, within-group std 0.069  -> weight ~11
#   short concrete prompts:      mean -5.06, within-group std 0.281  -> weight ~2.7
# (matching videoalign_ta, whose std is 0.75 at weight 1). `spectra_pmi` shifts
# the mean to ~0 but leaves the within-group std unchanged, so same weight.

# Kept deliberately plain: the paper's signal is p(prompt | sample), so the
# instruction should elicit a description without steering the wording.
VIDEO_INSTRUCTION = "Describe this video."
IMAGE_INSTRUCTION = "Describe this image."


class SpectraRewardScorer:
    """Frozen-MLLM reward: mean log p(prompt | generated sample)."""

    def __init__(
        self,
        device: str = "cuda",
        dtype=torch.bfloat16,
        fps: float = 2.0,
        model_id: str = DEFAULT_MODEL,
        max_pixels: int | None = 401408,  # cap vision tokens (~512x28x28)
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.fps = fps
        self.model_id = model_id

        logger.info("Loading SpectraReward backbone %s ...", model_id)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        proc_kwargs = {"trust_remote_code": True}
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)
        self.tokenizer = self.processor.tokenizer
        self._prior_cache: dict[tuple, float] = {}

    # ---------------------------------------------------------------- helpers
    def _messages(self, media, prompt_is_video: bool):
        instruction = VIDEO_INSTRUCTION if prompt_is_video else IMAGE_INSTRUCTION
        media_item = (
            {"type": "video", "video": media, "fps": self.fps}
            if prompt_is_video
            else {"type": "image", "image": media}
        )
        return [{"role": "user", "content": [media_item, {"type": "text", "text": instruction}]}]

    @torch.no_grad()
    def _score(self, media, prompt: str, is_video: bool, return_spectrum: bool = False):
        from qwen_vl_utils import process_vision_info

        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("SpectraReward needs a non-empty prompt")

        messages = self._messages(media, is_video)
        prefix = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        def encode(text):
            kwargs = dict(text=[text], images=image_inputs, videos=video_inputs,
                          padding=False, return_tensors="pt")
            if is_video:
                kwargs["fps"] = self.fps
            return self.processor(**kwargs).to(self.device)

        # ONE vision preprocess (decoding + resizing the clip is the dominant
        # cost). The scored-span length comes from a text-only tokenization
        # difference, which is exact because both strings contain the SAME
        # single vision placeholder and so receive the same token expansion.
        full_inputs = encode(prefix + prompt)
        input_ids = full_inputs["input_ids"]
        n_full = input_ids.shape[1]
        n_scored = (len(self.tokenizer(prefix + prompt, add_special_tokens=False).input_ids)
                    - len(self.tokenizer(prefix, add_special_tokens=False).input_ids))
        if n_scored < 1:
            raise ValueError(f"empty scored span (n_scored={n_scored}, full {n_full})")
        # Guard the one way that could go wrong: a BPE merge across the
        # prefix/prompt boundary would shift the span by a token. If the decoded
        # tail is not exactly the prompt, fall back to encoding the prefix with
        # the same vision inputs and using its true length.
        if self.tokenizer.decode(input_ids[0, -n_scored:]).strip() != prompt.strip():
            n_scored = n_full - encode(prefix)["input_ids"].shape[1]
            if n_scored < 1:
                raise ValueError("empty scored span after boundary fallback")
        n_prefix = n_full - n_scored

        logits = self.model(**full_inputs).logits  # [1, L, V]
        # Predict token i from logits at i-1 (teacher forcing, single pass).
        target = input_ids[:, n_prefix:]                       # [1, n_scored]
        pred_logits = logits[:, n_prefix - 1: n_full - 1, :]   # aligned
        logprobs = torch.log_softmax(pred_logits.float(), dim=-1)
        tok_lp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1).squeeze(0)  # [n_scored]

        out = {
            "spectra": float(tok_lp.mean()),
            "n_tokens": int(n_scored),
            "ppl": float(torch.exp(-tok_lp.mean())),
        }
        if return_spectrum:
            out["token_logprobs"] = [float(v) for v in tok_lp]
            out["tokens"] = self.tokenizer.convert_ids_to_tokens(target[0].tolist())
        return out

    @torch.no_grad()
    def _prior(self, prompt: str, is_video: bool) -> float:
        """Mean log p(prompt) under the SAME template with NO media attached.

        The published reward is dominated by this term: on our clips the spread
        across prompts is ~0.5 nats while the spread across videos of one prompt
        is ~0.05, so raw `spectra` mostly ranks prompts by their own
        predictability. That is harmless inside a GRPO group (one prompt, so the
        prior is a constant that the advantage subtracts) but makes cross-prompt
        comparison meaningless — hence the PMI variant below.
        """
        key = (prompt, is_video)
        if key in self._prior_cache:
            return self._prior_cache[key]
        instruction = VIDEO_INSTRUCTION if is_video else IMAGE_INSTRUCTION
        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
        prefix = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prefix + prompt], padding=False,
                                return_tensors="pt").to(self.device)
        ids = inputs["input_ids"]
        n_scored = (len(self.tokenizer(prefix + prompt, add_special_tokens=False).input_ids)
                    - len(self.tokenizer(prefix, add_special_tokens=False).input_ids))
        logits = self.model(**inputs).logits
        target = ids[:, -n_scored:]
        lp = torch.log_softmax(logits[:, -n_scored - 1: -1, :].float(), dim=-1)
        val = float(lp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean())
        self._prior_cache[key] = val
        return val

    # ------------------------------------------------------------------- API
    def score_video(self, video_path: str, prompt: str, return_spectrum: bool = False,
                    with_prior: bool = False) -> dict:
        """Mean log p(prompt | clip). Higher = the prompt reads back better.

        `with_prior=True` additionally returns `prior` = mean log p(prompt) with
        no clip attached and `spectra_pmi` = spectra - prior, the pointwise
        mutual information between clip and prompt. PMI is the variant to use
        when comparing videos that have DIFFERENT prompts (our evaluation
        suites); raw spectra is the published quantity and is what a same-prompt
        GRPO group sees. Priors are cached per prompt, so the extra cost is one
        text-only forward per unique prompt, not per clip.
        """
        out = self._score(video_path, prompt, is_video=True, return_spectrum=return_spectrum)
        if with_prior:
            out["prior"] = self._prior(prompt, is_video=True)
            out["spectra_pmi"] = out["spectra"] - out["prior"]
        return out

    def score_image(self, image, prompt: str, return_spectrum: bool = False,
                    with_prior: bool = False) -> dict:
        """`image`: PIL.Image, path, or anything qwen_vl_utils accepts."""
        out = self._score(image, prompt, is_video=False, return_spectrum=return_spectrum)
        if with_prior:
            out["prior"] = self._prior(prompt, is_video=False)
            out["spectra_pmi"] = out["spectra"] - out["prior"]
        return out

    def score_frames(self, frames, prompt: str, return_spectrum: bool = False) -> dict:
        """`frames`: list of PIL images / paths treated as one clip."""
        return self._score(list(frames), prompt, is_video=True, return_spectrum=return_spectrum)


_SHARED: dict[str, SpectraRewardScorer] = {}


def get_shared_spectra_scorer(device="cuda", dtype=torch.bfloat16, fps: float = 2.0,
                              model_id: str = DEFAULT_MODEL) -> SpectraRewardScorer:
    """One backbone per (model, device, fps) — the 7B weights are ~16 GB."""
    key = f"{model_id}|{device}|{fps}"
    if key not in _SHARED:
        _SHARED[key] = SpectraRewardScorer(device=device, dtype=dtype, fps=fps, model_id=model_id)
    return _SHARED[key]
