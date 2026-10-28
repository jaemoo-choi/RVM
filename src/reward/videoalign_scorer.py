from __future__ import annotations

import glob
import json
import logging
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import imageio
import numpy as np
import safetensors
import torch
import torchvision
from packaging import version
from peft import LoraConfig, get_peft_model
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .reward_ckpt_path import CKPT_PATH

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
VIDEOSCORE_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the frames of a given video and see the text prompt for generating the video,
then give scores based on its {dimension_name}, i.e., {dimension_description}.
Output a float number from 1.0 to 5.0 for this dimension,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 5.0 means Perfect/Real (the video is like a real video).
The text prompt used for generation is "{text_prompt}".
"""
DIMENSION_DESCRIPTIONS = {
	"VQ": ["visual quality", "the quality of the video in terms of clearness, resolution, brightness, and color"],
	"TA": ["text-to-video alignment", "the alignment between the text prompt and the video content and motion"],
	"MQ": ["motion quality", "the quality of the motion in terms of consistency, smoothness, and completeness"],
	"Overall": [
		"Overall Performance",
		"the overall performance of the video in terms of visual quality, text-to-video alignment, and motion quality",
	],
}
SIMPLE_PROMPT = """
Please evaluate the {dimension_name} of a generated video. Consider {dimension_description}.
The text prompt used for generation is "{text_prompt}".
"""
DETAILED_PROMPT_WITH_SPECIAL_TOKEN = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

Please provide the ratings of Visual Quality: <|VQ_reward|>
END

**Motion Quality:**
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

Please provide the ratings of Motion Quality: <|MQ_reward|>
END

**Text Alignment:**
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Text Alignment: <|TA_reward|>
END
"""
DETAILED_PROMPT = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

**Motion Quality:**
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

**Text Alignment:**
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Visual Quality, Motion Quality, and Text Alignment.
"""


@dataclass
class DataConfig:
	meta_data: str = "/path/to/dataset/meta_data.csv"
	data_dir: str = "/path/to/dataset"
	meta_data_test: str = None
	max_frame_pixels: int = 240 * 320
	num_frames: float = None
	fps: float = 2.0
	p_shuffle_frames: float = 0.0
	p_color_jitter: float = 0.0
	eval_dim: str | list[str] = "VQ"
	prompt_template_type: str = "none"
	add_noise: bool = False
	sample_type: str = "uniform"
	use_tied_data: bool = True


@dataclass
class TrainingConfig:
	load_from_pretrained: str = ""
	load_from_pretrained_step: int = -1
	gradient_checkpointing: bool = False
	disable_flash_attn2: bool = False
	bf16: bool = False
	fp16: bool = False
	output_dir: str = ""


@dataclass
class PEFTLoraConfig:
	lora_enable: bool = False
	vision_lora: bool = False
	lora_r: int = 16
	lora_alpha: int = 32
	lora_dropout: float = 0.05
	lora_target_modules: Optional[list[str]] = None
	lora_namespan_exclude: Optional[list[str]] = None
	lora_modules_to_save: Optional[list[str]] = None
	lora_task_type: str = "CAUSAL_LM"
	use_rslora: bool = False
	num_lora_modules: int = -1

	def __post_init__(self):
		if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
			self.lora_target_modules = self.lora_target_modules[0]
		if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
			self.lora_namespan_exclude = self.lora_namespan_exclude[0]


@dataclass
class ModelConfig:
	model_name_or_path: Optional[str] = None
	model_revision: str = "main"
	output_dim: int = 1
	use_special_tokens: bool = False
	freeze_vision_tower: bool = False
	freeze_llm: bool = False
	tune_merger: bool = False
	torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = None
	trust_remote_code: bool = False
	attn_implementation: Optional[str] = None
	load_in_8bit: bool = False
	load_in_4bit: bool = False
	bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
	use_bnb_nested_quant: bool = False
	reward_token: Literal["last", "mean", "special"] = "last"
	loss_type: Literal["bt", "reg", "btt", "margin", "constant_margin", "scaled"] = "regular"

	def __post_init__(self):
		if self.load_in_8bit and self.load_in_4bit:
			raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


def _resolve_videoalign_ckpt_path() -> Path:
	base = Path(CKPT_PATH).expanduser().resolve()
	candidates = [base / "VideoReward", base / "videoalign", base / "videoalign_ckpt", base]
	for candidate in candidates:
		if (candidate / "model_config.json").exists():
			return candidate
	raise FileNotFoundError(
		f"VideoAlign checkpoint not found in {base}. "
		"Expected model_config.json under one of: VideoReward, videoalign, videoalign_ckpt, or reward_ckpts root."
	)


def load_configs_from_json(config_path):
	with open(config_path, "r") as file:
		config_dict = json.load(file)
	config_dict["data_config"].pop("meta_data", None)
	config_dict["data_config"].pop("data_dir", None)
	return (
		config_dict["data_config"],
		None,
		config_dict["model_config"],
		config_dict["peft_lora_config"],
		config_dict.get("inference_config"),
	)


def build_prompt(prompt, dimension, template_type):
	if isinstance(dimension, list) and len(dimension) > 1:
		dimension_name = ", ".join([DIMENSION_DESCRIPTIONS[d][0] for d in dimension])
		dimension_name = f"overall performance({dimension_name})"
		dimension_description = "the overall performance of the video"
	else:
		if isinstance(dimension, list):
			dimension = dimension[0]
		dimension_name = DIMENSION_DESCRIPTIONS[dimension][0]
		dimension_description = DIMENSION_DESCRIPTIONS[dimension][1]

	if template_type == "none":
		return prompt
	if template_type == "simple":
		return SIMPLE_PROMPT.format(
			dimension_name=dimension_name,
			dimension_description=dimension_description,
			text_prompt=prompt,
		)
	if template_type == "video_score":
		return VIDEOSCORE_QUERY_PROMPT.format(
			dimension_name=dimension_name,
			dimension_description=dimension_description,
			text_prompt=prompt,
		)
	if template_type == "detailed_special":
		return DETAILED_PROMPT_WITH_SPECIAL_TOKEN.format(text_prompt=prompt)
	if template_type == "detailed":
		return DETAILED_PROMPT.format(text_prompt=prompt)
	raise ValueError("Invalid template type")


def round_by_factor(number: int, factor: int) -> int:
	return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
	return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
	return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28) -> tuple[int, int]:
	if max(height, width) / min(height, width) > 200:
		raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
	h_bar = max(factor, round_by_factor(height, factor))
	w_bar = max(factor, round_by_factor(width, factor))
	if h_bar * w_bar > max_pixels:
		beta = math.sqrt((height * width) / max_pixels)
		h_bar = floor_by_factor(height / beta, factor)
		w_bar = floor_by_factor(width / beta, factor)
	elif h_bar * w_bar < min_pixels:
		beta = math.sqrt(min_pixels / (height * width))
		h_bar = ceil_by_factor(height * beta, factor)
		w_bar = ceil_by_factor(width * beta, factor)
	return h_bar, w_bar


def smart_nframes(ele: dict, total_frames: int, video_fps: int | float) -> int:
	assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
	if "nframes" in ele:
		nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
	else:
		fps = ele.get("fps", FPS)
		min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
		max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
		nframes = total_frames / video_fps * fps
		nframes = min(max(nframes, min_frames), max_frames)
		nframes = round_by_factor(nframes, FRAME_FACTOR)
	if nframes > total_frames:
		nframes = total_frames
	if not (FRAME_FACTOR <= nframes <= total_frames):
		raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
	return nframes


def _select_frame_indices(ele: dict, total_frames: int, video_fps: float) -> list[int]:
	"""Frame indices VideoAlign feeds to Qwen2-VL for a T-frame clip at `video_fps`.

	Shared by the mp4 readers (torchvision/decord) AND the in-memory path so
	that all three pick byte-identical frames; the only thing the readers add
	on top is the H.264 encode/decode round trip.
	"""
	if ele["sample_type"] == "uniform":
		nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
		return torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
	if ele["sample_type"] == "multi_pts":
		frames_each_pts = 6
		num_pts = 4
		fps = 8
		nframes = int(total_frames * fps // video_fps)
		frames_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
		start_pt = int(frames_each_pts // 2)
		end_pt = int(nframes - frames_each_pts // 2 - 1)
		pts = torch.linspace(start_pt, end_pt, num_pts).round().long().tolist()
		idx = []
		for pt in pts:
			idx.extend(frames_idx[pt - frames_each_pts // 2: pt + frames_each_pts // 2])
		return idx
	raise ValueError(f"Unsupported sample_type: {ele['sample_type']}")


def _read_video_torchvision(ele: dict) -> torch.Tensor:
	video_path = ele["video"]
	if version.parse(torchvision.__version__) < version.parse("0.19.0") and "file://" in video_path:
		video_path = video_path[7:]
	video, _, info = io.read_video(
		video_path,
		start_pts=ele.get("video_start", 0.0),
		end_pts=ele.get("video_end", None),
		pts_unit="sec",
		output_format="TCHW",
	)
	total_frames, video_fps = video.size(0), info["video_fps"]
	idx = _select_frame_indices(ele, total_frames, video_fps)
	return video[idx]


class InMemoryVideo:
	"""Frames handed to VideoAlign without an mp4 round trip.

	`frames` is a uint8 tensor [T, H, W, C] exactly as the sampler produced it;
	`fps` is the value the mp4 header WOULD have carried (the scorer's
	`video_fps`), so `smart_nframes` selects the same indices the file path does.
	"""

	__slots__ = ("frames", "fps")

	def __init__(self, frames: torch.Tensor, fps: float):
		self.frames = frames
		self.fps = fps


def _read_video_in_memory(ele: dict) -> torch.Tensor:
	mem: InMemoryVideo = ele["video"]
	if "video_start" in ele or "video_end" in ele:
		raise NotImplementedError("in-memory VideoAlign path does not support video_start/video_end")
	video = mem.frames
	if video.ndim != 4 or video.shape[-1] != 3:
		raise ValueError(f"InMemoryVideo.frames must be [T, H, W, 3], got {tuple(video.shape)}")
	# Same layout/dtype torchvision.io.read_video(output_format="TCHW") returns,
	# so the downstream resize executes the identical (uint8) code path.
	video = video.permute(0, 3, 1, 2).contiguous()
	idx = _select_frame_indices(ele, video.shape[0], mem.fps)
	return video[idx]


def _read_video_decord(ele: dict) -> torch.Tensor:
	import decord

	video_path = ele["video"]
	vr = decord.VideoReader(video_path)
	if "video_start" in ele or "video_end" in ele:
		raise NotImplementedError("not support start_pts and end_pts in decord for now.")
	total_frames, video_fps = len(vr), vr.get_avg_fps()
	idx = _select_frame_indices(ele, total_frames, video_fps)
	video = vr.get_batch(idx).asnumpy()
	return torch.tensor(video).permute(0, 3, 1, 2)


VIDEO_READER_BACKENDS = {
	"decord": _read_video_decord,
	"torchvision": _read_video_torchvision,
}
FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
	video_reader_backend = FORCE_QWENVL_VIDEO_READER if FORCE_QWENVL_VIDEO_READER is not None else "torchvision"
	print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
	return video_reader_backend


def _read_video_robust(ele: dict) -> torch.Tensor:
	"""Decode with retry + backend fallback.

	The mp4 is written by the sampler moments before the scorer reads it; on a
	busy NFS the file is occasionally not yet fully visible, and torchvision
	then raises ``KeyError: 'video_fps'`` right after ffmpeg logs
	"Error submitting packet to decoder". Historically this killed the whole
	4-GPU job over ONE unreadable file (seen across runs since at least
	w22t2v_ta3_diffusionnft_low, and 3x on b1e3 on 2026-09-14).

	Retry the primary backend with a short backoff, then try the other backend.
	Reward semantics are unchanged — this only converts a transient read
	failure into a successful read.
	"""
	backend = get_video_reader_backend()
	others = [b for b in VIDEO_READER_BACKENDS if b != backend]
	last_err = None
	for attempt in range(4):
		try:
			return VIDEO_READER_BACKENDS[backend](ele)
		except Exception as e:  # noqa: BLE001 - decode errors are backend-specific
			last_err = e
			if attempt < 3:
				time.sleep(1.5 * (attempt + 1))
	for alt in others:
		try:
			video = VIDEO_READER_BACKENDS[alt](ele)
			print(f"[videoalign] {backend} failed on {ele.get('video')} ({type(last_err).__name__}); "
				  f"{alt} fallback succeeded", file=sys.stderr, flush=True)
			return video
		except Exception as e:  # noqa: BLE001
			last_err = e
	raise RuntimeError(
		f"[videoalign] could not decode {ele.get('video')} after retries and backend fallback"
	) from last_err


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor:
	if isinstance(ele["video"], InMemoryVideo):
		video = _read_video_in_memory(ele)
	elif isinstance(ele["video"], str):
		video = _read_video_robust(ele)
	else:
		raise ValueError("VideoAlignScorer expects a video path or an InMemoryVideo")
	nframes, _, height, width = video.shape
	min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
	total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
	max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
	max_pixels = ele.get("max_pixels", max_pixels)
	if "resized_height" in ele and "resized_width" in ele:
		resized_height, resized_width = smart_resize(ele["resized_height"], ele["resized_width"], factor=image_factor)
	else:
		resized_height, resized_width = smart_resize(
			height,
			width,
			factor=image_factor,
			min_pixels=min_pixels,
			max_pixels=max_pixels,
		)
	return transforms.functional.resize(
		video,
		[resized_height, resized_width],
		interpolation=InterpolationMode.BICUBIC,
		antialias=True,
	).float()


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
	vision_infos = []
	if isinstance(conversations[0], dict):
		conversations = [conversations]
	for conversation in conversations:
		for message in conversation:
			if isinstance(message["content"], list):
				for ele in message["content"]:
					if "video" in ele or ele.get("type") == "video":
						vision_infos.append(ele)
	return vision_infos


def process_vision_info(conversations: list[dict] | list[list[dict]]):
	vision_infos = extract_vision_info(conversations)
	image_inputs = []
	video_inputs = []
	for vision_info in vision_infos:
		if "image" in vision_info or "image_url" in vision_info:
			raise ValueError("VideoAlignScorer only supports video inputs, not images")
		elif "video" in vision_info:
			video_inputs.append(fetch_video(vision_info))
		else:
			raise ValueError("image, image_url or video should in content.")
	if len(image_inputs) == 0:
		image_inputs = None
	if len(video_inputs) == 0:
		video_inputs = None
	return image_inputs, video_inputs


class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
	def __init__(self, config, output_dim=4, reward_token="last", special_token_ids=None):
		super().__init__(config)
		self.output_dim = output_dim
		hidden_size = getattr(config, "hidden_size", None) or config.text_config.hidden_size
		self.rm_head = torch.nn.Linear(hidden_size, output_dim, bias=False)
		self.reward_token = reward_token

		self.special_token_ids = special_token_ids
		if self.special_token_ids is not None:
			self.reward_token = "special"

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[list[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		pixel_values: Optional[torch.Tensor] = None,
		pixel_values_videos: Optional[torch.FloatTensor] = None,
		image_grid_thw: Optional[torch.LongTensor] = None,
		video_grid_thw: Optional[torch.LongTensor] = None,
		rope_deltas: Optional[torch.LongTensor] = None,
		**kwargs,
	):
		## modified from the origin class Qwen2VLForConditionalGeneration
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if inputs_embeds is None:
			inputs_embeds = self.model.embed_tokens(input_ids)
			if pixel_values is not None:
				pixel_values = pixel_values.type(self.visual.get_dtype())
				image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
				image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
				image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
				inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

			if pixel_values_videos is not None:
				pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
				video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
				video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
				video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
				inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

			if attention_mask is not None:
				attention_mask = attention_mask.to(inputs_embeds.device)

		outputs = self.model(
			input_ids=None,
			position_ids=position_ids,
			attention_mask=attention_mask,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]  # [B, L, D]

		logits = self.rm_head(hidden_states)    # [B, L, N]

		if input_ids is not None:
			batch_size = input_ids.shape[0]
		else:
			batch_size = inputs_embeds.shape[0]

		## get sequence length
		if self.config.pad_token_id is None and batch_size != 1:
			raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
		if self.config.pad_token_id is None:
			sequence_lengths = -1
		else:
			if input_ids is not None:
				# if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
				sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
				sequence_lengths = sequence_lengths % input_ids.shape[-1]
				sequence_lengths = sequence_lengths.to(logits.device)
			else:
				sequence_lengths = -1

		## get the last token's logits
		if self.reward_token == "last":
			pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
		elif self.reward_token == "mean":
			## get the mean of all valid tokens' logits
			valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
			pooled_logits = torch.stack([logits[i, :valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
		elif self.reward_token == "special":
			special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
			for special_token_id in self.special_token_ids:
				special_token_mask = special_token_mask | (input_ids == special_token_id)
			pooled_logits = logits[special_token_mask, ...]
			pooled_logits = pooled_logits.view(batch_size, 3, -1)   # [B, 3, N] assert 3 attributes
			if self.output_dim == 3:
				pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
			pooled_logits = pooled_logits.view(batch_size, -1)
		else:
			raise ValueError("Invalid reward_token")

		return {"logits": pooled_logits}


def _find_target_linear_names(model, num_lora_modules: int = -1, lora_namespan_exclude: Optional[list[str]] = None):
	if lora_namespan_exclude is None:
		lora_namespan_exclude = []
	lora_module_names = []
	for name, module in model.named_modules():
		if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
			continue
		if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
			lora_module_names.append(name)
	if num_lora_modules > 0:
		lora_module_names = lora_module_names[-num_lora_modules:]
	return lora_module_names


def create_model_and_processor(model_config, peft_lora_config, training_args, cache_dir=None):
	model_name_or_path = model_config.model_name_or_path or "Qwen/Qwen2-VL-2B-Instruct"
	processor = AutoProcessor.from_pretrained(model_name_or_path, padding_side="right", cache_dir=cache_dir)

	special_token_ids = None
	if model_config.use_special_tokens:
		special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
		processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
		special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

	attn_impl = "sdpa"
	if not training_args.disable_flash_attn2:
		try:
			import flash_attn  # noqa: F401
			attn_impl = "flash_attention_2"
		except Exception:
			pass

	model = Qwen2VLRewardModelBT.from_pretrained(
		model_name_or_path,
		output_dim=model_config.output_dim,
		reward_token=model_config.reward_token,
		special_token_ids=special_token_ids,
		attn_implementation=attn_impl,
		cache_dir=cache_dir,
	)

	if model_config.use_special_tokens:
		model.resize_token_embeddings(len(processor.tokenizer))
	if training_args.bf16:
		model.to(torch.bfloat16)
	if training_args.fp16:
		model.to(torch.float16)

	if peft_lora_config.lora_enable:
		target_modules = _find_target_linear_names(
			model,
			num_lora_modules=peft_lora_config.num_lora_modules,
			lora_namespan_exclude=peft_lora_config.lora_namespan_exclude,
		)
		peft_config = LoraConfig(
			target_modules=target_modules,
			r=peft_lora_config.lora_r,
			lora_alpha=peft_lora_config.lora_alpha,
			lora_dropout=peft_lora_config.lora_dropout,
			task_type=peft_lora_config.lora_task_type,
			use_rslora=peft_lora_config.use_rslora,
			bias="none",
			modules_to_save=peft_lora_config.lora_modules_to_save,
		)
		model = get_peft_model(model, peft_config)
	else:
		peft_config = None

	model.config.tokenizer_padding_side = processor.tokenizer.padding_side
	model.config.pad_token_id = processor.tokenizer.pad_token_id
	return model, processor, peft_config


def _insert_adapter_name_into_state_dict(state_dict: dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str) -> dict[str, torch.Tensor]:
	peft_model_state_dict = {}
	for key, value in state_dict.items():
		if parameter_prefix in key:
			suffix = key.split(parameter_prefix)[1]
			if "." in suffix:
				suffix_to_replace = ".".join(suffix.split(".")[1:])
				key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
			else:
				key = f"{key}.{adapter_name}"
			peft_model_state_dict[key] = value
		else:
			peft_model_state_dict[key] = value
	return peft_model_state_dict


def load_model_from_checkpoint(model, checkpoint_dir, checkpoint_step):
	checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
	checkpoint_paths.sort(key=lambda path: int(path.split("-")[-1]), reverse=True)
	if checkpoint_step is None or checkpoint_step == -1:
		checkpoint_path = checkpoint_paths[0]
	else:
		checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_step}")
		if checkpoint_path not in checkpoint_paths:
			checkpoint_path = checkpoint_paths[0]
	checkpoint_step = checkpoint_path.split("checkpoint-")[-1].split("/")[0]

	full_ckpt = os.path.join(checkpoint_path, "model.pth")
	lora_ckpt = os.path.join(checkpoint_path, "adapter_model.safetensors")
	non_lora_ckpt = os.path.join(checkpoint_path, "non_lora_state_dict.pth")
	if os.path.exists(full_ckpt):
		model_state_dict = torch.load(full_ckpt, map_location="cpu")
		model.load_state_dict(model_state_dict, strict=True)
	else:
		lora_state_dict = safetensors.torch.load_file(lora_ckpt)
		non_lora_state_dict = torch.load(non_lora_ckpt, map_location="cpu")
		lora_state_dict = _insert_adapter_name_into_state_dict(lora_state_dict, adapter_name="default", parameter_prefix="lora_")

		# Verify rm_head.weight is provided by the checkpoint — if missing,
		# it would silently stay random-initialized, producing garbage rewards.
		all_checkpoint_keys = set(non_lora_state_dict.keys()) | set(lora_state_dict.keys())
		model_keys = set(model.state_dict().keys())
		missing_from_checkpoint = model_keys - all_checkpoint_keys
		# Keys from the base pretrained model (loaded via from_pretrained) are expected
		# to not be in the LoRA checkpoint. But rm_head and any other non-pretrained
		# keys MUST be in the checkpoint.
		critical_missing = [k for k in missing_from_checkpoint if "rm_head" in k]
		if critical_missing:
			raise RuntimeError(
				f"VideoAlign LoRA checkpoint is missing critical reward head weights: {critical_missing}. "
				f"These would remain randomly initialized, producing invalid reward scores. "
				f"Checkpoint: {checkpoint_path}"
			)

		model_state_dict = model.state_dict()
		model_state_dict.update(non_lora_state_dict)
		model_state_dict.update(lora_state_dict)
		model.load_state_dict(model_state_dict)
	return model, checkpoint_step


class VideoVLMRewardInference:
	def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device="cuda", dtype=torch.bfloat16):
		config_path = os.path.join(load_from_pretrained, "model_config.json")
		data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
		self.data_config = DataConfig(**data_config)
		model_config = ModelConfig(**model_config)
		peft_lora_config = PEFTLoraConfig(**peft_lora_config)
		training_args = TrainingConfig(
			load_from_pretrained=load_from_pretrained,
			load_from_pretrained_step=load_from_pretrained_step,
			gradient_checkpointing=False,
			disable_flash_attn2=False,
			bf16=dtype == torch.bfloat16,
			fp16=dtype == torch.float16,
			output_dir="",
		)
		model, processor, _ = create_model_and_processor(
			model_config=model_config,
			peft_lora_config=peft_lora_config,
			training_args=training_args,
		)
		self.device = device
		model, _ = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
		model.eval()
		self.model = model.to(self.device)
		self.processor = processor
		self.inference_config = inference_config

	def _norm(self, reward):
		if self.inference_config is None:
			return reward
		reward["VQ"] = (reward["VQ"] - self.inference_config["VQ_mean"]) / self.inference_config["VQ_std"]
		reward["MQ"] = (reward["MQ"] - self.inference_config["MQ_mean"]) / self.inference_config["MQ_std"]
		reward["TA"] = (reward["TA"] - self.inference_config["TA_mean"]) / self.inference_config["TA_std"]
		return reward

	def _prepare_input(self, data):
		if isinstance(data, Mapping):
			return type(data)({key: self._prepare_input(value) for key, value in data.items()})
		if isinstance(data, (tuple, list)):
			return type(data)(self._prepare_input(value) for value in data)
		if isinstance(data, torch.Tensor):
			return data.to(device=self.device)
		return data

	def _prepare_inputs(self, inputs):
		inputs = self._prepare_input(inputs)
		if len(inputs) == 0:
			raise ValueError
		return inputs

	def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None):
		fps = self.data_config.fps if fps is None else fps
		num_frames = self.data_config.num_frames if num_frames is None else num_frames
		max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels
		def _video_ref(v):
			# A path becomes the usual file:// URI; an InMemoryVideo is passed
			# through untouched and read by fetch_video without touching disk.
			return v if isinstance(v, InMemoryVideo) else f"file://{v}"

		if num_frames is None:
			chat_data = [[{
				"role": "user",
				"content": [
					{
						"type": "video",
						"video": _video_ref(video_path),
						"max_pixels": max_pixels,
						"fps": fps,
						"sample_type": self.data_config.sample_type,
					},
					{"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
				],
			}] for video_path, prompt in zip(video_paths, prompts)]
		else:
			chat_data = [[{
				"role": "user",
				"content": [
					{
						"type": "video",
						"video": _video_ref(video_path),
						"max_pixels": max_pixels,
						"nframes": num_frames,
						"sample_type": self.data_config.sample_type,
					},
					{"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
				],
			}] for video_path, prompt in zip(video_paths, prompts)]
		image_inputs, video_inputs = process_vision_info(chat_data)
		batch = self.processor(
			text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
			images=image_inputs,
			videos=video_inputs,
			padding=True,
			return_tensors="pt",
			videos_kwargs={"do_rescale": True},
		)
		return self._prepare_inputs(batch)

	def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
		assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
		# Process videos one at a time to avoid Qwen2VL vision encoder OOM.
		# Batching multiple long videos causes the attention over the
		# concatenated token sequence to request >100 GiB of VRAM.
		rewards = []
		for vp, pr in zip(video_paths, prompts):
			batch = self.prepare_batch([vp], [pr], fps, num_frames, max_pixels)
			logits = self.model(return_dict=True, **batch)["logits"]
			assert logits.shape[-1] == 3, f"expected logits last dim=3 (VQ/MQ/TA), got {tuple(logits.shape)}"
			reward = {"VQ": logits[0][0].item(), "MQ": logits[0][1].item(), "TA": logits[0][2].item()}
			if use_norm:
				reward = self._norm(reward)
			reward["Overall"] = reward["VQ"] + reward["MQ"] + reward["TA"]
			rewards.append(reward)
		return rewards


# Per-process cache so reward keys sharing the same VideoAlign backbone
# (videoalign_mq + videoalign_ta + videoalign_vq) reuse one ~14 GB Qwen2-VL
# instance instead of loading three separate copies. Keyed only on
# (device, video_fps, eval_num_frames) — reward_key is picked per call by the
# wrapping closure since the scorer always returns all dim scores.
_VIDEOALIGN_SCORER_CACHE: dict[tuple, "VideoAlignScorer"] = {}


def get_shared_videoalign_scorer(
	device, video_fps: int = 16, eval_num_frames: int | None = None
) -> "VideoAlignScorer":
	"""Return the per-(device, fps, frames) shared VideoAlignScorer instance."""
	key = (str(device), int(video_fps), eval_num_frames)
	scorer = _VIDEOALIGN_SCORER_CACHE.get(key)
	if scorer is None:
		scorer = VideoAlignScorer(
			device=device,
			video_fps=video_fps,
			reward_key="Overall",  # placeholder; closures pick dim from dim_scores
			eval_num_frames=eval_num_frames,
			dimension_weights=None,
		)
		_VIDEOALIGN_SCORER_CACHE[key] = scorer
	return scorer



def _videoalign_in_memory() -> bool:
	"""VIDEOALIGN_INMEMORY=1 -> score frames without the mp4 round trip (default: off)."""
	return os.getenv("VIDEOALIGN_INMEMORY", "0") == "1"


def _videoalign_encode_mode() -> str:
	"""VIDEOALIGN_ENCODE=file (default) | pipe (legacy imageio stdin pipe).

	`file` runs the SAME ffmpeg binary with the SAME arguments imageio-ffmpeg
	0.6.0 uses (rawvideo rgb24 -> libx264 yuv420p -crf 25, macro-block 16),
	only reading the frames from a temp .rgb file instead of `-i -`. The mp4
	bytes — and therefore the reward — are identical; what disappears is the
	stdin pipe that delivered a truncated first packet on b1e3 (5x, 3 nodes).
	"""
	# Default flipped pipe->file on 2026-09-14 after the A/B job showed
	# max|A-D| = 0.00e+00 on every VideoAlign dim (8 videos) and md5-identical
	# mp4s offline; the pipe is what truncated b1e3's sample 4 on 5 jobs.
	return os.getenv("VIDEOALIGN_ENCODE", "file")


def _encode_mp4_file_input(video_thwc: np.ndarray, out_path: Path, fps: float) -> None:
	import subprocess

	import imageio_ffmpeg

	v = np.ascontiguousarray(video_thwc)
	if v.dtype != np.uint8 or v.ndim != 4 or v.shape[-1] != 3:
		raise ValueError(f"expected uint8 [T,H,W,3], got {v.dtype} {tuple(v.shape)}")
	_, h, w, _ = v.shape
	raw = Path(str(out_path) + ".rgb")
	raw.write_bytes(v.tobytes())
	# Mirror what imageio's v2 FFMPEG plugin actually passes to
	# imageio_ffmpeg.write_frames for mimsave(codec="libx264"): quality=None
	# (so NO -crf — libx264's own default applies), pix_fmt_out None -> yuv420p,
	# log level warning. Verified by wrapping write_frames; an earlier draft
	# copied the function default quality=5 (-crf 25) and produced a 12%
	# smaller, different mp4 (D != A in the A/B job).
	cmd = [
		imageio_ffmpeg.get_ffmpeg_exe(), "-y",
		"-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}",
		"-pix_fmt", "rgb24", "-r", "{:.02f}".format(fps),
		"-i", str(raw), "-an",
		"-vcodec", "libx264", "-pix_fmt", "yuv420p",
	]
	if w % 16 or h % 16:  # same macro_block_size=16 padding rule as imageio
		cmd += ["-vf", "scale={}:{}".format(w + (16 - w % 16) % 16, h + (16 - h % 16) % 16)]
	cmd += ["-v", "warning", str(out_path)]
	try:
		res = subprocess.run(cmd, capture_output=True, timeout=900)
		if res.returncode != 0:
			raise RuntimeError(
				f"ffmpeg(file-input) rc={res.returncode}: {res.stderr.decode(errors='replace')[-400:]}"
			)
	finally:
		raw.unlink(missing_ok=True)


def _encode_frames(out_path: Path, video_thwc: np.ndarray, frames: list, fps: float, codec: str) -> None:
	if _videoalign_encode_mode() == "file" and codec == "libx264":
		_encode_mp4_file_input(video_thwc, out_path, fps)
	else:
		imageio.mimsave(str(out_path), frames, fps=fps, codec=codec, format="FFMPEG")

class VideoAlignScorer(torch.nn.Module):
	"""Self-contained VideoAlign reward scorer.

	Expected input videos: numpy array with shape [B, T, H, W, C] and uint8 dtype.

	Args:
		device: torch device for inference.
		video_fps: fps used when saving temp videos for the reward model.
		reward_key: which dimension to return as the scalar score.
			One of "Overall", "VQ", "MQ", "TA".
		eval_num_frames: if set, uniformly subsample the input video to this
			many frames before scoring (reduces VRAM and speeds up eval).
			Must be even (FRAME_FACTOR=2).  None keeps all input frames.
	"""

	def __init__(
		self,
		device: str | torch.device = "cuda",
		video_fps: int = 16,
		reward_key: str = "Overall",
		eval_num_frames: int | None = None,
		dimension_weights: dict[str, float] | None = None,
	):
		super().__init__()
		self.device = str(device)
		ckpt_path = _resolve_videoalign_ckpt_path()
		self.video_fps = video_fps
		self.reward_key = reward_key
		self.eval_num_frames = eval_num_frames
		self.dimension_weights = dimension_weights
		self.inferencer = VideoVLMRewardInference(
			load_from_pretrained=str(ckpt_path),
			device=self.device,
			dtype=torch.bfloat16,
		)
		self._warmup()

	def _warmup(self) -> None:
		"""Pre-compile Qwen2-VL CUDA kernels to eliminate the 15s cold-start.

		Diagnosis: the very first Qwen2-VL forward call triggers CUDA kernel
		autotune that takes ~15s on H100 even on an idle GPU. Under heavy
		contention with concurrent WAN2.1 sampling on the same device, this
		can amplify 4-10x and blow past the per-video timeout. Running a
		dummy forward at init time (when the GPU is otherwise free) makes
		every subsequent production call hit warm kernels (~0.3s).

		Failure here is non-fatal — if warmup itself errors, fall through to
		the regular code path so the cold-start cost is paid in production.
		"""
		try:
			t0 = time.monotonic()
			# Production-shaped dummy: 53 frames @ 480x832 (matches scripts'
			# `frames` and `height`/`width` config). Black frames are fine —
			# we only care about exercising the kernels, not the score.
			dummy = np.zeros((53, 480, 832, 3), dtype=np.uint8)
			with tempfile.TemporaryDirectory(prefix="videoalign_warmup_") as tmpdir:
				out_path = Path(tmpdir) / "warmup.mp4"
				imageio.mimsave(
					str(out_path), list(dummy), fps=self.video_fps,
					codec="libx264", format="FFMPEG",
				)
				with torch.no_grad():
					self.inferencer.reward([str(out_path)], ["warmup"], use_norm=False)
			logger.info("VideoAlign: warmup completed in %.2fs", time.monotonic() - t0)
		except Exception as exc:  # noqa: BLE001
			logger.warning("VideoAlign: warmup failed (non-fatal): %s", exc)

	def _subsample_frames(self, video: np.ndarray) -> np.ndarray:
		"""Uniformly subsample T frames from a [T, H, W, C] video."""
		if self.eval_num_frames is None or video.shape[0] <= self.eval_num_frames:
			return video
		indices = np.linspace(0, video.shape[0] - 1, self.eval_num_frames).round().astype(int)
		return video[indices]

	@torch.no_grad()
	def __call__(self, videos: np.ndarray, prompts: list[str], metadata) -> list[float]:
		if not isinstance(videos, np.ndarray):
			raise ValueError(f"Expected numpy.ndarray for videos, got {type(videos)}")
		if videos.ndim != 5:
			raise ValueError(f"Expected videos shape [B, T, H, W, C], got {videos.shape}")
		if videos.shape[-1] != 3:
			raise ValueError(f"Expected RGB videos with C=3, got shape {videos.shape}")
		if videos.dtype != np.uint8:
			# np.clip does NOT remove NaN (clip(nan,0,255) == nan) and casting
			# NaN/Inf to uint8 is undefined in numpy — the resulting frames made
			# libx264 reject the raw pipe ("Error submitting packet to decoder:
			# Invalid argument"), leaving a headerless mp4 that later blew up in
			# _read_video_torchvision with KeyError 'video_fps' and killed the
			# whole 4-GPU sample job (b1e3, 3x on 3 different nodes 2026-09-14).
			n_bad = int((~np.isfinite(videos)).sum())
			if n_bad:
				per_sample = [int((~np.isfinite(v)).sum()) for v in videos]
				logger.error(
					"VideoAlign: %d non-finite pixels in this batch (per-sample: %s) — "
					"model output contains NaN/Inf; sanitizing to keep the job alive, "
					"but the affected samples' rewards are meaningless.",
					n_bad, per_sample,
				)
				videos = np.nan_to_num(videos, nan=0.0, posinf=255.0, neginf=0.0)
			videos = np.clip(videos, 0, 255).astype(np.uint8)

		if _videoalign_in_memory():
			# In-memory path (opt-in, VIDEOALIGN_INMEMORY=1): hand the sampler's
			# frames straight to Qwen2-VL. Removes the imageio->ffmpeg->/tmp->
			# torchvision round trip (the failure point that killed b1e3 5x on
			# 2026-09-14) and the H.264 loss on the pixels the reward sees.
			# Frame selection/resize are the exact same functions as the file path
			# (see _select_frame_indices / fetch_video).
			mem = []
			for idx, video in enumerate(videos):
				video = self._subsample_frames(video)
				if video.shape[0] == 0:
					raise RuntimeError(f"[videoalign] sample {idx} has 0 frames (shape={video.shape})")
				mem.append(InMemoryVideo(torch.from_numpy(np.ascontiguousarray(video)), self.video_fps))
			rewards = self.inferencer.reward(video_paths=mem, prompts=prompts, use_norm=True)
		else:
			with tempfile.TemporaryDirectory(prefix="videoalign_") as tmpdir:
				video_paths = []
				for idx, video in enumerate(videos):
					video = self._subsample_frames(video)
					if video.shape[0] == 0:
						raise RuntimeError(f"[videoalign] sample {idx} has 0 frames (shape={video.shape})")
					out_path = Path(tmpdir) / f"sample_{idx}.mp4"
					# Encode with retries: the frames are valid (checked above), but the
					# imageio->ffmpeg stdin pipe occasionally delivers a truncated first
					# packet ("Invalid buffer size, packet size 18432 < expected
					# frame_size 1198080"), leaving a ~261B headerless mp4 that only
					# explodes later at read time and kills the whole 4-GPU job
					# (b1e3, 3x on 3 nodes 2026-09-14; /tmp and host RAM were both fine).
					# Since VIDEOALIGN_ENCODE=file became the default (2026-09-14, user
					# adopted it as THE reward path) every retry stays libx264 with the
					# identical arguments: a codec switch (the old mpeg4 last resort)
					# would silently hand the model a different video and thus a
					# different reward for that sample. pyav (same libx264, in-process)
					# below is the only remaining fallback and it warns loudly.
					frames = list(video)
					last_err = None
					for attempt, codec in enumerate(("libx264", "libx264", "libx264")):
						try:
							_encode_frames(out_path, video, frames, self.video_fps, codec)
						except Exception as exc:  # noqa: BLE001 - retry any writer error
							last_err = exc
						sz = out_path.stat().st_size if out_path.exists() else 0
						if sz >= 1024:
							if attempt:
								logger.warning(
									"VideoAlign: sample %d encoded only on attempt %d (codec=%s, %dB)",
									idx, attempt + 1, codec, sz,
								)
							break
						logger.warning(
							"VideoAlign: sample %d encode attempt %d (codec=%s) produced %dB — retrying",
							idx, attempt + 1, codec, sz,
						)
						out_path.unlink(missing_ok=True)
					else:
						# Last resort: pyav writes the frames IN-PROCESS, with no
						# subprocess stdin pipe at all. Every imageio attempt above
						# dies with ffmpeg reporting a truncated first packet
						# ("packet size 18432 < expected frame_size 1198080"), which
						# is a pipe-transport symptom — so bypassing the pipe is the
						# one fallback that addresses the observed failure directly.
						try:
							import imageio.v3 as _iio
							_iio.imwrite(str(out_path), video, plugin="pyav", codec="libx264", fps=self.video_fps)
							sz = out_path.stat().st_size if out_path.exists() else 0
							if sz >= 1024:
								logger.warning(
									"VideoAlign: sample %d encoded via pyav in-process fallback (%dB) "
									"after 3 imageio/ffmpeg-pipe failures", idx, sz,
								)
								video_paths.append(str(out_path))
								continue
							last_err = RuntimeError(f"pyav produced {sz}B")
						except Exception as exc:  # noqa: BLE001
							last_err = exc
						# Forensics: the frames look valid by every cheap check
						# (shape/dtype/range) yet ffmpeg reports a truncated first
						# packet deterministically for this one sample, on every node
						# and both codecs. Dump the actual array so it can be replayed
						# offline instead of guessing from log lines.
						try:
							dump_dir = Path(os.environ.get("VIDEOALIGN_DEBUG_DUMP_DIR", "/tmp"))
							dump_dir.mkdir(parents=True, exist_ok=True)
							dump = dump_dir / f"encfail_rank{os.environ.get('RANK','?')}_sample{idx}_{os.getpid()}.npy"
							np.save(str(dump), video)
							logger.error("VideoAlign: dumped failing frames to %s", dump)
						except Exception as dexc:  # noqa: BLE001
							logger.error("VideoAlign: frame dump failed: %s", dexc)
						raise RuntimeError(
							f"[videoalign] encode failed for sample {idx} after 3 attempts "
							f"(frames={video.shape}, dtype={video.dtype}, "
							f"contig={video.flags['C_CONTIGUOUS']}, "
							f"min={int(video.min())}, max={int(video.max())}, "
							f"nuniq={int(np.unique(video[0]).size)}, last_err={last_err})"
						)
					video_paths.append(str(out_path))
				rewards = self.inferencer.reward(video_paths=video_paths, prompts=prompts, use_norm=True)

		scores = []
		dim_scores = {"VQ": [], "MQ": [], "TA": [], "Overall": []}
		for reward in rewards:
			if isinstance(reward, dict):
				for dim_key in dim_scores:
					if dim_key in reward:
						dim_scores[dim_key].append(float(reward[dim_key]))
				if self.dimension_weights:
					score = sum(w * reward[k] for k, w in self.dimension_weights.items())
					scores.append(float(score))
				else:
					if self.reward_key not in reward:
						available = ", ".join(sorted(reward.keys()))
						raise KeyError(f"Requested reward_key='{self.reward_key}' not found. Available keys: {available}")
					scores.append(float(reward[self.reward_key]))
			else:
				scores.append(float(reward))
		return scores, dim_scores
