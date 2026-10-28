from paddleocr import PaddleOCR
import torch
import numpy as np
import os
import threading
from concurrent import futures
from Levenshtein import distance
from typing import List, Union, Tuple
from PIL import Image


def _cpu_plan() -> Tuple[int, int]:
    """(workers, cpu_threads) per rank for CPU-side PaddleOCR.

    Scoring is the OCR runs' bottleneck: ~40 of every 46-min epoch, entirely on
    CPU (PaddleOCR has no GPU here by choice). Every torchrun rank gets an equal
    slice of --cpus-per-task, so budget within that slice. Inter-image
    parallelism (workers) scales better than intra-op threads on 512px crops,
    so default to 1 thread per worker. Override with OCR_WORKERS /
    OCR_CPU_THREADS; both fall back to the old single-stream behaviour if set
    to 1 and 10 respectively.
    """
    cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))
    ranks = int(os.getenv("NUM_GPUS", "1")) or 1
    per_rank = max(1, cpus // ranks)
    workers = int(os.getenv("OCR_WORKERS", str(per_rank)))
    threads = int(os.getenv("OCR_CPU_THREADS", "1" if workers > 1 else str(per_rank)))
    return max(1, workers), max(1, threads)


class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        self._workers, self._cpu_threads = _cpu_plan()
        self._use_gpu = use_gpu
        # PaddleOCR predictors are NOT thread-safe, so each worker thread builds
        # and keeps its own instance rather than sharing self.ocr.
        self._local = threading.local()
        self._pool = (
            futures.ThreadPoolExecutor(max_workers=self._workers) if self._workers > 1 else None
        )
        self.ocr = self._build()

    def _build(self) -> PaddleOCR:
        return PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=self._use_gpu,
            show_log=False,  # Disable unnecessary log output
            # enable_mkldnn defaults to False upstream, which leaves CPU
            # inference on the unaccelerated path; cpu_threads defaults to 10
            # per instance, which oversubscribes the rank's CPU slice.
            enable_mkldnn=(not self._use_gpu) and os.getenv("OCR_MKLDNN", "1") != "0",
            cpu_threads=self._cpu_threads,
        )

    def _predictor(self) -> PaddleOCR:
        if self._pool is None:
            return self.ocr
        got = getattr(self._local, "ocr", None)
        if got is None:
            got = self._build()
            self._local.ocr = got
        return got

    @torch.no_grad()
    def __call__(self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]) -> torch.Tensor:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        if self._pool is None:
            return [self._score_one(i, p) for i, p in zip(images, prompts)]
        # map() preserves input order, so rewards stay aligned with prompts.
        return list(self._pool.map(self._score_one, images, prompts))

    def _score_one(self, img: Union[Image.Image, np.ndarray], prompt: str) -> float:
        # Convert image format
        if isinstance(img, Image.Image):
            img = np.array(img)

        try:
            # OCR recognition
            result = self._predictor().ocr(img, cls=False)
            # Extract recognized text (handle possible multi-line results)
            recognized_text = (
                "".join([res[1][0] if res[1][1] > 0 else "" for res in result[0]]) if result[0] else ""
            )

            recognized_text = recognized_text.replace(" ", "").lower()
            prompt = prompt.replace(" ", "").lower()
            if prompt in recognized_text:
                dist = 0
            else:
                dist = distance(recognized_text, prompt)
            # Recognized many unrelated characters, only add one character penalty
            if dist > len(prompt):
                dist = len(prompt)

        except Exception as e:
            # Error handling (e.g., OCR parsing failure)
            print(f"OCR processing failed: {str(e)}")
            dist = len(prompt)  # Maximum penalty
        return 1 - dist / (len(prompt))

class OcrScorer_video_or_image:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False  # Disable unnecessary log output
        )
        self.frame_interval = 4

    @torch.no_grad()
    def __call__(self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]) -> Tuple[List[float], torch.Tensor]:
        """
        :param images: List of images or videos (each video as np.ndarray of shape [F, H, W, C])
        :param prompts: List of prompts containing target text
        :return: (List of OCR rewards, Tensor of attention regions)
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        assert len(images) == len(prompts), "Mismatch between images and prompts."

        rewards = []
        for img, prompt in zip(images, prompts):
            prompt = prompt.replace(' ', '').lower()
            frame_rewards = []

            # Handle video: shape (F, H, W, C)
            if isinstance(img, np.ndarray) and img.ndim == 4:
                sampled_frames = img[::self.frame_interval]
            else:
                sampled_frames = [img]

            for frame in sampled_frames:
                region = None
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                try:
                    result = self.ocr.ocr(frame, cls=False)
                    text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                    text = text.replace(' ', '').lower()

                    dist = distance(text, prompt)
                    dist = min(dist, len(prompt))
    
                except Exception as e:
                    print(f"OCR failed on frame: {e}")
                    dist = len(prompt)

                reward = 1 - dist / len(prompt)
                if reward > 0:
                    frame_rewards.append(reward)

            if frame_rewards:
                rewards.append(sum(frame_rewards) / len(frame_rewards))
            else:
                rewards.append(0.0)

        return rewards
    
if __name__ == "__main__":
    example_image_path = "test_cases/hello world.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    # Instantiate scorer
    scorer = OcrScorer(use_gpu=False)

    # Call scorer and print result
    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")
