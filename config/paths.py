"""config/paths.py — every external path the Python code uses, with the same defaults as config/paths.sh
(environment variables override them)."""
import os

# transformers>=4.49 pip overlay for the Qwen2.5-VL judge rewards
VS2_OVERLAY = os.environ.get("VS2_OVERLAY", os.path.join(os.path.expanduser("~"), "vs2_overlay"))
# VBench model cache: VBench itself reads ${VBENCH_CACHE_DIR:-~/.cache/vbench} (vbench/utils.py).
VBENCH_CACHE_DIR = os.environ.get("VBENCH_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "vbench"))

# HuggingFace cache directory
HF_HOME = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

# HuggingFace token search paths
HF_TOKEN_PATHS = [
    os.path.join(HF_HOME, "token"),
    os.path.expanduser("~/.cache/huggingface/token"),
]

# Weights & Biases entity (None: your default entity)
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None
