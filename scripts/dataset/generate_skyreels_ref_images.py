"""Generate reference images for SkyReels-I2V using Flux.

Pre-generates reference images from text prompts using the Flux model,
so training doesn't need to load Flux on-the-fly. Supports multi-GPU
parallel generation via torchrun.

Usage (single GPU):
    python scripts/dataset/generate_skyreels_ref_images.py \
        --prompt_file dataset/skyreels_i2v/train.txt \
        --output_dir dataset/skyreels_i2v/ref_images_train \
        --height 400 --width 640

Usage (multi-GPU):
    torchrun --nproc_per_node=4 scripts/dataset/generate_skyreels_ref_images.py \
        --prompt_file dataset/skyreels_i2v/train.txt \
        --output_dir dataset/skyreels_i2v/ref_images_train \
        --height 400 --width 640

The script saves:
    - ref_images_{split}/{idx:06d}.png  (one image per prompt)
    - ref_images_{split}/metadata.jsonl  (prompt-to-image mapping)
"""

import argparse
import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from diffusers import FluxPipeline
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Flux reference images for SkyReels-I2V")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to prompt .txt file (one prompt per line)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated images and metadata")
    parser.add_argument("--flux_model", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="Flux model name or path")
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU (Flux typically needs batch_size=1 for memory)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip prompts that already have generated images")
    return parser.parse_args()


def main():
    args = parse_args()

    # Distributed setup. Default 10-min NCCL watchdog is too short for the
    # post-generation barrier when --skip_existing leaves ranks with uneven
    # workloads (one rank can finish 1000+ images later than another).
    # 2h covers the worst case and is still well under SLURM wall-time caps.
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read prompts
    with open(args.prompt_file, "r") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    if rank == 0:
        print(f"Loaded {len(all_prompts)} prompts from {args.prompt_file}")
        os.makedirs(args.output_dir, exist_ok=True)

    if distributed:
        dist.barrier()
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # Shard prompts across GPUs
    indices = list(range(len(all_prompts)))
    my_indices = indices[rank::world_size]

    # Filter out already-generated if requested
    if args.skip_existing:
        filtered = []
        for idx in my_indices:
            img_path = os.path.join(args.output_dir, f"{idx:06d}.png")
            if not os.path.exists(img_path):
                filtered.append(idx)
        skipped = len(my_indices) - len(filtered)
        if skipped > 0 and rank == 0:
            print(f"Skipping {skipped} already-generated images")
        my_indices = filtered

    if rank == 0:
        print(f"Rank {rank}: generating {len(my_indices)} images")

    # Load Flux
    pipe = FluxPipeline.from_pretrained(args.flux_model, torch_dtype=torch.bfloat16)
    pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Generate images
    metadata_entries = []
    pbar = tqdm(my_indices, desc=f"[Rank {rank}] Generating", disable=(rank != 0))

    for idx in pbar:
        prompt = all_prompts[idx]
        img_path = os.path.join(args.output_dir, f"{idx:06d}.png")

        with torch.no_grad():
            result = pipe(
                prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=args.max_sequence_length,
                generator=generator,
            )
        image = result.images[0]
        image.save(img_path)

        metadata_entries.append({
            "index": idx,
            "prompt": prompt,
            "image_path": f"{idx:06d}.png",
        })

    # Save per-rank metadata
    rank_meta_path = os.path.join(args.output_dir, f"metadata_rank{rank}.jsonl")
    with open(rank_meta_path, "w") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + "\n")

    if distributed:
        dist.barrier()

    # Rank 0 rebuilds metadata.jsonl from what is ON DISK, not from this
    # invocation's entries. The old code merged only the per-rank lists and
    # wrote with mode "w", so a --skip_existing resume (where every image
    # already exists, so every rank contributes zero entries) OVERWROTE a
    # complete metadata.jsonl with an empty one. Scanning the directory is
    # authoritative and idempotent: rerunning can only ever complete the file.
    if rank == 0:
        for r in range(world_size):
            rpath = os.path.join(args.output_dir, f"metadata_rank{r}.jsonl")
            if os.path.exists(rpath):
                os.remove(rpath)

        all_entries = []
        for idx in range(len(all_prompts)):
            img_name = f"{idx:06d}.png"
            if os.path.exists(os.path.join(args.output_dir, img_name)):
                all_entries.append({
                    "index": idx,
                    "prompt": all_prompts[idx],
                    "image_path": img_name,
                })

        meta_path = os.path.join(args.output_dir, "metadata.jsonl")
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w") as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + "\n")
        os.replace(tmp_path, meta_path)  # atomic: never leave a partial file

        missing = len(all_prompts) - len(all_entries)
        print(f"Done! {len(all_entries)}/{len(all_prompts)} images present in {args.output_dir}")
        print(f"Metadata saved to {meta_path}")
        if missing:
            print(f"INCOMPLETE: {missing} images still missing -- resubmit to continue.")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
