"""Rebuild ref_images_{split}/metadata.jsonl from the PNGs actually on disk.

generate_skyreels_ref_images.py used to write metadata.jsonl from only the
images produced in that invocation, so a --skip_existing resume could truncate
(or empty) a complete file. That is fixed in the generator, but this script
lets you repair or re-derive the mapping at any time without touching a GPU.

The mapping is positional and therefore exact: image ``{idx:06d}.png``
corresponds to line ``idx`` of the prompt file.

    python scripts/dataset/rebuild_ref_metadata.py dataset/vidprom_sf_i2v_544
"""

import argparse
import json
import os


def rebuild(dataset_dir, split):
    prompt_file = os.path.join(dataset_dir, f"{split}.txt")
    out_dir = os.path.join(dataset_dir, f"ref_images_{split}")
    if not os.path.isdir(out_dir):
        print(f"[{split}] no {out_dir}, skipping")
        return
    with open(prompt_file) as f:
        prompts = [line.strip() for line in f if line.strip()]

    entries = [
        {"index": i, "prompt": p, "image_path": f"{i:06d}.png"}
        for i, p in enumerate(prompts)
        if os.path.exists(os.path.join(out_dir, f"{i:06d}.png"))
    ]

    meta_path = os.path.join(out_dir, "metadata.jsonl")
    tmp_path = meta_path + ".tmp"
    with open(tmp_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    os.replace(tmp_path, meta_path)

    missing = len(prompts) - len(entries)
    status = "COMPLETE" if not missing else f"INCOMPLETE ({missing} missing)"
    print(f"[{split}] {len(entries)}/{len(prompts)} -> {meta_path}  {status}")
    return missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    a = ap.parse_args()
    for s in a.splits:
        rebuild(a.dataset_dir, s)
