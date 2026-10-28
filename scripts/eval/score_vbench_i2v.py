"""Official VBench++ (VBench-I2V) scoring on a directory of generated mp4s.

Scores the 10 VBench-I2V dimensions on videos produced by our I2V models
(SkyReels-I2V, Wan2.2-A14B-I2V) for the `dataset/vbench_i2v_a14b` suite:

  I2V dims:      i2v_subject (DINO), i2v_background (DreamSim), camera_motion (CoTracker)
  Quality dims:  subject_consistency, background_consistency, aesthetic_quality,
                 imaging_quality, temporal_flickering, motion_smoothness, dynamic_degree

then aggregates with the official VBench-I2V leaderboard math
(`src/reward/vbench_aggregate.leaderboard_scores_i2v`): min-max normalize,
camera_motion weight 0.1, dynamic_degree 0.5, temporal_flickering excluded from
the groups, Total = (I2V + Quality) / 2.

The installed `vbench2_beta_i2v.VBenchI2V.evaluate()` always runs in
`mode="vbench_standard"` (it forwards `custom_prompt` into **kwargs, which the
parent ignores), so we drive it by:
  * building our OWN full-info JSON from `metadata.jsonl`, where each entry's
    `image_name` is the ABSOLUTE path to the exact conditioning image the model
    saw — `os.path.join(image_root, abs_path)` returns `abs_path`, so VBench's
    relative `vbench2_beta_i2v/data/crop/<res>/` root is bypassed and we never
    need the official crop images.
  * symlinking each `<idx>.mp4` to `<prompt_en>-0.mp4` (the name VBench's
    standard loader expects). camera_motion's `get_type()` reads the motion
    phrase from this filename, so the metadata prompt (which carries
    "..., camera pans left") must survive into `prompt_en`.

Run in the `video_elbo` env AFTER `scripts/setup/setup_vbench_i2v.sh`.

Usage:
  python scripts/eval/score_vbench_i2v.py \
      --videos_dir   logs/<run>/evaluation_vbench_i2v/images \
      --metadata_jsonl dataset/vbench_i2v_a14b/ref_images_test/metadata.jsonl \
      --output_dir   logs/<run>/evaluation_vbench_i2v/vbench_i2v_eval
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.reward.vbench_aggregate import leaderboard_scores_i2v  # noqa: E402
from src.reward.vbench_scorer import _vbench_single_process  # noqa: E402

# All 10 VBench-I2V dims. Per-entry `vbench_dim` in the metadata decides which
# ones actually apply to each video; VBench intersects with this list.
I2V_DIMS = ["i2v_subject", "i2v_background", "camera_motion"]
QUALITY_DIMS = [
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "imaging_quality", "temporal_flickering", "motion_smoothness", "dynamic_degree",
]
ALL_I2V_DIMS = I2V_DIMS + QUALITY_DIMS


def _video_path(videos_dir: Path, index: int):
    """Locate the generated mp4 for a metadata index (tries common paddings)."""
    for cand in (f"{index:05d}.mp4", f"{index:06d}.mp4", f"{index}.mp4"):
        p = videos_dir / cand
        if p.exists():
            return p
    return None


def _safe(prompt: str) -> str:
    # Filenames can't contain os.sep; everything else (commas, motion phrase)
    # must survive so camera_motion's get_type() can read it.
    return prompt.replace("/", " ").strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos_dir", required=True)
    p.add_argument("--metadata_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ref_images_dir", default=None,
                   help="Dir of conditioning images (default: metadata's dir).")
    p.add_argument("--name", default="vbench_i2v")
    p.add_argument("--dimensions", default=None,
                   help="Comma list to restrict dims (default: all 10).")
    p.add_argument("--gpu_id", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    videos_dir = Path(args.videos_dir).resolve()
    meta_path = Path(args.metadata_jsonl).resolve()
    ref_dir = Path(args.ref_images_dir).resolve() if args.ref_images_dir else meta_path.parent
    score_dims = args.dimensions.split(",") if args.dimensions else list(ALL_I2V_DIMS)

    # --- build full-info entries + the prompt-named symlink dir ---
    entries, missing = [], 0
    with tempfile.TemporaryDirectory(prefix="vbench_i2v_links_") as link_dir, \
         tempfile.TemporaryDirectory(prefix="vbench_i2v_info_") as info_dir:
        link_dir, info_dir = Path(link_dir), Path(info_dir)
        seen_names = set()
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                idx = int(e["index"])
                vp = _video_path(videos_dir, idx)
                if vp is None:
                    missing += 1
                    continue
                # unique, filesystem-safe prompt_en that retains the motion phrase.
                # Prefer orig_prompt when present (prompt-augmented eval sets like
                # vbench_i2v_a14b_extended store the long rewrite in `prompt` and the
                # original short VBench prompt in `orig_prompt`): the original keeps
                # the camera clause intact for camera_motion get_type() and stays under
                # the 255-byte filename limit, and matches the baseline's naming.
                label = e.get("orig_prompt") or e["prompt"]
                prompt_en = f"{idx:05d}__{_safe(label)}"
                if prompt_en in seen_names:
                    prompt_en = f"{prompt_en}__{len(seen_names)}"
                seen_names.add(prompt_en)
                (link_dir / f"{prompt_en}-0.mp4").symlink_to(vp)
                entries.append({
                    "prompt_en": prompt_en,
                    "dimension": e.get("vbench_dim", e.get("dimension", [])),
                    # absolute path -> overrides VBench's relative image_root
                    "image_name": str((ref_dir / e["image_path"]).resolve()),
                })
        print(f"[i2v] matched {len(entries)} videos ({missing} metadata rows had no video)")
        if not entries:
            raise RuntimeError("no videos matched metadata indices; check --videos_dir")

        info_json = info_dir / "vbench_i2v_custom_full_info.json"
        with open(info_json, "w") as f:
            json.dump(entries, f)

        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        from vbench2_beta_i2v import VBenchI2V
        # torch.device (not str): vbench's GRiT path / quality dims do device.type
        runner = VBenchI2V(device, str(info_json), args.output_dir)

        # Bulk first; on failure isolate per-dim so a broken dim (most likely
        # camera_motion) doesn't sink the rest. Merge into one results JSON.
        def _eval(name, dims):
            with _vbench_single_process():
                runner.evaluate(videos_path=str(link_dir), name=name,
                                dimension_list=dims, local=True)

        merged, ok = {}, []
        try:
            _eval(args.name, score_dims)
            rj = Path(args.output_dir) / f"{args.name}_eval_results.json"
            with open(rj) as f:
                merged = json.load(f)
            ok = list(merged.keys())
        except Exception as bulk_err:  # noqa: BLE001
            print(f"[i2v] bulk evaluate failed ({bulk_err}); retrying per-dim.")
            for d in score_dims:
                try:
                    _eval(f"{args.name}_{d}", [d])
                    pj = Path(args.output_dir) / f"{args.name}_{d}_eval_results.json"
                    with open(pj) as f:
                        merged.update(json.load(f))
                    ok.append(d)
                except Exception as de:  # noqa: BLE001
                    print(f"[i2v]   dim={d} failed: {de}. Dropping.")
            with open(Path(args.output_dir) / f"{args.name}_eval_results.json", "w") as f:
                json.dump(merged, f, indent=2)
        print(f"[i2v] scored {len(ok)} dims: {sorted(ok)}")

    # --- aggregate ---
    per_dim_avg = {}
    for dim, payload in merged.items():
        if isinstance(payload, (list, tuple)) and payload:
            try:
                per_dim_avg[dim] = float(payload[0])
            except (TypeError, ValueError):
                pass

    lb = leaderboard_scores_i2v(per_dim_avg)
    scaled = {f"{d}_x100": v * 100.0 for d, v in per_dim_avg.items()}
    for k in ("I2V", "Quality", "Total"):
        if lb[k] is not None:
            scaled[f"{k}_x100"] = lb[k] * 100.0

    agg = {
        "raw_per_dim_avg": per_dim_avg,
        "normalized_per_dim": lb["normalized_per_dim"],
        "leaderboard_scale_x100": scaled,
        "scored_dims": lb["scored_dims"],
        "missing_dims": lb["missing_dims"],
        "num_videos": len(entries),
        "videos_dir": str(videos_dir),
    }
    agg_path = os.path.join(args.output_dir, "average_scores_vbench_i2v.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print("\n--- VBench-I2V (×100, leaderboard form) ---")
    for k in sorted(scaled):
        print(f"  {k:<40}: {scaled[k]:.2f}")
    print(f"[done] {agg_path}")


if __name__ == "__main__":
    main()
