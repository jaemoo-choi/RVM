"""Score a directory of pre-generated VBench mp4s with VideoAlign (TA/VQ/MQ/Overall)
and HPSv3 — no regeneration. Pairs each mp4 (named by the short VBench prompt) with
the EXTENDED generation prompt (self_forcing_extended), matching our eval protocol.

Usage:
  python scripts/eval/score_videoalign_hpsv3.py \
      --videos_dir logs/.../evaluation_vbench_official/videos \
      --output_json logs/.../evaluation_vbench_official/videoalign_hpsv3_scores.json

VideoAlign uses video_fps=8 (matches the videoalign_ta/mq closures in
src/reward/rewards.py). HPSv3 = caption-based video score (video_hpsv3_score):
per-frame HPSv3 vs the extended caption, averaged over 4 uniformly-sampled frames.
"""
import argparse
import glob
import json
import os
import re
import sys

# Ensure repo root is importable (so `import src.reward...` works when run as a script).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--prompt_aug_file", default="dataset/vbench/self_forcing_extended.txt")
    p.add_argument("--full_info_json", default="",
                   help="trimmed full_info json paired 1:1 with --prompt_aug_file")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--videoalign_fps", type=int, default=8)
    p.add_argument("--hpsv3_frames", type=int, default=4)
    p.add_argument("--hpsv3_top_k_pct", type=float, default=0.3)
    p.add_argument("--per_video_csv", default="",
                   help="Also write one row per mp4 (file + all scores). Default: "
                        "<output_json dir>/videoalign_hpsv3_per_video.csv")
    p.add_argument("--dynamic_degree", type=int, default=0,
                   help="1 = also score RAFT dynamic degree per video (binary "
                        "VBench rule num_pairs=8/pass_rate=0.25 + continuous flow)")
    return p.parse_args()


def build_prompt_map(aug_file, full_info_json=""):
    """short VBench prompt (mp4 filename stem) -> extended generation prompt."""
    import vbench

    full_info_path = full_info_json or os.path.join(
        os.path.dirname(vbench.__file__), "VBench_full_info.json")
    with open(full_info_path) as f:
        full = json.load(f)
    short = [e["prompt_en"] for e in full]
    with open(aug_file) as f:
        ext = [ln.rstrip("\n") for ln in f if ln.strip()]
    if len(ext) != len(short):
        raise ValueError(f"extended {len(ext)} != full_info {len(short)} — misaligned")
    return dict(zip(short, ext))


def decode_mp4(path):
    """-> np.uint8 [T, H, W, C]."""
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    return vr.get_batch(range(len(vr))).asnumpy()


def main():
    args = parse_args()
    device = "cuda"
    pmap = build_prompt_map(args.prompt_aug_file, args.full_info_json)

    from src.reward.videoalign_scorer import get_shared_videoalign_scorer
    from src.reward.hpsv3_scorer import get_shared_hpsv3_scorer

    va = get_shared_videoalign_scorer(device, video_fps=args.videoalign_fps)
    hps = get_shared_hpsv3_scorer(dtype=torch.float32, device=device)
    dd = None
    if args.dynamic_degree:
        from src.reward.dynamic_degree_scorer import get_shared_dynamic_degree_scorer
        dd = get_shared_dynamic_degree_scorer(dtype=torch.float32, device=device)

    files = sorted(glob.glob(os.path.join(args.videos_dir, "*.mp4")))
    print(f"[score] {len(files)} mp4s in {args.videos_dir}", flush=True)

    score_keys = ["TA", "VQ", "MQ", "Overall", "HPSv3", "HPSv3_pct"]
    if dd is not None:
        score_keys += ["DD_binary", "DD_flow"]
    acc = {k: [] for k in score_keys}
    acc_files = []
    n_match = n_miss = 0

    def flush(frames_list, prompts, fnames):
        # frames_list: list of np [T,H,W,C] uint8 with identical T within the batch
        vids = np.stack(frames_list, axis=0)  # [B,T,H,W,C]
        _, dim = va(vids, prompts, None)
        for k in ("TA", "VQ", "MQ", "Overall"):
            acc[k].extend([float(x) for x in dim[k]])
        # HPSv3: [B,T,C,H,W] float[0,1]
        t = torch.from_numpy(vids).float().div_(255.0).permute(0, 1, 4, 2, 3).contiguous().to(device)
        with torch.no_grad():
            hs = hps.score_video(t, prompts, num_sample_frames=args.hpsv3_frames)
            hp = hps.score_video_percentile(t, prompts, num_sample_frames=args.hpsv3_frames,
                                            top_k_pct=args.hpsv3_top_k_pct)
        acc["HPSv3"].extend([float(x) for x in hs])
        acc["HPSv3_pct"].extend([float(x) for x in hp])
        if dd is not None:
            # Same rule as the dynamic_degree training reward closure.
            binary = dd.score_video_vbench_binary(t, num_pairs=8, pass_rate=0.25)
            with torch.no_grad():
                flow = dd.score_video_no_grad(t, num_pairs=8)
            acc["DD_binary"].extend([float(x) for x in binary])
            acc["DD_flow"].extend([float(x) for x in flow])
        acc_files.extend(fnames)

    buf_frames, buf_prompts, buf_files, buf_T = [], [], [], None
    for i, f in enumerate(files):
        stem = re.sub(r"-\d+\.mp4$", "", os.path.basename(f))
        if stem not in pmap:
            n_miss += 1
            continue
        try:
            fr = decode_mp4(f)
        except Exception as e:
            print(f"[warn] decode failed {os.path.basename(f)}: {e}", flush=True)
            n_miss += 1
            continue
        # Flush on batch-full or frame-count change (videoalign needs uniform T).
        if buf_frames and (len(buf_frames) >= args.batch_size or fr.shape[0] != buf_T):
            flush(buf_frames, buf_prompts, buf_files)
            buf_frames, buf_prompts, buf_files, buf_T = [], [], [], None
        buf_frames.append(fr)
        buf_prompts.append(pmap[stem])
        buf_files.append(os.path.basename(f))
        buf_T = fr.shape[0]
        n_match += 1
        if (i + 1) % 100 == 0:
            print(f"[score] {i+1}/{len(files)} (matched {n_match}, missed {n_miss})", flush=True)
    if buf_frames:
        flush(buf_frames, buf_prompts, buf_files)

    per_video_csv = args.per_video_csv or os.path.join(
        os.path.dirname(args.output_json), "videoalign_hpsv3_per_video.csv")
    import csv as _csv
    with open(per_video_csv, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["file"] + score_keys)
        for j, fn in enumerate(acc_files):
            w.writerow([fn] + [f"{acc[k][j]:.6f}" for k in score_keys])
    print(f"[score] per-video scores -> {per_video_csv}", flush=True)

    out = {
        "videos_dir": args.videos_dir,
        "prompt_aug_file": args.prompt_aug_file,
        "n_matched": n_match,
        "n_missed": n_miss,
        "videoalign_fps": args.videoalign_fps,
        "hpsv3_frames": args.hpsv3_frames,
        "TA": float(np.mean(acc["TA"])) if acc["TA"] else None,
        "VQ": float(np.mean(acc["VQ"])) if acc["VQ"] else None,
        "MQ": float(np.mean(acc["MQ"])) if acc["MQ"] else None,
        "Overall": float(np.mean(acc["Overall"])) if acc["Overall"] else None,
        "HPSv3": float(np.mean(acc["HPSv3"])) if acc["HPSv3"] else None,
        "HPSv3_percentile": float(np.mean(acc["HPSv3_pct"])) if acc["HPSv3_pct"] else None,
        "per_video_csv": per_video_csv,
    }
    if dd is not None:
        out["DD_binary"] = float(np.mean(acc["DD_binary"])) if acc["DD_binary"] else None
        out["DD_flow"] = float(np.mean(acc["DD_flow"])) if acc["DD_flow"] else None
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] matched={n_match} missed={n_miss}", flush=True)
    print(json.dumps({k: out[k] for k in ("TA", "VQ", "MQ", "Overall", "HPSv3", "HPSv3_percentile")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
