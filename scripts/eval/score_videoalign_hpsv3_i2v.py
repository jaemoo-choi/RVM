"""Score pre-generated SkyReels/Wan2.2 I2V mp4s with VideoAlign (TA/VQ/MQ/Overall),
HPSv3, and HPSv3-percentile — no regeneration.

Unlike the T2V scorer (scripts/eval/score_videoalign_hpsv3.py, which maps VBench
short-prompt filenames to self_forcing_extended), the I2V videos are named by
index (images/00000.mp4) and their captions live in the eval results jsonl
written by evaluation.py. We read (video_path, prompt) pairs straight from that
jsonl, so pairing is exactly what the eval used to compute videoalign_mq.

Usage:
  python scripts/eval/score_videoalign_hpsv3_i2v.py --run_dir logs/skyreels_default/<run>
  # -> <run_dir>/evaluation_vbench_i2v/videoalign_hpsv3_scores.json
"""
import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="training run dir (contains evaluation_vbench_i2v/)")
    p.add_argument("--eval_subdir", default="evaluation_vbench_i2v")
    p.add_argument("--output_json", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--videoalign_fps", type=int, default=8)
    p.add_argument("--hpsv3_frames", type=int, default=4)
    p.add_argument("--hpsv3_top_k_pct", type=float, default=0.3)
    # Sharding: split the (video,prompt) list into num_shards interleaved slices
    # and score only shard_id's slice. Each shard writes its own partial json
    # (means + n_scored); merge with --merge afterward. Lets the (CPU-bound)
    # scorer run as N parallel single-GPU jobs instead of one ~6h serial pass.
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    return p.parse_args()


def find_eval_jsonl(eval_dir):
    # Prefer the a14b (1118-prompt) results, else the plain vbench_i2v results.
    for name in ("evaluation_results_vbench_i2v_a14b.jsonl",
                 "evaluation_results_vbench_i2v.jsonl"):
        p = os.path.join(eval_dir, name)
        if os.path.exists(p):
            return p
    # Base-eval dirs use suffixed variants (…_a14b_extended.jsonl); take the longest
    # prompt set so the baseline covers the same prompts as the trained runs.
    cands = sorted(glob.glob(os.path.join(eval_dir, "evaluation_results_vbench_i2v*.jsonl")))
    if cands:
        return max(cands, key=lambda p: sum(1 for _ in open(p)))
    raise FileNotFoundError(f"no evaluation_results_vbench_i2v*.jsonl in {eval_dir}")


def load_pairs(jsonl_path, repo_root):
    """-> list of (abs_video_path, prompt), de-duplicated by video_path."""
    pairs, seen = [], set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            vp = d.get("video_path")
            pr = d.get("prompt")
            if not vp or pr is None:
                continue
            if not os.path.isabs(vp):
                vp = os.path.join(repo_root, vp)
            if vp in seen or not os.path.exists(vp):
                continue
            seen.add(vp)
            pairs.append((vp, pr))
    return pairs


def decode_mp4(path):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    return vr.get_batch(range(len(vr))).asnumpy()  # [T,H,W,C] uint8


def main():
    args = parse_args()
    repo_root = REPO_ROOT
    eval_dir = os.path.join(args.run_dir, args.eval_subdir)
    # Base-eval dirs (logs/*_base_eval/<variant>/) hold the jsonl directly, with no
    # evaluation_vbench_i2v/ subdir — fall back to the given dir itself.
    if not os.path.isdir(eval_dir):
        eval_dir = args.run_dir
    jsonl = find_eval_jsonl(eval_dir)
    pairs = load_pairs(jsonl, repo_root)
    print(f"[score-i2v] {len(pairs)} (video,prompt) pairs from {jsonl}", flush=True)
    if not pairs:
        raise SystemExit("no usable (video,prompt) pairs found")

    if args.num_shards > 1:
        pairs = pairs[args.shard_id::args.num_shards]
        default_out = os.path.join(
            eval_dir, f"videoalign_hpsv3_scores.shard{args.shard_id}of{args.num_shards}.json")
        print(f"[score-i2v] shard {args.shard_id}/{args.num_shards}: {len(pairs)} pairs", flush=True)
    else:
        default_out = os.path.join(eval_dir, "videoalign_hpsv3_scores.json")
    out_json = args.output_json or default_out

    device = "cuda"
    from src.reward.videoalign_scorer import get_shared_videoalign_scorer
    from src.reward.hpsv3_scorer import get_shared_hpsv3_scorer
    va = get_shared_videoalign_scorer(device, video_fps=args.videoalign_fps)
    hps = get_shared_hpsv3_scorer(dtype=torch.float32, device=device)

    acc = {k: [] for k in ("TA", "VQ", "MQ", "Overall",
                            "TA_raw", "VQ_raw", "MQ_raw", "Overall_raw",
                            "HPSv3", "HPSv3_pct")}

    # VideoAlign dim scores are z-normalized (norm = (raw - mean) / std) using
    # the checkpoint's inference_config. We also report the un-normalized raw
    # head outputs: raw = norm * std + mean.  (Overall_raw = VQ_raw+MQ_raw+TA_raw)
    norm_cfg = va.inferencer.inference_config or {}

    def _to_raw(dim_key, norm_vals):
        m = norm_cfg.get(f"{dim_key}_mean")
        s = norm_cfg.get(f"{dim_key}_std")
        if m is None or s is None:  # no inference_config -> already raw
            return [float(x) for x in norm_vals]
        return [float(x) * float(s) + float(m) for x in norm_vals]

    def flush(frames_list, prompts):
        vids = np.stack(frames_list, axis=0)  # [B,T,H,W,C]
        _, dim = va(vids, prompts, None)
        for k in ("TA", "VQ", "MQ", "Overall"):
            acc[k].extend([float(x) for x in dim[k]])
        ta_r = _to_raw("TA", dim["TA"])
        vq_r = _to_raw("VQ", dim["VQ"])
        mq_r = _to_raw("MQ", dim["MQ"])
        acc["TA_raw"].extend(ta_r)
        acc["VQ_raw"].extend(vq_r)
        acc["MQ_raw"].extend(mq_r)
        acc["Overall_raw"].extend([a + b + c for a, b, c in zip(ta_r, vq_r, mq_r)])
        t = torch.from_numpy(vids).float().div_(255.0).permute(0, 1, 4, 2, 3).contiguous().to(device)
        with torch.no_grad():
            hs = hps.score_video(t, prompts, num_sample_frames=args.hpsv3_frames)
            hp = hps.score_video_percentile(t, prompts, num_sample_frames=args.hpsv3_frames,
                                            top_k_pct=args.hpsv3_top_k_pct)
        acc["HPSv3"].extend([float(x) for x in hs])
        acc["HPSv3_pct"].extend([float(x) for x in hp])

    buf_f, buf_p, buf_T = [], [], None
    n_ok = n_bad = 0
    for i, (vp, pr) in enumerate(pairs):
        try:
            fr = decode_mp4(vp)
        except Exception as e:
            print(f"[warn] decode failed {os.path.basename(vp)}: {e}", flush=True)
            n_bad += 1
            continue
        if buf_f and (len(buf_f) >= args.batch_size or fr.shape[0] != buf_T):
            flush(buf_f, buf_p)
            buf_f, buf_p, buf_T = [], [], None
        buf_f.append(fr); buf_p.append(pr); buf_T = fr.shape[0]; n_ok += 1
        if (i + 1) % 100 == 0:
            print(f"[score-i2v] {i+1}/{len(pairs)} (ok {n_ok}, bad {n_bad})", flush=True)
    if buf_f:
        flush(buf_f, buf_p)

    mean = lambda k: (float(np.mean(acc[k])) if acc[k] else None)
    out = {
        "run_dir": args.run_dir,
        "eval_jsonl": jsonl,
        "n_scored": n_ok,
        "n_failed": n_bad,
        "videoalign_fps": args.videoalign_fps,
        "hpsv3_frames": args.hpsv3_frames,
        "hpsv3_top_k_pct": args.hpsv3_top_k_pct,
        "TA": mean("TA"), "VQ": mean("VQ"), "MQ": mean("MQ"), "Overall": mean("Overall"),
        "TA_raw": mean("TA_raw"), "VQ_raw": mean("VQ_raw"), "MQ_raw": mean("MQ_raw"),
        "Overall_raw": mean("Overall_raw"),
        "HPSv3": mean("HPSv3"), "HPSv3_percentile": mean("HPSv3_pct"),
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", out_json, flush=True)
    print(json.dumps({k: out[k] for k in (
        "TA", "VQ", "MQ", "Overall",
        "TA_raw", "VQ_raw", "MQ_raw", "Overall_raw",
        "HPSv3", "HPSv3_percentile")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
