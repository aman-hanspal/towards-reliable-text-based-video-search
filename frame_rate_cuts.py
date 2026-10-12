"""
frame_rate_cuts.py  --  what if the detector only looked at every k-th frame?   (CPU only, no GPU, no re-run)

Experiment plan item 5 ("Fewer frames: keep every 1st/2nd/5th/10th frame, computed from saved boxes").

Idea
  The full runs ran YOLO on EVERY frame and saved every box. Looking at only every k-th frame is the same as
  throwing away the boxes on the other frames. So for k = 1, 2, 5, 10 this script keeps only frames
  0, k, 2k, ... (both the detector boxes AND the dataset labels on those frames) and scores them with the
  SAME scoring code as before (score_video from score_runs.py). Nothing on disk is changed.

What it reports per run, per k (threshold 0.25 unless --thr):
  found in video     share of present prompts found at least once in the frames looked at.
                     Denominator = the same prompts as k=1 (if every labelled frame of an object is skipped,
                     it counts as NOT found -- a user would miss it too).
  all labelled frames skipped   how often that happened (explains part of any drop)
  frame hit rate     share of the LOOKED-AT labelled frames where the object was found (should barely move;
                     it is a check, not a result)
  easy / hard / strict-hard false alarms   share of absent prompts that fired anywhere in the looked-at frames
  hard misnaming     share of hard prompts that fired ON a real object of another kind
  detector GPU time  inference time from timing.csv divided by k (every frame costs about the same).
                     Decode time is listed separately: a video file is still read start to end.
  Each change vs k=1 gets a 95% paired bootstrap interval (resampling videos, 2000 times, fixed seed).

CHECK: the k=1 row must equal the earlier scores (medium 640 separate at 0.25: found 50.7%, frame hit 38.5%,
easy 1.1%, hard 23.9%). If it does not, stop -- something is wrong.

Outputs in --out (default frame_rate_cuts/):
  <run>_rows.csv     one row per (k, video, prompt, threshold) -- the raw table
  <run>_summary.md   the readable table
  gt_cache.json.gz   copy of scores_full/gt_cache.json.gz (the original is never written to)

Usage (from the OVT-B-Dataset folder, needs score_runs.py next to it):
  python frame_rate_cuts.py                                   # medium 640 separate only (main)
  python frame_rate_cuts.py --only yolov8s-worldv2_img640_fp16_separate,yolov8l-worldv2_img640_fp16_separate
"""
import argparse
import csv
import gzip
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from score_runs import load_gt, score_video

MAIN_RUN = "yolov8m-worldv2_img640_fp16_separate"
METRICS = (  # (key, label, kind of prompt it is measured on)
    ("found", "found in video", "present"),
    ("skipped", "all labelled frames skipped", "present"),
    ("frame_hit", "frame hit rate (looked-at frames)", "present"),
    ("easy", "easy false alarms", "absent"),
    ("hard", "hard false alarms", "absent"),
    ("strict", "strict-hard false alarms", "absent"),
    ("hard_mis", "hard misnaming (on a real object)", "absent"),
)
ROW_FIELDS = ["k", "video_id", "prompt_idx", "prompt", "kind", "strict_hard", "threshold", "n_boxes", "gt_frames",
              "gt_boxes", "frame_hit_rate", "video_found", "fired", "fired_misnaming"]


def cut(rec, gt_v, k):
    """Keep only frames 0, k, 2k, ... in both the detections and the labels."""
    if k == 1:
        return rec, gt_v
    r = dict(rec)
    r["detections"] = [d for d in rec["detections"] if d[0] % k == 0]
    return r, {fi: b for fi, b in gt_v.items() if fi % k == 0}


def score_run(files, qs, gt, steps, thr, rows_path):
    """Returns vals[k][metric] = {video_id: [sum, count]}."""
    vals = {k: {m[0]: {} for m in METRICS} for k in steps}

    def add(k, m, vid, x):
        s = vals[k][m].setdefault(vid, [0.0, 0])
        s[0] += x
        s[1] += 1

    with open(rows_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        t0 = time.time()
        for n, vf in enumerate(files, 1):
            rec = json.loads(gzip.decompress(vf.read_bytes()))
            vid = rec["video_id"]
            qv, gt_v = qs[vid], gt.get(vid, {})
            full_has = {}  # present prompt_idx -> had labelled boxes in the full video (k=1)
            for k in steps:
                r, g = cut(rec, gt_v, k)
                for row in score_video(r, qv, g):
                    row["k"] = k
                    w.writerow(row)
                    if row["threshold"] != thr:
                        continue
                    pi, kind = row["prompt_idx"], row["kind"]
                    if kind == "present":
                        if k == 1:
                            full_has[pi] = bool(row["gt_boxes"])
                        if full_has.get(pi):
                            looked = bool(row["gt_boxes"])
                            add(k, "found", vid, row["video_found"] if looked else 0)
                            add(k, "skipped", vid, 0 if looked else 1)
                            if looked:
                                add(k, "frame_hit", vid, row["frame_hit_rate"])
                    elif kind == "easy_absent":
                        add(k, "easy", vid, row["fired"])
                    elif kind == "hard_absent":
                        add(k, "hard", vid, row["fired"])
                        add(k, "hard_mis", vid, row["fired_misnaming"])
                        if row["strict_hard"]:
                            add(k, "strict", vid, row["fired"])
            del rec
            if n % 200 == 0:
                print(f"  {n}/{len(files)} videos, {(time.time() - t0) / 60:.1f} min so far", flush=True)
    return vals


def summarise(vals, video_ids, steps, boot=2000, seed=20261011):
    """Mean per k and metric, plus paired bootstrap interval of (k - k=1) over videos."""
    idx = {v: i for i, v in enumerate(video_ids)}
    n = len(video_ids)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(boot, n))
    out = {}
    for m, _, _ in METRICS:
        arrs = {}
        for k in steps:
            s, c = np.zeros(n), np.zeros(n)
            for vid, (a, b) in vals[k][m].items():
                s[idx[vid]], c[idx[vid]] = a, b
            arrs[k] = (s, c)
        s1, c1 = arrs[steps[0]]
        base_b = s1[draw].sum(1) / np.maximum(c1[draw].sum(1), 1)
        for k in steps:
            s, c = arrs[k]
            mean = s.sum() / c.sum() if c.sum() else None
            if k == steps[0] or mean is None:
                out[(k, m)] = (mean, None, int(c.sum()))
            else:
                d = s[draw].sum(1) / np.maximum(c[draw].sum(1), 1) - base_b
                out[(k, m)] = (mean, (np.percentile(d, 2.5), np.percentile(d, 97.5)), int(c.sum()))
    return out


def timing(run_dir):
    inf = dec = 0.0
    p = run_dir / "timing.csv"
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            inf += float(r["infer_s"])
            dec += float(r["decode_s"])
    return inf, dec


def write_summary(run, res, steps, thr, inf, dec, n_videos, path):
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    L = [f"# Frame-rate cuts: {run}\n",
         f"Threshold {thr}. {n_videos} videos. Every k-th frame kept (frames 0, k, 2k, ...). "
         f"Change vs k=1 with 95% paired bootstrap interval (videos resampled). "
         f"k=1 must equal the earlier scores.\n",
         "| measure | " + " | ".join(f"k={k}" for k in steps) + " |",
         "|---|" + "---:|" * len(steps)]
    for m, label, _ in METRICS:
        cells = []
        for k in steps:
            mean, ci, cnt = res[(k, m)]
            if ci is None:
                cells.append(f"{pct(mean)} (n={cnt})" if k == steps[0] else pct(mean))
            else:
                b = res[(steps[0], m)][0]
                cells.append(f"{pct(mean)} ({100 * (mean - b):+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}])")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("| detector GPU time (h) | " + " | ".join(f"{inf / k / 3600:.2f}" for k in steps) + " |")
    L.append("| frames looked at | " + " | ".join(f"{100 / k:.0f}%" for k in steps) + " |")
    L += ["", f"Decode time (whole run, reading every frame of the file): {dec / 3600:.2f} h. "
              f"It does not shrink with k unless frames are skipped without decoding (not measured here)."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--runs", default="full_runs")
    ap.add_argument("--only", default=MAIN_RUN, help="comma-separated run folder names")
    ap.add_argument("--steps", default="1,2,5,10")
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--query-set", default="query_set_v3/query_set.json")
    ap.add_argument("--ann", default="ovtb_ann.json")
    ap.add_argument("--gt-cache", default="scores_full/gt_cache.json.gz", help="copied, never written to")
    ap.add_argument("--out", default="frame_rate_cuts")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.root) / a.out
    out.mkdir(parents=True, exist_ok=True)
    steps = sorted({int(x) for x in a.steps.split(",")})
    if steps[0] != 1:
        raise SystemExit("--steps must include 1 (the full-frame reference).")
    qs = {v["video_id"]: v for v in json.loads((root / a.query_set).read_text(encoding="utf-8"))["videos"]}

    run_dirs = [root / a.runs / r.strip() for r in a.only.split(",") if r.strip()]
    for rd in run_dirs:
        if not (rd / "videos").is_dir():
            raise SystemExit(f"Run folder not found: {rd}")
    files = {rd: sorted((rd / "videos").glob("*.json.gz"), key=lambda f: int(f.name.split(".")[0]))
             for rd in run_dirs}
    video_ids = sorted({int(f.name.split(".")[0]) for fs in files.values() for f in fs})

    cache = out / "gt_cache.json.gz"
    src = root / a.gt_cache
    if not cache.exists() and src.exists():
        shutil.copy(src, cache)
    gt = load_gt(root, root / a.ann, set(video_ids), cache)

    for rd, fs in files.items():
        print(f"\nScoring {rd.name}: {len(fs)} videos x k={steps} ...", flush=True)
        vals = score_run(fs, qs, gt, steps, a.thr, out / f"{rd.name}_rows.csv")
        vids = sorted({int(f.name.split('.')[0]) for f in fs})
        res = summarise(vals, vids, steps)
        inf, dec = timing(rd)
        write_summary(rd.name, res, steps, a.thr, inf, dec, len(fs), out / f"{rd.name}_summary.md")
    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
