"""
compare_runs.py  --  Gate test 1: does the new runner give the same boxes as the old one?

Compares two run folders box by box, for the videos both contain.
A box (score >= 0.05) in run A counts as "matched" if run B has a box for the same video,
same frame, same prompt, with IoU >= 0.9 and a score within 0.02.
Checked in both directions (A -> B and B -> A).

PASS rule (fixed before running): >= 99% of boxes matched in BOTH directions.

Usage:
  python compare_runs.py gate_check1/yolov8m-worldv2_img640_fp32_separate mini_runs/yolov8m-worldv2_img640_fp32_separate
"""
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

MIN_SCORE, IOU_MIN, SCORE_TOL, PASS = 0.05, 0.9, 0.02, 0.99


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def load(run):
    return {int(f.name.split(".")[0]): json.loads(gzip.decompress(f.read_bytes()))
            for f in (Path(run) / "videos").glob("*.json.gz")}


def grouped(rec):
    g = defaultdict(list)
    for fi, pi, sc, *b in rec["detections"]:
        if sc >= MIN_SCORE:
            g[(fi, pi)].append((sc, b))
    return g


def matched_share(ga, gb):
    hit = tot = 0
    for key, boxes in ga.items():
        other = gb.get(key, [])
        for sc, b in boxes:
            tot += 1
            hit += any(iou(b, b2) >= IOU_MIN and abs(sc - s2) <= SCORE_TOL for s2, b2 in other)
    return hit, tot


def decode_total(run, ids):
    p = Path(run) / "timing.csv"
    if not p.exists():
        return None
    rows = [r for r in csv.DictReader(open(p)) if int(r["video_id"]) in ids]
    return sum(float(r["decode_s"]) for r in rows), sum(float(r["infer_s"]) for r in rows), \
        sum(float(r["wall_s"]) for r in rows) if rows and "wall_s" in rows[0] else None


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a_run, b_run = sys.argv[1], sys.argv[2]
    A, B = load(a_run), load(b_run)
    common = sorted(set(A) & set(B))
    if not common:
        sys.exit("No videos in common.")
    ab = [0, 0]
    ba = [0, 0]
    print(f"A = {a_run}\nB = {b_run}\n")
    print("| video | frames A/B | prompts same? | boxes A | boxes B | A found in B | B found in A |")
    print("|---:|---|---|---:|---:|---:|---:|")
    for vid in common:
        a, b = A[vid], B[vid]
        same_prompts = a["prompts"] == b["prompts"]
        ga, gb = grouped(a), grouped(b)
        h1, t1 = matched_share(ga, gb)
        h2, t2 = matched_share(gb, ga)
        ab[0] += h1; ab[1] += t1; ba[0] += h2; ba[1] += t2
        print(f"| {vid} | {a['frames_decoded']}/{b['frames_decoded']} | {same_prompts} | {t1} | {t2} | "
              f"{h1 / t1:.2%} | {h2 / t2:.2%} |" if t1 and t2 else f"| {vid} | - | {same_prompts} | {t1} | {t2} | - | - |")
    r1 = ab[0] / ab[1] if ab[1] else 1.0
    r2 = ba[0] / ba[1] if ba[1] else 1.0
    print(f"\nAll videos: A found in B {r1:.2%} ({ab[0]}/{ab[1]}),  B found in A {r2:.2%} ({ba[0]}/{ba[1]})")
    ta, tb = decode_total(a_run, set(common)), decode_total(b_run, set(common))
    if ta and tb:
        print(f"Decode time: A {ta[0]:.1f}s vs B {tb[0]:.1f}s  |  inference time: A {ta[1]:.1f}s vs B {tb[1]:.1f}s")
    ok = r1 >= PASS and r2 >= PASS
    print(f"\nGATE TEST 1: {'PASS' if ok else 'FAIL'}  (rule: both directions >= {PASS:.0%})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
