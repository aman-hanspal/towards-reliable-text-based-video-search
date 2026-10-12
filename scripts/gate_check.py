"""
gate_check.py  --  Gate tests 2 and 3, before the full runs.

TEST 2 (accuracy): does FP16 (new runner) give the same answers as FP32 (old runner) in separate mode?
  For each model, compares scores_gate (FP16) with scores_mini (FP32), at thresholds 0.10 and 0.25:
    present: mean frame hit rate              PASS if the difference is <= 3.0 points
    hard absent: per-prompt "fired" yes/no    PASS if at most 2 of the 40 prompts flip
    easy absent: per-prompt "fired" yes/no    PASS if at most 1 of the 40 prompts flips
  (rules fixed before running; 1 prompt = 2.5 points, so "2 flips" ~ 5 points)

TEST 3 (time and disk): from the gate runs' real timing and file sizes, estimates the full runs
  (all videos in the query set) for the 6 planned settings.
  480 settings are scaled from 640 using the mini-run joint FP16 speed ratios.

Usage:
  python gate_check.py --old scores_mini/prompt_scores.csv --new scores_gate/prompt_scores.csv --gate-runs gate_runs
"""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

THRESHOLDS = ("0.1", "0.25")
MAX_PRESENT_DIFF, MAX_HARD_FLIPS, MAX_EASY_FLIPS = 3.0, 2, 1
# mini-run joint FP16 ms/frame: 480 vs 640 (from mini_run_report.md)
RATIO_480 = {"yolov8s-worldv2": 26.0 / 27.5, "yolov8m-worldv2": 24.7 / 29.9, "yolov8l-worldv2": 34.5 / 44.5}
PLAN = [("yolov8m-worldv2", 640), ("yolov8l-worldv2", 640), ("yolov8s-worldv2", 640),
        ("yolov8m-worldv2", 480), ("yolov8l-worldv2", 480), ("yolov8s-worldv2", 480)]


def load_scores(path):
    runs = defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        runs[r["run"]].append(r)
    return runs


def f(x):
    return None if x in ("", "None") else float(x)


def summary(rows, t):
    pres = [f(r["frame_hit_rate"]) for r in rows if r["kind"] == "present" and r["threshold"] == t
            and f(r["gt_boxes"])]
    pres = [x for x in pres if x is not None]
    fired = {k: {(r["video_id"], r["prompt_idx"]): int(r["fired"]) for r in rows
                 if r["kind"] == k and r["threshold"] == t} for k in ("hard_absent", "easy_absent")}
    return 100 * sum(pres) / len(pres), fired


def test2(old, new):
    print("## TEST 2: FP16 (new runner) vs FP32 (old runner), separate mode, mini20\n")
    print("| model | thr | present frame hit FP32 -> FP16 | diff | hard fired FP32 -> FP16 | hard flips | easy flips | result |")
    print("|---|---:|---|---:|---|---:|---:|---|")
    all_ok, seen = True, 0
    for run in sorted(new):
        if "_fp16_separate" not in run:
            continue
        ref = run.replace("_fp16_", "_fp32_")
        if ref not in old:
            print(f"| {run} | - | no FP32 reference ({ref}) in --old | | | | | SKIP |")
            continue
        seen += 1
        for t in THRESHOLDS:
            p_old, f_old = summary(old[ref], t)
            p_new, f_new = summary(new[run], t)
            flips = {}
            for k in f_old:
                keys = set(f_old[k]) & set(f_new[k])
                flips[k] = sum(f_old[k][x] != f_new[k][x] for x in keys)
            h_old = 100 * sum(f_old["hard_absent"].values()) / max(1, len(f_old["hard_absent"]))
            h_new = 100 * sum(f_new["hard_absent"].values()) / max(1, len(f_new["hard_absent"]))
            ok = (abs(p_new - p_old) <= MAX_PRESENT_DIFF and flips["hard_absent"] <= MAX_HARD_FLIPS
                  and flips["easy_absent"] <= MAX_EASY_FLIPS)
            all_ok &= ok
            print(f"| {run.split('_')[0]} | {t} | {p_old:.1f}% -> {p_new:.1f}% | {p_new - p_old:+.1f} | "
                  f"{h_old:.1f}% -> {h_new:.1f}% | {flips['hard_absent']} | {flips['easy_absent']} | "
                  f"{'pass' if ok else 'FAIL'} |")
    if not seen:
        print("\nNo FP16 separate runs found in --new.")
        return False
    print(f"\nGATE TEST 2: {'PASS' if all_ok else 'FAIL'}  (rules: present diff <= {MAX_PRESENT_DIFF} pts, "
          f"hard flips <= {MAX_HARD_FLIPS}, easy flips <= {MAX_EASY_FLIPS})\n")
    return all_ok


def test3(gate_runs, query_set, root):
    print("## TEST 3: estimated full runs (FP16, separate, decode-once)\n")
    qs = json.loads(Path(query_set).read_text(encoding="utf-8"))["videos"]
    total_fp = 0
    for v in qs:
        prompts = [q["prompt"] for k in ("present", "easy_absent", "hard_absent") for q in v[k]]
        total_fp += v["frame_count"] * len(set(prompts))
    print(f"Full set: {len(qs)} videos, {sum(v['frame_count'] for v in qs):,} frames, "
          f"{total_fp:,} frame-passes (frames x prompts)\n")
    rates = {}
    for run in sorted(Path(gate_runs).iterdir()):
        tp = run / "timing.csv"
        if not tp.exists() or "_fp16_separate" not in run.name:
            continue
        rows = list(csv.DictReader(open(tp)))
        fp = sum(int(r["frames"]) * int(r["passes"]) for r in rows)
        wall = sum(float(r["wall_s"]) for r in rows if r.get("wall_s"))
        disk = sum(p.stat().st_size for p in (run / "videos").glob("*.json.gz"))
        dec = sum(float(r["decode_s"]) for r in rows)
        rates[run.name.split("_")[0]] = (wall / fp, disk / fp, dec / max(wall, 1e-9))
    if not rates:
        print("No gate runs with timing found.")
        return
    print("| order | setting | ms per frame-pass (wall) | est. hours | est. disk |")
    print("|---:|---|---:|---:|---:|")
    tot_h = tot_gb = 0.0
    for i, (m, s) in enumerate(PLAN, 1):
        if m not in rates:
            print(f"| {i} | {m} {s} | (no gate run) | - | - |")
            continue
        sec, byt, _ = rates[m]
        sec = sec * (RATIO_480[m] if s == 480 else 1.0)
        h, gb = sec * total_fp / 3600, byt * total_fp / 1e9
        tot_h += h
        tot_gb += gb
        print(f"| {i} | {m} {s} FP16 | {1000 * sec:.1f} | {h:.1f} | {gb:.1f} GB |")
    free = shutil.disk_usage(root).free / 1e9
    print(f"\nTotal: ~{tot_h:.0f} h (~{tot_h / 24:.1f} days nonstop), ~{tot_gb:.0f} GB.  Free disk here: {free:.0f} GB "
          f"-> {'OK' if free > 1.5 * tot_gb else 'NOT ENOUGH, tell Claude'}")
    for m, (_, _, d) in rates.items():
        print(f"  {m}: decoding is now {100 * d:.1f}% of wall time")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--old", default="scores_mini/prompt_scores.csv")
    ap.add_argument("--new", default="scores_gate/prompt_scores.csv")
    ap.add_argument("--gate-runs", default="gate_runs")
    ap.add_argument("--query-set", default="query_set_v3/query_set.json")
    args = ap.parse_args()
    root = Path(args.root)
    test2(load_scores(root / args.old), load_scores(root / args.new))
    test3(root / args.gate_runs, root / args.query_set, root)


if __name__ == "__main__":
    main()
