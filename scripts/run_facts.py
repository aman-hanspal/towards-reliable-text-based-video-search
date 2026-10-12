"""
run_facts.py  --  speed, memory and GPU conditions per full run, plus the zero-box check.   (no GPU used)

Reads, for every run folder in --runs (default full_runs):
  timing.csv          one row per video: frames, prompt passes, decode / inference / wall seconds, peak VRAM
  videos/*.json.gz    the saved boxes (only to count them)
and, if present, the monitor logs written by full_runs.sh:
  monitor/run_windows.csv   start / end time of each run (a run stopped and resumed has several windows)
  monitor/gpu_log.csv       every 10 s: GPU busy %, memory used, temperature, clock, power

Writes to --out (default run_facts/):
  speed_table.md     per run: total hours, ms per frame-pass (wall and inference-only), videos/hour,
                     peak GPU memory (PyTorch's own count and nvidia-smi's), and the GPU's state while it ran
  zero_boxes.csv     every (run, video) with no box at all (score >= 0.01), and how many boxes the
                     SAME video got in each of the other runs -> tells "odd video" from "real miss"
  box_counts.csv     boxes per (run, video) at >= 0.01 and >= 0.25, for any later check

Definitions
  frame-pass  = one frame shown to YOLO with one vocabulary (separate: one prompt; joint: all prompts at once)
  wall time   = everything for a video (decode + inference + saving); inference-only excludes decoding
  peak_vram   = most GPU memory PyTorch had allocated for that video; nvidia-smi also counts the CUDA
                context and cache, so its number is higher and closer to "what the card must have"

Usage:  python run_facts.py
        python run_facts.py --runs full_runs --out run_facts
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


def parse_time(s):
    s = s.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def num(s):
    """'17.53 W' / '780 MHz' / '91 %' -> 17.53 / 780 / 91 ; anything else -> None"""
    try:
        return float(s.strip().split()[0])
    except (ValueError, IndexError):
        return None


def load_gpu_log(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 7 or r[0].startswith("timestamp"):
                continue  # header lines (written again after each restart) or partial lines
            t = parse_time(r[0])
            if t is None:
                continue
            rows.append({"t": t, "util": num(r[1]), "mem": num(r[2]), "mem_total": num(r[3]),
                         "temp": num(r[4]), "clock": num(r[5]), "power": num(r[6])})
    return rows


def load_windows(path):
    win = defaultdict(list)
    if not path.exists():
        return win
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            a, b = parse_time(r["start"]), parse_time(r["end"])
            if a and b:
                win[r["setting"]].append((a, b, r["status"]))
    return win


def gpu_during(gpu, windows):
    sel = [g for g in gpu if any(a <= g["t"] <= b for a, b, _ in windows)]
    if not sel:
        return None

    def med(k):
        v = [g[k] for g in sel if g[k] is not None]
        return median(v) if v else None

    mem = [g["mem"] for g in sel if g["mem"] is not None]
    return {"samples": len(sel), "util": med("util"), "clock": med("clock"), "power": med("power"),
            "temp": med("temp"), "mem_max": max(mem) if mem else None,
            "mem_total": sel[0]["mem_total"]}


def f1(x, unit=""):
    return "-" if x is None else f"{x:.1f}{unit}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="full_runs")
    ap.add_argument("--out", default="run_facts")
    args = ap.parse_args()
    runs_dir, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in runs_dir.iterdir() if (p / "timing.csv").exists())
    gpu = load_gpu_log(runs_dir / "monitor" / "gpu_log.csv")
    windows = load_windows(runs_dir / "monitor" / "run_windows.csv")
    print(f"{len(runs)} runs; GPU log samples: {len(gpu)}; run windows for {len(windows)} settings")

    # ---------- speed and memory
    L = ["# Speed, memory and GPU conditions per run\n",
         "frame-pass = one frame x one vocabulary. Times summed over all videos. "
         "GPU columns are medians over the 10-second samples taken while that run was going.\n",
         "| run | videos | frame-passes | hours (wall) | ms/frame-pass wall | ms/frame-pass inference | "
         "frame-passes/s | peak VRAM PyTorch MB | peak VRAM nvidia-smi MB | GPU busy % | clock MHz | power W | temp C |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    seen_ids = {}
    for run in runs:
        rows = list(csv.DictReader(open(run / "timing.csv")))
        by_vid = {}
        for r in rows:  # a resumed run never repeats a finished video, but keep the last row just in case
            by_vid[r["video_id"]] = r
        rows = list(by_vid.values())
        seen_ids[run.name] = set(by_vid)
        fp = sum(int(r["frames"]) * int(r["passes"]) for r in rows)
        wall = sum(float(r["wall_s"]) for r in rows)
        inf = sum(float(r["infer_s"]) for r in rows)
        vram = [float(r["peak_vram_mb"]) for r in rows if r["peak_vram_mb"] not in ("", "None")]
        g = gpu_during(gpu, windows.get(run.name, []))
        L.append(f"| {run.name} | {len(rows)} | {fp:,} | {wall / 3600:.2f} | {1000 * wall / fp:.1f} | "
                 f"{1000 * inf / fp:.1f} | {fp / wall:.0f} | {f1(max(vram) if vram else None)} | "
                 f"{f1(g and g['mem_max'])} | {f1(g and g['util'])} | {f1(g and g['clock'])} | "
                 f"{f1(g and g['power'])} | {f1(g and g['temp'])} |")
    if gpu:
        tot = gpu[0]["mem_total"]
        L.append(f"\nGPU memory available: {f1(tot)} MB. nvidia-smi memory includes everything on the card "
                 f"(other programs too, e.g. Windows), so treat it as an upper bound for YOLO.")
    else:
        L.append("\nNo GPU log found: GPU columns are empty.")
    (out / "speed_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # ---------- box counts and zero-box check
    print("\nCounting boxes in every saved video file (reads all files once, a few minutes) ...", flush=True)
    counts = {}  # (run, video_id) -> (n_all, n_025, sequence, frames, prompts)
    for run in runs:
        for k, vf in enumerate(sorted((run / "videos").glob("*.json.gz")), 1):
            rec = json.loads(gzip.decompress(vf.read_bytes()))
            n_all = len(rec["detections"])
            n25 = sum(1 for d in rec["detections"] if d[2] >= 0.25)
            counts[(run.name, rec["video_id"])] = (n_all, n25, rec["sequence"], rec["frames_decoded"],
                                                    len(rec["prompts"]))
        print(f"  {run.name}: {k} videos", flush=True)
    with open(out / "box_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "video_id", "sequence", "frames", "prompts", "boxes_ge_0.01", "boxes_ge_0.25"])
        for (run, vid), (n, n25, seq, fr, pr) in sorted(counts.items()):
            w.writerow([run, vid, seq, fr, pr, n, n25])

    run_names = [r.name for r in runs]
    zero = sorted({vid for (run, vid), c in counts.items() if c[0] == 0})
    with open(out / "zero_boxes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "sequence", "frames", "prompts", "runs_with_zero_boxes"]
                   + [f"boxes_{r}" for r in run_names])
        for vid in zero:
            per = [counts.get((r, vid), (None,))[0] for r in run_names]
            any_c = next(c for (r, v), c in counts.items() if v == vid)
            w.writerow([vid, any_c[2], any_c[3], any_c[4], sum(1 for p in per if p == 0)] + per)
    print(f"\nVideos with zero boxes in at least one run: {len(zero)}  (details: {out / 'zero_boxes.csv'})")
    for vid in zero:
        any_c = next(c for (r, v), c in counts.items() if v == vid)
        per = {r.split("-worldv2_img")[0][-1] + r.split("_img")[1].split("_fp16")[0] + r.split("_fp16_")[1][0]:
               counts.get((r, vid), (None,))[0] for r in run_names}
        print(f"  {vid} {any_c[2]} ({any_c[3]} frames, {any_c[4]} prompts): "
              + ", ".join(f"{k}={v}" for k, v in per.items()))
    print("  (key: model letter + image size + j/s for joint/separate, e.g. m640s = medium 640 separate)")
    print(f"\nWritten: {out / 'speed_table.md'}, {out / 'zero_boxes.csv'}, {out / 'box_counts.csv'}")


if __name__ == "__main__":
    main()
