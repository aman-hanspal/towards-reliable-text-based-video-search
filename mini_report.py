"""mini_report.py -- summarise the mini-run: speed/memory per setting, and joint vs separate agreement."""
import argparse, csv, gzip, json
from collections import defaultdict
from pathlib import Path
from statistics import median

THRESHOLDS = (0.05, 0.10, 0.25)


def load(cfg_dir):
    out = {}
    for f in (cfg_dir / "videos").glob("*.json.gz"):
        r = json.loads(gzip.decompress(f.read_bytes()))
        out[r["video_id"]] = r
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0.0


def compare(joint, sep):
    """For each prompt in each video: does it 'fire' (any box >= t) in joint vs separate?"""
    agree = {t: defaultdict(lambda: [0, 0, 0, 0]) for t in THRESHOLDS}  # kind -> [both, sep_only, joint_only, neither]
    box_match = defaultdict(lambda: [0, 0])                             # kind -> [separate boxes matched by joint, total]
    for vid, s in sep.items():
        if vid not in joint:
            continue
        j = joint[vid]
        kinds = [p["kind"] for p in s["prompts"]]
        by = lambda r: defaultdict(list, _group(r["detections"]))
        js, ss = by(j), by(s)
        for pi, kind in enumerate(kinds):
            jmax = max((d[2] for d in js[pi]), default=0.0)
            smax = max((d[2] for d in ss[pi]), default=0.0)
            for t in THRESHOLDS:
                a, b = smax >= t, jmax >= t
                agree[t][kind][0 if a and b else 1 if a else 2 if b else 3] += 1
            jf = defaultdict(list)
            for d in js[pi]:
                if d[2] >= 0.10: jf[d[0]].append(d[3:])
            for d in ss[pi]:
                if d[2] >= 0.10:
                    box_match[kind][1] += 1
                    if any(iou(d[3:], b) >= 0.5 for b in jf[d[0]]):
                        box_match[kind][0] += 1
    return agree, box_match


def _group(dets):
    g = defaultdict(list)
    for d in dets:
        g[d[1]].append(d)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="mini_runs")
    args = ap.parse_args()
    runs = Path(args.runs)
    cfgs = sorted(p for p in runs.iterdir() if (p / "timing.csv").exists())
    L = ["# Mini-run report\n", "## Speed and memory per setting\n",
         "Inference only (decode excluded), median over videos. ms/frame is per prompt-pass.\n",
         "| setting | videos | median ms/frame | frames/s | peak VRAM MB | saturated frames | frame-count problems |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for c in cfgs:
        rows = list(csv.DictReader(open(c / "timing.csv")))
        if not rows: continue
        ms = median(float(r["infer_ms_per_frame_pass"]) for r in rows)
        vram = [float(r["peak_vram_mb"]) for r in rows if r["peak_vram_mb"] not in ("", "None")]
        sat = sum(int(r["saturated_frames"]) for r in rows)
        bad = sum(r["frame_count_ok"] != "True" for r in rows)
        L.append(f"| {c.name} | {len(rows)} | {ms:.1f} | {1000/ms:.0f} | {max(vram) if vram else 'n/a'} | {sat} | {bad} |")
    L += ["", "## Joint vs separate prompting\n",
          "Does a prompt 'fire' (any box at or above the threshold) the same way when YOLO is given all "
          "prompts at once (joint) as when it is asked one at a time (separate)?\n"]
    for c in cfgs:
        if not c.name.endswith("_separate"): continue
        jc = runs / c.name.replace("_separate", "_joint")
        if not jc.exists():
            L.append(f"- {c.name}: no matching joint run"); continue
        agree, bm = compare(load(jc), load(c))
        L += [f"### {c.name.replace('_separate', '')}\n",
              "| threshold | kind | both fire | only separate fires | only joint fires | neither | agreement |",
              "|---:|---|---:|---:|---:|---:|---:|"]
        for t in THRESHOLDS:
            for kind in ("present", "easy_absent", "hard_absent"):
                b, so, jo, n = agree[t][kind]
                tot = b + so + jo + n
                L.append(f"| {t} | {kind} | {b} | {so} | {jo} | {n} | {(b+n)/tot:.1%} |" if tot else f"| {t} | {kind} | - | - | - | - | - |")
        L += ["", "Box level (score >= 0.10): share of separate-mode boxes that joint mode also finds (IoU >= 0.5)\n"]
        for kind, (m, tot) in bm.items():
            L.append(f"- {kind}: {m}/{tot} = {m/tot:.1%}" if tot else f"- {kind}: no boxes")
        L.append("")
    (runs / "mini_run_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWritten to {runs/'mini_run_report.md'}")


if __name__ == "__main__":
    main()
