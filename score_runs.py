"""
score_runs.py  --  compare detector runs against the dataset's labels (ground truth).   [version 2]

For every run folder (e.g. full_runs/yolov8m-worldv2_img640_fp16_separate) it scores:

PRESENT prompts (object is labelled in the video)
  frame_hit_rate   share of labelled frames where >=1 labelled box was found
  box_recall       share of labelled boxes matched by a detection (greedy, one-to-one)
  video_found      the object was found at least once in the video
  Matching: strict IoU >= 0.5 and lenient IoU >= 0.3 (lenient = "box roughly in the right place").

ABSENT prompts (object not labelled in the video)
  fired            any box at or above the threshold
  Every such box is sorted automatically:
    misnaming        box overlaps (IoU >= 0.5) a labelled object of ANOTHER category -> confirmed error
    unlabelled area  box overlaps no label -> hallucination OR real unlabelled object -> goes to Stage C review
  Per prompt: fired_misnaming (>=1 misnaming box), fired_unlabelled_only (fired, but only in unlabelled areas)

Version 2 changes (scoring rules unchanged):
  --only     score only the named run folders (comma-separated), e.g. while other runs are still going
  streaming  videos are read one at a time, so full runs (millions of boxes) fit in memory
  skips      run folders without a finished run are refused unless --allow-partial

Outputs in --out:
  gt_cache.json.gz           labels for the scored videos (built once, reused)
  prompt_scores.csv          one row per (run, video, prompt, threshold): the raw table for all analysis
  score_summary.md           readable tables per run
  review_candidates.csv      for absent prompts that fired only in unlabelled areas: the strongest box,
                             for Stage C eye review (threshold 0.25)

Usage:
  python score_runs.py --runs mini_runs --out scores_mini
  python score_runs.py --runs full_runs --out scores_full --only yolov8m-worldv2_img640_fp16_separate,yolov8m-worldv2_img640_fp16_joint
"""
import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

THRESHOLDS = (0.05, 0.10, 0.25, 0.40)
STRICT_IOU, LENIENT_IOU, MISNAME_IOU = 0.5, 0.3, 0.5
STRICT_HARD_WUP = 0.85
SIZE_BINS = ((0.01, "small (<1% of frame)"), (0.10, "medium (1-10%)"), (9e9, "large (>10%)"))
SUMMARY_FIELDS = ("kind", "threshold", "gt_boxes", "video_found", "frame_hit_rate", "frame_hit_rate_lenient",
                  "box_recall", "box_recall_lenient", "split", "size_bin", "source", "strict_hard", "fired",
                  "fired_misnaming", "fired_unlabelled_only")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(dets, gts, thr):
    """dets: [(score, box)] sorted desc; gts: [box]. Returns number of gts matched (one-to-one)."""
    used, n = set(), 0
    for _, d in dets:
        best, bi = thr, None
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            v = iou(d, g)
            if v >= best:
                best, bi = v, gi
        if bi is not None:
            used.add(bi)
            n += 1
    return n


def size_bin(rel_area):
    if rel_area is None:
        return "unknown"
    for lim, name in SIZE_BINS:
        if rel_area < lim:
            return name
    return SIZE_BINS[-1][1]


# ------------------------------------------------------------------ ground truth
def load_gt(root, ann_path, video_ids, cache_path):
    """{video_id: {frame_idx: [(category_id, [x1,y1,x2,y2])]}}  using manifests for frame -> image_id."""
    key = hashlib.sha256((",".join(map(str, sorted(video_ids))) + str(ann_path)).encode()).hexdigest()[:16]
    if cache_path.exists():
        c = json.loads(gzip.decompress(cache_path.read_bytes()))
        if c.get("key") == key:
            return {int(v): {int(f): b for f, b in fr.items()} for v, fr in c["gt"].items()}
    img_to_vf = {}
    for mf in (root / "prepared_videos").glob("*/*/*.manifest.json"):
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m["video_id"] in video_ids:
            for fr in m["frames"]:
                img_to_vf[fr["image_id"]] = (m["video_id"], fr["video_frame_index"])
    print(f"Reading labels for {len(video_ids)} videos from {ann_path.name} (streams the big file, a few minutes) ...",
          flush=True)
    try:
        import ijson
        f = open(ann_path, "rb")
        anns = ijson.items(f, "annotations.item", use_float=True)
    except ImportError:
        anns = json.load(open(ann_path, encoding="utf-8"))["annotations"]
    gt = defaultdict(lambda: defaultdict(list))
    for a in anns:
        vf = img_to_vf.get(a["image_id"])
        if vf is None or a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        gt[vf[0]][vf[1]].append((a["category_id"], [x, y, x + w, y + h]))
    gt = {v: dict(fr) for v, fr in gt.items()}
    cache_path.write_bytes(gzip.compress(json.dumps({"key": key, "gt": gt}).encode()))
    return gt


# ------------------------------------------------------------------ scoring
def score_video(rec, qv, gt_v):
    """Yields one dict per (prompt, threshold)."""
    by_prompt = defaultdict(list)
    for fi, pi, sc, x1, y1, x2, y2 in rec["detections"]:
        by_prompt[pi].append((fi, sc, (x1, y1, x2, y2)))
    # query-set details per prompt row (same order as runner: present, easy_absent, hard_absent)
    qrows = [(k, q) for k in ("present", "easy_absent", "hard_absent") for q in qv[k]]
    for pi, p in enumerate(rec["prompts"]):
        kind, cid = p["kind"], p["category_id"]
        q = qrows[pi][1] if pi < len(qrows) else {}
        base = {"video_id": rec["video_id"], "source": rec["sequence"].split("/")[0], "prompt_idx": pi,
                "prompt": p["prompt"], "kind": kind, "category_id": cid, "split": q.get("split", ""),
                "strict_hard": kind == "hard_absent" and (q.get("wup_to_anchor") or 0) >= STRICT_HARD_WUP,
                "size_bin": size_bin(q.get("median_rel_box_area")) if kind == "present" else ""}
        dets = by_prompt.get(pi, [])
        for t in THRESHOLDS:
            kept = [d for d in dets if d[1] >= t]
            row = dict(base, threshold=t, n_boxes=len(kept))
            if kind == "present":
                per_frame = defaultdict(list)
                for fi, sc, b in kept:
                    per_frame[fi].append((sc, b))
                gt_frames = {fi: [b for c, b in boxes if c == cid] for fi, boxes in gt_v.items()}
                gt_frames = {fi: bs for fi, bs in gt_frames.items() if bs}
                n_gt = sum(len(bs) for bs in gt_frames.values())
                m_strict = m_len = hit_s = hit_l = 0
                for fi, bs in gt_frames.items():
                    ds = sorted(per_frame.get(fi, []), key=lambda d: -d[0])
                    a = greedy_match(ds, bs, STRICT_IOU)
                    b2 = greedy_match(ds, bs, LENIENT_IOU)
                    m_strict += a; m_len += b2
                    hit_s += a > 0; hit_l += b2 > 0
                nf = len(gt_frames)
                row.update(gt_frames=nf, gt_boxes=n_gt,
                           frame_hit_rate=hit_s / nf if nf else None,
                           frame_hit_rate_lenient=hit_l / nf if nf else None,
                           box_recall=m_strict / n_gt if n_gt else None,
                           box_recall_lenient=m_len / n_gt if n_gt else None,
                           video_found=int(hit_s > 0), video_found_lenient=int(hit_l > 0))
            else:
                mis = unl = 0
                best_unl = None
                for fi, sc, b in kept:
                    labels = gt_v.get(fi, [])
                    if any(iou(b, g) >= MISNAME_IOU for c, g in labels if c != cid):
                        mis += 1
                    else:
                        unl += 1
                        if best_unl is None or sc > best_unl[1]:
                            best_unl = (fi, sc, b)
                row.update(fired=int(len(kept) > 0), misnaming_boxes=mis, unlabelled_boxes=unl,
                           fired_misnaming=int(mis > 0), fired_unlabelled_only=int(mis == 0 and unl > 0),
                           best_unlabelled=best_unl)
            yield row


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, pct=True):
    return "-" if x is None else (f"{100 * x:.1f}%" if pct else f"{x}")


def summarise(rows_by_run, out):
    L = ["# Score summary (against dataset labels)\n",
         "Present: strict = box overlap IoU >= 0.5, lenient = IoU >= 0.3. Absent: 'misnaming' = the box sits on a "
         "labelled object of another kind (confirmed error); 'unlabelled only' = fired only where nothing is labelled "
         "(needs Stage C eye check).\n",
         "**Runs ending in _joint are biased** (each box keeps only its best prompt). Use _separate runs for absent prompts.\n"]
    for run, rows in sorted(rows_by_run.items()):
        L += [f"## {run}\n", "### Present prompts\n",
              "| threshold | prompts | video found | frame hit rate | frame hit (lenient) | box recall | box recall (lenient) |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for t in THRESHOLDS:
            r = [x for x in rows if x["kind"] == "present" and x["threshold"] == t and x["gt_boxes"]]
            L.append(f"| {t} | {len(r)} | {fmt(mean([x['video_found'] for x in r]))} | {fmt(mean([x['frame_hit_rate'] for x in r]))} | "
                     f"{fmt(mean([x['frame_hit_rate_lenient'] for x in r]))} | {fmt(mean([x['box_recall'] for x in r]))} | "
                     f"{fmt(mean([x['box_recall_lenient'] for x in r]))} |")
        L += ["", "Present, frame hit rate at threshold 0.25, by group:\n", "| group | prompts | frame hit rate |", "|---|---:|---:|"]
        r25 = [x for x in rows if x["kind"] == "present" and x["threshold"] == 0.25 and x["gt_boxes"]]
        for key in ("split", "size_bin", "source"):
            for g in sorted({x[key] for x in r25}):
                rr = [x for x in r25 if x[key] == g]
                L.append(f"| {key}: {g} | {len(rr)} | {fmt(mean([x['frame_hit_rate'] for x in rr]))} |")
        L += ["", "### Absent prompts (share of prompts)\n",
              "| threshold | group | prompts | fired | misnaming (confirmed) | unlabelled only (needs review) |",
              "|---:|---|---:|---:|---:|---:|"]
        for t in THRESHOLDS:
            for name, f in (("easy", lambda x: x["kind"] == "easy_absent"),
                            ("hard", lambda x: x["kind"] == "hard_absent"),
                            ("strict-hard", lambda x: x["strict_hard"])):
                r = [x for x in rows if x["threshold"] == t and x["kind"] != "present" and f(x)]
                if r:
                    L.append(f"| {t} | {name} | {len(r)} | {fmt(mean([x['fired'] for x in r]))} | "
                             f"{fmt(mean([x['fired_misnaming'] for x in r]))} | {fmt(mean([x['fired_unlabelled_only'] for x in r]))} |")
        L.append("")
    (out / "score_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def run_is_finished(run_dir, qs):
    """A run is finished when it has a file for every selected video."""
    meta = json.loads((run_dir / "run_meta.json").read_text()) if (run_dir / "run_meta.json").exists() else {}
    want = meta.get("selected_video_ids") or list(qs)
    have = {int(f.name.split(".")[0]) for f in (run_dir / "videos").glob("*.json.gz")}
    return len(have), len(want), set(want) <= have


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--runs", default="mini_runs", help="folder containing run folders")
    ap.add_argument("--only", default="", help="comma-separated run folder names to score (default: all)")
    ap.add_argument("--allow-partial", action="store_true", help="also score runs that are not finished")
    ap.add_argument("--query-set", default="query_set_v3/query_set.json")
    ap.add_argument("--ann", default="ovtb_ann.json")
    ap.add_argument("--out", default="scores_mini")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.root) / args.out
    out.mkdir(parents=True, exist_ok=True)

    qs = {v["video_id"]: v for v in json.loads((root / args.query_set).read_text(encoding="utf-8"))["videos"]}
    runs = sorted(p for p in (root / args.runs).iterdir() if (p / "videos").is_dir())
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        missing = [w for w in wanted if w not in {r.name for r in runs}]
        if missing:
            raise SystemExit(f"Run folder(s) not found in {args.runs}: {missing}")
        runs = [r for r in runs if r.name in wanted]
    keep = []
    for r in runs:
        have, want, done = run_is_finished(r, qs)
        if done or args.allow_partial:
            keep.append(r)
            print(f"  {r.name}: {have}/{want} videos{'' if done else '  (PARTIAL)'}")
        else:
            print(f"  {r.name}: {have}/{want} videos -> not finished, skipped (use --allow-partial to include)")
    runs = keep
    if not runs:
        raise SystemExit("Nothing to score.")
    files = {r.name: sorted((r / "videos").glob("*.json.gz"), key=lambda f: int(f.name.split(".")[0])) for r in runs}
    video_ids = {int(f.name.split(".")[0]) for fs in files.values() for f in fs}
    gt = load_gt(root, root / args.ann, video_ids, out / "gt_cache.json.gz")

    fields = ["run", "video_id", "source", "prompt_idx", "prompt", "kind", "category_id", "split", "strict_hard",
              "size_bin", "threshold", "n_boxes", "gt_frames", "gt_boxes", "frame_hit_rate", "frame_hit_rate_lenient",
              "box_recall", "box_recall_lenient", "video_found", "video_found_lenient", "fired", "misnaming_boxes",
              "unlabelled_boxes", "fired_misnaming", "fired_unlabelled_only"]
    rows_by_run, review = defaultdict(list), []
    with open(out / "prompt_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for run, fs in files.items():
            t0 = time.time()
            print(f"Scoring {run} ({len(fs)} videos) ...", flush=True)
            for k, vf in enumerate(fs, 1):
                rec = json.loads(gzip.decompress(vf.read_bytes()))
                for row in score_video(rec, qs[rec["video_id"]], gt.get(rec["video_id"], {})):
                    row["run"] = run
                    w.writerow(row)
                    rows_by_run[run].append({k2: row.get(k2) for k2 in SUMMARY_FIELDS})
                    if (row["kind"] != "present" and row["threshold"] == 0.25 and row["fired_unlabelled_only"]
                            and run.endswith("_separate")):
                        fi, sc, b = row["best_unlabelled"]
                        review.append({"run": run, "video_id": row["video_id"], "kind": row["kind"],
                                       "prompt": row["prompt"], "frame": fi, "score": round(sc, 4),
                                       "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]})
                del rec
                if k % 200 == 0:
                    el = time.time() - t0
                    print(f"  {k}/{len(fs)} videos, {el / 60:.1f} min so far", flush=True)
    with open(out / "review_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "video_id", "kind", "prompt", "frame", "score", "x1", "y1", "x2", "y2"])
        w.writeheader()
        w.writerows(sorted(review, key=lambda r: -r["score"]))
    L = summarise(rows_by_run, out)
    print(f"\nFull summary: {out/'score_summary.md'}  |  raw table: {out/'prompt_scores.csv'}  |  "
          f"review list: {out/'review_candidates.csv'} ({len(review)} items)")


if __name__ == "__main__":
    main()
