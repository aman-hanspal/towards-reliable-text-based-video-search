"""
export_vlm_sets.py  --  build the VLM test sets A / B / C from one detector run and cut the crops.   (no GPU)

Unit = one (video, prompt) that YOLO answered "yes" to at --threshold (default 0.25).
For each such prompt we take ONE box (the strongest of the right type), cut it out of the frame with a margin,
and save it as a JPEG. The same crop files are later sent to the VLM on every machine.

  Set A  lookalike errors   absent prompt; its strongest box sits on a labelled object of ANOTHER kind
                            (IoU >= 0.5)                              -> correct VLM answer: "no"
  Set B  real detections    present prompt; its strongest box that matches a label of the SAME kind
                            (IoU >= 0.5)                              -> correct VLM answer: "yes"
  Set C  unlabelled area    absent prompt; fired, but every box sits where nothing is labelled
                            -> truth unknown until eye check ("unknown")
  pilot                     a few A and B items, drawn first and kept OUT of the results; only used to check
                            that the VLM answers in the expected format

Sizes (defaults): A = all, B = random --n-b (600), C = all, pilot = --n-pilot of each of A and B (10).
Random choices use a fixed seed, so the same command always picks the same items.

Crop: box + --margin (20%) of the box size on every side, clipped to the frame, at least --min-side (48) px
per side. Crops larger than --max-side (768) px on their long side are shrunk once, here, so every machine
receives byte-identical files and very large crops cannot exhaust a 4 GB GPU. Smaller crops are not resized.

Outputs in --out/<run name>/:
  crops/<item_id>.jpg
  items.csv     one row per crop: set, expected answer, video, prompt, YOLO score, box, crop box, sizes,
                base/novel, strict-hard, and for set A the labelled object the box sits on
  FROZEN.md     settings, counts and the SHA-256 of items.csv (the test set must not change after the pilot)

Usage (from the OVT-B-Dataset folder):
  python export_vlm_sets.py --run full_runs/yolov8m-worldv2_img640_fp16_separate
Needs score_runs.py in the same folder (reuses its label loading) and scores_full/gt_cache.json.gz.
"""
import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from score_runs import STRICT_HARD_WUP, iou, load_gt

MATCH_IOU = 0.5
FIELDS = ["item_id", "set", "expected", "video_id", "sequence", "source", "prompt_idx", "kind", "prompt",
          "category_id", "split", "strict_hard", "frame", "yolo_score", "x1", "y1", "x2", "y2",
          "crop_x1", "crop_y1", "crop_x2", "crop_y2", "crop_w", "crop_h", "saved_w", "saved_h", "box_rel_area",
          "prompt_max_score", "sits_on"]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop_box(b, w, h, margin, min_side):
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    x1, y1, x2, y2 = x1 - margin * bw, y1 - margin * bh, x2 + margin * bw, y2 + margin * bh
    for lo, hi, lim in ((0, 2, w), (1, 3, h)):  # widen to min_side around the centre
        c = [x1, y1, x2, y2]
        if c[hi] - c[lo] < min_side:
            mid = (c[hi] + c[lo]) / 2
            c[lo], c[hi] = mid - min_side / 2, mid + min_side / 2
        x1, y1, x2, y2 = c
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
    return x1, y1, x2, y2


def candidates(rec, qv, gt_v, thr):
    """One candidate per fired (video, prompt): (set, expected, row dict)."""
    qrows = [q for k in ("present", "easy_absent", "hard_absent") for q in qv[k]]
    names = {q["category_id"]: q["prompt"] for q in qv["present"]}  # labelled objects are present categories
    by_prompt = defaultdict(list)
    for fi, pi, sc, x1, y1, x2, y2 in rec["detections"]:
        if sc >= thr:
            by_prompt[pi].append((sc, fi, (x1, y1, x2, y2)))
    out = []
    for pi, dets in by_prompt.items():
        p = rec["prompts"][pi]
        q = qrows[pi] if pi < len(qrows) else {}
        kind, cid = p["kind"], p["category_id"]
        base = {"video_id": rec["video_id"], "sequence": rec["sequence"], "source": rec["sequence"].split("/")[0],
                "prompt_idx": pi, "kind": kind, "prompt": p["prompt"], "category_id": cid,
                "split": q.get("split", ""),
                "strict_hard": kind == "hard_absent" and (q.get("wup_to_anchor") or 0) >= STRICT_HARD_WUP}
        dets.sort(key=lambda d: -d[0])
        base["prompt_max_score"] = round(dets[0][0], 4)  # strongest box of this prompt, any location
        if kind == "present":
            for sc, fi, b in dets:
                if any(c == cid and iou(b, g) >= MATCH_IOU for c, g in gt_v.get(fi, [])):
                    out.append(("B", "yes", dict(base, frame=fi, yolo_score=sc, box=b, sits_on=p["prompt"])))
                    break
        else:
            mis = None
            for sc, fi, b in dets:
                hits = [(iou(b, g), c) for c, g in gt_v.get(fi, []) if c != cid]
                best = max(hits, default=(0, None))
                if best[0] >= MATCH_IOU:
                    mis = (sc, fi, b, names.get(best[1], str(best[1])))
                    break
            if mis:
                sc, fi, b, on = mis
                out.append(("A", "no", dict(base, frame=fi, yolo_score=sc, box=b, sits_on=on)))
            else:
                sc, fi, b = dets[0]
                out.append(("C", "unknown", dict(base, frame=fi, yolo_score=sc, box=b, sits_on="")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--run", required=True, help="e.g. full_runs/yolov8m-worldv2_img640_fp16_separate")
    ap.add_argument("--query-set", default="query_set_v3/query_set.json")
    ap.add_argument("--ann", default="ovtb_ann.json")
    ap.add_argument("--gt-cache", default="scores_full/gt_cache.json.gz")
    ap.add_argument("--out", default="vlm_sets")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--n-b", type=int, default=600, help="random sample size for set B (A and C: all)")
    ap.add_argument("--n-pilot", type=int, default=10, help="pilot items taken from each of A and B")
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--min-side", type=int, default=48)
    ap.add_argument("--max-side", type=int, default=768, help="shrink crops whose long side is larger than this")
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()

    import cv2
    root, run = Path(args.root), Path(args.root) / args.run
    if "_separate" not in run.name:
        raise SystemExit("Use a _separate run: joint runs keep only each box's best prompt.")
    out = root / args.out / run.name
    if (out / "items.csv").exists():
        raise SystemExit(f"{out/'items.csv'} already exists. A test set is never overwritten; use a new --out.")
    (out / "crops").mkdir(parents=True, exist_ok=True)

    qs = {v["video_id"]: v for v in json.loads((root / args.query_set).read_text(encoding="utf-8"))["videos"]}
    files = sorted((run / "videos").glob("*.json.gz"), key=lambda f: int(f.name.split(".")[0]))
    video_ids = {int(f.name.split(".")[0]) for f in files}
    gt = load_gt(root, root / args.ann, video_ids, root / args.gt_cache)

    print(f"Finding candidates in {len(files)} videos (threshold {args.threshold}) ...", flush=True)
    pools = defaultdict(list)
    for k, vf in enumerate(files, 1):
        rec = json.loads(gzip.decompress(vf.read_bytes()))
        for s, exp, row in candidates(rec, qs[rec["video_id"]], gt.get(rec["video_id"], {}), args.threshold):
            row.update(set=s, expected=exp, mp4=qs[rec["video_id"]]["mp4"])
            pools[s].append(row)
        if k % 400 == 0:
            print(f"  {k}/{len(files)} videos", flush=True)
    for s in "ABC":
        pools[s].sort(key=lambda r: (r["video_id"], r["prompt_idx"]))
        print(f"  candidates: set {s} = {len(pools[s])}")

    rng = random.Random(args.seed)
    chosen = []
    for s in "AB":
        pool = pools[s][:]
        rng.shuffle(pool)
        pilot, rest = pool[:args.n_pilot], pool[args.n_pilot:]
        for r in pilot:
            r["set"] = "pilot_" + s
        if s == "B":
            rest = rest[:args.n_b]
        chosen += pilot + rest
    chosen += pools["C"]

    # cut crops: read each needed video once, in frame order
    by_video = defaultdict(list)
    for r in chosen:
        r["item_id"] = f"{r['set']}_{r['video_id']}_{r['prompt_idx']}"
        by_video[r["video_id"]].append(r)
    print(f"Cutting {len(chosen)} crops from {len(by_video)} videos ...", flush=True)
    missing = 0
    for k, (vid, rows) in enumerate(sorted(by_video.items()), 1):
        need = defaultdict(list)
        for r in rows:
            need[r["frame"]].append(r)
        cap = cv2.VideoCapture(str(root / rows[0]["mp4"]))
        fi, last = 0, max(need)
        while fi <= last:
            ok, frame = cap.read()  # sequential read: exact frame numbers (seeking in h264 is not exact)
            if not ok:
                break
            for r in need.get(fi, []):
                h, w = frame.shape[:2]
                cb = crop_box(r["box"], w, h, args.margin, args.min_side)
                img = frame[cb[1]:cb[3], cb[0]:cb[2]]
                long_side = max(img.shape[:2])
                if long_side > args.max_side:
                    s = args.max_side / long_side
                    img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / "crops" / f"{r['item_id']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                r.update(saved_w=img.shape[1], saved_h=img.shape[0])
                x1, y1, x2, y2 = r["box"]
                r.update(x1=x1, y1=y1, x2=x2, y2=y2, crop_x1=cb[0], crop_y1=cb[1], crop_x2=cb[2], crop_y2=cb[3],
                         crop_w=cb[2] - cb[0], crop_h=cb[3] - cb[1],
                         box_rel_area=round((x2 - x1) * (y2 - y1) / (w * h), 6), yolo_score=round(r["yolo_score"], 4))
                r["done"] = True
            fi += 1
        cap.release()
        for r in rows:
            if not r.get("done"):
                missing += 1
                print(f"  !! could not read frame {r['frame']} of {r['sequence']}")
        if k % 200 == 0:
            print(f"  {k}/{len(by_video)} videos", flush=True)

    rows = sorted((r for r in chosen if r.get("done")), key=lambda r: r["item_id"])
    with open(out / "items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    counts = defaultdict(int)
    for r in rows:
        counts[r["set"]] += 1
    sha = sha256_file(out / "items.csv")
    L = ["# VLM test set (frozen)\n",
         f"- created (UTC): {datetime.now(timezone.utc).isoformat()}",
         f"- detector run: {run.name}", f"- threshold: {args.threshold}; match IoU: {MATCH_IOU}",
         f"- crop margin: {args.margin}; min side: {args.min_side}px; long side shrunk to <= {args.max_side}px; "
         f"JPEG quality 95",
         f"- seed: {args.seed}; B sample: {args.n_b}; pilot per set: {args.n_pilot}",
         f"- candidates before sampling: A {len(pools['A'])}, B {len(pools['B'])}, C {len(pools['C'])}",
         "- items: " + ", ".join(f"{s} {counts[s]}" for s in sorted(counts)),
         f"- unreadable frames: {missing}",
         f"- items.csv SHA-256: `{sha}`",
         f"- export script SHA-256: `{sha256_file(__file__)}`", "",
         "Pilot items only check the VLM's answer format; they are excluded from all results."]
    (out / "FROZEN.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nCrops in {out/'crops'}  |  list: {out/'items.csv'}")


if __name__ == "__main__":
    main()
