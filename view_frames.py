"""
view_frames.py -- draw YOLO detections (and the dataset's labels) on a few frames.

Colours:  BLUE = detection for a PRESENT prompt   RED = detection for an ABSENT prompt
          GREEN = the dataset's real label (ground truth), when --gt is given

Auto mode picks 4 revealing frames:
  1. the seal video (1947): best frame for the correct prompt
  2. most confident HARD-absent detection in the run   (lookalike false-alarm candidate)
  3. most confident EASY-absent detection in the run   (unrelated hallucination candidate)
  4. a crowd video (penguins), all confident boxes

Usage:
  python view_frames.py --run smoke_test/yolov8m-worldv2_img640_fp16_joint --gt
  python view_frames.py --run ... --video 1947 --frame 12 --gt      (one specific frame)
Images are written to <run>/frames/. Upload them to Claude to look at together.
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import cv2

BLUE, RED, GREEN = (255, 120, 0), (0, 0, 255), (0, 200, 0)


def load_run(run):
    vids = {}
    for f in (run / "videos").glob("*.json.gz"):
        r = json.loads(gzip.decompress(f.read_bytes()))
        vids[r["video_id"]] = r
    return vids


def read_frame(mp4, idx):
    cap = cv2.VideoCapture(str(mp4))
    frame = None
    for _ in range(idx + 1):  # sequential read: exact, unlike seeking in h264
        ok, frame = cap.read()
        if not ok:
            frame = None
            break
    cap.release()
    return frame


def load_gt(root, ann, wanted):
    """{(video_id, video_frame_index): [(category_name, bbox_xywh)]} for the wanted (video, frame) pairs."""
    img_to_vf = {}
    vids = {v for v, _ in wanted}
    for mf in (root / "prepared_videos").glob("*/*/*.manifest.json"):
        m = json.loads(mf.read_text())
        if m["video_id"] in vids:
            for fr in m["frames"]:
                if (m["video_id"], fr["video_frame_index"]) in wanted:
                    img_to_vf[fr["image_id"]] = (m["video_id"], fr["video_frame_index"])
    try:
        import ijson
        def items(key):
            with open(ann, "rb") as f:
                yield from ijson.items(f, f"{key}.item", use_float=True)
    except ImportError:
        data = json.load(open(ann))
        def items(key):
            yield from data.get(key, [])
    names = {c["id"]: c["name"] for c in items("categories")}
    gt = defaultdict(list)
    print("Reading ground-truth boxes (streams the big annotation file, ~1-3 min) ...")
    for a in items("annotations"):
        if a["image_id"] in img_to_vf:
            gt[img_to_vf[a["image_id"]]].append((names.get(a["category_id"], "?"), a["bbox"]))
    return gt


def draw(root, rec, frame_idx, min_score, gt, title, out_path):
    frame = read_frame(root / f"prepared_videos/{rec['sequence']}/{rec['sequence'].split('/')[-1]}.mp4", frame_idx)
    if frame is None:
        print(f"  could not read frame {frame_idx} of {rec['sequence']}")
        return
    lines = []
    for name, (x, y, w, h) in gt.get((rec["video_id"], frame_idx), []):
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), GREEN, 2)
        cv2.putText(frame, f"LABEL {name}", (int(x), int(y + h) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
    dets = [d for d in rec["detections"] if d[0] == frame_idx and d[2] >= min_score]
    for _, pi, sc, x1, y1, x2, y2 in sorted(dets, key=lambda d: d[2]):
        p = rec["prompts"][pi]
        col = BLUE if p["kind"] == "present" else RED
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(frame, f"{p['prompt']} {sc:.2f}", (int(x1), max(15, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        lines.append(f"    {p['kind']:12s} {p['prompt']:20s} {sc:.2f}")
    cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    cv2.imwrite(str(out_path), frame)
    print(f"  saved {out_path}  ({rec['sequence']}, frame {frame_idx}, {len(dets)} boxes >= {min_score})")
    for l in lines[-8:]:
        print(l)


def best(vids, kind, exclude=()):
    top = None
    for vid, r in vids.items():
        if vid in exclude:
            continue
        for d in r["detections"]:
            if r["prompts"][d[1]]["kind"] == kind and (top is None or d[2] > top[2][2]):
                top = (vid, r, d)
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ann", default="ovtb_ann.json")
    ap.add_argument("--gt", action="store_true", help="also draw the dataset's real labels (green)")
    ap.add_argument("--video", type=int)
    ap.add_argument("--frame", type=int)
    ap.add_argument("--min-score", type=float, default=0.25)
    args = ap.parse_args()
    root, run = Path(args.root), Path(args.run)
    vids = load_run(run)
    out = run / "frames"
    out.mkdir(exist_ok=True)

    picks = []  # (video_id, frame, title, filename)
    if args.video is not None:
        picks.append((args.video, args.frame or 0, f"video {args.video} frame {args.frame or 0}",
                      f"v{args.video}_f{args.frame or 0}.jpg"))
    else:
        if 1947 in vids:
            r = vids[1947]
            pres = [d for d in r["detections"] if r["prompts"][d[1]]["kind"] == "present"]
            if pres:
                d = max(pres, key=lambda d: d[2])
                picks.append((1947, d[0], f"1 seal video: best CORRECT-prompt box ({d[2]:.2f})", "1_seal_best_present.jpg"))
            else:
                picks.append((1947, 12, "1 seal video: correct prompt NEVER fired", "1_seal_no_present.jpg"))
        for n, kind, label in ((2, "hard_absent", "HARD-absent"), (3, "easy_absent", "EASY-absent")):
            t = best(vids, kind, exclude={1947})
            if t:
                vid, r, d = t
                picks.append((vid, d[0], f"{n} top {label}: '{r['prompts'][d[1]]['prompt']}' {d[2]:.2f}",
                              f"{n}_top_{kind}.jpg"))
        crowd = next((v for v, r in vids.items() if "penguin" in r["sequence"]), None)
        if crowd is not None:
            picks.append((crowd, 100, "4 crowd video (penguins), frame 100", "4_crowd_penguins.jpg"))

    gt = load_gt(root, root / args.ann, {(v, f) for v, f, _, _ in picks}) if args.gt else {}
    for vid, f, title, name in picks:
        if vid not in vids:
            print(f"  video {vid} not in this run")
            continue
        print(f"\n{title}")
        draw(root, vids[vid], f, args.min_score, gt, title, out / name)
    print(f"\nImages are in {out}/  -- upload them to Claude.")


if __name__ == "__main__":
    main()
