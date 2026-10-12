"""
run_detector.py  --  YOLO-World takes the exam (Stage B).   [version 2: decode-once]

Runs ONE configuration (model, image size, precision, prompt mode) over videos
from the FROZEN query set and saves every detection, so all scoring and every
confidence threshold can be analysed later WITHOUT re-running the GPU.

Prompt modes
  joint     : all of a video's prompts given to YOLO at once (one pass per video).
              Each box is reported only under its best-matching prompt.
  separate  : one prompt at a time (one pass per prompt) -- what a real search
              user does; the faithful reference.

Decoding (new in version 2)
  once      : (default) each video is decoded ONCE. Frames are read in chunks that fit in
              --chunk-mb of RAM; every prompt is run on a chunk before the next chunk is read.
              Same frames, same prompts, same model -> same detections, less CPU time.
  legacy    : the version-1 behaviour (re-decode the whole video for every prompt).
              Kept only to verify that 'once' gives identical results.

Selections
  mini20    : 20 videos, up to 3 per source, fixed seed (for the mini-run)
  miniN     : the first N videos of mini20 by video_id (e.g. mini3, for quick checks)
  all       : every video in the query set
  1,2,3     : comma-separated video ids

Output: <out>/<config_name>/
  run_meta.json              versions, hashes, GPU, settings
  videos/<video_id>.json.gz  detections: [frame, prompt_idx, score, x1, y1, x2, y2]
  timing.csv                 per-video decode time, inference time, wall time, peak VRAM
Re-running the same command resumes: finished videos are skipped.

Example:
  python run_detector.py --model yolov8m-worldv2.pt --imgsz 640 --half --mode separate --select all --out full_runs
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

FROZEN_SHA = "6d08f1a2408ade15446917896f7d053527263de291cea4e3dfc4b14342e27713"
RUNNER_VERSION = 2


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def select_videos(videos, how, seed=20261001):
    if how == "all":
        return videos
    if how.startswith("mini"):
        n = int(how[4:])
        by_src = defaultdict(list)
        for v in videos:
            by_src[v["source"]].append(v)
        rng = random.Random(f"{seed}:mini20")
        picked = []
        for src in sorted(by_src):
            pool = sorted(by_src[src], key=lambda v: v["video_id"])
            picked += rng.sample(pool, min(3, len(pool)))
        picked = sorted(picked, key=lambda v: v["video_id"])
        mini20 = rng.sample(picked, min(20, len(picked))) if len(picked) > 20 else picked
        if n == 20:
            return mini20                      # identical to version 1 (same order)
        return sorted(mini20, key=lambda v: v["video_id"])[:n]
    ids = {int(x) for x in how.split(",")}
    return [v for v in videos if v["video_id"] in ids]


def prompt_table(v):
    """All prompts of a video, in a fixed order: present, easy, hard."""
    rows = []
    for kind in ("present", "easy_absent", "hard_absent"):
        for q in v[kind]:
            rows.append({"kind": kind, "category_id": q["category_id"], "prompt": q["prompt"]})
    return rows


# ------------------------------------------------------------------ detectors
class FakeDetector:
    """Deterministic stand-in so the pipeline can be tested without a GPU."""
    max_det = 300

    def __init__(self, *a, **k):
        self.classes = []

    def set_classes(self, classes):
        self.classes = list(classes)

    def predict(self, frame, frame_idx):
        rng = random.Random(f"{len(self.classes)}:{frame_idx}:{','.join(self.classes)}:{int(frame.mean())}")
        h, w = frame.shape[:2]
        out = []
        for ci in range(len(self.classes)):
            if rng.random() < 0.3:
                x1, y1 = rng.uniform(0, w / 2), rng.uniform(0, h / 2)
                out.append((ci, rng.uniform(0.01, 0.9), x1, y1, x1 + w / 4, y1 + h / 4))
        return out

    def sync(self):
        pass

    def reset_peak(self):
        pass

    def peak_mb(self):
        return None


class YoloWorldDetector:
    max_det = 300

    def __init__(self, model_path, imgsz, half, device, conf):
        import torch
        from ultralytics import YOLOWorld
        self.torch = torch
        self.model = YOLOWorld(model_path)
        self.imgsz, self.half, self.device, self.conf = imgsz, half, device, conf
        # Ultralytics >= 8.4.80 replaced 'half' with 'quantize' (16 = FP16, 32 = FP32)
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            has_quantize = "quantize" in DEFAULT_CFG_DICT
        except Exception:
            has_quantize = False
        self.precision_kw = {"quantize": 16 if half else 32} if has_quantize else {"half": half}

    def set_classes(self, classes):
        self.model.set_classes(list(classes))

    def predict(self, frame, frame_idx):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, **self.precision_kw,
                               device=self.device, max_det=self.max_det, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy, conf, cls = b.xyxy.cpu().tolist(), b.conf.cpu().tolist(), b.cls.cpu().tolist()
        return [(int(c), s, *bb) for bb, s, c in zip(xyxy, conf, cls)]

    def actual_fp16(self):
        """What the loaded backend really uses, not what we asked for."""
        try:
            return bool(self.model.predictor.model.fp16)
        except Exception:
            return None

    def sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def reset_peak(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

    def peak_mb(self):
        if self.torch.cuda.is_available():
            return round(self.torch.cuda.max_memory_allocated() / 2**20, 1)
        return None


# ------------------------------------------------------------------ core
def detect_legacy(det, mp4, vocabularies):
    """Version-1 behaviour: one full decode of the video per vocabulary."""
    import cv2
    out = []
    decode_s = infer_s = 0.0
    sat = n = 0
    for vocab in vocabularies:
        det.set_classes(vocab)
        cap = cv2.VideoCapture(mp4)
        dets, n = [], 0
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            decode_s += time.perf_counter() - t0
            if not ok:
                break
            t1 = time.perf_counter()
            o = det.predict(frame, n)
            det.sync()
            infer_s += time.perf_counter() - t1
            sat += len(o) >= det.max_det
            dets += [(n, ci, s, *bb) for ci, s, *bb in o]
            n += 1
        cap.release()
        out.append(dets)
    return out, decode_s, infer_s, n, sat, len(vocabularies)


def detect_once(det, mp4, vocabularies, chunk_mb):
    """Version 2: decode each frame once; run every vocabulary on a chunk of frames held in RAM."""
    import cv2
    out = [[] for _ in vocabularies]
    decode_s = infer_s = 0.0
    sat = n = 0
    single = len(vocabularies) == 1
    if single:
        det.set_classes(vocabularies[0])
    cap = cv2.VideoCapture(mp4)
    chunk_frames, finished = None, False
    while not finished:
        chunk = []
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            decode_s += time.perf_counter() - t0
            if not ok:
                finished = True
                break
            chunk.append(frame)
            if chunk_frames is None:
                chunk_frames = max(1, int(chunk_mb * 2**20 // frame.nbytes))
            if len(chunk) >= chunk_frames:
                break
        if not chunk:
            break
        for vi, vocab in enumerate(vocabularies):
            if not single:
                det.set_classes(vocab)
            for k, frame in enumerate(chunk):
                t1 = time.perf_counter()
                o = det.predict(frame, n + k)
                det.sync()
                infer_s += time.perf_counter() - t1
                sat += len(o) >= det.max_det
                out[vi] += [(n + k, ci, s, *bb) for ci, s, *bb in o]
        n += len(chunk)
    cap.release()
    return out, decode_s, infer_s, n, sat, chunk_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--query-set", default="query_set_v3/query_set.json")
    ap.add_argument("--model", required=True, help="e.g. yolov8s-worldv2.pt / yolov8m-worldv2.pt / yolov8l-worldv2.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 inference")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.01,
                    help="save everything above this; real thresholds are applied at analysis time")
    ap.add_argument("--mode", choices=["joint", "separate"], default="joint")
    ap.add_argument("--decode", choices=["once", "legacy"], default="once")
    ap.add_argument("--chunk-mb", type=int, default=1024, help="RAM for decoded frames held at once (decode=once)")
    ap.add_argument("--select", default="mini20", help="mini20 | miniN | all | comma-separated video ids")
    ap.add_argument("--out", default="detector_runs")
    ap.add_argument("--fake", action="store_true", help="use a fake detector (testing only)")
    args = ap.parse_args()

    import cv2
    root = Path(args.root)
    qs_path = root / args.query_set
    qs_sha = sha256_file(qs_path)
    if qs_sha != FROZEN_SHA and not args.fake:
        sys.exit(f"Query set hash {qs_sha[:12]}... is not the frozen set {FROZEN_SHA[:12]}... Stopping.")
    qs = json.loads(qs_path.read_text(encoding="utf-8"))
    videos = select_videos(qs["videos"], args.select)

    model_stem = Path(args.model).stem
    config = f"{model_stem}_img{args.imgsz}_{'fp16' if args.half else 'fp32'}_{args.mode}" + ("_FAKE" if args.fake else "")
    out_dir = root / args.out / config
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    det = (FakeDetector() if args.fake
           else YoloWorldDetector(args.model, args.imgsz, args.half, args.device, args.conf))

    meta_path = out_dir / "run_meta.json"
    meta = {
        "config": config, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_version": RUNNER_VERSION, "decode": args.decode, "chunk_mb": args.chunk_mb,
        "query_set_sha256": qs_sha, "selection": args.select,
        "selected_video_ids": [v["video_id"] for v in videos],
        "model": args.model, "imgsz": args.imgsz, "half": args.half, "device": args.device,
        "save_conf": args.conf, "mode": args.mode, "max_det": det.max_det, "fake": args.fake,
        "script_sha256": sha256_file(__file__), "python": platform.python_version(),
        "opencv": cv2.__version__,
    }
    if not args.fake:
        import torch
        import ultralytics
        meta.update({"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "checkpoint_sha256": sha256_file(det.model.ckpt_path) if getattr(det.model, "ckpt_path", None)
                     and Path(det.model.ckpt_path).exists() else None})
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for k in ("query_set_sha256", "model", "imgsz", "half", "mode", "save_conf", "runner_version", "decode"):
            if old.get(k) != meta.get(k):
                sys.exit(f"{out_dir} was created with different settings ({k}). Use a new --out.")
    else:
        atomic_write_bytes(meta_path, json.dumps(meta, indent=1).encode())

    # warm-up (excluded from timing): first selected video, first frame
    first = cv2.VideoCapture(str(root / videos[0]["mp4"]))
    ok, frame = first.read()
    first.release()
    if ok:
        det.set_classes([p["prompt"] for p in prompt_table(videos[0])][:2] or ["object"])
        for _ in range(3):
            det.predict(frame, 0)
        det.sync()
    if not args.fake:
        actual = det.actual_fp16()
        meta_now = json.loads(meta_path.read_text())
        meta_now.update({"precision_argument": det.precision_kw, "backend_fp16_actual": actual})
        atomic_write_bytes(meta_path, json.dumps(meta_now, indent=1).encode())
        print(f"Precision requested {det.precision_kw} -> backend actually FP16: {actual}")
        if actual is not None and actual != args.half:
            sys.exit("Requested precision was NOT applied by Ultralytics. Stopping so results are not mislabelled.")

    timing_path = out_dir / "timing.csv"
    new_timing = not timing_path.exists()
    todo = [v for v in videos if not (out_dir / "videos" / f"{v['video_id']}.json.gz").exists()]
    if len(todo) < len(videos):
        print(f"{len(videos) - len(todo)} videos already done (resuming); {len(todo)} to go.")
    t_start, done_now = time.time(), 0
    with open(timing_path, "a", newline="") as tf:
        tw = csv.writer(tf)
        if new_timing:
            tw.writerow(["video_id", "source", "frames", "prompts", "passes", "decode_s", "infer_s",
                         "infer_ms_per_frame_pass", "peak_vram_mb", "saturated_frames", "frame_count_ok",
                         "wall_s", "chunk_frames"])
        for i, v in enumerate(todo, 1):
            vpath = out_dir / "videos" / f"{v['video_id']}.json.gz"
            prompts = prompt_table(v)
            unique = list(dict.fromkeys(p["prompt"] for p in prompts))
            to_rows = defaultdict(list)  # unique prompt index -> prompt-table rows sharing that text
            for ri, p in enumerate(prompts):
                to_rows[unique.index(p["prompt"])].append(ri)
            mp4 = str(root / v["mp4"])
            vocabularies = [unique] if args.mode == "joint" else [[u] for u in unique]

            det.reset_peak()
            t_video = time.perf_counter()
            if args.decode == "legacy":
                per_vocab, dec, inf, frames, sat, chunk_frames = detect_legacy(det, mp4, vocabularies)
                chunk_frames = None
            else:
                per_vocab, dec, inf, frames, sat, chunk_frames = detect_once(det, mp4, vocabularies, args.chunk_mb)
            wall = time.perf_counter() - t_video
            passes = len(vocabularies)

            all_dets = []
            for vi, d in enumerate(per_vocab):
                offset = 0 if args.mode == "joint" else vi
                for fi, ci, sc, x1, y1, x2, y2 in d:
                    for ri in to_rows[ci + offset]:
                        all_dets.append([fi, ri, round(sc, 4), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
            ok_count = frames == v["frame_count"]
            record = {"video_id": v["video_id"], "sequence": v["sequence"], "config": config,
                      "frames_decoded": frames, "frame_count_expected": v["frame_count"],
                      "prompts": prompts, "saturated_frames": sat,
                      "detection_fields": ["frame", "prompt_idx", "score", "x1", "y1", "x2", "y2"],
                      "detections": all_dets}
            atomic_write_bytes(vpath, gzip.compress(json.dumps(record).encode()))
            ms = 1000 * inf / max(1, frames * passes)
            tw.writerow([v["video_id"], v["source"], frames, len(prompts), passes, round(dec, 3), round(inf, 3),
                         round(ms, 2), det.peak_mb(), sat, ok_count, round(wall, 3), chunk_frames])
            tf.flush()
            done_now += 1
            eta_h = (time.time() - t_start) / done_now * (len(todo) - i) / 3600
            flag = "" if ok_count else "  !! frame count mismatch"
            print(f"[{i}/{len(todo)}] {v['sequence']}: {frames} frames x {passes} pass(es), "
                  f"{ms:.1f} ms/frame, {wall:.0f}s, {len(all_dets)} boxes | ~{eta_h:.1f} h left{flag}", flush=True)
    print(f"\nFinished {config}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
