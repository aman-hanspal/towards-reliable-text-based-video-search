#!/usr/bin/env python3
"""Read-only OVT-B inspection; writes only a NEW diagnostic JSON report.

Python >= 3.10. Install dependency: python -m pip install ijson
Save this file beside ovtb_ann.json and the OVT-B/ image directory, then run:
    python inspect_ovtb.py --root .

No model, GPU, video conversion, network calls, or dataset edits are performed.
Two streaming annotation passes retain small indexes and the selected sequence,
not the complete annotations array. File existence checks cover ONLY that sequence.
Timing fields are reported as unverified evidence, never converted into assumed FPS.
Exit codes: 0 = report written (check warnings), 1 = input/processing/write failure,
2 = invalid command-line arguments. Existing reports are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time


SECTIONS = {"images", "videos", "tracks", "annotations", "categories", "licenses"}
REQUIRED = SECTIONS - {"licenses"}
TIMING = re.compile(r"fps|frame.?rate|frames_per_second|timestamp|time.?base|duration|\bpts\b|frame.?time|presentation.?time", re.I)
LIMIT = 12


def say(message):
    print(message, file=sys.stderr, flush=True)


def compact(value):
    """Keep example values small without altering the source."""
    if isinstance(value, dict):
        return {str(k): compact(v) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        return [compact(v) for v in value[:LIMIT]]
    if isinstance(value, str):
        return value[:600]
    return value


def normalized_path(name):
    return str(name).replace("\\", "/").removeprefix("./")


def stream_records(path, evidence=None):
    """Build only one top-level array record at a time, regardless of key order."""
    import ijson
    from ijson.common import ObjectBuilder

    builder = None
    active = None
    depth = 0
    roots = set()
    arrays = set()
    with path.open("rb") as handle:
        events = ijson.parse(handle, use_float=True)
        first = next(events, None)
        if first != ("", "start_map", None):
            raise ValueError("Annotation JSON must be a top-level object.")
        for prefix, event, value in events:
            if prefix == "" and event == "map_key":
                if value in roots:
                    raise ValueError(f"Duplicate top-level JSON key: {value}")
                roots.add(value)
            if prefix in SECTIONS and event == "start_array":
                arrays.add(prefix)
            if evidence is not None and event in {"number", "string", "boolean", "null"}:
                if any(TIMING.search(part) for part in prefix.split(".")):
                    row = evidence.setdefault(prefix, {"occurrences": 0, "examples": []})
                    row["occurrences"] += 1
                    sample = compact(value)
                    if len(row["examples"]) < 3 and sample not in row["examples"]:
                        row["examples"].append(sample)
            if builder is None:
                parts = prefix.split(".")
                if len(parts) == 2 and parts[0] in SECTIONS and parts[1] == "item":
                    if event != "start_map":
                        raise ValueError(f"Expected an object at {prefix}, got {event}.")
                    active = parts[0]
                    builder = ObjectBuilder()
                    depth = 0
            if builder is not None:
                builder.event(event, value)
                if event in {"start_map", "start_array"}:
                    depth += 1
                elif event in {"end_map", "end_array"}:
                    depth -= 1
                if depth == 0:
                    yield active, builder.value
                    builder = None
                    active = None
        missing = REQUIRED - arrays
        if missing:
            raise ValueError(f"Required top-level arrays missing or invalid: {sorted(missing)}")


def safe_image_path(image_root, name):
    rel = PurePosixPath(normalized_path(name))
    if rel.is_absolute() or ".." in rel.parts or ":" in str(rel):
        return None
    candidate = (image_root / Path(*rel.parts)).resolve()
    return candidate if candidate.is_relative_to(image_root) else None


def inspect_classes(path, categories):
    if not path.is_file():
        return {"path": str(path), "found": False}
    if path.stat().st_size > 2 * 1024 * 1024:
        return {"path": str(path), "found": True, "skipped": "Text file exceeds 2 MiB."}
    raw = path.read_text(encoding="utf-8-sig").splitlines()
    entries = [s.strip() for s in raw if s.strip() and not s.lstrip().startswith("#")]
    normalize = lambda s: " ".join(s.casefold().split())
    names = Counter(normalize(s) for s in entries)
    json_names = {normalize(str(c.get("name", ""))) for c in categories.values()}
    return {
        "path": str(path), "found": True, "line_count": len(raw),
        "nonblank_noncomment_lines": len(entries), "unique_normalized_lines": len(names),
        "duplicate_lines": {k: v for k, v in names.items() if v > 1},
        "first_lines": entries[:LIMIT],
        "text_only_count": len(set(names) - json_names),
        "text_only_examples": sorted(set(names) - json_names)[:30],
        "json_only_count": len(json_names - set(names)),
        "json_only_examples": sorted(json_names - set(names))[:30],
        "comparison_rule": "Literal names ignoring case and whitespace; no synonym or numeric-prefix rewriting.",
        "warning": "Text line positions are NOT dataset category IDs. Differences may be formatting or aliases.",
    }


def sidecar_evidence(root, image_root, sequence_dir, annotation_path):
    """Inspect only small text sidecars in four specified directories; never execute .py/.pt."""
    candidates = set()
    for folder in {root, image_root, sequence_dir.parent, sequence_dir}:
        if folder.is_dir():
            candidates.update(p for p in folder.iterdir() if p.is_file())
    findings = []
    inspected = []
    suffixes = {".txt", ".ini", ".json", ".yaml", ".yml", ".csv", ".md"}
    for path in sorted(candidates):
        if path == annotation_path or path.suffix.lower() not in suffixes:
            continue
        if path.stat().st_size > 256 * 1024:
            continue
        inspected.append(str(path))
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (UnicodeError, OSError):
            continue
        matches = [{"line": i, "text": line[:400]} for i, line in enumerate(lines, 1)
                   if TIMING.search(line)]
        if matches:
            findings.append({"file": str(path), "matching_line_count": len(matches),
                             "examples": matches[:LIMIT]})
    return {"inspected_files": inspected, "findings": findings,
            "scope": "Root, image root, selected source directory and selected sequence directory only; <=256 KiB text files."}


def inspect(root, annotation_path, image_root, sequence, classes_path):
    started = time.perf_counter()
    original_stat = annotation_path.stat()
    counts = Counter()
    duplicate_ids = Counter()
    schemas = defaultdict(set)
    sources = Counter()
    categories, videos, tracks, image_video = {}, {}, {}, {}
    selected_images = []
    used_categories = Counter()
    evidence = {}
    say("Pass 1/2: streaming annotation records and building compact indexes...")
    for section, row in stream_records(annotation_path, evidence):
        counts[section] += 1
        schemas[section].update(row)
        if section in {"categories", "videos", "tracks", "images"}:
            rid = row.get("id")
            if not isinstance(rid, int) or isinstance(rid, bool):
                raise ValueError(f"Missing/non-integer ID in {section}: {compact(row)}")
            target = {"categories": categories, "videos": videos, "tracks": tracks,
                      "images": image_video}[section]
            if rid in target:
                duplicate_ids[section] += 1
            target[rid] = row.get("video_id") if section == "images" else row
        if section == "images":
            filename = normalized_path(row.get("file_name", ""))
            sources[filename.split("/")[0]] += 1
            if str(PurePosixPath(filename).parent) == sequence:
                selected_images.append(row)
        elif section == "annotations":
            used_categories[row.get("category_id")] += 1
            if counts[section] % 100000 == 0:
                say(f"  {counts[section]:,} annotations scanned")

    if not selected_images:
        raise ValueError(f"No image records found for {sequence!r}. Known image sources: {dict(sources)}")
    selected_vids = {r.get("video_id") for r in selected_images}
    selected_ids = {r["id"] for r in selected_images}
    selected_tracks = {tid: row for tid, row in tracks.items() if row.get("video_id") in selected_vids}
    warnings = []
    if len(selected_vids) != 1:
        warnings.append("Selected folder maps to multiple video IDs; do not convert until resolved.")
    indices_ok = all(isinstance(r.get("frame_index"), int) and not isinstance(r["frame_index"], bool)
                     for r in selected_images)
    if indices_ok:
        selected_images.sort(key=lambda r: (r["frame_index"], r["id"]))
    else:
        warnings.append("Some frame_index values are missing/non-integer; ordering is NOT verified.")
    index_counts = Counter(r.get("frame_index") for r in selected_images)
    duplicate_indices = [k for k, v in index_counts.items() if v > 1]
    if duplicate_indices:
        warnings.append("Selected sequence has duplicate frame indices.")
    deltas = Counter(b["frame_index"] - a["frame_index"] for a, b in zip(selected_images, selected_images[1:])) if indices_ok else {}
    if any(k != 1 for k in deltas):
        warnings.append("Frame-index increments are not uniformly 1; this alone does not prove missing files.")

    missing, unsafe, zero_bytes = [], [], []
    for row in selected_images:
        path = safe_image_path(image_root, row["file_name"])
        if path is None:
            unsafe.append(row["file_name"])
        elif not path.is_file():
            missing.append(row["file_name"])
        elif path.stat().st_size == 0:
            zero_bytes.append(row["file_name"])
    sequence_dir = image_root / sequence
    disk_jpegs = {p.name for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}} if sequence_dir.is_dir() else set()
    referenced_names = {PurePosixPath(normalized_path(r["file_name"])).name for r in selected_images}
    if missing or unsafe or zero_bytes:
        warnings.append("Selected sequence has missing, empty or unsafe image paths; see file_checks.")

    say("Pass 2/2: checking annotation references and summarizing selected-sequence boxes...")
    errors = Counter()
    error_examples = defaultdict(list)
    seq_categories = Counter()
    seq_annotations = 0
    ann_examples = []
    frames_with_boxes = set()

    def issue(key, row):
        errors[key] += 1
        if len(error_examples[key]) < 3:
            error_examples[key].append(compact(row))

    for section, ann in stream_records(annotation_path):
        if section != "annotations":
            continue
        iid, tid, cid = ann.get("image_id"), ann.get("track_id"), ann.get("category_id")
        if iid not in image_video:
            issue("unknown_image_id", ann)
        if tid not in tracks:
            issue("unknown_track_id", ann)
        if cid not in categories:
            issue("unknown_category_id", ann)
        if iid in image_video and "video_id" in ann and ann["video_id"] != image_video[iid]:
            issue("annotation_image_video_mismatch", ann)
        if tid in tracks:
            if cid != tracks[tid].get("category_id"):
                issue("annotation_track_category_mismatch", ann)
            if iid in image_video and image_video[iid] != tracks[tid].get("video_id"):
                issue("annotation_track_video_mismatch", ann)
        bbox = ann.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4 and
                all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox) and
                bbox[2] > 0 and bbox[3] > 0):
            issue("invalid_bbox_shape_or_size", ann)
        if iid in selected_ids:
            seq_annotations += 1
            frames_with_boxes.add(iid)
            seq_categories[cid] += 1
            if len(ann_examples) < LIMIT:
                ann_examples.append(compact(ann))
    if errors:
        warnings.append("Annotation reference/box checks found issues; inspect integrity_checks.")
    if duplicate_ids:
        warnings.append("Duplicate record IDs found; indexes retain the last record. Do not evaluate until resolved.")
    for vid in selected_vids:
        if vid not in videos:
            warnings.append(f"Selected sequence references missing video record {vid}.")

    selected_video_records = [videos[v] for v in sorted(selected_vids) if v in videos]
    sidecars = sidecar_evidence(root, image_root, sequence_dir, annotation_path)
    if annotation_path.stat().st_size != original_stat.st_size or annotation_path.stat().st_mtime_ns != original_stat.st_mtime_ns:
        raise ValueError("Annotation file changed during inspection. Run again with a stable input.")
    timing_status = "candidate_fields_found_requires_review" if evidence or sidecars["findings"] else "not_found_in_inspected_scope"
    warnings.append("No FPS has been verified. No real-world timestamps or durations were inferred.")
    warnings.append("Filesystem checks cover the selected sequence only; JPEG decoding/corruption checks and full-dataset completeness are not tested.")
    warnings.append("Missing boxes are not proof of absence. Inspect neg_category_ids and not_exhaustive_category_ids before evaluation.")
    return {
        "report_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "inspection_complete", "elapsed_seconds": round(time.perf_counter() - started, 3),
        "inputs": {"root": str(root), "annotation_path": str(annotation_path), "image_root": str(image_root),
                   "annotation_bytes": original_stat.st_size, "annotation_mib": round(original_stat.st_size / 1024**2, 2),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "dataset": {"counts": {k: counts[k] for k in sorted(SECTIONS)}, "duplicate_record_ids": dict(duplicate_ids),
                    "record_fields": {k: sorted(v) for k, v in schemas.items()}, "images_by_source": dict(sources),
                    "unique_annotated_category_ids": len(used_categories),
                    "unused_category_ids": sorted(set(categories) - set(used_categories)),
                    "category_examples": [compact(c) for c in list(categories.values())[:LIMIT]],
                    "text_classes": inspect_classes(classes_path, categories)},
        "selected_sequence": {
            "name": sequence, "video_ids": sorted(selected_vids), "video_records": selected_video_records,
            "image_count": len(selected_images), "annotation_count": seq_annotations,
            "track_count": len(selected_tracks), "frames_with_annotations": len(frames_with_boxes),
            "frames_without_annotations": len(selected_ids - frames_with_boxes),
            "ordering": {"sorted_by": "frame_index_then_image_id" if indices_ok else "input_order_unverified",
                         "frame_index_step_counts": {str(k): v for k, v in deltas.items()},
                         "duplicate_frame_index_count": len(duplicate_indices), "duplicate_frame_index_examples": duplicate_indices[:LIMIT]},
            "dimension_counts": dict(Counter(f"{r.get('width')}x{r.get('height')}" for r in selected_images)),
            "first_images": selected_images[:5], "last_images": selected_images[-5:],
            "category_summary": [{"category_id": k, "name": categories.get(k, {}).get("name"), "boxes": v}
                                 for k, v in seq_categories.most_common()],
            "track_examples": [compact(t) for t in list(selected_tracks.values())[:LIMIT]],
            "annotation_examples": ann_examples,
            "file_checks": {"scope": "selected sequence only; existence and nonzero size, not image decoding",
                            "missing_count": len(missing), "missing_examples": missing[:LIMIT],
                            "unsafe_path_count": len(unsafe), "unsafe_examples": unsafe[:LIMIT],
                            "empty_file_count": len(zero_bytes), "empty_examples": zero_bytes[:LIMIT],
                            "jpeg_files_in_folder": len(disk_jpegs),
                            "unreferenced_jpeg_count": len(disk_jpegs - referenced_names),
                            "unreferenced_jpeg_examples": sorted(disk_jpegs - referenced_names)[:LIMIT]},
        },
        "integrity_checks": {"annotation_issue_counts": dict(errors), "examples": dict(error_examples),
                             "scope": "All annotation image/track/category references and bbox shape/positive size; not a complete benchmark validation."},
        "timing": {"status": timing_status, "verified_fps": None, "json_field_evidence": evidence,
                   "sidecar_evidence": sidecars,
                   "note": "Candidate fields can belong to another sequence or mean something other than frame timing. Human review required."},
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Dataset folder containing ovtb_ann.json and OVT-B/ (default: script folder)")
    parser.add_argument("--sequence", default="AnimalTrack/chicken_1", help="Sequence path relative to image root")
    parser.add_argument("--annotations", type=Path, help="Annotation path, relative to root unless absolute")
    parser.add_argument("--image-root", type=Path, help="Image directory, relative to root unless absolute")
    parser.add_argument("--classes-file", type=Path, help="Classes text path, relative to root unless absolute")
    parser.add_argument("--output", type=Path, help="NEW JSON file; relative paths are relative to root")
    args = parser.parse_args()
    try:
        import ijson  # noqa: F401
    except ImportError:
        parser.exit(1, "Missing dependency. Run: python -m pip install ijson\n")
    root = args.root.expanduser().resolve()
    resolve = lambda p, default: (root / (p or Path(default))).resolve()
    annotation_path = resolve(args.annotations, "ovtb_ann.json")
    image_root = resolve(args.image_root, "OVT-B")
    classes_path = resolve(args.classes_file, "ovtb_classes.txt")
    sequence = normalized_path(args.sequence).rstrip("/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = resolve(args.output, f"inspection_reports/ovtb_inspection_{stamp}.json")
    if not root.is_dir() or not annotation_path.is_file() or not image_root.is_dir():
        parser.exit(1, f"Check root, annotation and image paths:\n{root}\n{annotation_path}\n{image_root}\n")
    if safe_image_path(image_root, sequence) is None or sequence in {"", "."}:
        parser.exit(1, "Sequence must be a safe relative directory path inside the image root.\n")
    if output.exists() or output.is_relative_to(image_root):
        parser.exit(1, "Output must be a NEW file outside the image directory. Nothing overwritten.\n")
    try:
        result = inspect(root, annotation_path, image_root, sequence, classes_path)
        payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects against another process creating it meanwhile.
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"Report saved: {output}")
        print("Counts:", result["dataset"]["counts"])
        print("Sequence:", sequence, "| frames:", result["selected_sequence"]["image_count"])
        print("Timing:", result["timing"]["status"], "| verified FPS: unknown")
        print("Share the JSON report. Warnings and limitations are included inside it.")
    except KeyboardInterrupt:
        parser.exit(1, "\nInspection interrupted; source data unchanged.\n")
    except Exception as exc:
        parser.exit(1, f"Inspection failed: {type(exc).__name__}: {exc}\nSource data unchanged.\n")


if __name__ == "__main__":
    main()
