#!/usr/bin/env python3
"""Optimised batch conversion of OVT-B JPEG sequences (WSL/Linux).

Python >=3.10; python -m pip install ijson Pillow; ffmpeg/ffprobe with libx264.
Save beside ovtb_ann.json and OVT-B/. Example:
  python prepare_ovtb_batch_optimised.py --root . --fps 30 --resume

Previews are OFF by default; --previews enables first/middle/last annotated PNGs
for newly converted videos. Existing outputs are preserved (no preview backfill).
Metadata is collected in one streaming pass with byte-progress messages. Optional
preview annotation lookup needs one additional pass. Verification and optional
preview extraction share ONE full decode. Encoding settings stay unchanged.
Stage timings and a 20-second activity message distinguish work from silence.
On resume, source JPEG hashes are checked. A matching saved MP4 SHA-256 plus a
previous successful full verification avoids redundant decoding; --reverify
forces full decoding again. Older manifests without a video hash are decoded.

Outputs: prepared_videos/<source>/<sequence>/{name.mp4,name.manifest.json,previews/}
Progress summaries: conversion_reports/batch_<UTC>.json (updated after each video).
--resume verifies existing outputs and source hashes before skipping. Invalid or
unrecognized existing outputs are reported as failures and NEVER overwritten.
Use a new --output-root for a fresh conversion. --limit N selects the first N
sequences, ordered by name; --sequence SOURCE/NAME selects a single sequence.

All FPS values are ASSUMED playback rates. Original timestamps remain unknown.
Every available frame is included once, ordered by frame_index, even when source
indices have gaps. Gaps are recorded; no missing source frames are invented.
Original pixels are not resized. Odd or varying dimensions are padded in black
on right/bottom to an even canvas. Source dimensions/placement are in the manifest.
MP4: H.264 CRF 18, yuv420p, no audio, sequential CPU encoding. Ground-truth preview
PNGs are separate; the MP4 has no boxes. Annotation JSON is scanned a fixed number
of times per batch, not per video. Only image metadata and preview boxes are indexed.

Ctrl+C records interruption; the active incomplete conversion is cleaned up.
Completed outputs survive and can be resumed. A power loss can leave an incomplete
folder; it is reported on resume, never silently deleted. Exit 0: all selected
sequences complete; 1: setup/sequence failures; 130: interrupted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager


@contextmanager
def stage(timings, name):
    started = time.monotonic()
    done = threading.Event()
    def heartbeat():
        while not done.wait(20):
            say(f'  {name}: still working ({time.monotonic() - started:.0f}s)')
    say(f'  {name}...')
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    finally:
        done.set()
        worker.join()
        seconds = time.monotonic() - started
        timings[name + '_seconds'] = round(seconds, 3)
        say(f'  {name}: {seconds:.2f}s')


def say(message):
    print(message, flush=True)


def records(path, section):
    import ijson
    with path.open('rb') as stream:
        yield from ijson.items(stream, section + '.item', use_float=True)


def integer(value, label):
    if type(value) is not int:
        raise ValueError(f'{label} must be an integer, got {value!r}')
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(base, name):
    path = Path(name).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def relative_name(value):
    if not isinstance(value, str):
        raise ValueError('Sequence and image filenames must be strings.')
    name = PurePosixPath(value.replace('\\', '/'))
    if not name.parts or name.is_absolute() or '..' in name.parts or ':' in str(name):
        raise ValueError(f'Unsafe relative path: {value!r}')
    return name


def run(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        process.kill()
        process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(f'{command[0]} failed:\n{stderr[-4000:]}')
    return stdout


def encode(images, image_root, destination, size, fps):
    from PIL import Image
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
               '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size',
               f'{size[0]}x{size[1]}', '-framerate', str(fps), '-i', 'pipe:0',
               '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough',
               '-movflags', '+faststart', str(destination)]
    mapping = []
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for index, row in enumerate(images):
                name = relative_name(row['file_name'])
                source = (image_root / str(name)).resolve()
                if not source.is_relative_to(image_root):
                    raise ValueError(f'Image path escapes image root: {name}')
                raw = source.read_bytes()
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != 'JPEG' or image.size != (row['width'], row['height']):
                        raise ValueError(f'{name}: expected JPEG dimensions {size}, got {image.format} {image.size}')
                    image.load()  # Fully decode; truncated/corrupt input must fail.
                    # Do not EXIF-transpose: annotation coordinates use stored pixels.
                    canvas = image.convert('RGB')
                    if canvas.size != size:
                        padded = Image.new('RGB', size, 'black')
                        padded.paste(canvas, (0, 0))
                        canvas = padded
                    process.stdin.write(canvas.tobytes())
                mapping.append({
                    'video_frame_index': index, 'image_id': row['id'],
                    'source_frame_index': row['frame_index'],
                    'source_frame_id': row.get('frame_id'),
                    'source_width': row['width'], 'source_height': row['height'],
                    'placement_xy': [0, 0], 'file_name': str(name), 'source_jpeg_sha256': hashlib.sha256(raw).hexdigest(),
                    'playback_timestamp_seconds': float(Fraction(index, 1) / fps),
                    'source_timestamp_seconds': None,
                })
                if (index + 1) % 100 == 0 or index + 1 == len(images):
                    say(f'  Encoded {index + 1}/{len(images)} frames')
            process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError('Encoding failed: ' + errors.read().decode(errors='replace')[-4000:])
        except BaseException as exc:
            if process.poll() is None:
                process.kill()
            process.wait()
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            if isinstance(exc, BrokenPipeError):
                errors.seek(0)
                raise RuntimeError('FFmpeg stopped: ' + errors.read().decode(errors='replace')[-4000:]) from exc
            raise
    return mapping


def verify(video, expected_count, size, fps, preview_folder=None, selected=None):
    data = json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-show_streams', '-show_format', '-of', 'json', str(video)]))
    stream = data['streams'][0]
    if (stream['width'], stream['height']) != size:
        raise ValueError('Encoded video dimensions do not match source images.')
    if Fraction(stream['avg_frame_rate']) != fps:
        raise ValueError('Encoded video frame rate differs from requested playback rate.')
    # Decode every output frame and inspect its presentation timestamp.
    command = ['ffmpeg', '-hide_banner', '-v', 'error', '-xerror', '-n', '-i', str(video)]
    if preview_folder is not None:
        preview_folder.mkdir()
        expression = '+'.join(f'eq(n,{index})' for index in selected)
        command += ['-filter_complex',
                    f"[0:v]split=2[check][preview];[preview]select='{expression}'[chosen]",
                    '-map', '[check]', '-an', '-fps_mode', 'passthrough', '-f', 'framemd5', '-',
                    '-map', '[chosen]', '-fps_mode', 'vfr', str(preview_folder / 'decoded_%03d.png')]
    else:
        command += ['-map', '0:v:0', '-an', '-fps_mode', 'passthrough', '-f', 'framemd5', '-']
    decoded = run(command)
    rows = [line.split(',') for line in decoded.splitlines() if line and not line.startswith('#')]
    if len(rows) != expected_count:
        raise ValueError(f'Decoded {len(rows)} frames; expected {expected_count}.')
    timebase = next((line.split(':', 1)[1].strip() for line in decoded.splitlines()
                     if line.startswith('#tb 0:')), None)
    if timebase is None:
        raise ValueError('Decoder did not report a time base.')
    for index, row in enumerate(rows):
        if Fraction(int(row[2])) * Fraction(timebase) != Fraction(index, 1) / fps:
            raise ValueError(f'Unexpected decoded timestamp at output frame {index}.')
    return {'fully_decoded_frames': len(rows), 'dimensions': list(size),
            'avg_frame_rate': stream['avg_frame_rate'], 'ordered_timestamps_checked': True,
            'container_duration_seconds': float(data['format']['duration']),
            'codec_name': stream['codec_name'], 'pixel_format': stream['pix_fmt']}


def previews(video, folder, selected, boxes, categories, size):
    from PIL import Image, ImageDraw
    # PNGs were already extracted during verify(); no second video decode.
    results = []
    for counter, (index, row) in enumerate(selected.items(), 1):
        source = folder / f'decoded_{counter:03d}.png'
        destination = folder / f'frame_{index:06d}_ground_truth.png'
        image_boxes = boxes[row['id']]
        size = (row['width'], row['height'])
        outside = 0
        with Image.open(source) as image:
            image = image.convert('RGB')
            draw = ImageDraw.Draw(image)
            for ann in image_boxes:
                x, y, w, h = ann['bbox']
                if x < 0 or y < 0 or x + w > size[0] or y + h > size[1]:
                    outside += 1
                # Clip drawings ONLY; never change original annotation coordinates.
                left, top = max(0, x), max(0, y)
                right, bottom = min(size[0] - 1, x + w), min(size[1] - 1, y + h)
                if right < left or bottom < top:
                    continue
                draw.rectangle((left, top, right, bottom), outline='lime', width=3)
                label = f"{categories.get(ann['category_id'], ann['category_id'])} track={ann.get('track_id', '?')}"
                label = label.encode('ascii', 'replace').decode()
                draw.text((left + 2, top + 2), label, fill='yellow', stroke_width=1, stroke_fill='black')
            image.save(destination)
        source.unlink()
        results.append({'video_frame_index': index, 'image_id': row['id'],
                        'file_name': 'previews/' + destination.name,
                        'ground_truth_box_count': len(image_boxes),
                        'boxes_outside_image_bounds': outside})
    return results


def collect_metadata(annotation):
    """One pass, arbitrary top-level key order, no annotation objects in memory."""
    import ijson
    from ijson.common import ObjectBuilder
    wanted = {'videos', 'images', 'categories'}
    output = {key: [] for key in wanted}
    arrays, keys = set(), set()
    digest = hashlib.sha256()
    total = annotation.stat().st_size
    class Reader:
        def __init__(self, handle):
            self.handle, self.consumed, self.last = handle, 0, time.monotonic()
        def read(self, size=-1):
            data = self.handle.read(size)
            digest.update(data)
            self.consumed += len(data)
            now = time.monotonic()
            if now - self.last >= 5:
                say(f'  Metadata read: {self.consumed / 2**20:.1f}/{total / 2**20:.1f} MiB')
                self.last = now
            return data
        def readinto(self, buffer):
            data = self.read(len(buffer))
            buffer[:len(data)] = data
            return len(data)
    builder, active, depth = None, None, 0
    with annotation.open('rb') as handle:
        events = ijson.parse(Reader(handle), use_float=True)
        if next(events, None) != ('', 'start_map', None):
            raise ValueError('Annotation JSON must contain a top-level object.')
        for prefix, event, value in events:
            if prefix == '' and event == 'map_key':
                if value in keys:
                    raise ValueError(f'Duplicate top-level JSON key: {value}')
                keys.add(value)
            if prefix in wanted and event == 'start_array':
                arrays.add(prefix)
            if builder is None and prefix in ('videos.item', 'images.item', 'categories.item'):
                if event != 'start_map':
                    raise ValueError(f'Expected object at {prefix}')
                builder, active, depth = ObjectBuilder(), prefix.split('.')[0], 0
            if builder is not None:
                builder.event(event, value)
                if event in ('start_map', 'start_array'):
                    depth += 1
                elif event in ('end_map', 'end_array'):
                    depth -= 1
                if depth == 0:
                    output[active].append(builder.value)
                    builder = None
    if wanted - arrays:
        raise ValueError(f'Missing metadata arrays: {sorted(wanted - arrays)}')
    say(f'  Metadata read complete: {total / 2**20:.1f} MiB')
    return output, digest.hexdigest()


def build_index(annotation, wanted, limit, with_previews=False):
    from collections import defaultdict
    metadata, annotation_hash = collect_metadata(annotation)
    videos, names = {}, set()
    for video in metadata['videos']:
        vid = integer(video['id'], 'video.id')
        name = str(relative_name(video['name']))
        if vid in videos or name in names:
            raise ValueError('Duplicate video IDs or names in annotation JSON.')
        videos[vid] = video
        names.add(name)
    if not videos:
        raise ValueError('No video records found.')
    selected = sorted(videos.values(), key=lambda row: row['name'])
    if wanted:
        selected = [video for video in selected if video['name'] == wanted]
        if not selected:
            raise ValueError(f'Sequence not found: {wanted}')
    if limit:
        selected = selected[:limit]
    ids = {video['id'] for video in selected}
    images = defaultdict(list)
    seen_images = set()
    for row in metadata['images']:
        if row.get('video_id') not in ids:
            continue
        image_id = integer(row['id'], 'image.id')
        if image_id in seen_images:
            raise ValueError(f'Duplicate image ID: {image_id}')
        seen_images.add(image_id)
        images[row['video_id']].append(row)
    preview_ids = set()
    index = []
    for video in selected:
        rows = images[video['id']]
        error, size, chosen, gaps = None, None, {}, []
        try:
            if not rows:
                raise ValueError('No image records in sequence.')
            for row in rows:
                for key in ('frame_index', 'width', 'height'):
                    integer(row[key], f'image.{key}')
                if min(row['width'], row['height']) <= 0:
                    raise ValueError('Image dimensions must be positive.')
                relative_name(row['file_name'])
            rows.sort(key=lambda row: row['frame_index'])
            if len({row['file_name'] for row in rows}) != len(rows):
                raise ValueError('Duplicate source filenames in sequence.')
            for previous, current in zip(rows, rows[1:]):
                step = current['frame_index'] - previous['frame_index']
                if step <= 0:
                    raise ValueError('Duplicate or reversed frame indices.')
                if step != 1:
                    gaps.append({'after_image_id': previous['id'], 'before_image_id': current['id'],
                                 'source_frame_index_step': step})
            width, height = max(r['width'] for r in rows), max(r['height'] for r in rows)
            size = (width + width % 2, height + height % 2)
            chosen = {i: rows[i] for i in sorted({0, len(rows)//2, len(rows)-1})}
            preview_ids.update(row['id'] for row in chosen.values())
        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)
        index.append(dict(video=video, images=rows, size=size, chosen=chosen, gaps=gaps, error=error))
    categories = {row['id']: row['name'] for row in metadata['categories']}
    boxes = {image_id: [] for image_id in preview_ids}
    if with_previews:
        say('  Reading annotations for optional previews (one additional pass)...')
        for count, ann in enumerate(records(annotation, 'annotations'), 1):
            if ann.get('image_id') in preview_ids:
                boxes[ann['image_id']].append(ann)
            if count % 200000 == 0:
                say(f'  Preview annotations: {count:,} records scanned')
    return index, boxes, categories, len(videos), annotation_hash


def validate_boxes(entry, boxes):
    for row in entry['chosen'].values():
        for ann in boxes[row['id']]:
            bbox = ann.get('bbox')
            if ann.get('video_id') != entry['video']['id']:
                raise ValueError('Preview annotation video ID differs from image video ID.')
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)
                    or bbox[2] <= 0 or bbox[3] <= 0):
                raise ValueError(f"Invalid preview box in annotation {ann.get('id')}")


def checked_source(image_root, row):
    source = (image_root / str(relative_name(row['file_name']))).resolve()
    if not source.is_relative_to(image_root):
        raise ValueError(f"Source escapes image directory: {row['file_name']}")
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f'Missing or empty source JPEG: {source}')
    return source


def resume_existing(folder, entry, context):
    """Support both the earlier single-video manifest and batch manifests."""
    name = PurePosixPath(entry['video']['name']).name
    manifest = json.loads((folder / (name + '.manifest.json')).read_text(encoding='utf-8'))
    if (manifest.get('status') != 'complete'
            or manifest.get('sequence') != entry['video']['name']
            or manifest.get('video_id') != entry['video']['id']
            or manifest.get('source', {}).get('annotation_sha256') != context['annotation_sha256']):
        raise ValueError('Existing manifest is incomplete or belongs to different source annotations.')
    timing = manifest['timing']
    if timing.get('playback_fps_status') != 'assumed' or Fraction(timing['playback_fps']) != context['fps']:
        raise ValueError('Existing output has different playback timing settings.')
    conversion = manifest['conversion']
    if ((conversion['width'], conversion['height']) != entry['size']
            or conversion['frame_count'] != len(entry['images'])
            or conversion.get('crf') != 18 or conversion.get('pixel_format') != 'yuv420p'
            or conversion.get('codec') != 'libx264' or conversion.get('preset') != 'medium'):
        raise ValueError('Existing output has different conversion settings.')
    frames = manifest['frames']
    if len(frames) != len(entry['images']):
        raise ValueError('Existing frame mapping has a different length.')
    for index, (frame, row) in enumerate(zip(frames, entry['images'])):
        if (frame['video_frame_index'] != index or frame['image_id'] != row['id']
                or frame['source_frame_index'] != row['frame_index']
                or frame['file_name'] != str(relative_name(row['file_name']))
                or frame['source_jpeg_sha256'] != sha256(checked_source(context['image_root'], row))):
            raise ValueError(f'Existing mapping/source mismatch at frame {index}.')
    video = folder / (name + '.mp4')
    if conversion.get('video_sha256') and sha256(video) != conversion['video_sha256']:
        raise ValueError('Existing video hash does not match the manifest.')
    prior = manifest.get('verification', {})
    # Rational rates are also supported; do not rely on one string representation.
    try:
        prior_verified = (prior.get('fully_decoded_frames') == len(frames)
                          and prior.get('dimensions') == list(entry['size'])
                          and prior.get('ordered_timestamps_checked') is True
                          and Fraction(prior.get('avg_frame_rate', '0')) == context['fps'])
    except (ValueError, ZeroDivisionError):
        prior_verified = False
    if context['reverify'] or not (conversion.get('video_sha256') and prior_verified):
        say('  Existing video needs full decode verification...')
        verify(video, len(frames), entry['size'], context['fps'])
    else:
        say('  MP4 hash matches previously verified output; repeat decoding avoided.')
    # Existing previews are optional and are never overwritten or backfilled.
    return 'skipped_verified'


def convert_one(entry, boxes, categories, context, resume):
    if entry['error']:
        raise ValueError(entry['error'])
    if context['previews']:
        validate_boxes(entry, boxes)
    sequence = str(relative_name(entry['video']['name']))
    destination = (context['output_root'] / sequence).resolve()
    if not destination.is_relative_to(context['output_root']):
        raise ValueError('Output sequence path escapes output root.')
    if destination.exists():
        if not resume:
            raise ValueError('Output already exists; use --resume to verify and skip it.')
        with stage(context['stage_timings'], 'resume_validation'):
            return resume_existing(destination, entry, context)
    with stage(context['stage_timings'], 'source_file_checks'):
        for row in entry['images']:
            checked_source(context['image_root'], row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    successful = False
    name = PurePosixPath(sequence).name
    try:
        with tempfile.TemporaryDirectory(prefix='ovtb-working-', dir=destination) as working:
            staging = Path(working)
            output_video = staging / (name + '.mp4')
            with stage(context['stage_timings'], 'encode'):
                mapping = encode(entry['images'], context['image_root'], output_video, entry['size'], context['fps'])
            preview_folder = staging / 'previews' if context['previews'] else None
            with stage(context['stage_timings'], 'verify_and_extract'):
                checks = verify(output_video, len(mapping), entry['size'], context['fps'],
                                preview_folder, entry['chosen'])
            preview_info = []
            if context['previews']:
                with stage(context['stage_timings'], 'draw_previews'):
                    preview_info = previews(output_video, preview_folder, entry['chosen'], boxes, categories, entry['size'])
            with stage(context['stage_timings'], 'video_hash'):
                video_hash = sha256(output_video)
            padding = any((row['width'], row['height']) != entry['size'] for row in entry['images'])
            manifest = {
                'schema_version': '1.2', 'status': 'complete',
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'sequence': sequence, 'video_id': entry['video']['id'],
                'source': {'annotation_path': str(context['annotation']),
                           'annotation_sha256': context['annotation_sha256'],
                           'image_root': str(context['image_root']), 'video_record': entry['video']},
                'conversion': {'video_file': output_video.name, 'video_sha256': video_hash,
                               'width': entry['size'][0], 'height': entry['size'][1],
                               'frame_count': len(mapping), 'codec': 'libx264', 'crf': 18,
                               'preset': 'medium', 'pixel_format': 'yuv420p', 'audio': False,
                               'resized': False, 'frames_interpolated': False,
                               'padding_applied': padding,
                               'padding_policy': 'Black right/bottom padding; source pixel origin remains (0,0).',
                               'ffmpeg_version': context['ffmpeg_version'],
                               'script_sha256': context['script_sha256']},
                'timing': {'playback_fps': str(context['fps']), 'playback_fps_status': 'assumed',
                           'verified_source_fps': None, 'source_duration_seconds': None,
                           'expected_playback_duration_seconds': float(Fraction(len(mapping)) / context['fps']),
                           'note': 'One available JPEG per playback frame; no original time intervals inferred.',
                           'source_frame_index_gaps': entry['gaps']},
                'verification': checks, 'frames': mapping, 'ground_truth_previews': preview_info,
                'previews_enabled': context['previews'], 'stage_timings': dict(context['stage_timings']),
                'limitations': ['Reconstructed lossy MP4; not the original recording or codec.',
                                'Assumed playback FPS; use frame indices until source timing is verified.',
                                'Previews are original dataset ground truth, not model predictions.',
                                'Out-of-bounds boxes are clipped for drawing only; no source labels are changed.',
                                'Visual annotation alignment needs human review.',
                                'Conversion does not score detections or establish annotation completeness.'],
            }
            (staging / (name + '.manifest.json')).write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
            for child in staging.iterdir():
                child.rename(destination / child.name)
        successful = True
    finally:
        if not successful:
            shutil.rmtree(destination)
    return 'converted'


def write_report(path, report):
    from collections import Counter
    report['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['counts'] = dict(Counter(row['status'] for row in report['sequences']))
    temp_path = path.with_suffix('.tmp')
    temp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp_path.replace(path)


def batch(args):
    import time
    root = args.root.expanduser().resolve()
    annotation = resolve_input(root, args.annotations)
    image_root = resolve_input(root, args.image_root)
    output_root = resolve_input(root, args.output_root)
    report_root = resolve_input(root, 'conversion_reports')
    if not annotation.is_file() or not image_root.is_dir():
        raise ValueError('Expected ovtb_ann.json and OVT-B/ under --root; check input paths.')
    for folder in (output_root, report_root):
        if (folder.is_relative_to(image_root) or image_root.is_relative_to(folder)
                or annotation.is_relative_to(folder)):
            raise ValueError('Output/report roots must be separate from source images and annotations.')
    if report_root == output_root or report_root.is_relative_to(output_root) or output_root.is_relative_to(report_root):
        raise ValueError('Output root must be separate from conversion_reports/.')
    fps = Fraction(args.fps)
    if fps <= 0 or fps > 1000:
        raise ValueError('--fps must be positive and no more than 1000.')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive.')
    for executable in ('ffmpeg', 'ffprobe'):
        if not shutil.which(executable):
            raise ValueError(f'{executable} not found on PATH.')
    if not any(line.split()[1:2] == ['libx264'] for line in run(['ffmpeg', '-hide_banner', '-encoders']).splitlines()):
        raise ValueError('FFmpeg must include libx264.')
    # Advisory file lock automatically releases on exit, including crashes (WSL/Linux).
    import fcntl
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / '.ovtb_conversion.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another batch conversion is using this output root.') from exc
        setup_timings = {}
        with stage(setup_timings, 'index_metadata'):
            index, boxes, categories, total, annotation_hash = build_index(
                annotation, args.sequence, args.limit, args.previews)
        context = {'annotation': annotation, 'image_root': image_root, 'output_root': output_root,
                   'annotation_sha256': annotation_hash, 'fps': fps,
                   'previews': args.previews, 'reverify': args.reverify, 'stage_timings': {},
                   'ffmpeg_version': run(['ffmpeg', '-version']).splitlines()[0],
                   'script_sha256': sha256(Path(__file__))}
        report_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        report_path = report_root / f'batch_optimised_{stamp}.json'
        report = {'schema_version': '1.0', 'status': 'running', 'dataset_video_count': total,
                  'selected_video_count': len(index), 'output_root': str(output_root),
                  'assumed_playback_fps': str(fps), 'annotation_sha256': context['annotation_sha256'],
                  'resume': args.resume, 'previews': args.previews,
                  'setup_timings': setup_timings, 'sequences': []}
        write_report(report_path, report)
        say(f'Selected {len(index):,}/{total:,} sequences. Free output space: {shutil.disk_usage(output_root).free / 2**30:.1f} GiB')
        say(f'Progress report: {report_path}')
        interrupted = False
        for position, entry in enumerate(index, 1):
            sequence = entry['video']['name']
            say(f'[{position}/{len(index)}] {sequence} | {len(entry["images"])} frames')
            started = time.monotonic()
            record = {'sequence': sequence, 'video_id': entry['video']['id'], 'frame_count': len(entry['images'])}
            context['stage_timings'] = {}
            try:
                record['status'] = convert_one(entry, boxes, categories, context, args.resume)
            except KeyboardInterrupt:
                record.update(status='interrupted', error='User interrupted; completed outputs preserved.')
                interrupted = True
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            record['elapsed_seconds'] = round(time.monotonic() - started, 3)
            record['stage_timings'] = dict(context['stage_timings'])
            report['sequences'].append(record)
            write_report(report_path, report)
            say(f'  {record["status"]}' + (f': {record["error"]}' if 'error' in record else ''))
            if interrupted:
                break
        failures = any(row['status'] == 'failed' for row in report['sequences'])
        report['status'] = 'interrupted' if interrupted else ('completed_with_failures' if failures else 'complete')
        write_report(report_path, report)
        say(f'Batch status: {report["status"]}; counts: {report["counts"]}')
        say(f'Report saved: {report_path}')
        return 130 if interrupted else (1 if failures else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--annotations', default='ovtb_ann.json')
    parser.add_argument('--image-root', default='OVT-B')
    parser.add_argument('--output-root', default='prepared_videos')
    parser.add_argument('--fps', default='30', help='ASSUMED playback FPS, e.g. 30 or 30000/1001')
    parser.add_argument('--resume', action='store_true', help='Verify complete existing outputs before skipping')
    parser.add_argument('--previews', action='store_true', help='Generate ground-truth PNGs for new conversions (default: off)')
    parser.add_argument('--reverify', action='store_true', help='Fully decode existing videos on resume even if verified hashes match')
    parser.add_argument('--sequence', help='Optional exact SOURCE/NAME; default is ALL sequences')
    parser.add_argument('--limit', type=int, help='Optional first N sequences ordered by name')
    args = parser.parse_args()
    try:
        import ijson  # noqa: F401
        import PIL  # noqa: F401
        return batch(args)
    except ImportError as exc:
        print(f'Missing dependency: {exc}. Use WSL/Linux; install Python packages with: python -m pip install ijson Pillow', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Interrupted during setup; completed videos are unchanged.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
