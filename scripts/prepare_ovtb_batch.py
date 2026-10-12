#!/usr/bin/env python3
"""Batch-convert OVT-B JPEG sequences. No model inference or dataset edits.

Python >=3.10; python -m pip install ijson Pillow; ffmpeg/ffprobe with libx264.
Save beside ovtb_ann.json and OVT-B/. Example:
  python prepare_ovtb_batch.py --root . --fps 30 --resume

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
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed:\n{result.stderr[-4000:]}')
    return result.stdout


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
                    canvas = Image.new('RGB', size, 'black')
                    canvas.paste(image.convert('RGB'), (0, 0))
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


def verify(video, expected_count, size, fps):
    data = json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-show_streams', '-show_format', '-of', 'json', str(video)]))
    stream = data['streams'][0]
    if (stream['width'], stream['height']) != size:
        raise ValueError('Encoded video dimensions do not match source images.')
    if Fraction(stream['avg_frame_rate']) != fps:
        raise ValueError('Encoded video frame rate differs from requested playback rate.')
    # Decode every output frame and inspect its presentation timestamp.
    decoded = run(['ffmpeg', '-hide_banner', '-v', 'error', '-xerror', '-i', str(video),
                   '-map', '0:v:0', '-an', '-fps_mode', 'passthrough', '-f', 'framemd5', '-'])
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
    folder.mkdir()
    expression = '+'.join(f'eq(n,{index})' for index in selected)
    run(['ffmpeg', '-hide_banner', '-v', 'error', '-xerror', '-n', '-i', str(video),
         '-vf', f"select='{expression}'", '-fps_mode', 'vfr', str(folder / 'decoded_%03d.png')])
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


def build_index(annotation, wanted, limit):
    from collections import defaultdict
    videos, names = {}, set()
    for video in records(annotation, 'videos'):
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
    for row in records(annotation, 'images'):
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
    categories = {row['id']: row['name'] for row in records(annotation, 'categories')}
    boxes = {image_id: [] for image_id in preview_ids}
    for count, ann in enumerate(records(annotation, 'annotations'), 1):
        if ann.get('image_id') in preview_ids:
            boxes[ann['image_id']].append(ann)
        if count % 200000 == 0:
            say(f'  Indexed preview annotations: {count:,} annotation records scanned')
    return index, boxes, categories, len(videos)


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
    from PIL import Image
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
            or conversion.get('codec') != 'libx264'):
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
    verify(video, len(frames), entry['size'], context['fps'])
    for frame_index in entry['chosen']:
        preview = folder / 'previews' / f'frame_{frame_index:06d}_ground_truth.png'
        with Image.open(preview) as image:
            image.load()
            if image.size != entry['size']:
                raise ValueError(f'Existing preview dimensions differ: {preview}')
    return 'skipped_verified'


def convert_one(entry, boxes, categories, context, resume):
    if entry['error']:
        raise ValueError(entry['error'])
    validate_boxes(entry, boxes)
    sequence = str(relative_name(entry['video']['name']))
    destination = (context['output_root'] / sequence).resolve()
    if not destination.is_relative_to(context['output_root']):
        raise ValueError('Output sequence path escapes output root.')
    if destination.exists():
        if not resume:
            raise ValueError('Output already exists; use --resume to verify and skip it.')
        return resume_existing(destination, entry, context)
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
            mapping = encode(entry['images'], context['image_root'], output_video, entry['size'], context['fps'])
            say('  Verifying complete output decoding and making ground-truth previews...')
            checks = verify(output_video, len(mapping), entry['size'], context['fps'])
            preview_info = previews(output_video, staging / 'previews', entry['chosen'], boxes, categories, entry['size'])
            padding = any((row['width'], row['height']) != entry['size'] for row in entry['images'])
            manifest = {
                'schema_version': '1.1', 'status': 'complete',
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'sequence': sequence, 'video_id': entry['video']['id'],
                'source': {'annotation_path': str(context['annotation']),
                           'annotation_sha256': context['annotation_sha256'],
                           'image_root': str(context['image_root']), 'video_record': entry['video']},
                'conversion': {'video_file': output_video.name, 'video_sha256': sha256(output_video),
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
        say('Indexing annotation metadata once for this batch (no model inference)...')
        index, boxes, categories, total = build_index(annotation, args.sequence, args.limit)
        context = {'annotation': annotation, 'image_root': image_root, 'output_root': output_root,
                   'annotation_sha256': sha256(annotation), 'fps': fps,
                   'ffmpeg_version': run(['ffmpeg', '-version']).splitlines()[0],
                   'script_sha256': sha256(Path(__file__))}
        report_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        report_path = report_root / f'batch_{stamp}.json'
        report = {'schema_version': '1.0', 'status': 'running', 'dataset_video_count': total,
                  'selected_video_count': len(index), 'output_root': str(output_root),
                  'assumed_playback_fps': str(fps), 'annotation_sha256': context['annotation_sha256'],
                  'resume': args.resume, 'sequences': []}
        write_report(report_path, report)
        say(f'Selected {len(index):,}/{total:,} sequences. Free output space: {shutil.disk_usage(output_root).free / 2**30:.1f} GiB')
        say(f'Progress report: {report_path}')
        interrupted = False
        for position, entry in enumerate(index, 1):
            sequence = entry['video']['name']
            say(f'[{position}/{len(index)}] {sequence} | {len(entry["images"])} frames')
            started = time.monotonic()
            record = {'sequence': sequence, 'video_id': entry['video']['id'], 'frame_count': len(entry['images'])}
            try:
                record['status'] = convert_one(entry, boxes, categories, context, args.resume)
            except KeyboardInterrupt:
                record.update(status='interrupted', error='User interrupted; completed outputs preserved.')
                interrupted = True
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            record['elapsed_seconds'] = round(time.monotonic() - started, 3)
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
