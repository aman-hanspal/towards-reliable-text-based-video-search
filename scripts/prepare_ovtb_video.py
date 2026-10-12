#!/usr/bin/env python3
"""Convert ONE OVT-B JPEG sequence to a clean MP4 with a frame manifest.

Python >=3.10; dependencies: ijson, Pillow; executables: ffmpeg, ffprobe
    python -m pip install ijson Pillow
    python prepare_ovtb_video.py --root . --sequence AnimalTrack/chicken_1 --fps 30

Save beside ovtb_ann.json and OVT-B/. Relative paths are resolved from --root.
Default output: prepared_videos/<sequence>/ (must not already exist).
All FPS values are explicitly ASSUMED playback rates, never verified source FPS.
No frames are interpolated, skipped or repeated. Original JPEGs are unchanged.
The MP4 uses lossy H.264 CRF 18, yuv420p, original dimensions and no audio.
Outputs: clean MP4, JSON manifest, first/middle/last annotated PNG previews.
Annotations are streamed; only selected sequence images and preview boxes stay
in memory. No model inference, predicted boxes, evaluation metrics or GPU use.
Exit codes: 0 success; 1 processing failure; 2 CLI error; 130 interrupted.
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
                    if image.format != 'JPEG' or image.size != size:
                        raise ValueError(f'{name}: expected JPEG dimensions {size}, got {image.format} {image.size}')
                    image.load()  # Fully decode; truncated/corrupt input must fail.
                    # Do not EXIF-transpose: annotation coordinates use stored pixels.
                    process.stdin.write(image.convert('RGB').tobytes())
                mapping.append({
                    'video_frame_index': index, 'image_id': row['id'],
                    'source_frame_index': row['frame_index'],
                    'source_frame_id': row.get('frame_id'),
                    'file_name': str(name), 'source_jpeg_sha256': hashlib.sha256(raw).hexdigest(),
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
                process.stdin.close()
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


def prepare(args):
    root = args.root.expanduser().resolve()
    annotation = resolve_input(root, args.annotations)
    image_root = resolve_input(root, args.image_root)
    sequence = str(relative_name(args.sequence))
    destination = resolve_input(root, args.output or f'prepared_videos/{sequence}')
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}. Choose a NEW --output directory.')
    if destination.is_relative_to(image_root) or annotation.is_relative_to(destination):
        raise ValueError('Output must be outside the image tree and must not contain the annotation file.')
    if not annotation.is_file() or not image_root.is_dir():
        raise ValueError('Expected ovtb_ann.json and OVT-B/ under --root; check the input paths.')
    for executable in ('ffmpeg', 'ffprobe'):
        if not shutil.which(executable):
            raise ValueError(f'{executable} was not found on PATH.')
    encoders = run(['ffmpeg', '-hide_banner', '-encoders'])
    if not any(line.split()[1:2] == ['libx264'] for line in encoders.splitlines()):
        raise ValueError('FFmpeg requires the libx264 encoder.')
    fps = Fraction(args.fps)
    if fps <= 0 or fps > 1000:
        raise ValueError('--fps must be positive and no more than 1000 (e.g. 30 or 30000/1001).')
    say('Reading selected sequence metadata (streaming the annotation JSON)...')
    videos = [row for row in records(annotation, 'videos') if row.get('name') == sequence]
    if len(videos) != 1:
        raise ValueError(f'Expected one video named {sequence!r}; found {len(videos)}.')
    video = videos[0]
    integer(video['id'], 'video.id')
    images = [row for row in records(annotation, 'images') if row.get('video_id') == video['id']]
    if not images:
        raise ValueError('Selected sequence has no annotated image records.')
    for row in images:
        for key in ('id', 'frame_index', 'width', 'height'):
            integer(row[key], f'image.{key}')
    images.sort(key=lambda row: row['frame_index'])
    if len({row['id'] for row in images}) != len(images):
        raise ValueError('Duplicate image IDs in selected sequence.')
    indices = [row['frame_index'] for row in images]
    if any(b - a != 1 for a, b in zip(indices, indices[1:])):
        raise ValueError('Frame indices have gaps or duplicates; this converter requires a consecutive sequence.')
    size = (images[0]['width'], images[0]['height'])
    if min(size) <= 0 or any(value % 2 for value in size):
        raise ValueError('H.264 yuv420p requires positive even dimensions; this script will not resize frames.')
    if any((row['width'], row['height']) != size for row in images):
        raise ValueError('Image dimensions vary within the selected sequence.')
    selected = {index: images[index] for index in sorted({0, len(images) // 2, len(images) - 1})}
    boxes = {row['id']: [] for row in selected.values()}
    categories = {row['id']: row['name'] for row in records(annotation, 'categories')}
    for ann in records(annotation, 'annotations'):
        if ann.get('image_id') in boxes:
            if ann.get('video_id') != video['id']:
                raise ValueError('Preview annotation video_id disagrees with image video_id.')
            bbox = ann.get('bbox')
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)
                    or bbox[2] <= 0 or bbox[3] <= 0):
                raise ValueError(f"Invalid preview bounding box: annotation {ann.get('id')}")
            boxes[ann['image_id']].append(ann)
    say(f'{sequence}: {len(images)} frames, {size[0]}x{size[1]}, assumed playback FPS={fps}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A fresh, exclusively created output directory prevents overwriting prior work.
    destination.mkdir(exist_ok=False)
    successful = False
    try:
        with tempfile.TemporaryDirectory(prefix='ovtb-working-', dir=destination) as working:
            staging = Path(working)
            name = PurePosixPath(sequence).name
            output_video = staging / (name + '.mp4')
            mapping = encode(images, image_root, output_video, size, fps)
            say('Decoding the complete output to verify frame count, dimensions and timestamps...')
            checks = verify(output_video, len(images), size, fps)
            preview_info = previews(output_video, staging / 'previews', selected, boxes, categories, size)
            manifest = {
                'schema_version': '1.0', 'status': 'complete',
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'sequence': sequence, 'video_id': video['id'],
                'source': {'annotation_path': str(annotation), 'annotation_sha256': sha256(annotation),
                           'image_root': str(image_root), 'video_record': video},
                'conversion': {'video_file': output_video.name, 'width': size[0], 'height': size[1],
                               'frame_count': len(images), 'codec': 'libx264', 'crf': 18,
                               'preset': 'medium', 'pixel_format': 'yuv420p', 'audio': False,
                               'resized': False, 'frames_interpolated': False,
                               'ffmpeg_version': run(['ffmpeg', '-version']).splitlines()[0],
                               'script_sha256': sha256(Path(__file__))},
                'timing': {'playback_fps': str(fps), 'playback_fps_status': 'assumed',
                           'verified_source_fps': None, 'source_duration_seconds': None,
                           'expected_playback_duration_seconds': float(Fraction(len(images)) / fps),
                           'note': 'Playback timestamps are generated; they are not verified original timestamps.'},
                'verification': checks, 'frames': mapping, 'ground_truth_previews': preview_info,
                'limitations': ['Reconstructed lossy MP4; not the original recording or codec.',
                                'Previews use existing ground truth, not model predictions.',
                                'Out-of-bounds boxes are clipped for drawing only.',
                                'Visual annotation alignment needs human review.',
                                'Conversion does not validate full-dataset completeness or score detections.'],
            }
            (staging / (name + '.manifest.json')).write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
            for child in staging.iterdir():
                child.rename(destination / child.name)
        successful = True
    finally:
        if not successful:
            shutil.rmtree(destination)  # Only this invocation's exclusively created directory.
    say(f'Saved: {destination / (name + ".mp4")}')
    say(f'Manifest: {destination / (name + ".manifest.json")}')
    say(f'Previews: {destination / "previews"}')
    say('Complete. FPS is assumed playback timing; inspect the three ground-truth previews.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--sequence', default='AnimalTrack/chicken_1')
    parser.add_argument('--annotations', default='ovtb_ann.json')
    parser.add_argument('--image-root', default='OVT-B')
    parser.add_argument('--output', help='NEW output directory, relative to --root or absolute')
    parser.add_argument('--fps', default='30', help='ASSUMED playback FPS, integer or fraction (default: 30)')
    args = parser.parse_args()
    try:
        import ijson  # noqa: F401
        import PIL  # noqa: F401
        prepare(args)
    except ImportError as exc:
        print(f'Missing dependency: {exc}. Run: python -m pip install ijson Pillow', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Interrupted; unfinished output removed.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
