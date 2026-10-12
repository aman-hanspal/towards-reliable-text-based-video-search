#!/usr/bin/env bash
# Mini-run: 20 videos, all settings. Re-running resumes where it stopped.
set -e
cd "$(dirname "$0")"
MODELS="yolov8s-worldv2.pt yolov8m-worldv2.pt yolov8l-worldv2.pt"
for m in $MODELS; do
  for s in 320 480 640; do
    python run_detector.py --model "$m" --imgsz "$s" --mode joint --select mini20 --out mini_runs
    python run_detector.py --model "$m" --imgsz "$s" --mode joint --select mini20 --out mini_runs --half
  done
  # the faithful one-prompt-at-a-time reference, for comparison with joint
  python run_detector.py --model "$m" --imgsz 640 --mode separate --select mini20 --out mini_runs
done
python mini_report.py --runs mini_runs
