#!/usr/bin/env bash
# Gate before the full runs. Stops at the first failed test.
# Run:  bash gate.sh 2>&1 | tee gate_log.txt
set -e
cd "$(dirname "$0")"

echo "=== TEST 1: new script (decode-once) vs old script, FP32, 3 videos (~3-5 min) ==="
python run_detector.py --model yolov8m-worldv2.pt --imgsz 640 --mode separate --select mini3 --out gate_check1
python compare_runs.py gate_check1/yolov8m-worldv2_img640_fp32_separate mini_runs/yolov8m-worldv2_img640_fp32_separate

echo "=== TEST 2: FP16 separate on mini20, medium / large / small at 640 (~45-60 min) ==="
for m in yolov8m yolov8l yolov8s; do
  python run_detector.py --model ${m}-worldv2.pt --imgsz 640 --half --mode separate --select mini20 --out gate_runs
done
mkdir -p scores_gate
cp -n scores_mini/gt_cache.json.gz scores_gate/ 2>/dev/null || true   # same 20 videos -> reuse labels
python score_runs.py --runs gate_runs --out scores_gate > /dev/null

echo "=== TESTS 2 + 3: compare with FP32, estimate full runs ==="
python gate_check.py --old scores_mini/prompt_scores.csv --new scores_gate/prompt_scores.csv --gate-runs gate_runs
