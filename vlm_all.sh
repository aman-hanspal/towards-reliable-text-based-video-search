#!/usr/bin/env bash
# vlm_all.sh -- VLM checker (qwen3-vl:4b-instruct) on the test sets of ALL 6 separate full runs, one after another.
#   1. medium 640 (main)      -> vlm_runs/rtx3050__qwen3-vl-4b-instruct/            (same folder as the pilot)
#   2-6. small 640, large 640, medium 480, small 480, large 480
#                             -> vlm_runs/by_run/<run name>/rtx3050__qwen3-vl-4b-instruct/
# For each run: build its frozen test set if missing (export_vlm_sets.py), then ask the VLM about every main item.
# Safety: verify_vlm.py retries when Ollama hiccups and stops on junk replies; each run gets a 2nd attempt after
# a 2-minute pause; a failed run never blocks the next one. Re-running this script resumes everything.
# Logs: vlm_all.log (console), vlm_monitor/gpu_log.csv (GPU every 10 s), vlm_monitor/run_windows.csv (start/end/status).
# Run:  bash vlm_all.sh 2>&1 | tee -a vlm_all.log      (inside tmux; laptop plugged in, sleep off)
cd "$(dirname "$0")"
MODEL=qwen3-vl:2b-instruct
MAIN=yolov8m-worldv2_img640_fp16_separate
OTHERS="yolov8s-worldv2_img640_fp16_separate yolov8l-worldv2_img640_fp16_separate
        yolov8m-worldv2_img480_fp16_separate yolov8s-worldv2_img480_fp16_separate
        yolov8l-worldv2_img480_fp16_separate"
MON=vlm_monitor
mkdir -p "$MON"
PIDS=()
if command -v nvidia-smi >/dev/null; then
  stdbuf -oL nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw \
    --format=csv -l 10 >> "$MON/gpu_log.csv" 2>&1 &
  PIDS+=($!)
fi
trap 'kill "${PIDS[@]}" 2>/dev/null' EXIT
trap 'echo "Interrupted."; exit 130' INT TERM
[ -f "$MON/run_windows.csv" ] || echo "run,start,end,status" > "$MON/run_windows.csv"

ollama list >/dev/null 2>&1 || { echo "Ollama is not reachable. Start it (ollama serve) and run this again."; exit 1; }

one () {  # run_name out_dir_or_empty
  local r="$1" out="$2" start status=error
  start="$(date '+%F %T')"
  echo "=== $start START $r ==="
  if [ ! -f "vlm_sets/$r/items.csv" ]; then
    python export_vlm_sets.py --run "full_runs/$r" || { echo "!!! export failed for $r"; echo "$r,$start,$(date '+%F %T'),export_failed" >> "$MON/run_windows.csv"; return; }
  fi
  for attempt in 1 2; do
    if python -u verify_vlm.py --set-dir "vlm_sets/$r" --model "$MODEL" --device rtx3050 --part main ${out:+--out "$out"}; then
      status=ok; break
    fi
    echo "!!! $r attempt $attempt stopped; waiting 2 min before the next attempt"
    sleep 120
  done
  echo "$r,$start,$(date '+%F %T'),$status" >> "$MON/run_windows.csv"
  echo "=== $(date '+%F %T') END $r ($status) ==="
}

one "$MAIN" ""
for r in $OTHERS; do one "$r" "vlm_runs/by_run/$r"; done
echo "=== $(date '+%F %T') ALL DONE ==="
cat "$MON/run_windows.csv"
