#!/usr/bin/env bash
# Full runs: FP16, all 1,972 videos.
#   Part 1 (~2.5 h): JOINT mode at 640 for medium, large, small  -> "does all-prompts-at-once hide errors?"
#   Part 2 (~28 h):  SEPARATE mode, 6 settings, most important first -> the main results
# Also records, every 10 s, GPU (busy %, memory, temperature, clock, power) and CPU/RAM,
# plus the start/end time of each run, all in full_runs/monitor/.
# Resumable: if anything stops (crash, reboot), run this script again; finished videos are skipped.
# Run:  bash full_runs.sh 2>&1 | tee -a full_runs.log      (keep the window open)
cd "$(dirname "cd "$(dirname "$0")"")/.."
MON=full_runs/monitor
mkdir -p "$MON"

# ---------- background monitors (stopped automatically when this script ends or is interrupted)
PIDS=()
if command -v nvidia-smi >/dev/null; then
  stdbuf -oL nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw \
    --format=csv -l 10 >> "$MON/gpu_log.csv" 2>&1 &
  PIDS+=($!)
else
  echo "WARNING: nvidia-smi not found, GPU monitor not running"
fi
stdbuf -oL vmstat -t 10 >> "$MON/cpu_log.txt" 2>&1 &
PIDS+=($!)
stop_monitors () { kill "${PIDS[@]}" 2>/dev/null; }
trap stop_monitors EXIT
trap 'echo "Interrupted."; exit 130' INT TERM
echo "Monitors running (GPU + CPU every 10 s) -> $MON/"
[ -f "$MON/run_windows.csv" ] || echo "setting,start,end,status" > "$MON/run_windows.csv"

# ---------- one detector run
run () {  # model imgsz mode
  local name="$(basename "$1" .pt)_img$2_fp16_$3" start end status
  start="$(date '+%F %T')"
  echo "=== $start START $name ==="
  if python -u scripts/run_detector.py --model "$1" --imgsz "$2" --half --mode "$3" --select all --out full_runs; then
    status=ok
  else
    status=error
    echo "!!! $name stopped with an error -- moving on; run this script again later to resume it"
  fi
  end="$(date '+%F %T')"
  echo "$name,$start,$end,$status" >> "$MON/run_windows.csv"
  echo "=== $end END $name ($status) ==="
}

# Part 1: joint
run yolov8m-worldv2.pt 640 joint
run yolov8l-worldv2.pt 640 joint
run yolov8s-worldv2.pt 640 joint
# Part 2: separate
run yolov8m-worldv2.pt 640 separate
run yolov8l-worldv2.pt 640 separate
run yolov8s-worldv2.pt 640 separate
run yolov8m-worldv2.pt 480 separate
run yolov8l-worldv2.pt 480 separate
run yolov8s-worldv2.pt 480 separate
echo "ALL FULL RUNS FINISHED $(date '+%F %T')"
