#!/usr/bin/env bash
# score_all_vlm.sh -- check and score the VLM answers of all 6 separate settings (CPU only, a few minutes).
# 1. completeness check per setting: every main item answered exactly once, no duplicates, how many 'unclear'
# 2. score_vlm.py per setting -> vlm_results/<run>/vlm_summary.md  (+ a blind c_eye_check.csv per setting)
# 3. all summaries joined into vlm_results/ALL_vlm_summaries.md  (upload this one file)
# Run from the OVT-B-Dataset folder:  bash score_all_vlm.sh 2>&1 | tee score_all_vlm.log
cd "$(dirname "cd "$(dirname "$0")"")/.."
TAG=rtx3050__qwen3-vl-2b-instruct
RUNS="yolov8m-worldv2_img640_fp16_separate yolov8s-worldv2_img640_fp16_separate yolov8l-worldv2_img640_fp16_separate
      yolov8m-worldv2_img480_fp16_separate yolov8s-worldv2_img480_fp16_separate yolov8l-worldv2_img480_fp16_separate"
mkdir -p vlm_results
ALL=vlm_results/ALL_vlm_summaries.md
echo "# VLM checker (qwen3-vl:2b-instruct) on all 6 separate settings" > "$ALL"
echo "=== completeness check ==="
for r in $RUNS; do
  if [ "$r" = yolov8m-worldv2_img640_fp16_separate ]; then ans="vlm_runs/$TAG"; else ans="vlm_runs/by_run/$r/$TAG"; fi
  python - "$r" "vlm_sets/$r/items.csv" "$ans/answers.csv" <<'EOF'
import csv, sys
run, items_p, ans_p = sys.argv[1:]
items = {r["item_id"] for r in csv.DictReader(open(items_p, encoding="utf-8")) if not r["set"].startswith("pilot_")}
rows = [r for r in csv.DictReader(open(ans_p, encoding="utf-8")) if r["item_id"] in items]
ids = [r["item_id"] for r in rows]
dup = len(ids) - len(set(ids))
missing = len(items - set(ids))
unclear = sum(r["answer"] == "unclear" for r in rows)
ok = "OK" if dup == 0 and missing == 0 else "CHECK!"
print(f"{ok:7s} {run}: main items {len(items)}, answered {len(set(ids))}, missing {missing}, duplicates {dup}, unclear {unclear}")
EOF
done
echo "=== scoring ==="
for r in $RUNS; do
  if [ "$r" = yolov8m-worldv2_img640_fp16_separate ]; then ans="vlm_runs/$TAG"; else ans="vlm_runs/by_run/$r/$TAG"; fi
  python scripts/score_vlm.py --set-dir "vlm_sets/$r" --runs "$ans" --out "vlm_results/$r" > /dev/null \
    && { echo; echo "---"; echo; cat "vlm_results/$r/vlm_summary.md"; } >> "$ALL" \
    && echo "scored $r" || echo "!!! scoring failed for $r"
done
echo "=== done: upload $ALL ==="
