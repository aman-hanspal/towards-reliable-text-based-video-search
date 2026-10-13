# Towards Reliable Text-Based Video Search: Evaluating Open-Vocabulary Detectors and VLM Verification

Code, frozen test sets and results for the paper above (preprint in preparation).

**Authors:** Aman Hanspal, Dr. Gargi Phadke

## What this study asks

Can open-vocabulary object detectors act as trustworthy video search engines on a
low-memory GPU (4 GB)? A search engine has to:

1. **find** objects that are in a video, and
2. **stay silent** about objects that are not.

We measure both, using present, easy-absent (unrelated) and hard-absent (lookalike)
text queries, and test whether a vision-language model (VLM) can verify detections
and remove lookalike false alarms.

- **Detector:** YOLO-World v2 (small / medium / large), via Ultralytics
- **Dataset:** OVT-B (1,972 of 1,973 videos; see `DATA.md`)
- **Verifier:** Qwen3-VL 2B instruct (`qwen3-vl:2b-instruct`), run locally through Ollama
- **Hardware:** RTX 3050 Laptop GPU, 4 GB VRAM (power-limited to 30 W in every run)

## Main results

Medium model, 640 px, score threshold 0.25, one search word at a time (full table: `paper_figures/table1_main.md`):

| Measure | Value |
|---|---|
| Present objects found somewhere in the video | 50.7% |
| False alarms for **unrelated** words (e.g. "violin" in a farm video) | 1.1% |
| False alarms for **lookalike** words (e.g. "duck" in a penguin video) | 23.9% |
| Lookalike false alarms that sit on a real object of another kind | 93-95% in all six settings |
| Lookalike false alarms hidden when all words are asked at once | 8.4 points (about a third) |
| Lookalike false alarms after the VLM check | 8.1% (a stricter threshold with the same loss of real detections: 18.7%) |
| Real detections the VLM wrongly rejects | 10.2% (clear VLM mistakes: 1.2%) |

A larger model, a stricter threshold, 480 px input or processing only every 10th frame do not remove lookalike
false alarms. The VLM check brings them to 8-9% in all six detector settings. Every number in the paper is
recomputed from the files in this repository by `scripts/make_figures.py` (see `paper_figures/check.md`).

## Repository layout

```
scripts/        all code (run from the repository root)
query_set_v3/   the frozen query set and its hash (decided before any detector ran)
mini_runs/      pilot report (21 settings, 20 videos)
scores_full/    scoring summaries; prompt_scores.csv.gz = every (run, video, word, threshold) score
vlm_sets/       frozen VLM test lists per setting (items.csv + FROZEN.md); crops not included
vlm_runs/       the VLM's answer for every item (answers.csv + run_meta.json with the weights hash)
vlm_results/    VLM verification summaries and the hand checks (hand_check/)
frame_rate_cuts/ every-2nd/5th/10th-frame results (summaries)
run_facts/      speed, memory and zero-box summaries
paper_figures/  figures, Table 1, numbers.csv (every value behind every figure), check.md
docs/           notes, including which script converted the videos
```

Raw detections, converted videos, VLM crops, model weights and the dataset itself are not
included (size and licensing). Each crop's video, frame and pixel
coordinates are listed in `vlm_sets/*/items.csv` (`frame`, `crop_x1`..`crop_y2`), so the crops can be cut
again from the converted videos. Before running `make_figures.py`, unpack the scores:
`gunzip -k scores_full/prompt_scores.csv.gz`.

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. Data | `scripts/inspect_ovtb.py` | Inspect OVT-B annotations and report issues |
| 1. Data | `scripts/prepare_ovtb_video.py` | Convert one JPEG sequence to MP4 + manifest |
| 1. Data | `scripts/prepare_ovtb_batch.py` | Batch conversion (first 20 videos) |
| 1. Data | `scripts/prepare_ovtb_batch_optimised.py` | Faster batch conversion (remaining 1,951 videos) |
| 2. Queries | `scripts/build_query_set.py` | Build the frozen present / easy-absent / hard-absent query set (CLIP + WordNet) |
| 2. Queries | `scripts/check_negatives.py` | Audit absent prompts for fairness |
| 3. Detection | `scripts/run_detector.py` | Run YOLO-World with one setting; save every box >= 0.01 |
| 3. Detection | `scripts/mini_run.sh`, `scripts/mini_report.py` | Pilot: 21 settings on 20 videos, and its report |
| 3. Detection | `scripts/gate.sh`, `scripts/gate_check.py`, `scripts/compare_runs.py` | Checks before full runs: new runner and FP16 must match old runner and FP32 |
| 3. Detection | `scripts/full_runs.sh`, `scripts/run_facts.py` | Full runs and run metadata |
| 4. Scoring | `scripts/score_runs.py` | Score against labels: present hit rates, absent false-alarm rates |
| 4. Scoring | `scripts/analyze_full_scores.py` | Headline results with paired bootstrap confidence intervals |
| 4. Scoring | `scripts/frame_rate_cuts.py` | Simulate searching only every k-th frame |
| 5. VLM | `scripts/export_vlm_sets.py`, `scripts/verify_vlm.py`, `scripts/vlm_all.sh` | Export crops and verify them with a local VLM |
| 5. VLM | `scripts/score_vlm.py`, `scripts/score_all_vlm.sh` | Score results after verification |
| 5. VLM | `scripts/vlm_hand_check.py`, `scripts/vlm_contact_sheet.py` | Manual review of VLM answers |
| 6. Figures | `scripts/make_figures.py`, `scripts/view_frames.py` | Paper figures and annotated frames |

## Reproducing

1. Download OVT-B from the official repository (see `DATA.md`) and place
   `ovtb_ann.json` and the `OVT-B/` image folders in the repository root.
2. Install dependencies: `pip install -r requirements.txt` (exact versions used are in
   `docs/environment.txt`).
3. Run everything **from the repository root**. The conversion and inspection scripts
   must be given `--root .` explicitly, e.g.
   `python scripts/prepare_ovtb_video.py --root . ...`
4. Shell scripts (`scripts/*.sh`) are bash; on Windows run them in WSL.

The detector runner refuses to run unless the query set matches the frozen hash in
`query_set_v3/`.

## Notes on the data

- OVT-B category `synset` fields are all `unknown`; categories are resolved in WordNet
  by name, keeping physical-object senses only.
- YouTube-VIS sequences keep every 5th frame, so all timing is reported per frame,
  never in seconds.
- One video (LVVIS/train_03031) failed conversion and is excluded.

## Acknowledgements

This work builds on OVT-B and the seven datasets it is drawn from, YOLO-World,
Ultralytics, CLIP, WordNet and Qwen3-VL. Full credits and citations are in `DATA.md`.
The study grew out of VisionQuery, a project built in Dr. Arjun Jain's AI Masterclass
(Cohort 1).

## Citation

```bibtex
@misc{hanspal2026towards,
  title  = {Towards Reliable Text-Based Video Search: Evaluating Open-Vocabulary Detectors and VLM Verification},
  author = {Hanspal, Aman and Phadke, Dr. Gargi},
  year   = {2026},
  note   = {Preprint, in preparation}
}
```

## License

Code: AGPL-3.0 (see `LICENSE`), matching Ultralytics, which this code depends on.
Datasets and models keep their own licenses; see `DATA.md`.