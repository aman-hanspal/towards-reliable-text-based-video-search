# Open-vocabulary detectors as video search engines on a low budget

Code, frozen query set and result summaries for a research paper in preparation
(working title; final title to be confirmed).

**Authors:** Aman Hanspal, Gargi

## What this study asks

Can open-vocabulary object detectors act as trustworthy video search engines on a
low-memory GPU (4 GB)? A search engine has to (1) find objects that are in a video and
(2) stay silent about objects that are not. We measure both, using present,
easy-absent (unrelated) and hard-absent (lookalike) text queries, and test whether a
vision-language model (VLM) can verify detections and remove lookalike false alarms.

- **Detector:** YOLO-World v2 (small / medium / large), via Ultralytics
- **Dataset:** OVT-B (1,972 of 1,973 videos; see `DATA.md`)
- **Hardware:** RTX 3050 Laptop GPU, 4 GB VRAM

## Repository layout

```
scripts/        all code (run from the repository root)
query_set_v3/   the frozen query set and its hash (decided before any detector ran)
mini_runs/      pilot report (21 settings, 20 videos)
scores_full/    scoring summaries for the full runs
vlm_results/    VLM verification summaries
run_facts/      run metadata summaries
paper_figures/  figures used in the paper
docs/           notes, including which script converted the videos
```

Raw detections, converted videos, model weights and the dataset itself are not
included (size and licensing).

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
| 4. Scoring | `scripts/analyze_full_scores.py` | Headline results with bootstrap confidence intervals |
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
Ultralytics, CLIP and WordNet. Full credits and citations are in `DATA.md`.
The study grew out of VisionQuery, a project built in Dr. Arjun Jain's AI Masterclass
(Cohort 1).

## License

Code: AGPL-3.0 (see `LICENSE`), matching Ultralytics, which this code depends on.
Datasets and models keep their own licenses; see `DATA.md`.
