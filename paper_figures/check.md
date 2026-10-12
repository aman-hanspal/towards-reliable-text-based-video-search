# Check: recomputed from raw files vs first draft

Tolerance 0.15 points (draft values were rounded to 0.1). Bootstrap intervals use a new seed, so they may differ in the last digit; only the centre values are checked.

| measure | size | res | draft | recomputed | status |
|---|---|---|---:|---:|---|
| found | small | 640 | 53.3 | 53.33 | OK |
| found | medium | 640 | 50.7 | 50.70 | OK |
| found | large | 640 | 52.2 | 52.23 | OK |
| found | small | 480 | 52.0 | 52.02 | OK |
| found | medium | 480 | 50.5 | 50.49 | OK |
| found | large | 480 | 51.9 | 51.93 | OK |
| easy | small | 640 | 2.5 | 2.51 | OK |
| easy | medium | 640 | 1.1 | 1.14 | OK |
| easy | large | 640 | 1.1 | 1.14 | OK |
| hard | small | 640 | 28.9 | 28.85 | OK |
| hard | medium | 640 | 23.9 | 23.86 | OK |
| hard | large | 640 | 24.6 | 24.65 | OK |
| hard | small | 480 | 27.1 | 27.05 | OK |
| hard | medium | 480 | 24.0 | 23.96 | OK |
| hard | large | 480 | 23.2 | 23.17 | OK |
| strict | small | 640 | 33.8 | 33.79 | OK |
| strict | medium | 640 | 31.0 | 30.98 | OK |
| strict | large | 640 | 31.6 | 31.59 | OK |
| frame_hit | medium | 640 | 38.5 | 38.51 | OK |
| misnaming_share | medium | 640 | 94.0 | 94.26 | DIFF |
| ms | small | 640 | 25.1 | 25.08 | OK |
| ms | medium | 640 | 34.1 | 34.07 | OK |
| ms | large | 640 | 48.2 | 48.17 | OK |
| ms | small | 480 | 25.3 | 25.31 | OK |
| ms | medium | 480 | 32.4 | 32.42 | OK |
| ms | large | 480 | 40.6 | 40.62 | OK |
| joint_diff | small | 640 | -9.9 | -9.94 | OK |
| joint_diff | medium | 640 | -8.4 | -8.42 | OK |
| joint_diff | large | 640 | -8.3 | -8.27 | OK |
| vlm_after | small | 640 | 9.1 | 9.13 | OK |
| vlm_after | medium | 640 | 8.1 | 8.07 | OK |
| vlm_after | large | 640 | 9.0 | 9.00 | OK |
| vlm_after | small | 480 | 8.7 | 8.69 | OK |
| vlm_after | medium | 480 | 8.3 | 8.33 | OK |
| vlm_after | large | 480 | 8.3 | 8.29 | OK |
| thr_matched | small | 640 | 20.7 | 20.72 | OK |
| thr_matched | medium | 640 | 18.7 | 18.74 | OK |
| thr_matched | large | 640 | 16.5 | 16.53 | OK |
| thr_matched | small | 480 | 19.4 | 19.42 | OK |
| thr_matched | medium | 480 | 17.4 | 17.42 | OK |
| thr_matched | large | 480 | 14.8 | 14.81 | OK |
| vlm_removed_A | medium | 640 | 69.0 | 68.98 | OK |
| vlm_lost_B | medium | 640 | 10.2 | 10.17 | OK |
| found_k10 | small | 640 | 45.8 | 45.77 | OK |
| found_k10 | medium | 640 | 44.5 | 44.54 | OK |
| found_k10 | large | 640 | 46.3 | 46.35 | OK |
| hard_k10 | small | 640 | 21.0 | 21.02 | OK |
| hard_k10 | medium | 640 | 17.8 | 17.85 | OK |
| hard_k10 | large | 640 | 18.6 | 18.64 | OK |

**1 difference(s).**
