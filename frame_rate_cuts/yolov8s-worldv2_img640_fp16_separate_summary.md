# Frame-rate cuts: yolov8s-worldv2_img640_fp16_separate

Threshold 0.25. 1972 videos. Every k-th frame kept (frames 0, k, 2k, ...). Change vs k=1 with 95% paired bootstrap interval (videos resampled). k=1 must equal the earlier scores.

| measure | k=1 | k=2 | k=5 | k=10 |
|---|---:|---:|---:|---:|
| found in video | 53.3% (n=7808) | 51.7% (-1.7 [-2.0, -1.4]) | 48.6% (-4.7 [-5.2, -4.3]) | 45.8% (-7.6 [-8.1, -7.0]) |
| all labelled frames skipped | 0.0% (n=7808) | 0.1% (+0.1 [+0.0, +0.2]) | 0.3% (+0.3 [+0.2, +0.5]) | 0.7% (+0.7 [+0.5, +0.9]) |
| frame hit rate (looked-at frames) | 39.1% (n=7808) | 39.1% (+0.0 [-0.0, +0.1]) | 39.2% (+0.2 [-0.0, +0.4]) | 39.4% (+0.4 [+0.1, +0.7]) |
| easy false alarms | 2.5% (n=3944) | 2.3% (-0.3 [-0.4, -0.1]) | 1.6% (-0.9 [-1.2, -0.6]) | 1.4% (-1.1 [-1.5, -0.8]) |
| hard false alarms | 28.9% (n=3944) | 26.9% (-2.0 [-2.4, -1.5]) | 23.8% (-5.0 [-5.8, -4.4]) | 21.0% (-7.8 [-8.7, -7.0]) |
| strict-hard false alarms | 33.8% (n=1456) | 32.1% (-1.7 [-2.4, -1.1]) | 28.3% (-5.5 [-6.8, -4.4]) | 25.9% (-7.9 [-9.4, -6.5]) |
| hard misnaming (on a real object) | 27.3% (n=3944) | 25.5% (-1.9 [-2.3, -1.4]) | 22.6% (-4.7 [-5.4, -4.1]) | 19.9% (-7.5 [-8.3, -6.6]) |
| detector GPU time (h) | 3.85 | 1.92 | 0.77 | 0.38 |
| frames looked at | 100% | 50% | 20% | 10% |

Decode time (whole run, reading every frame of the file): 0.05 h. It does not shrink with k unless frames are skipped without decoding (not measured here).
