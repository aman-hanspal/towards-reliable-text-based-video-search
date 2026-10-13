# Frame-rate cuts: yolov8l-worldv2_img640_fp16_separate

Threshold 0.25. 1972 videos. Every k-th frame kept (frames 0, k, 2k, ...). Change vs k=1 with 95% paired bootstrap interval (videos resampled). k=1 must equal the earlier scores.

| measure | k=1 | k=2 | k=5 | k=10 |
|---|---:|---:|---:|---:|
| found in video | 52.2% (n=7808) | 51.0% (-1.3 [-1.5, -1.0]) | 48.8% (-3.5 [-3.9, -3.1]) | 46.3% (-5.9 [-6.4, -5.4]) |
| all labelled frames skipped | 0.0% (n=7808) | 0.1% (+0.1 [+0.0, +0.2]) | 0.3% (+0.3 [+0.2, +0.5]) | 0.7% (+0.7 [+0.5, +0.9]) |
| frame hit rate (looked-at frames) | 40.5% (n=7808) | 40.6% (+0.1 [-0.0, +0.2]) | 40.9% (+0.4 [+0.2, +0.5]) | 40.9% (+0.4 [+0.1, +0.7]) |
| easy false alarms | 1.1% (n=3944) | 1.0% (-0.1 [-0.3, -0.0]) | 0.9% (-0.3 [-0.5, -0.1]) | 0.6% (-0.6 [-0.8, -0.3]) |
| hard false alarms | 24.6% (n=3944) | 23.1% (-1.5 [-1.9, -1.1]) | 20.5% (-4.1 [-4.8, -3.5]) | 18.6% (-6.0 [-6.8, -5.3]) |
| strict-hard false alarms | 31.6% (n=1456) | 30.0% (-1.6 [-2.3, -1.0]) | 27.0% (-4.6 [-5.8, -3.5]) | 24.8% (-6.8 [-8.2, -5.4]) |
| hard misnaming (on a real object) | 22.9% (n=3944) | 21.5% (-1.3 [-1.7, -1.0]) | 19.2% (-3.7 [-4.3, -3.1]) | 17.4% (-5.5 [-6.2, -4.8]) |
| detector GPU time (h) | 7.65 | 3.83 | 1.53 | 0.77 |
| frames looked at | 100% | 50% | 20% | 10% |

Decode time (whole run, reading every frame of the file): 0.05 h. It does not shrink with k unless frames are skipped without decoding (not measured here).
