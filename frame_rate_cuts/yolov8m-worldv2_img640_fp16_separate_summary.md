# Frame-rate cuts: yolov8m-worldv2_img640_fp16_separate

Threshold 0.25. 1972 videos. Every k-th frame kept (frames 0, k, 2k, ...). Change vs k=1 with 95% paired bootstrap interval (videos resampled). k=1 must equal the earlier scores.

| measure | k=1 | k=2 | k=5 | k=10 |
|---|---:|---:|---:|---:|
| found in video | 50.7% (n=7808) | 49.2% (-1.5 [-1.8, -1.3]) | 46.9% (-3.8 [-4.3, -3.4]) | 44.5% (-6.2 [-6.7, -5.6]) |
| all labelled frames skipped | 0.0% (n=7808) | 0.1% (+0.1 [+0.0, +0.2]) | 0.3% (+0.3 [+0.2, +0.5]) | 0.7% (+0.7 [+0.5, +0.9]) |
| frame hit rate (looked-at frames) | 38.5% (n=7808) | 38.6% (+0.0 [-0.0, +0.1]) | 38.9% (+0.4 [+0.2, +0.5]) | 39.0% (+0.5 [+0.2, +0.8]) |
| easy false alarms | 1.1% (n=3944) | 1.0% (-0.1 [-0.3, -0.0]) | 0.8% (-0.4 [-0.6, -0.2]) | 0.7% (-0.5 [-0.7, -0.3]) |
| hard false alarms | 23.9% (n=3944) | 22.7% (-1.2 [-1.5, -0.9]) | 19.9% (-4.0 [-4.6, -3.3]) | 17.8% (-6.0 [-6.8, -5.2]) |
| strict-hard false alarms | 31.0% (n=1456) | 29.8% (-1.2 [-1.7, -0.7]) | 25.8% (-5.2 [-6.3, -4.0]) | 23.9% (-7.1 [-8.4, -5.8]) |
| hard misnaming (on a real object) | 22.5% (n=3944) | 21.3% (-1.1 [-1.5, -0.8]) | 18.9% (-3.6 [-4.2, -3.0]) | 16.9% (-5.6 [-6.3, -4.8]) |
| detector GPU time (h) | 5.34 | 2.67 | 1.07 | 0.53 |
| frames looked at | 100% | 50% | 20% | 10% |

Decode time (whole run, reading every frame of the file): 0.05 h. It does not shrink with k unless frames are skipped without decoding (not measured here).
