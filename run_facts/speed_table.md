# Speed, memory and GPU conditions per run

frame-pass = one frame x one vocabulary. Times summed over all videos. GPU columns are medians over the 10-second samples taken while that run was going.

| run | videos | frame-passes | hours (wall) | ms/frame-pass wall | ms/frame-pass inference | frame-passes/s | peak VRAM PyTorch MB | peak VRAM nvidia-smi MB | GPU busy % | clock MHz | power W | temp C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolov8l-worldv2_img480_fp16_separate | 1972 | 593,320 | 6.70 | 40.6 | 38.9 | 25 | 1790.8 | 2160.0 | 86.0 | 780.0 | 14.7 | 67.0 |
| yolov8l-worldv2_img640_fp16_joint | 1972 | 88,066 | 1.30 | 53.1 | 45.9 | 19 | 1601.7 | 1960.0 | 89.0 | 780.0 | 15.0 | 66.0 |
| yolov8l-worldv2_img640_fp16_separate | 1972 | 593,320 | 7.94 | 48.2 | 46.4 | 21 | 1770.3 | 2312.0 | 91.0 | 780.0 | 15.8 | 67.0 |
| yolov8m-worldv2_img480_fp16_separate | 1972 | 593,320 | 5.34 | 32.4 | 30.7 | 31 | 1458.1 | 1903.0 | 71.0 | 780.0 | 13.5 | 69.0 |
| yolov8m-worldv2_img640_fp16_joint | 1972 | 88,066 | 0.98 | 40.2 | 32.6 | 25 | 1313.1 | 1991.0 | 81.0 | 780.0 | 13.4 | 66.0 |
| yolov8m-worldv2_img640_fp16_separate | 1972 | 593,320 | 5.61 | 34.1 | 32.4 | 29 | 1483.6 | 1831.0 | 85.0 | 780.0 | 15.1 | 67.0 |
| yolov8s-worldv2_img480_fp16_separate | 1972 | 593,320 | 4.17 | 25.3 | 23.6 | 40 | 1033.2 | 1763.0 | 48.0 | 780.0 | 11.7 | 68.0 |
| yolov8s-worldv2_img640_fp16_joint | 1972 | 88,066 | 0.72 | 29.3 | 22.0 | 34 | 984.3 | 1331.0 | 54.0 | 780.0 | 11.5 | 67.0 |
| yolov8s-worldv2_img640_fp16_separate | 1972 | 593,320 | 4.13 | 25.1 | 23.3 | 40 | 1081.7 | 1439.0 | 54.0 | 780.0 | 12.7 | 68.0 |

GPU memory available: 4096.0 MB. nvidia-smi memory includes everything on the card (other programs too, e.g. Windows), so treat it as an upper bound for YOLO.
