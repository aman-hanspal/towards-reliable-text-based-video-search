# Mini-run report

## Speed and memory per setting

Inference only (decode excluded), median over videos. ms/frame is per prompt-pass.

| setting | videos | median ms/frame | frames/s | peak VRAM MB | saturated frames | frame-count problems |
|---|---:|---:|---:|---:|---:|---:|
| yolov8l-worldv2_img320_fp16_joint | 20 | 25.6 | 39 | 1451.1 | 0 | 0 |
| yolov8l-worldv2_img320_fp32_joint | 20 | 30.1 | 33 | 1840.5 | 0 | 0 |
| yolov8l-worldv2_img480_fp16_joint | 20 | 34.5 | 29 | 1402.3 | 0 | 0 |
| yolov8l-worldv2_img480_fp32_joint | 20 | 48.0 | 21 | 1972.7 | 0 | 0 |
| yolov8l-worldv2_img640_fp16_joint | 20 | 44.5 | 22 | 1500.5 | 0 | 0 |
| yolov8l-worldv2_img640_fp32_joint | 20 | 65.2 | 15 | 2010.9 | 0 | 0 |
| yolov8l-worldv2_img640_fp32_separate | 20 | 63.9 | 16 | 2357.2 | 0 | 0 |
| yolov8m-worldv2_img320_fp16_joint | 20 | 21.2 | 47 | 1115.5 | 0 | 0 |
| yolov8m-worldv2_img320_fp32_joint | 20 | 27.9 | 36 | 1475.6 | 0 | 0 |
| yolov8m-worldv2_img480_fp16_joint | 20 | 24.7 | 41 | 1131.9 | 0 | 0 |
| yolov8m-worldv2_img480_fp32_joint | 20 | 31.2 | 32 | 1523.5 | 0 | 0 |
| yolov8m-worldv2_img640_fp16_joint | 20 | 29.9 | 33 | 1114.2 | 0 | 0 |
| yolov8m-worldv2_img640_fp32_joint | 20 | 40.1 | 25 | 1511.2 | 0 | 0 |
| yolov8m-worldv2_img640_fp32_separate | 20 | 40.1 | 25 | 1809.5 | 0 | 0 |
| yolov8s-worldv2_img320_fp16_joint | 20 | 27.4 | 36 | 860.8 | 0 | 0 |
| yolov8s-worldv2_img320_fp32_joint | 20 | 26.8 | 37 | 980.4 | 0 | 0 |
| yolov8s-worldv2_img480_fp16_joint | 20 | 26.0 | 39 | 873.6 | 0 | 0 |
| yolov8s-worldv2_img480_fp32_joint | 20 | 25.0 | 40 | 1050.1 | 0 | 0 |
| yolov8s-worldv2_img640_fp16_joint | 20 | 27.5 | 36 | 847.9 | 0 | 0 |
| yolov8s-worldv2_img640_fp32_joint | 20 | 28.6 | 35 | 1032.4 | 0 | 0 |
| yolov8s-worldv2_img640_fp32_separate | 20 | 26.5 | 38 | 1240.9 | 0 | 0 |

## Joint vs separate prompting

Does a prompt 'fire' (any box at or above the threshold) the same way when YOLO is given all prompts at once (joint) as when it is asked one at a time (separate)?

### yolov8l-worldv2_img640_fp32

| threshold | kind | both fire | only separate fires | only joint fires | neither | agreement |
|---:|---|---:|---:|---:|---:|---:|
| 0.05 | present | 35 | 2 | 0 | 1 | 94.7% |
| 0.05 | easy_absent | 2 | 3 | 0 | 35 | 92.5% |
| 0.05 | hard_absent | 21 | 10 | 1 | 8 | 72.5% |
| 0.1 | present | 34 | 1 | 0 | 3 | 97.4% |
| 0.1 | easy_absent | 2 | 0 | 0 | 38 | 100.0% |
| 0.1 | hard_absent | 17 | 11 | 0 | 12 | 72.5% |
| 0.25 | present | 31 | 1 | 1 | 5 | 94.7% |
| 0.25 | easy_absent | 0 | 1 | 0 | 39 | 97.5% |
| 0.25 | hard_absent | 12 | 10 | 0 | 18 | 75.0% |

Box level (score >= 0.10): share of separate-mode boxes that joint mode also finds (IoU >= 0.5)

- present: 20732/22917 = 90.5%
- hard_absent: 2706/10572 = 25.6%
- easy_absent: 2/13 = 15.4%

### yolov8m-worldv2_img640_fp32

| threshold | kind | both fire | only separate fires | only joint fires | neither | agreement |
|---:|---|---:|---:|---:|---:|---:|
| 0.05 | present | 33 | 3 | 0 | 2 | 92.1% |
| 0.05 | easy_absent | 2 | 0 | 0 | 38 | 100.0% |
| 0.05 | hard_absent | 21 | 9 | 1 | 9 | 75.0% |
| 0.1 | present | 31 | 4 | 0 | 3 | 89.5% |
| 0.1 | easy_absent | 0 | 0 | 2 | 38 | 95.0% |
| 0.1 | hard_absent | 17 | 8 | 1 | 14 | 77.5% |
| 0.25 | present | 30 | 2 | 0 | 6 | 94.7% |
| 0.25 | easy_absent | 0 | 0 | 0 | 40 | 100.0% |
| 0.25 | hard_absent | 12 | 7 | 1 | 20 | 80.0% |

Box level (score >= 0.10): share of separate-mode boxes that joint mode also finds (IoU >= 0.5)

- present: 20924/24390 = 85.8%
- hard_absent: 8664/18714 = 46.3%

### yolov8s-worldv2_img640_fp32

| threshold | kind | both fire | only separate fires | only joint fires | neither | agreement |
|---:|---|---:|---:|---:|---:|---:|
| 0.05 | present | 32 | 1 | 0 | 5 | 97.4% |
| 0.05 | easy_absent | 1 | 1 | 2 | 36 | 92.5% |
| 0.05 | hard_absent | 20 | 9 | 0 | 11 | 77.5% |
| 0.1 | present | 31 | 1 | 0 | 6 | 97.4% |
| 0.1 | easy_absent | 1 | 1 | 1 | 37 | 95.0% |
| 0.1 | hard_absent | 18 | 8 | 0 | 14 | 80.0% |
| 0.25 | present | 30 | 2 | 0 | 6 | 94.7% |
| 0.25 | easy_absent | 1 | 0 | 0 | 39 | 100.0% |
| 0.25 | hard_absent | 14 | 8 | 0 | 18 | 80.0% |

Box level (score >= 0.10): share of separate-mode boxes that joint mode also finds (IoU >= 0.5)

- present: 17725/22046 = 80.4%
- easy_absent: 6/32 = 18.8%
- hard_absent: 11672/23130 = 50.5%

