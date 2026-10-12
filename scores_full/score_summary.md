# Score summary (against dataset labels)

Present: strict = box overlap IoU >= 0.5, lenient = IoU >= 0.3. Absent: 'misnaming' = the box sits on a labelled object of another kind (confirmed error); 'unlabelled only' = fired only where nothing is labelled (needs Stage C eye check).

**Runs ending in _joint are biased** (each box keeps only its best prompt). Use _separate runs for absent prompts.

## yolov8l-worldv2_img480_fp16_separate

### Present prompts

| threshold | prompts | video found | frame hit rate | frame hit (lenient) | box recall | box recall (lenient) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 7808 | 66.5% | 55.3% | 57.0% | 52.2% | 54.0% |
| 0.1 | 7808 | 61.1% | 49.6% | 51.2% | 46.6% | 48.2% |
| 0.25 | 7808 | 51.9% | 40.5% | 41.8% | 37.8% | 39.1% |
| 0.4 | 7808 | 45.7% | 34.1% | 35.2% | 31.8% | 32.7% |

Present, frame hit rate at threshold 0.25, by group:

| group | prompts | frame hit rate |
|---|---:|---:|
| split: base | 4728 | 49.8% |
| split: novel | 3080 | 26.2% |
| size_bin: large (>10%) | 3338 | 52.4% |
| size_bin: medium (1-10%) | 2824 | 33.9% |
| size_bin: small (<1% of frame) | 1646 | 27.6% |
| source: AnimalTrack | 29 | 63.0% |
| source: GMOT-40 | 15 | 48.1% |
| source: ImageNet-VID | 93 | 46.5% |
| source: LVVIS | 7482 | 39.9% |
| source: OVIS | 27 | 87.1% |
| source: UVO | 113 | 44.9% |
| source: YouTube-VIS-2021 | 49 | 66.9% |

### Absent prompts (share of prompts)

| threshold | group | prompts | fired | misnaming (confirmed) | unlabelled only (needs review) |
|---:|---|---:|---:|---:|---:|
| 0.05 | easy | 3944 | 5.7% | 4.3% | 1.4% |
| 0.05 | hard | 3944 | 44.1% | 40.8% | 3.3% |
| 0.05 | strict-hard | 1456 | 51.8% | 48.2% | 3.6% |
| 0.1 | easy | 3944 | 3.1% | 2.2% | 0.9% |
| 0.1 | hard | 3944 | 35.1% | 32.5% | 2.6% |
| 0.1 | strict-hard | 1456 | 43.2% | 40.2% | 3.0% |
| 0.25 | easy | 3944 | 1.2% | 0.9% | 0.3% |
| 0.25 | hard | 3944 | 23.2% | 21.9% | 1.3% |
| 0.25 | strict-hard | 1456 | 30.1% | 28.8% | 1.3% |
| 0.4 | easy | 3944 | 0.5% | 0.3% | 0.2% |
| 0.4 | hard | 3944 | 16.6% | 15.7% | 0.9% |
| 0.4 | strict-hard | 1456 | 22.7% | 21.6% | 1.2% |

## yolov8l-worldv2_img640_fp16_joint

### Present prompts

| threshold | prompts | video found | frame hit rate | frame hit (lenient) | box recall | box recall (lenient) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 7808 | 62.1% | 50.7% | 52.4% | 48.1% | 49.8% |
| 0.1 | 7808 | 57.5% | 46.1% | 47.7% | 43.4% | 45.1% |
| 0.25 | 7808 | 49.6% | 38.4% | 39.8% | 35.9% | 37.4% |
| 0.4 | 7808 | 43.5% | 32.6% | 33.8% | 30.2% | 31.5% |

Present, frame hit rate at threshold 0.25, by group:

| group | prompts | frame hit rate |
|---|---:|---:|
| split: base | 4728 | 47.9% |
| split: novel | 3080 | 23.8% |
| size_bin: large (>10%) | 3338 | 46.7% |
| size_bin: medium (1-10%) | 2824 | 32.7% |
| size_bin: small (<1% of frame) | 1646 | 31.2% |
| source: AnimalTrack | 29 | 64.7% |
| source: GMOT-40 | 15 | 48.5% |
| source: ImageNet-VID | 93 | 37.0% |
| source: LVVIS | 7482 | 37.9% |
| source: OVIS | 27 | 85.0% |
| source: UVO | 113 | 45.4% |
| source: YouTube-VIS-2021 | 49 | 61.7% |

### Absent prompts (share of prompts)

| threshold | group | prompts | fired | misnaming (confirmed) | unlabelled only (needs review) |
|---:|---|---:|---:|---:|---:|
| 0.05 | easy | 3944 | 3.5% | 1.8% | 1.7% |
| 0.05 | hard | 3944 | 30.3% | 24.5% | 5.8% |
| 0.05 | strict-hard | 1456 | 36.9% | 30.7% | 6.2% |
| 0.1 | easy | 3944 | 2.2% | 1.1% | 1.1% |
| 0.1 | hard | 3944 | 25.1% | 21.0% | 4.1% |
| 0.1 | strict-hard | 1456 | 31.5% | 27.1% | 4.5% |
| 0.25 | easy | 3944 | 0.9% | 0.4% | 0.5% |
| 0.25 | hard | 3944 | 16.4% | 14.3% | 2.1% |
| 0.25 | strict-hard | 1456 | 21.6% | 19.2% | 2.5% |
| 0.4 | easy | 3944 | 0.4% | 0.2% | 0.2% |
| 0.4 | hard | 3944 | 11.9% | 10.6% | 1.3% |
| 0.4 | strict-hard | 1456 | 16.2% | 14.8% | 1.4% |

## yolov8l-worldv2_img640_fp16_separate

### Present prompts

| threshold | prompts | video found | frame hit rate | frame hit (lenient) | box recall | box recall (lenient) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 7808 | 67.4% | 55.8% | 57.6% | 52.8% | 54.7% |
| 0.1 | 7808 | 61.7% | 49.8% | 51.5% | 46.9% | 48.6% |
| 0.25 | 7808 | 52.2% | 40.5% | 41.9% | 37.8% | 39.2% |
| 0.4 | 7808 | 45.8% | 34.1% | 35.2% | 31.6% | 32.7% |

Present, frame hit rate at threshold 0.25, by group:

| group | prompts | frame hit rate |
|---|---:|---:|
| split: base | 4728 | 49.7% |
| split: novel | 3080 | 26.5% |
| size_bin: large (>10%) | 3338 | 50.7% |
| size_bin: medium (1-10%) | 2824 | 34.3% |
| size_bin: small (<1% of frame) | 1646 | 30.7% |
| source: AnimalTrack | 29 | 69.5% |
| source: GMOT-40 | 15 | 53.5% |
| source: ImageNet-VID | 93 | 43.4% |
| source: LVVIS | 7482 | 40.0% |
| source: OVIS | 27 | 86.3% |
| source: UVO | 113 | 44.0% |
| source: YouTube-VIS-2021 | 49 | 67.3% |

### Absent prompts (share of prompts)

| threshold | group | prompts | fired | misnaming (confirmed) | unlabelled only (needs review) |
|---:|---|---:|---:|---:|---:|
| 0.05 | easy | 3944 | 6.0% | 4.4% | 1.6% |
| 0.05 | hard | 3944 | 45.8% | 42.4% | 3.4% |
| 0.05 | strict-hard | 1456 | 53.6% | 50.3% | 3.4% |
| 0.1 | easy | 3944 | 3.3% | 2.3% | 1.0% |
| 0.1 | hard | 3944 | 35.9% | 33.1% | 2.8% |
| 0.1 | strict-hard | 1456 | 43.5% | 40.5% | 3.0% |
| 0.25 | easy | 3944 | 1.1% | 0.8% | 0.4% |
| 0.25 | hard | 3944 | 24.6% | 22.9% | 1.8% |
| 0.25 | strict-hard | 1456 | 31.6% | 29.7% | 1.9% |
| 0.4 | easy | 3944 | 0.5% | 0.3% | 0.2% |
| 0.4 | hard | 3944 | 17.1% | 16.1% | 1.0% |
| 0.4 | strict-hard | 1456 | 22.9% | 21.8% | 1.2% |

## yolov8m-worldv2_img480_fp16_separate

### Present prompts

| threshold | prompts | video found | frame hit rate | frame hit (lenient) | box recall | box recall (lenient) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 7808 | 65.9% | 53.9% | 55.7% | 50.7% | 52.6% |
| 0.1 | 7808 | 60.0% | 48.2% | 49.8% | 45.0% | 46.8% |
| 0.25 | 7808 | 50.5% | 38.8% | 40.1% | 35.9% | 37.3% |
| 0.4 | 7808 | 43.7% | 31.8% | 33.0% | 29.4% | 30.5% |

Present, frame hit rate at threshold 0.25, by group:

| group | prompts | frame hit rate |
|---|---:|---:|
| split: base | 4728 | 47.3% |
| split: novel | 3080 | 25.6% |
| size_bin: large (>10%) | 3338 | 50.7% |
| size_bin: medium (1-10%) | 2824 | 32.1% |
| size_bin: small (<1% of frame) | 1646 | 26.0% |
| source: AnimalTrack | 29 | 60.9% |
| source: GMOT-40 | 15 | 52.4% |
| source: ImageNet-VID | 93 | 50.3% |
| source: LVVIS | 7482 | 38.1% |
| source: OVIS | 27 | 79.0% |
| source: UVO | 113 | 45.0% |
| source: YouTube-VIS-2021 | 49 | 62.3% |

### Absent prompts (share of prompts)

| threshold | group | prompts | fired | misnaming (confirmed) | unlabelled only (needs review) |
|---:|---|---:|---:|---:|---:|
| 0.05 | easy | 3944 | 5.3% | 3.7% | 1.6% |
| 0.05 | hard | 3944 | 44.1% | 40.5% | 3.5% |
| 0.05 | strict-hard | 1456 | 51.3% | 47.6% | 3.7% |
| 0.1 | easy | 3944 | 2.8% | 2.1% | 0.7% |
| 0.1 | hard | 3944 | 35.2% | 33.0% | 2.3% |
| 0.1 | strict-hard | 1456 | 42.7% | 40.3% | 2.3% |
| 0.25 | easy | 3944 | 1.2% | 1.0% | 0.3% |
| 0.25 | hard | 3944 | 24.0% | 22.6% | 1.4% |
| 0.25 | strict-hard | 1456 | 30.4% | 28.6% | 1.8% |
| 0.4 | easy | 3944 | 0.7% | 0.5% | 0.2% |
| 0.4 | hard | 3944 | 16.6% | 15.9% | 0.7% |
| 0.4 | strict-hard | 1456 | 22.1% | 21.0% | 1.1% |

## yolov8m-worldv2_img640_fp16_joint

### Present prompts

| threshold | prompts | video found | frame hit rate | frame hit (lenient) | box recall | box recall (lenient) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 7808 | 60.7% | 48.9% | 50.5% | 46.2% | 47.9% |
| 0.1 | 7808 | 55.7% | 44.2% | 45.7% | 41.5% | 43.1% |
| 0.25 | 7808 | 47.6% | 36.4% | 37.9% | 33.8% | 35.3% |
| 0.4 | 7808 | 41.5% | 30.3% | 31.5% | 28.0% | 29.2% |

Present, frame hit rate at threshold 0.25, by group:

| group | prompts | frame hit rate |
|---|---:|---:|
| split: base | 4728 | 45.5% |
| split: novel | 3080 | 22.3% |
| size_bin: large (>10%) | 3338 | 44.9% |
| size_bin: medium (1-10%) | 2824 | 29.8% |
| size_bin: small (<1% of frame) | 1646 | 30.3% |
| source: AnimalTrack | 29 | 61.7% |
| source: GMOT-40 | 15 | 49.1% |
| source: ImageNet-VID | 93 | 37.7% |
| source: LVVIS | 7482 | 35.8% |
| source: OVIS | 27 | 73.5% |
| source: UVO | 113 | 46.4% |
| source: YouTube-VIS-2021 | 49 | 58.6% |

### Absent prompts (share of prompts)

| threshold | group | prompts | fired | misnaming (confirmed) | unlabelled only (needs review) |
|---:|---|---:|---:|---:|---:|
| 0.05 | easy | 3944 | 3.9% | 1.7% | 2.2% |
| 0.05 | hard | 3944 | 30.7% | 25.1% | 5.6% |
| 0.05 | strict-hard | 1456 | 36.4% | 30.8% | 5.6% |
| 0.1 | easy | 3944 | 1.9% | 0.9% | 1.1% |
| 0.1 | hard | 3944 | 24.0% | 20.2% | 3.8% |
| 0.1 | strict-hard | 1456 | 29.7% | 25.9% | 3.8% |
| 0.25 | easy | 3944 | 0.9% | 0.4% | 0.5% |
| 0.25 | hard | 3944 | 15.4% | 13.8% | 1.6% |
| 0.25 | strict-hard | 1456 | 20.3% | 18.5% | 1.8% |
| 0.4 | easy | 3944 | 0.5% | 0.2% | 0.3% |
| 0.4 | hard | 3944 | 11.0% | 10.0% | 1.0% |
| 0.4 | strict-hard | 1456 | 15.1% | 13.8% | 1.3% |

## yolov8m-worldv2_img640_fp16_separate

### Present prompts

| threshold | prompts | video found | frame hit rate | frame hit (lenient) | box recall | box recall (lenient) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 7808 | 66.7% | 54.2% | 56.1% | 50.9% | 52.9% |
| 0.1 | 7808 | 60.7% | 48.3% | 50.0% | 45.1% | 47.0% |
| 0.25 | 7808 | 50.7% | 38.5% | 40.0% | 35.7% | 37.2% |
| 0.4 | 7808 | 43.5% | 31.7% | 32.8% | 29.2% | 30.3% |

Present, frame hit rate at threshold 0.25, by group:

| group | prompts | frame hit rate |
|---|---:|---:|
| split: base | 4728 | 46.9% |
| split: novel | 3080 | 25.6% |
| size_bin: large (>10%) | 3338 | 49.1% |
| size_bin: medium (1-10%) | 2824 | 31.5% |
| size_bin: small (<1% of frame) | 1646 | 29.1% |
| source: AnimalTrack | 29 | 69.1% |
| source: GMOT-40 | 15 | 59.2% |
| source: ImageNet-VID | 93 | 50.0% |
| source: LVVIS | 7482 | 37.8% |
| source: OVIS | 27 | 79.3% |
| source: UVO | 113 | 44.8% |
| source: YouTube-VIS-2021 | 49 | 60.8% |

### Absent prompts (share of prompts)

| threshold | group | prompts | fired | misnaming (confirmed) | unlabelled only (needs review) |
|---:|---|---:|---:|---:|---:|
| 0.05 | easy | 3944 | 6.7% | 4.8% | 1.9% |
| 0.05 | hard | 3944 | 45.4% | 41.9% | 3.5% |
| 0.05 | strict-hard | 1456 | 52.3% | 48.6% | 3.7% |
| 0.1 | easy | 3944 | 3.6% | 2.5% | 1.1% |
| 0.1 | hard | 3944 | 36.4% | 33.7% | 2.7% |
| 0.1 | strict-hard | 1456 | 43.5% | 40.2% | 3.2% |
| 0.25 | easy | 3944 | 1.1% | 0.8% | 0.3% |
| 0.25 | hard | 3944 | 23.9% | 22.5% | 1.4% |
| 0.25 | strict-hard | 1456 | 31.0% | 29.3% | 1.6% |
| 0.4 | easy | 3944 | 0.6% | 0.4% | 0.2% |
| 0.4 | hard | 3944 | 17.5% | 16.6% | 0.9% |
| 0.4 | strict-hard | 1456 | 23.4% | 22.3% | 1.1% |

## yolov8s-worldv2_img480_fp16_separate

### Present prompts

| threshold | prompts | video found | frame hit rate | frame hit (lenient) | box recall | box recall (lenient) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 7808 | 68.9% | 55.3% | 57.5% | 51.9% | 54.2% |
| 0.1 | 7808 | 62.8% | 48.9% | 50.9% | 45.6% | 47.6% |
| 0.25 | 7808 | 52.0% | 38.5% | 40.0% | 35.7% | 37.2% |
| 0.4 | 7808 | 44.5% | 31.3% | 32.5% | 28.9% | 30.1% |

Present, frame hit rate at threshold 0.25, by group:

| group | prompts | frame hit rate |
|---|---:|---:|
| split: base | 4728 | 46.6% |
| split: novel | 3080 | 26.0% |
| size_bin: large (>10%) | 3338 | 53.6% |
| size_bin: medium (1-10%) | 2824 | 31.8% |
| size_bin: small (<1% of frame) | 1646 | 19.0% |
| source: AnimalTrack | 29 | 48.6% |
| source: GMOT-40 | 15 | 30.4% |
| source: ImageNet-VID | 93 | 42.4% |
| source: LVVIS | 7482 | 38.1% |
| source: OVIS | 27 | 72.3% |
| source: UVO | 113 | 41.5% |
| source: YouTube-VIS-2021 | 49 | 56.6% |

### Absent prompts (share of prompts)

| threshold | group | prompts | fired | misnaming (confirmed) | unlabelled only (needs review) |
|---:|---|---:|---:|---:|---:|
| 0.05 | easy | 3944 | 10.2% | 8.5% | 1.7% |
| 0.05 | hard | 3944 | 51.9% | 48.4% | 3.5% |
| 0.05 | strict-hard | 1456 | 57.6% | 53.9% | 3.6% |
| 0.1 | easy | 3944 | 6.0% | 4.9% | 1.1% |
| 0.1 | hard | 3944 | 41.5% | 38.6% | 2.9% |
| 0.1 | strict-hard | 1456 | 46.5% | 43.3% | 3.2% |
| 0.25 | easy | 3944 | 2.6% | 2.0% | 0.6% |
| 0.25 | hard | 3944 | 27.1% | 25.5% | 1.6% |
| 0.25 | strict-hard | 1456 | 31.9% | 29.9% | 1.9% |
| 0.4 | easy | 3944 | 1.3% | 1.0% | 0.3% |
| 0.4 | hard | 3944 | 19.4% | 18.5% | 0.9% |
| 0.4 | strict-hard | 1456 | 23.8% | 22.7% | 1.1% |

## yolov8s-worldv2_img640_fp16_joint

### Present prompts

| threshold | prompts | video found | frame hit rate | frame hit (lenient) | box recall | box recall (lenient) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 7808 | 69.2% | 55.2% | 57.5% | 51.8% | 54.1% |
| 0.1 | 7808 | 63.7% | 49.3% | 51.5% | 45.8% | 48.0% |
| 0.25 | 7808 | 52.4% | 37.8% | 39.6% | 34.7% | 36.4% |
| 0.4 | 7808 | 43.2% | 29.0% | 30.4% | 26.5% | 27.8% |

Present, frame hit rate at threshold 0.25, by group:

| group | prompts | frame hit rate |
|---|---:|---:|
| split: base | 4728 | 44.9% |
| split: novel | 3080 | 27.0% |
| size_bin: large (>10%) | 3338 | 47.5% |
| size_bin: medium (1-10%) | 2824 | 33.6% |
| size_bin: small (<1% of frame) | 1646 | 25.6% |
| source: AnimalTrack | 29 | 55.3% |
| source: GMOT-40 | 15 | 39.2% |
| source: ImageNet-VID | 93 | 35.7% |
| source: LVVIS | 7482 | 37.4% |
| source: OVIS | 27 | 69.3% |
| source: UVO | 113 | 43.9% |
| source: YouTube-VIS-2021 | 49 | 60.1% |

### Absent prompts (share of prompts)

| threshold | group | prompts | fired | misnaming (confirmed) | unlabelled only (needs review) |
|---:|---|---:|---:|---:|---:|
| 0.05 | easy | 3944 | 8.3% | 4.5% | 3.8% |
| 0.05 | hard | 3944 | 38.5% | 31.8% | 6.6% |
| 0.05 | strict-hard | 1456 | 43.1% | 36.1% | 7.0% |
| 0.1 | easy | 3944 | 4.8% | 2.9% | 1.9% |
| 0.1 | hard | 3944 | 30.1% | 25.5% | 4.7% |
| 0.1 | strict-hard | 1456 | 35.2% | 30.4% | 4.7% |
| 0.25 | easy | 3944 | 1.8% | 1.1% | 0.7% |
| 0.25 | hard | 3944 | 18.9% | 16.8% | 2.1% |
| 0.25 | strict-hard | 1456 | 23.1% | 20.8% | 2.3% |
| 0.4 | easy | 3944 | 0.8% | 0.4% | 0.4% |
| 0.4 | hard | 3944 | 12.9% | 11.8% | 1.2% |
| 0.4 | strict-hard | 1456 | 16.4% | 14.8% | 1.6% |

## yolov8s-worldv2_img640_fp16_separate

### Present prompts

| threshold | prompts | video found | frame hit rate | frame hit (lenient) | box recall | box recall (lenient) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 7808 | 70.5% | 56.7% | 59.1% | 53.2% | 55.7% |
| 0.1 | 7808 | 64.2% | 50.0% | 52.1% | 46.7% | 48.8% |
| 0.25 | 7808 | 53.3% | 39.1% | 40.9% | 36.1% | 38.0% |
| 0.4 | 7808 | 45.3% | 32.0% | 33.4% | 29.5% | 30.9% |

Present, frame hit rate at threshold 0.25, by group:

| group | prompts | frame hit rate |
|---|---:|---:|
| split: base | 4728 | 46.5% |
| split: novel | 3080 | 27.6% |
| size_bin: large (>10%) | 3338 | 52.3% |
| size_bin: medium (1-10%) | 2824 | 32.7% |
| size_bin: small (<1% of frame) | 1646 | 23.1% |
| source: AnimalTrack | 29 | 62.4% |
| source: GMOT-40 | 15 | 43.2% |
| source: ImageNet-VID | 93 | 39.8% |
| source: LVVIS | 7482 | 38.6% |
| source: OVIS | 27 | 73.3% |
| source: UVO | 113 | 43.0% |
| source: YouTube-VIS-2021 | 49 | 61.4% |

### Absent prompts (share of prompts)

| threshold | group | prompts | fired | misnaming (confirmed) | unlabelled only (needs review) |
|---:|---|---:|---:|---:|---:|
| 0.05 | easy | 3944 | 12.0% | 9.9% | 2.1% |
| 0.05 | hard | 3944 | 53.7% | 50.0% | 3.7% |
| 0.05 | strict-hard | 1456 | 59.4% | 55.3% | 4.1% |
| 0.1 | easy | 3944 | 6.8% | 5.6% | 1.2% |
| 0.1 | hard | 3944 | 43.6% | 40.5% | 3.0% |
| 0.1 | strict-hard | 1456 | 48.3% | 45.4% | 2.9% |
| 0.25 | easy | 3944 | 2.5% | 1.9% | 0.6% |
| 0.25 | hard | 3944 | 28.9% | 27.3% | 1.5% |
| 0.25 | strict-hard | 1456 | 33.8% | 32.1% | 1.6% |
| 0.4 | easy | 3944 | 1.5% | 1.1% | 0.4% |
| 0.4 | hard | 3944 | 20.6% | 19.6% | 1.0% |
| 0.4 | strict-hard | 1456 | 24.7% | 23.4% | 1.2% |

