# VLM checker (qwen3-vl:2b-instruct) on all 6 separate settings

---

# VLM checker results

- test set: yolov8m-worldv2_img640_fp16_separate, detector threshold 0.25
- items: A 909, B 600, C 67, pilot_A 10, pilot_B 10
- pilot items excluded; 'unclear' answers count as 'no'; [95% Wilson interval]

## rtx3050__qwen3-vl-2b-instruct

- model qwen3-vl:2b-instruct (weights 4b22b97d9986), Ollama 0.40.2, device rtx3050
- model in memory 1949 MB, on GPU 1754 MB (share 0.9); thinking switch False
- answers so far: 1576 main items; unclear: 0

### 1-2. Accuracy on the sets with known answers

| group | share |
|---|---|
| **A removed** (said no to a lookalike) - all | 69.0% [65.9, 71.9] (n=909) |
| A removed - hard prompts | 68.1% [64.9, 71.1] (n=877) |
| A removed - strict-hard | 59.9% [55.2, 64.5] (n=424) |
| A removed - easy prompts | 93.8% [79.9, 98.3] (n=32) |
| A removed - novel prompt | 67.6% [62.1, 72.6] (n=299) |
| A removed - base prompt | 69.7% [65.9, 73.2] (n=610) |
| **B lost** (said no to a real object) - all | 10.2% [8.0, 12.8] (n=600) |
| B lost - small (<1%) object | 2.3% [0.6, 8.0] (n=87) |
| B lost - medium (1-10%) object | 4.9% [2.6, 9.1] (n=183) |
| B lost - large (>10%) object | 15.2% [11.7, 19.4] (n=330) |
| B lost - novel | 3.9% [1.9, 7.8] (n=180) |
| B lost - base | 12.9% [10.0, 16.4] (n=420) |

### 3. Set C (fired where nothing is labelled)

- VLM said yes: 59.7% [47.7, 70.6] (n=67)
- eye check not filled in yet (c_eye_check.csv)

### 4. Before -> after, detector threshold 0.25

| prompt type | before | after VLM check |
|---|---:|---:|
| present: found in video | 50.7% | 45.5% |
| hard (lookalike) false alarm | 23.9% | 8.1% |
| easy (unrelated) false alarm | 1.1% | 0.2% |

(after = before x share the VLM keeps; misnaming and unlabelled-area alarms weighted by how many there are)

### 5. Same loss of real detections with a stricter YOLO threshold instead

- VLM loses 10.2% of real detections; a YOLO threshold of 0.361 loses about the same
- hard false-alarm rate at that threshold: 18.7% (VLM: see table above; before: 23.9%)

### 6. Cost

- seconds per crop (wall): median 1.71, mean 7.75, 90th percentile 1.80; total 203.5 min for 1576 crops


---

# VLM checker results

- test set: yolov8s-worldv2_img640_fp16_separate, detector threshold 0.25
- items: A 1144, B 600, C 83, pilot_A 10, pilot_B 10
- pilot items excluded; 'unclear' answers count as 'no'; [95% Wilson interval]

## rtx3050__qwen3-vl-2b-instruct

- model qwen3-vl:2b-instruct (weights 4b22b97d9986), Ollama 0.40.2, device rtx3050
- model in memory 1949 MB, on GPU 1754 MB (share 0.9); thinking switch False
- answers so far: 1827 main items; unclear: 0

### 1-2. Accuracy on the sets with known answers

| group | share |
|---|---|
| **A removed** (said no to a lookalike) - all | 71.0% [68.3, 73.5] (n=1144) |
| A removed - hard prompts | 69.5% [66.6, 72.2] (n=1068) |
| A removed - strict-hard | 60.5% [56.0, 64.8] (n=466) |
| A removed - easy prompts | 92.1% [83.8, 96.3] (n=76) |
| A removed - novel prompt | 69.2% [64.5, 73.5] (n=402) |
| A removed - base prompt | 72.0% [68.6, 75.1] (n=742) |
| **B lost** (said no to a real object) - all | 13.0% [10.5, 15.9] (n=600) |
| B lost - small (<1%) object | 1.8% [0.3, 9.3] (n=57) |
| B lost - medium (1-10%) object | 3.4% [1.6, 7.2] (n=178) |
| B lost - large (>10%) object | 19.5% [15.7, 23.8] (n=365) |
| B lost - novel | 7.4% [4.4, 12.0] (n=190) |
| B lost - base | 15.6% [12.4, 19.4] (n=410) |

### 3. Set C (fired where nothing is labelled)

- VLM said yes: 53.0% [42.4, 63.4] (n=83)
- eye check not filled in yet (c_eye_check.csv)

### 4. Before -> after, detector threshold 0.25

| prompt type | before | after VLM check |
|---|---:|---:|
| present: found in video | 53.3% | 46.4% |
| hard (lookalike) false alarm | 28.9% | 9.1% |
| easy (unrelated) false alarm | 2.5% | 0.5% |

(after = before x share the VLM keeps; misnaming and unlabelled-area alarms weighted by how many there are)

### 5. Same loss of real detections with a stricter YOLO threshold instead

- VLM loses 13.0% of real detections; a YOLO threshold of 0.398 loses about the same
- hard false-alarm rate at that threshold: 20.7% (VLM: see table above; before: 28.9%)

### 6. Cost

- seconds per crop (wall): median 1.70, mean 4.85, 90th percentile 1.76; total 147.8 min for 1827 crops


---

# VLM checker results

- test set: yolov8l-worldv2_img640_fp16_separate, detector threshold 0.25
- items: A 922, B 600, C 85, pilot_A 10, pilot_B 10
- pilot items excluded; 'unclear' answers count as 'no'; [95% Wilson interval]

## rtx3050__qwen3-vl-2b-instruct

- model qwen3-vl:2b-instruct (weights 4b22b97d9986), Ollama 0.40.2, device rtx3050
- model in memory 1949 MB, on GPU 1754 MB (share 0.9); thinking switch False
- answers so far: 1607 main items; unclear: 0

### 1-2. Accuracy on the sets with known answers

| group | share |
|---|---|
| **A removed** (said no to a lookalike) - all | 66.3% [63.2, 69.2] (n=922) |
| A removed - hard prompts | 65.4% [62.2, 68.4] (n=893) |
| A removed - strict-hard | 59.5% [54.8, 64.0] (n=427) |
| A removed - easy prompts | 93.1% [78.0, 98.1] (n=29) |
| A removed - novel prompt | 65.3% [59.7, 70.5] (n=297) |
| A removed - base prompt | 66.7% [62.9, 70.3] (n=625) |
| **B lost** (said no to a real object) - all | 11.0% [8.7, 13.8] (n=600) |
| B lost - small (<1%) object | 2.2% [0.6, 7.8] (n=89) |
| B lost - medium (1-10%) object | 4.4% [2.3, 8.5] (n=181) |
| B lost - large (>10%) object | 17.0% [13.3, 21.4] (n=330) |
| B lost - novel | 4.3% [2.2, 8.2] (n=188) |
| B lost - base | 14.1% [11.1, 17.8] (n=412) |

### 3. Set C (fired where nothing is labelled)

- VLM said yes: 57.6% [47.0, 67.6] (n=85)
- eye check not filled in yet (c_eye_check.csv)

### 4. Before -> after, detector threshold 0.25

| prompt type | before | after VLM check |
|---|---:|---:|
| present: found in video | 52.2% | 46.5% |
| hard (lookalike) false alarm | 24.6% | 9.0% |
| easy (unrelated) false alarm | 1.1% | 0.2% |

(after = before x share the VLM keeps; misnaming and unlabelled-area alarms weighted by how many there are)

### 5. Same loss of real detections with a stricter YOLO threshold instead

- VLM loses 11.0% of real detections; a YOLO threshold of 0.413 loses about the same
- hard false-alarm rate at that threshold: 16.5% (VLM: see table above; before: 24.6%)

### 6. Cost

- seconds per crop (wall): median 1.71, mean 5.70, 90th percentile 1.80; total 152.5 min for 1607 crops


---

# VLM checker results

- test set: yolov8m-worldv2_img480_fp16_separate, detector threshold 0.25
- items: A 918, B 600, C 65, pilot_A 10, pilot_B 10
- pilot items excluded; 'unclear' answers count as 'no'; [95% Wilson interval]

## rtx3050__qwen3-vl-2b-instruct

- model qwen3-vl:2b-instruct (weights 4b22b97d9986), Ollama 0.40.2, device rtx3050
- model in memory 1949 MB, on GPU 1754 MB (share 0.9); thinking switch False
- answers so far: 1583 main items; unclear: 0

### 1-2. Accuracy on the sets with known answers

| group | share |
|---|---|
| **A removed** (said no to a lookalike) - all | 67.4% [64.3, 70.4] (n=918) |
| A removed - hard prompts | 66.5% [63.3, 69.5] (n=880) |
| A removed - strict-hard | 58.6% [53.8, 63.3] (n=411) |
| A removed - easy prompts | 89.5% [75.9, 95.8] (n=38) |
| A removed - novel prompt | 64.3% [58.7, 69.5] (n=294) |
| A removed - base prompt | 68.9% [65.2, 72.4] (n=624) |
| **B lost** (said no to a real object) - all | 11.3% [9.0, 14.1] (n=600) |
| B lost - small (<1%) object | 0.0% [0.0, 5.1] (n=72) |
| B lost - medium (1-10%) object | 5.1% [2.8, 9.1] (n=196) |
| B lost - large (>10%) object | 17.5% [13.8, 21.9] (n=332) |
| B lost - novel | 3.4% [1.6, 7.2] (n=176) |
| B lost - base | 14.6% [11.6, 18.3] (n=424) |

### 3. Set C (fired where nothing is labelled)

- VLM said yes: 52.3% [40.4, 64.0] (n=65)
- eye check not filled in yet (c_eye_check.csv)

### 4. Before -> after, detector threshold 0.25

| prompt type | before | after VLM check |
|---|---:|---:|
| present: found in video | 50.5% | 44.8% |
| hard (lookalike) false alarm | 24.0% | 8.3% |
| easy (unrelated) false alarm | 1.2% | 0.2% |

(after = before x share the VLM keeps; misnaming and unlabelled-area alarms weighted by how many there are)

### 5. Same loss of real detections with a stricter YOLO threshold instead

- VLM loses 11.3% of real detections; a YOLO threshold of 0.372 loses about the same
- hard false-alarm rate at that threshold: 17.4% (VLM: see table above; before: 24.0%)

### 6. Cost

- seconds per crop (wall): median 1.75, mean 3.93, 90th percentile 1.91; total 103.7 min for 1583 crops


---

# VLM checker results

- test set: yolov8s-worldv2_img480_fp16_separate, detector threshold 0.25
- items: A 1073, B 600, C 86, pilot_A 10, pilot_B 10
- pilot items excluded; 'unclear' answers count as 'no'; [95% Wilson interval]

## rtx3050__qwen3-vl-2b-instruct

- model qwen3-vl:2b-instruct (weights 4b22b97d9986), Ollama 0.40.2, device rtx3050
- model in memory 1949 MB, on GPU 1754 MB (share 0.9); thinking switch False
- answers so far: 1759 main items; unclear: 0

### 1-2. Accuracy on the sets with known answers

| group | share |
|---|---|
| **A removed** (said no to a lookalike) - all | 70.4% [67.6, 73.0] (n=1073) |
| A removed - hard prompts | 68.7% [65.7, 71.5] (n=996) |
| A removed - strict-hard | 59.6% [54.9, 64.2] (n=431) |
| A removed - easy prompts | 92.2% [84.0, 96.4] (n=77) |
| A removed - novel prompt | 69.1% [64.3, 73.6] (n=379) |
| A removed - base prompt | 71.0% [67.6, 74.3] (n=694) |
| **B lost** (said no to a real object) - all | 15.7% [13.0, 18.8] (n=600) |
| B lost - small (<1%) object | 5.1% [1.7, 13.9] (n=59) |
| B lost - medium (1-10%) object | 7.7% [4.7, 12.2] (n=196) |
| B lost - large (>10%) object | 22.0% [18.0, 26.7] (n=345) |
| B lost - novel | 10.2% [6.7, 15.2] (n=197) |
| B lost - base | 18.4% [14.9, 22.4] (n=403) |

### 3. Set C (fired where nothing is labelled)

- VLM said yes: 43.0% [33.1, 53.6] (n=86)
- eye check not filled in yet (c_eye_check.csv)

### 4. Before -> after, detector threshold 0.25

| prompt type | before | after VLM check |
|---|---:|---:|
| present: found in video | 52.0% | 43.9% |
| hard (lookalike) false alarm | 27.1% | 8.7% |
| easy (unrelated) false alarm | 2.6% | 0.4% |

(after = before x share the VLM keeps; misnaming and unlabelled-area alarms weighted by how many there are)

### 5. Same loss of real detections with a stricter YOLO threshold instead

- VLM loses 15.7% of real detections; a YOLO threshold of 0.398 loses about the same
- hard false-alarm rate at that threshold: 19.4% (VLM: see table above; before: 27.1%)

### 6. Cost

- seconds per crop (wall): median 1.76, mean 1.90, 90th percentile 2.37; total 55.8 min for 1759 crops


---

# VLM checker results

- test set: yolov8l-worldv2_img480_fp16_separate, detector threshold 0.25
- items: A 888, B 600, C 64, pilot_A 10, pilot_B 10
- pilot items excluded; 'unclear' answers count as 'no'; [95% Wilson interval]

## rtx3050__qwen3-vl-2b-instruct

- model qwen3-vl:2b-instruct (weights 4b22b97d9986), Ollama 0.40.2, device rtx3050
- model in memory 1949 MB, on GPU 1754 MB (share 0.9); thinking switch False
- answers so far: 1552 main items; unclear: 0

### 1-2. Accuracy on the sets with known answers

| group | share |
|---|---|
| **A removed** (said no to a lookalike) - all | 66.8% [63.6, 69.8] (n=888) |
| A removed - hard prompts | 65.9% [62.6, 69.0] (n=853) |
| A removed - strict-hard | 59.0% [54.2, 63.7] (n=415) |
| A removed - easy prompts | 88.6% [74.0, 95.5] (n=35) |
| A removed - novel prompt | 65.7% [60.2, 70.8] (n=306) |
| A removed - base prompt | 67.4% [63.4, 71.0] (n=582) |
| **B lost** (said no to a real object) - all | 12.8% [10.4, 15.7] (n=600) |
| B lost - small (<1%) object | 1.2% [0.2, 6.7] (n=81) |
| B lost - medium (1-10%) object | 5.6% [3.1, 10.0] (n=178) |
| B lost - large (>10%) object | 19.4% [15.5, 23.9] (n=341) |
| B lost - novel | 6.2% [3.5, 10.8] (n=177) |
| B lost - base | 15.6% [12.5, 19.4] (n=423) |

### 3. Set C (fired where nothing is labelled)

- VLM said yes: 59.4% [47.1, 70.5] (n=64)
- eye check not filled in yet (c_eye_check.csv)

### 4. Before -> after, detector threshold 0.25

| prompt type | before | after VLM check |
|---|---:|---:|
| present: found in video | 51.9% | 45.3% |
| hard (lookalike) false alarm | 23.2% | 8.3% |
| easy (unrelated) false alarm | 1.2% | 0.2% |

(after = before x share the VLM keeps; misnaming and unlabelled-area alarms weighted by how many there are)

### 5. Same loss of real detections with a stricter YOLO threshold instead

- VLM loses 12.8% of real detections; a YOLO threshold of 0.446 loses about the same
- hard false-alarm rate at that threshold: 14.8% (VLM: see table above; before: 23.2%)

### 6. Cost

- seconds per crop (wall): median 1.72, mean 1.73, 90th percentile 1.78; total 44.8 min for 1552 crops

