# VLM test set (frozen)

- created (UTC): 2026-10-10T01:14:30.311911+00:00
- detector run: yolov8l-worldv2_img640_fp16_separate
- threshold: 0.25; match IoU: 0.5
- crop margin: 0.2; min side: 48px; long side shrunk to <= 768px; JPEG quality 95
- seed: 20261009; B sample: 600; pilot per set: 10
- candidates before sampling: A 932, B 4078, C 85
- items: A 922, B 600, C 85, pilot_A 10, pilot_B 10
- unreadable frames: 0
- items.csv SHA-256: `3752a139e2b90b22d3f5e347f8ca7e3c5eccc75c68ab6952a51b981e5a9a3595`
- export script SHA-256: `b55df73cd0ce210d8a1110c332a0892a2871986ef17f4447f3fadd387fcd678e`

Pilot items only check the VLM's answer format; they are excluded from all results.
