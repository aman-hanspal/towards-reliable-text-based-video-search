# VLM test set (frozen)

- created (UTC): 2026-10-10T05:55:56.562032+00:00
- detector run: yolov8s-worldv2_img480_fp16_separate
- threshold: 0.25; match IoU: 0.5
- crop margin: 0.2; min side: 48px; long side shrunk to <= 768px; JPEG quality 95
- seed: 20261009; B sample: 600; pilot per set: 10
- candidates before sampling: A 1083, B 4062, C 86
- items: A 1073, B 600, C 86, pilot_A 10, pilot_B 10
- unreadable frames: 0
- items.csv SHA-256: `f9d817a9e51433c1e4cd4d9566d098151d341b1ab7c8b2d944d4d2568614d406`
- export script SHA-256: `b55df73cd0ce210d8a1110c332a0892a2871986ef17f4447f3fadd387fcd678e`

Pilot items only check the VLM's answer format; they are excluded from all results.
