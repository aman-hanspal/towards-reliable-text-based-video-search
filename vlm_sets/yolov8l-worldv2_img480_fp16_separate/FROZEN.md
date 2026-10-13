# VLM test set (frozen)

- created (UTC): 2026-10-10T06:57:03.323905+00:00
- detector run: yolov8l-worldv2_img480_fp16_separate
- threshold: 0.25; match IoU: 0.5
- crop margin: 0.2; min side: 48px; long side shrunk to <= 768px; JPEG quality 95
- seed: 20261009; B sample: 600; pilot per set: 10
- candidates before sampling: A 898, B 4055, C 64
- items: A 888, B 600, C 64, pilot_A 10, pilot_B 10
- unreadable frames: 0
- items.csv SHA-256: `c3820c93fc50e9eacda2a8fe735375eb27ba6abc5d10b3cf21ac55382a2e3fff`
- export script SHA-256: `b55df73cd0ce210d8a1110c332a0892a2871986ef17f4447f3fadd387fcd678e`

Pilot items only check the VLM's answer format; they are excluded from all results.
