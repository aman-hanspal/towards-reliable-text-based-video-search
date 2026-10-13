# VLM test set (frozen)

- created (UTC): 2026-10-09T22:41:22.074307+00:00
- detector run: yolov8s-worldv2_img640_fp16_separate
- threshold: 0.25; match IoU: 0.5
- crop margin: 0.2; min side: 48px; long side shrunk to <= 768px; JPEG quality 95
- seed: 20261009; B sample: 600; pilot per set: 10
- candidates before sampling: A 1154, B 4164, C 83
- items: A 1144, B 600, C 83, pilot_A 10, pilot_B 10
- unreadable frames: 0
- items.csv SHA-256: `68ea2bf2318a01053dc901116431997079f03187e4b5bd74136eca93826df44a`
- export script SHA-256: `b55df73cd0ce210d8a1110c332a0892a2871986ef17f4447f3fadd387fcd678e`

Pilot items only check the VLM's answer format; they are excluded from all results.
