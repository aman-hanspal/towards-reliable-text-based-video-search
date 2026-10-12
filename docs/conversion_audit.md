# Which script converted the videos

Counted from the `script_sha256` field of all 1,972 manifests in `prepared_videos/`.

| Videos | Script | SHA-256 |
|---:|---|---|
| 1 | `prepare_ovtb_video.py` | `ab8c5714bc5591180b8f32c6eaf6ebfb8b93757ac470bdcb0d089a312f015561` |
| 20 | `prepare_ovtb_batch.py` | `9629fcdded1c6a8f6b637e2855567ac84d86443ed7bfbe43fda78797063194dd` |
| 1,951 | `prepare_ovtb_batch_optimised.py` | `a4e7837dea449cce4009904a9ca3ba7e59828bb790057177b79d21b752a9f977` |

Every hash matches the script file in `scripts/`, so no conversion script was changed after use.
