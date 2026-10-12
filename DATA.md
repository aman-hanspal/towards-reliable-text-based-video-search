# Data, models and credits

## Dataset: OVT-B

This repository does **not** redistribute any images, videos or annotations.
Download OVT-B from its official repository and follow its terms:

- Repository: https://github.com/Coo1Sea/OVT-B-Dataset
- Paper: Haiji Liang and Ruize Han. *OVT-B: A New Large-Scale Benchmark for
  Open-Vocabulary Multi-Object Tracking.* NeurIPS 2024, Datasets and Benchmarks Track.
  arXiv:2410.17534

OVT-B (1,973 videos, 1,048 categories) is built from seven existing datasets. Each keeps
its own license and terms of use, which apply to any use of OVT-B:

| Source dataset | Original task |
|---|---|
| AnimalTrack | multi-object tracking (animals) |
| GMOT-40 | generic multi-object tracking |
| LV-VIS | large-vocabulary video instance segmentation |
| OVIS | occluded video instance segmentation |
| UVO | unidentified video objects |
| YouTube-VIS | video instance segmentation |
| ImageNet-VID | video object detection |

Please cite OVT-B and the relevant source datasets if you use this work.

```bibtex
@inproceedings{liang2024ovtb,
  title     = {OVT-B: A New Large-Scale Benchmark for Open-Vocabulary Multi-Object Tracking},
  author    = {Liang, Haiji and Han, Ruize},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2024}
}
```

## What we do share

- Our code (`scripts/`)
- The frozen query set (`query_set_v3/`): text prompts per video, derived from OVT-B
  category names, with no images or boxes
- Result summaries (reports and tables)

## Models and tools

| Component | Use | Credit / license |
|---|---|---|
| YOLO-World v2 | open-vocabulary detector | Cheng et al., *YOLO-World: Real-Time Open-Vocabulary Object Detection*, CVPR 2024 |
| Ultralytics | detector runtime and weights | AGPL-3.0, https://github.com/ultralytics/ultralytics |
| CLIP (ViT-B/32) | text similarity for building the query set | Radford et al., ICML 2021; MIT license |
| WordNet (via NLTK) | word senses and Wu-Palmer similarity | Princeton University, WordNet license |
| Local VLM (via Ollama) | yes/no verification of detections | MODEL NAME AND LICENSE TO ADD |
