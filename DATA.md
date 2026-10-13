# Data, models and credits

## Dataset: OVT-B

This repository does **not** redistribute any images, videos or annotations.
Download OVT-B from its official repository and follow its terms:

- Repository: https://github.com/Coo1Sea/OVT-B-Dataset
- Paper: Haiji Liang and Ruize Han. *OVT-B: A New Large-Scale Benchmark for
  Open-Vocabulary Multi-Object Tracking.* NeurIPS 2024, Datasets and Benchmarks Track.
  arXiv:2410.17534

OVT-B (1,973 videos; 1,048 categories according to the OVT-B paper, 1,053 category entries in the
released annotation file, of which 535 base and 518 novel; we use the annotation file) is built from seven existing datasets. Each keeps
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
- The frozen VLM test lists (`vlm_sets/*/items.csv`): which detection box of which video, frame and word was
  checked, with its expected answer; no images
- The VLM's answers (`vlm_runs/`) and the first author's hand checks (`vlm_results/hand_check/`)
- All detector scores (`scores_full/prompt_scores.csv.gz`), result summaries, Table 1 and figures

## Models and tools

| Component | Use | Credit / license |
|---|---|---|
| YOLO-World v2 | open-vocabulary detector | Cheng et al., *YOLO-World: Real-Time Open-Vocabulary Object Detection*, CVPR 2024 |
| Ultralytics | detector runtime and weights | AGPL-3.0, https://github.com/ultralytics/ultralytics |
| CLIP (ViT-B/32) | text similarity for building the query set | Radford et al., ICML 2021; MIT license |
| WordNet (via NLTK) | word senses and Wu-Palmer similarity | Princeton University, WordNet license |
| Qwen3-VL 2B instruct (`qwen3-vl:2b-instruct`, via Ollama 0.40.2) | yes/no verification of detections | Qwen Team, *Qwen3-VL Technical Report*, arXiv:2511.21631, 2025; Apache-2.0 license |
| Ollama | local model server | MIT license, https://ollama.com |