# ECCIL

Official research code for **“ECCIL: Element Contribution-Guided Cross-Modal Interaction Learning for Multimodal Fake News Detection.”** ECCIL estimates the influence of textual and visual elements through input perturbations and uses the resulting signals to guide bidirectional cross-modal interaction for binary fake-news detection.

## Overview

ECCIL is organized around three components:

1. **Dual-path representation learning:** BERT and Swin Transformer encode the original text-image pair and element-masked variants.
2. **Element-level contribution estimation (ELCE):** prediction changes caused by masking selected textual elements or detected image regions provide signed guidance scores.
3. **Contribution-guided cross-modal interaction:** bidirectional text-image interaction combines base, contribution-conditioned, similarity, and discrepancy views using a learned gate before classification.

## Framework

![ECCIL framework](method.png)

The upper part of the figure illustrates original and perturbed text/image encoding and contribution estimation. The lower part shows bidirectional multi-channel interaction, gated fusion, and binary classification.

## Requirements
Dependencies imported by the formal execution path are:

| Dependency | Version |
| --- | --- |
| Python | 3.9.25 |
| PyTorch | 2.8.0 |
| torchvision | 0.22.0 |

The model assets expected by the manuscript and configuration are:

- `bert-base-chinese` for Weibo and `bert-base-uncased` for PHEME and PolitiFact;
- `swin-base-patch4-window7-224` for images;
- `zh_core_web_sm` for Weibo and `en_core_web_sm` for PHEME and PolitiFact;

## Datasets

| Dataset | Language/domain | Train | Test | Fake | Real | Images |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Weibo | Chinese social-media posts | 6,858 | 2,243 | 4,532 | 4,611 | 9,143 |
| PHEME | English breaking-news tweets | 1,712 | 541 | 634 | 1,619 | 2,253 |
| PolitiFact | English political news | 273 | 77 | 147 | 203 | 350 |

### Weibo

The formal data loader expects the following configured layout. Image paths are read from the CSV and may be absolute or relative to `dataset.root`.

```text
/PATH/TO/WEIBO/
├── train_weibo_final3.csv
├── test_weibo_final3.csv
└── ... image files referenced by the CSV ...
```

The default positional CSV mapping in `configs/weibo.yaml` is:

| Field | Zero-based column index |
| --- | ---: |
| sample ID | 0 |
| image path(s) | 1 |
| label | 2 |
| text | 3 |
| has-image flag | 6 |

Multiple image paths are separated by `|`. With `image_selection: seeded`, one path is selected deterministically from the split seed and sample ID. A sample with no image receives a blank 224 x 224 RGB image. The default `missing_image_policy: error` raises an error when a referenced image cannot be read.

Text is padded or truncated to 300 tokens by the configured fast BERT tokenizer. Images are processed by the `AutoImageProcessor` stored with the local Swin model. The current code selects up to five spaCy named entities and up to five Faster R-CNN detections with confidence at least 0.7, masks one selected span or blackens one detected box at a time, and aligns the resulting signed scores to BERT tokens or Swin patches.

If `dataset.val_csv` is unset, the formal loader creates a fixed, non-stratified 10% validation subset from the training CSV with `training.split_seed: 2024`. The official test CSV is evaluated only after validation-based model selection. Exact split indices are saved with each run.

### PHEME

The manuscript reports 1,712 training and 541 test samples, uses `bert-base-uncased` and `en_core_web_sm`, and sets batch size 32 and task learning rate `1e-3`.

**TODO:** No PHEME data adapter, directory convention, column schema, validation protocol, configuration file, or executable training/evaluation command is included in the repository.

### PolitiFact

The manuscript reports 273 training and 77 test samples, uses `bert-base-uncased` and `en_core_web_sm`, and sets batch size 16 and task learning rate `1e-3`.

**TODO:** No PolitiFact data adapter, directory convention, column schema, validation protocol, configuration file, or executable training/evaluation command is included in the repository.

## Project Structure

```text
ECCIL/
├── configs/
│   └── weibo.yaml                 # Formal Weibo experiment configuration
├── eccil/
│   ├── __init__.py
│   ├── config.py                  # YAML loading and validation
│   ├── data.py                    # Weibo records, preprocessing, and data splits
│   ├── model.py                   # Formal ECCIL model and local asset loading
│   └── variants.py                # Full model and clean ablation definitions
├── tests/
│   └── test_eccil_core.py         # Asset-free formal-core tests
├── exp/
│   ├── multi_channel_1.py         # Preserved legacy model snapshot
│   ├── train spacy_1.py           # Preserved legacy trainer snapshot
│   └── weibo_dataset_1.py         # Preserved legacy Weibo loader snapshot
├── phase1_backup_20260826/        # Earlier implementation backup
├── readme/                        # Four README files used only as style references
├── train_eccil.py                 # Formal leakage-free training/evaluation entry point
├── multi_channel.py               # Legacy-compatible implementation; not the formal entry point
├── train spacy.py                 # Legacy trainer; not recommended for formal runs
├── eccil_sanity_check.py          # Legacy/refactor sanity checks
├── method.png                     # Manuscript framework figure
├── weibo.png                      # Manuscript channel-ablation figure
├── pheme.png                      # Manuscript channel-ablation figure
├── politi.png                     # Manuscript channel-ablation figure
├── sn-article.tex                 # Manuscript source
└── sn-article.pdf                 # Compiled manuscript
```
### Formal Weibo configuration

| Parameter | configs |
| --- | ---: |
| Batch size | 32 |
| Maximum epochs | 50 |
| Task learning rate | 1e-3 |
| BERT learning rate | 1e-5 |
| Swin learning rate | 1e-5 |
| Optimizer | AdamW |

## Contact

Project GitHub: [https://github.com/zhangyanan-maker/ECCIL](https://github.com/zhangyanan-maker/ECCIL)

