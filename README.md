# ECCIL

Official research code for **“ECCIL: Element Contribution-Guided Cross-Modal Interaction Learning for Multimodal Fake News Detection.”** ECCIL estimates the influence of textual and visual elements through input perturbations and uses the resulting signals to guide bidirectional cross-modal interaction for binary fake-news detection.

## Overview

ECCIL is organized around three components:

1. **Dual-path representation learning:** BERT and Swin Transformer encode the original text-image pair and element-masked variants.
2. **Element-level contribution estimation (ELCE):** prediction changes caused by masking selected textual elements or detected image regions provide signed guidance scores.
3. **Contribution-guided cross-modal interaction:** bidirectional text-image interaction combines base, contribution-conditioned, similarity, and discrepancy views using a learned gate before classification.

## Framework

![ECCIL framework](./method.png)

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

## Dataset

The datasets used in this study are publicly available from the following repositories:

- **Weibo**: [MMFN-fake-news-detection](https://github.com/zhouyangming/MMFN-fake-news-detection)
- **PHEME**: [Figshare](https://doi.org/10.6084/m9.figshare.4010619.v1)
- **PolitiFact**: [MMFN-fake-news-detection](https://github.com/zhouyangming/MMFN-fake-news-detection)

Please download the datasets from the original sources and organize them according to the directory structure required by the training scripts.

## Project Structure

```text
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

