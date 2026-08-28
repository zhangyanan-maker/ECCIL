# ECCIL

Official research code for **“ECCIL: Element Contribution-Guided Cross-Modal Interaction Learning for Multimodal Fake News Detection.”** ECCIL estimates the influence of textual and visual elements through input perturbations and uses the resulting signals to guide bidirectional cross-modal interaction for binary fake-news detection.

## Overview

ECCIL is organized around three components:

1. **Dual-path representation learning:** BERT and Swin Transformer encode the original text-image pair and element-masked variants.
2. **Element-level contribution estimation (ELCE):** prediction changes caused by masking selected textual elements or detected image regions provide signed guidance scores.
3. **Contribution-guided cross-modal interaction:** bidirectional text-image interaction combines base, contribution-conditioned, similarity, and discrepancy views using a learned gate before classification.

The formal implementation projects the BERT token sequence and Swin patch sequence into a 512-dimensional shared space. It retains their sequence lengths, aligns text spans to tokenizer offsets and detector boxes to image patches, and uses the same `ECCILModel` definition for the full model and all ablations.

## Framework

![ECCIL framework](method.png)

The upper part of the figure illustrates original and perturbed text/image encoding and contribution estimation. The lower part shows bidirectional multi-channel interaction, gated fusion, and binary classification.

> **TODO:** The conditional labels attached to the masked-text and masked-image branches in `method.png` are reversed relative to the corresponding inputs and the formal code. Correct the figure before public release.

## Requirements

The manuscript reports training on one NVIDIA RTX 5090D GPU with 32 GB memory and a maximum observed GPU-memory use of 20 GB. It does not report exact Python, PyTorch, torchvision, or CUDA versions, and this repository does not currently include `requirements.txt`, `environment.yml`, or another environment lock file.

Dependencies imported by the formal execution path are:

| Dependency | Version |
| --- | --- |
| Python | TODO |
| PyTorch | TODO |
| torchvision | TODO |
| CUDA | TODO |
| transformers | TODO |
| spaCy | TODO |
| NumPy | TODO |
| pandas | TODO |
| Pillow | TODO |
| scikit-learn | TODO |
| PyYAML | TODO |

The model assets expected by the manuscript and configuration are:

- `bert-base-chinese` for Weibo and `bert-base-uncased` for PHEME and PolitiFact;
- `swin-base-patch4-window7-224` for images;
- `zh_core_web_sm` for Weibo and `en_core_web_sm` for PHEME and PolitiFact;
- a local state-dict checkpoint compatible with `torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)`.

`configs/weibo.yaml` sets `local_files_only: true`; the formal code does not download model assets or the detector checkpoint.

**TODO:** Record the exact revisions/checksums of the BERT, Swin, spaCy, and Faster R-CNN assets used for the reported experiments, and document the detector checkpoint's training source and class definition.

## Datasets

No dataset is distributed in this repository, and the manuscript does not provide dataset download URLs. **TODO:** Add verified acquisition and licensing instructions for all three datasets. The following counts are copied from the final manuscript.

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

### Dataset issues requiring confirmation

- **TODO:** The manuscript's Weibo split totals 9,101 samples (6,858 + 2,243), whereas its statistics table reports 9,143 labeled samples and 9,143 images. Confirm the preprocessing or filtering that accounts for the 42-sample difference.
- **TODO:** Confirm the exact class-index mapping. `configs/weibo.yaml` leaves both names unverified and the code uses fixed class index 1 for contribution estimation; the manuscript's label description and legacy evaluation code do not establish a consistent mapping.
- **TODO:** Confirm the six positional Weibo CSV columns above against the release data.

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

The formal public path is `train_eccil.py` -> `eccil/`. The root legacy scripts and backups are retained for provenance and should not be used to reproduce the formal protocol.

## Training

First replace every placeholder in `configs/weibo.yaml`:

```yaml
dataset.root: /PATH/TO/WEIBO
assets.bert_path: /PATH/TO/BERT
assets.swin_path: /PATH/TO/SWIN
assets.detector_path: /PATH/TO/DETECTOR_CHECKPOINT
training.output_dir: /PATH/TO/OUTPUT
```

### Weibo

The only currently supported formal training command is:

```bash
python train_eccil.py --config configs/weibo.yaml --variant full --seeds 42 2024 666
```

The CLI also accepts `--device`, for example `--device cuda:0`. Without this option, `device: auto` selects `cuda:0` when CUDA is available and otherwise selects CPU.

### PHEME

```text
TODO: no runnable PHEME training command exists in the current repository.
```

### PolitiFact

```text
TODO: no runnable PolitiFact training command exists in the current repository.
```

### Formal Weibo configuration

| Parameter | `configs/weibo.yaml` |
| --- | ---: |
| Batch size | 32 |
| Maximum epochs | 50 |
| Task learning rate | `1e-3` |
| BERT learning rate | `1e-5` |
| Swin learning rate | `1e-5` |
| Optimizer | AdamW |
| Weight decay | 0.0 |
| LR scheduler | ReduceLROnPlateau, factor 0.5, patience 3 |
| Early-stopping patience | 5 |
| Gradient clipping norm | 5.0 |
| Seeds | 42, 2024, 666 |
| Split seed | 2024 |
| Shared dimension | 512 |
| Attention heads / layers | 8 / 2 |
| Dropout | 0.1 |
| Maximum text length | 300 |

## Evaluation

`train_eccil.py` performs validation internally after every epoch. It monitors validation accuracy for checkpoint selection and early stopping, uses validation loss for `ReduceLROnPlateau`, restores `best.pt`, and evaluates the official test set exactly once. Therefore, the Weibo training command above is also the only supported formal test command.

The evaluator records accuracy, per-class precision, recall and F1, support, confusion matrix, mean loss, and sample count. The manuscript reports accuracy and fake/real class precision, recall, and F1.

```text
TODO: the repository has no standalone checkpoint-only evaluation command.
TODO: the repository has no formal PHEME or PolitiFact evaluation command.
```

## Experimental Results

The following are the **final values reported in the manuscript**, not newly generated results from this repository state.

| Dataset | Accuracy | Fake precision | Fake recall | Fake F1 | Real precision | Real recall | Real F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Weibo | 0.937 | 0.918 | 0.959 | 0.938 | 0.957 | 0.915 | 0.936 |
| PHEME | 0.932 | 0.946 | 0.962 | 0.954 | 0.891 | 0.847 | 0.868 |
| PolitiFact | 0.922 | 0.979 | 0.904 | 0.940 | 0.828 | 0.960 | 0.889 |

The manuscript reports single values rather than mean +/- standard deviation. The current formal configuration runs three seeds and writes an aggregate accuracy summary, but no corresponding formal-run artifacts are included here.

## Ablation Study

### Module ablation

These values are copied from the manuscript's module-ablation table.

| Dataset | Variant | Accuracy | Fake precision | Fake recall | Fake F1 | Real precision | Real recall | Real F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Weibo | w/o ELCE | 0.923 | 0.885 | 0.973 | 0.927 | 0.970 | 0.872 | 0.918 |
| Weibo | w/o MCI | 0.918 | 0.885 | 0.962 | 0.922 | 0.958 | 0.874 | 0.914 |
| Weibo | w/o CAT | 0.924 | 0.894 | 0.963 | 0.927 | 0.959 | 0.885 | 0.921 |
| Weibo | ECCIL | 0.937 | 0.918 | 0.959 | 0.938 | 0.957 | 0.915 | 0.936 |
| PHEME | w/o ELCE | 0.911 | 0.944 | 0.935 | 0.939 | 0.824 | 0.847 | 0.836 |
| PHEME | w/o MCI | 0.902 | 0.962 | 0.902 | 0.931 | 0.769 | 0.903 | 0.831 |
| PHEME | w/o CAT | 0.919 | 0.938 | 0.952 | 0.945 | 0.862 | 0.826 | 0.844 |
| PHEME | ECCIL | 0.932 | 0.946 | 0.962 | 0.954 | 0.891 | 0.847 | 0.868 |
| PolitiFact | w/o ELCE | 0.896 | 0.923 | 0.923 | 0.923 | 0.840 | 0.840 | 0.840 |
| PolitiFact | w/o MCI | 0.870 | 0.889 | 0.923 | 0.906 | 0.826 | 0.760 | 0.792 |
| PolitiFact | w/o CAT | 0.909 | 0.978 | 0.885 | 0.929 | 0.800 | 0.960 | 0.873 |
| PolitiFact | ECCIL | 0.922 | 0.979 | 0.904 | 0.940 | 0.828 | 0.960 | 0.889 |

### Interaction-channel ablation

These values are read from the final manuscript figures `weibo.png`, `pheme.png`, and `politi.png`.

| Dataset | Variant | Accuracy | Fake F1 | Real F1 |
| --- | --- | ---: | ---: | ---: |
| Weibo | w/o Con | 0.925 | 0.924 | 0.926 |
| Weibo | w/o Base | 0.928 | 0.931 | 0.925 |
| Weibo | w/o Sim | 0.929 | 0.931 | 0.928 |
| Weibo | w/o Dis | 0.930 | 0.932 | 0.928 |
| Weibo | ECCIL | 0.937 | 0.938 | 0.936 |
| PHEME | w/o Con | 0.897 | 0.929 | 0.803 |
| PHEME | w/o Base | 0.900 | 0.932 | 0.814 |
| PHEME | w/o Sim | 0.897 | 0.929 | 0.803 |
| PHEME | w/o Dis | 0.908 | 0.938 | 0.820 |
| PHEME | ECCIL | 0.932 | 0.954 | 0.868 |
| PolitiFact | w/o Con | 0.870 | 0.906 | 0.792 |
| PolitiFact | w/o Base | 0.870 | 0.904 | 0.800 |
| PolitiFact | w/o Sim | 0.883 | 0.914 | 0.816 |
| PolitiFact | w/o Dis | 0.896 | 0.923 | 0.840 |
| PolitiFact | ECCIL | 0.922 | 0.940 | 0.889 |

> **TODO:** The full-model bar is labeled `CCL-MFND` rather than `ECCIL` in all three channel-ablation image files. Confirm that these bars are the final ECCIL results and update the figure labels before release.

The formal variant names in `eccil/variants.py` are `full`, `no_elce`, `no_mci`, `no_cat`, `no_base`, `no_con`, `no_sim`, and `no_dis`. For example, the available Weibo ablations can be launched with:

```bash
python train_eccil.py --config configs/weibo.yaml --variant no_elce --seeds 42 2024 666
python train_eccil.py --config configs/weibo.yaml --variant no_mci  --seeds 42 2024 666
python train_eccil.py --config configs/weibo.yaml --variant no_cat  --seeds 42 2024 666
python train_eccil.py --config configs/weibo.yaml --variant no_base --seeds 42 2024 666
python train_eccil.py --config configs/weibo.yaml --variant no_con  --seeds 42 2024 666
python train_eccil.py --config configs/weibo.yaml --variant no_sim  --seeds 42 2024 666
python train_eccil.py --config configs/weibo.yaml --variant no_dis  --seeds 42 2024 666
```

## Reproducibility

For each seed, the formal trainer creates:

```text
<output_dir>/<variant>/seed_<seed>_<UTC timestamp>/
├── config.yaml       # Exact resolved configuration snapshot
├── split.json        # Split policy, indices, seed, and sample counts
├── epochs.jsonl      # Per-epoch training and validation metrics/LRs
├── best.pt           # Best validation-selected checkpoint and metadata
└── provenance.json   # Dataset, seed, split, metrics, environment, and checkpoint

<output_dir>/<variant>_summary.json  # Multi-seed accuracy mean and sample SD
```

Deterministic mode seeds Python, NumPy, PyTorch, CUDA, DataLoader workers, the training generator, validation splitting, and multi-image selection. With the default config, experiment seeds are 42, 2024, and 666, while the split seed is 2024. Exact library versions and hardware information are captured in each completed run's provenance file.

Before a formal run:

1. Fill all local paths in `configs/weibo.yaml`.
2. Confirm the CSV positions and label semantics.
3. Confirm that the fast tokenizer provides offset mappings and a mask token.
4. Confirm detector-checkpoint compatibility with the installed torchvision version.
5. Run the asset-free tests and a one-batch server smoke test.
6. Preserve the generated configuration, split, log, checkpoint, provenance, and summary files with any reported result.

The asset-free test command intended by the project is:

```bash
python -m unittest discover -s tests -v
```

The suite passes all 8 tests in an available local validation environment (Python 3.9.19, PyTorch 2.0.0+cu117, CUDA build 11.7). These versions are reported only as a README-validation environment and are not claimed to be the manuscript's training environment. The system-default Python used during review does not have PyTorch installed.

### Paper / code consistency notes

No choice is made silently when the manuscript and formal code disagree:

| Item | Manuscript | Current formal code/config | Required action |
| --- | --- | --- | --- |
| Text elements | spaCy POS-tagged nouns | spaCy `doc.ents` named entities | TODO: choose one definition and align paper, figure, and code |
| Contribution modulation | Min-max-normalized scores multiply logits by `1 + alpha*s` | Raw signed scores are added to logits as `logits + alpha*s` | TODO: align the method description and implementation; rerun affected experiments |
| Similarity view | Mean-pooled global cosine value broadcast over token pairs | Pairwise normalized token/patch cosine followed by softmax | TODO: align the method and implementation; rerun affected experiments |
| Channel gate | Conv3D followed by sigmoid | `Conv3d(4, 4, 1)` followed by channel softmax | TODO: align the method and implementation; rerun affected experiments |
| Optimizer | Adam | AdamW | TODO: confirm the final optimizer |
| Maximum epochs | 40 | 50 with early stopping | TODO: confirm the final protocol |
| Weibo task LR | `1e-4` | `1e-3` | TODO: confirm the final value |
| Validation split | Only train/test counts are reported | Fixed 10% split from training data | TODO: document the validation protocol used for final results |
| Seeds/reporting | No seeds or aggregation method reported | Seeds 42/2024/666; accuracy mean and sample SD | TODO: state which run protocol produced the manuscript table |
| Result provenance | Final tables are present | No formal checkpoints or logs are included; the refactor report requires retraining | TODO: rerun and archive formal artifacts before claiming code-level reproduction |

The manuscript's PHEME and PolitiFact settings are therefore descriptive only until corresponding formal configurations and data loaders are added and verified.

## Citation

Citation will be updated after publication.

```bibtex
@article{TODO_eccil,
  title   = {ECCIL: Element Contribution-Guided Cross-Modal Interaction Learning for Multimodal Fake News Detection},
  author  = {Zhang, Yanan and Li, Yuanqing and Li, Qiyue and Wang, Dianwei},
  journal = {TODO},
  year    = {TODO}
}
```

## Contact

Project GitHub: [https://github.com/zhangyanan-maker/ECCIL](https://github.com/zhangyanan-maker/ECCIL)

## License

**TODO:** Add the intended open-source license before publishing the repository.
