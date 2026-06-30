# Animal Recognition Challenge
**SEP: Computer Vision & Deep Learning — Group Project**

> **Status: active development — data pipeline complete, model training next**

---

## What this system does

Given an input image, the system returns a single integer:
- `0–19` — one of 20 cat/dog breed indices
- `-1` — reject (no target species present, or confidence below threshold)

The evaluation interface is fixed at `inference.py`. Run it as:
```bash
python inference.py --image-folder <path-to-folder>
```
The folder must contain images and a `labels.csv` with columns `filename,label`.

---

## Pipeline

```
Input image (PIL)
      │
      ▼
AnimalDetector (YOLOv8m, off-the-shelf)
      │  → finds all cats/dogs in the image
      │  → no detections → return -1
      │  → selects largest bounding box by area
      │  → crops and resizes to 224×224
      ▼
Classifier (BaselineCNN or TransferClassifier)
      │  → raw logits over 20 breed classes
      ▼
OODGate
      │  → max(softmax(logits)) < τ → return -1
      │  → else → return argmax
      ▼
{-1, 0, …, 19}
```

Swap the classifier and OOD method by editing `config.yaml` — no code changes needed.

---

## Class mapping

| Index | Class | Index | Class |
|---|---|---|---|
| 0 | Abyssinian | 10 | Beagle |
| 1 | Bengal | 11 | Pug |
| 2 | Birman | 12 | Boxer |
| 3 | Bombay | 13 | Shiba\_Inu |
| 4 | British\_Shorthair | 14 | Samoyed |
| 5 | Maine\_Coon | 15 | Golden\_Retriever |
| 6 | Ragdoll | 16 | German\_Shepherd |
| 7 | Sphynx | 17 | Siberian\_Husky |
| 8 | Tabby | 18 | Dalmatian |
| 9 | Tiger\_Cat | 19 | Rottweiler |

---

## Repository layout

```
.
├── config.yaml                        # pipeline routing + all hyperparameters
├── inference.py                       # fixed evaluation interface (do not change outer structure)
├── test_dataset.py                    # sanity checks for data pipeline
├── requirements.txt
├── README.md
└── animal_recognition/
    ├── data/
    │   ├── raw/                       # training images — one subfolder per breed (not committed)
    │   ├── processed/                 # train/val split after preprocessing (not committed)
    │   └── confounders/               # OOD images labelled -1 (not committed)
    └── src/
        ├── config.py                  # load_config() → dot-accessible config namespace
        ├── data/
        │   ├── dataset.py             # AnimalDataset (torch Dataset)
        │   ├── augmentations.py       # get_train_transforms / get_val_transforms
        │   ├── reddit_downloader.py   # gallery-dl scraper for breed subreddits
        │   └── tiger_cat_downloader.py # ImageNet-1k streaming for Tiger Cat class
        ├── models/
        │   ├── detector.py            # AnimalDetector (YOLOv8 wrapper)
        │   ├── baseline_cnn.py        # BaselineCNN — ResNet-style, trained from scratch
        │   └── transfer_model.py      # TransferClassifier — timm backbone (TODO)
        ├── training/
        │   └── trainer.py             # Trainer — training loop (TODO)
        ├── ood/
        │   └── gate.py                # OODGate — softmax threshold / energy (TODO)
        ├── evaluation/
        │   └── evaluator.py           # Evaluator — per-class metrics (TODO)
        └── xai/
            └── gradcam_wrapper.py     # GradCAMExplainer (TODO)
```

---

## Setup

Requires **Python 3.11**. Python 3.14 fails to build numpy/scipy wheels.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**AMD GPU (ROCm) users** — replace the pip step with:
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm7.2
```
Adjust the ROCm version suffix to match your installed ROCm (e.g. `rocm6.0`).

---

## Getting the data

### Breed images (19 classes via Reddit)

Requires `gallery-dl` with a Firefox session cookie:
```bash
pip install gallery-dl
python animal_recognition/src/data/reddit_downloader.py
```
Downloads up to 200 images per class into `animal_recognition/data/raw/<ClassName>/`. Skips classes that already have 200+ images.

### Tiger Cat (ImageNet-1k)

Tiger Cat has no dedicated subreddit. Images are streamed from the gated ImageNet-1k dataset on HuggingFace (label 282, synset n02123159):
```bash
huggingface-cli login      # one-time: accept terms at huggingface.co/datasets/ILSVRC/imagenet-1k first
python animal_recognition/src/data/tiger_cat_downloader.py
```

### Confounder images

Not yet collected. Planned sources:
- ImageNet-1k non-target-animal classes
- OpenImages v7 non-target species
- iNaturalist wild animals

Source URLs will be tracked in `animal_recognition/data/confounders/sources.txt`.

---

## Configuration

All pipeline routing and hyperparameters live in `config.yaml`:

```yaml
pipeline:
  classifier: baseline_cnn      # 'baseline_cnn' | 'transfer'
  ood_gate: softmax_threshold   # 'softmax_threshold' | 'temperature_scaling' | 'energy'

classifier:
  transfer:
    backbone: efficientnet_b3   # any timm model name
```

Change `classifier: baseline_cnn` to `classifier: transfer` to route through the transfer model. No other changes needed.

---

## Running the sanity tests

With mock data already in place (`animal_recognition/data/raw/`), run:
```bash
python test_dataset.py
```

Checks that:
- Dataset loads and finds all 20 classes
- `__getitem__` returns the correct tensor shape `[3, 224, 224]` and dtype `float32`
- Normalisation is applied (values outside `[0, 1]`)
- Train transforms are random (same image → different tensor)
- Confounders load with label `-1`

---

## Current status

| Component | Status |
|---|---|
| `AnimalDetector` (YOLOv8) | Done |
| `BaselineCNN` (from scratch) | Done |
| `AnimalDataset` | Done |
| `augmentations.py` | Done |
| `config.yaml` + `config.py` | Done |
| `TransferClassifier` (timm) | TODO |
| `Trainer` | TODO |
| `OODGate` | TODO |
| `Evaluator` | TODO |
| `GradCAMExplainer` | TODO |
| Wire `inference.py::Model` | TODO |
| Download training data | TODO |

---

## Deadlines

| Milestone | Date |
|---|---|
| Preliminary report | 25 June 2026 ✓ |
| Final presentation | 16 July 2026 |
| Final submission | 2 August 2026 at 23:59 |

---

## References

- He et al. (2016) — *Deep Residual Learning* · [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Tan & Le (2019) — *EfficientNet* · [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Liu et al. (2022) — *ConvNeXt* · [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Dosovitskiy et al. (2021) — *ViT* · [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Selvaraju et al. (2017) — *Grad-CAM* · [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Liu et al. (2020) — *Energy-based OOD Detection* · [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)
- Jocher et al. (2023) — *YOLOv8* · [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Zhang et al. (2018) — *Mixup* · [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- `timm` — [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- `albumentations` 2.x — [albumentations.ai](https://albumentations.ai)
