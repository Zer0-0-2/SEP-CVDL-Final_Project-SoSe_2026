# Animal Recognition Challenge
**SEP: Computer Vision & Deep Learning — Group Project**

> **Status: pipeline trained and working. Final submission due 2 August 2026, 23:59 — see [Known issues](#known-issues-before-submission) below.**

---

## What this system does

Given an input image, the system returns a single integer:
- `0–19` — one of 20 cat/dog breed indices
- `-1` — reject (no target species present, or confidence below threshold)

The evaluation interface the chair specified is `inference.py` at the project root:
```bash
python inference.py --image-folder <path-to-folder>
```
The folder must contain images and a `labels.csv` with columns `filename,label`.

**This root file currently runs the old, retired pipeline (YOLOv8m + BaselineCNN/TransferClassifier) and its weights path in `config.yaml` doesn't exist — running it as-is raises `FileNotFoundError`.** The actually trained and validated pipeline (YoloWorld + ConvNeXt/GCViT + OOD gate) lives in [`animal_recognition/src/evaluation/InferenceBackup.py`](animal_recognition/src/evaluation/InferenceBackup.py) — see [Running inference](#running-inference) for the working command, and [Known issues](#known-issues-before-submission) for what needs to happen before submission.

---

## Pipeline

Training and inference are two separate steps: classifiers are trained once and saved as `.pt` weights; inference combines the (frozen) detector, those weights, and the OOD gate into one forward pass.

```
Input image (PIL)
      │
      ▼
YoloWorldDetector (YOLO-World, open-vocabulary, off-the-shelf)
      │  → open-vocabulary prompts restricted to the 20 accepted species
      │  → no valid detection → return -1
      │  → crops the best detection, resized to 224×224 (or the checkpoint's trained size)
      ▼
Classifier (ConvNeXt or GCViT, TIMM backbone, trained via animal_recognition/src/training/)
      │  → raw logits over 20 breed classes
      ▼
OODGate (optional, off by default — see config.yaml: ood.enabled)
      │  softmax_threshold:    max(softmax(logits)) < threshold        → reject
      │  energy / temperature_scaling:  -T·logsumexp(logits/T) > threshold → reject
      │  else → argmax
      ▼
{-1, 0, …, 19}
```

Swap detector/classifier/OOD settings by editing `config.yaml` — no code changes needed for the scripts that read it (`InferenceBackup.py`, training scripts). The root `inference.py` and `evaluation/inference.py` only partially read `config.yaml` (see Known issues).

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
├── config.yaml                         # pipeline routing + all hyperparameters (source of truth)
├── inference.py                        # chair's fixed evaluation interface — currently BROKEN, see Known issues
├── test_dataset.py                     # sanity checks for AnimalDataset — currently FAILS (data/raw/ is empty locally)
├── requirements.txt
├── README.md
└── animal_recognition/
    ├── models/
    │   ├── yolov8x-worldv2.pt          # YOLO-World detector weights (auto-downloaded, gitignored)
    │   └── weights/                    # misc weights (CLIP, gitignored); NOT where classifier checkpoints live
    ├── data/
    │   ├── raw/                        # training images, one subfolder per breed (gitignored)
    │   ├── processed/                  # output of sanitize_scraped_data.py
    │   │   ├── rejected/               # images the detector/sanitizer rejected
    │   │   └── accepted/               # images actually used for training
    │   ├── confounders/                # OOD images labelled -1 (gitignored)
    │   ├── images/                     # flat eval set (labels.csv + 0000.jpg…), 143 imgs incl. real confounders (gitignored)
    │   └── eval_flat/                  # flat eval set built from test_dataset/ via build_eval_folder.py (gitignored)
    └── src/
        ├── config.py                   # load_config() → dot-accessible config.yaml namespace
        ├── data/
        │   ├── dataset.py              # AnimalDataset (torch Dataset), expects data/raw/<Breed>/*.jpg
        │   ├── augmentations.py / _mild.py / _stronger.py / _vetted.py   # augmentation presets, all used across the training scripts
        │   ├── downloader_reddit.py    # gallery-dl scraper for breed subreddits (needs Firefox session cookie)
        │   ├── downloader_tiger_cat.py # streams the Tiger Cat class from ImageNet-1k (gated HF dataset)
        │   ├── downloader_test_dataset.py  # one-off: built the 100-img/breed held-out test set from Oxford Pets/Stanford Dogs/ImageNet-1k
        │   └── sanitize_scraped_data.py    # runs YoloWorld over raw/ to split scraped images into processed/accepted vs rejected
        ├── models/
        │   ├── yoloworld.py             # YoloWorldDetector — detector actually used in the pipeline
        │   ├── classifier_convnext.py   # ConvNextClassifier (TIMM `convnext_*`) — used in the pipeline
        │   ├── classifier_gcvit.py      # GCViTClassifier (TIMM `gcvit_*`) — used in the pipeline, current best model
        │   ├── yolo.py                  # closed-vocabulary YOLO detector, evaluated but not used in the deployed pipeline
        │   ├── detector.py              # old AnimalDetector (YOLOv8m) — superseded by yoloworld.py, only referenced by root inference.py
        │   ├── baseline_cnn.py          # first classifier baseline, superseded, still used by src/xai/ scripts
        │   ├── classifier_resnet50.py   # trained once for comparison, not used further, not imported anywhere
        │   ├── classifier_swin.py       # Swin classifier, has its own training script, not part of the main pipeline
        │   └── transfer_model.py        # TIMM-backbone wrapper, only used by root inference.py
        ├── training/
        │   ├── train_classifier.py             # main parameterized trainer (schedulers, augmentation presets, etc.)
        │   ├── train_classifier_top_5.py / _2nd_attempt.py   # experiments narrowing to the 5 best configs
        │   ├── train_bitfit_tiny.py / train_bitfit_base.py   # BitFit fine-tuning (bias/norm/head-only) for gcvit_tiny / gcvit_base
        │   └── train_swin_classifier.py        # trains classifier_swin.py
        ├── ood/
        │   └── gate.py                  # OODGate — softmax_threshold / energy / temperature_scaling (all implemented)
        ├── evaluation/
        │   ├── InferenceBackup.py              # ✅ recommended: full pipeline + switchable OOD gate, tested working
        │   ├── inference.py                     # ❌ pipeline works but has no OOD gate wired in, broken defaults, crashes on CPU-only machines
        │   ├── inference_batch.py               # multi-model leaderboard via Tkinter file picker — needs a Tk-enabled Python, only detects gcvit/convnextv2 checkpoints by filename
        │   ├── ood_gate_accuracy_comparison.py  # ✅ compares accuracy with vs. without the OOD gate, saves a bar chart
        │   ├── build_eval_folder.py             # flattens data/test_dataset/<Breed>/*.jpg into the flat labels.csv format
        │   ├── detector_eval_yoloworld.py       # sweeps YoloWorld confidence thresholds / prompt sets
        │   └── detector_eval.ipynb              # notebook version of the above
        └── xai/
            ├── run_xai.py                # CLI entry point, supports baseline_cnn / convnext / gcvit / swin via a registry
            ├── gradcam_xai.py             # Grad-CAM
            ├── occlusion_xai.py           # occlusion sensitivity maps
            └── layer_activation_xai.py    # layer activation visualization
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**AMD GPU (ROCm) users** — replace the last step with:
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm7.2
```
Adjust the ROCm version suffix to match your installed ROCm (e.g. `rocm6.0`).

`inference_batch.py` additionally needs a Python built with Tk support (`import tkinter` must work) — it opens a native file picker. All other scripts in this repo do not need Tkinter.

---

## Getting the data

### Breed images (19 classes via Reddit)

Requires `gallery-dl` with a Firefox session cookie:
```bash
python animal_recognition/src/data/downloader_reddit.py
```
Downloads up to 200 images per class into `animal_recognition/data/raw/<ClassName>/`. Skips classes that already have 200+ images.

### Tiger Cat (ImageNet-1k)

Tiger Cat has no dedicated subreddit. Images are streamed from the gated ImageNet-1k dataset on HuggingFace.
```bash
huggingface-cli login      # one-time: accept terms at huggingface.co/datasets/ILSVRC/imagenet-1k first
python animal_recognition/src/data/downloader_tiger_cat.py
```

### Sanitizing scraped data

`sanitize_scraped_data.py` runs the YoloWorld detector over `data/raw/` and splits images into `data/processed/accepted/` (used for training) and `data/processed/rejected/`.

---

## Running inference

All commands below assume `source venv/bin/activate` first, and are run from the project root.

### Recommended: `InferenceBackup.py` (full pipeline, switchable OOD gate)

```bash
python -m animal_recognition.src.evaluation.InferenceBackup \
    --image-folder animal_recognition/data/images \
    --classifier-type gcvit \
    --model-name gcvit_tiny \
    --weights-path animal_recognition/src/models/models_weights/<weights_file>.pt \
    --ood-gate softmax_threshold \
    --ood-threshold 0.5
```
`--classifier-type` is `convnext` or `gcvit`; `--model-name` must match the checkpoint's architecture (e.g. `convnext_tiny`, `convnext_small`, `gcvit_tiny`). `--ood-gate` is `softmax_threshold` | `energy` | `temperature_scaling`; defaults for `--ood-gate`/`--ood-threshold`/`--ood-temperature` come from `config.yaml` if omitted. Prints accuracy, a full classification report, and the confusion matrix. Tested working end-to-end (CPU and GPU).

### Comparing accuracy with vs. without the OOD gate

```bash
python -m animal_recognition.src.evaluation.ood_gate_accuracy_comparison \
    --image-folder animal_recognition/data/images \
    --classifier-type gcvit \
    --model-name gcvit_tiny \
    --weights-path animal_recognition/src/models/models_weights/<weights_file>.pt \
    --ood-gate energy \
    --ood-threshold -3.9
```
Reuses `InferenceBackup.Model` internally, prints per-image predictions and accuracy with/without the gate, and saves a bar chart to `oodGate_stats/`. Tested working.

### Multi-model leaderboard (`inference_batch.py`)

```bash
python -m animal_recognition.src.evaluation.inference_batch --image-folder animal_recognition/data/images
```
Opens a native file picker (Tkinter) to select one or more `.pt` files, then ranks them by macro F1 in a Rich table. **Only infers `gcvit` or `convnextv2` architectures from the filename** — plain `convnext_*` checkpoints will raise `ValueError`. Needs a Tk-enabled Python; failed in this environment with `ModuleNotFoundError: No module named '_tkinter'`.

### `evaluation/inference.py` — not currently recommended

Same detector/classifier pipeline as `InferenceBackup.py` but without the OOD gate, and with two bugs confirmed by testing: the default `--weights-path` doesn't exist, and loading any checkpoint crashes on a CPU-only machine (`torch.load` is missing `map_location`). Use `InferenceBackup.py` instead until this is fixed.

---


## run Explainable AI

Run the following comand with any picture:
```bash
python -m animal_recognition.src.xai.run_xai --image <path> --weights-path <pattern or path> --classifier <type> --classifier-type <modellname>
```
classifiers are either convnext or gcvit.
For weigths you can either use a specific path or a part of the weights name.

---

## Configuration

All pipeline routing and hyperparameters live in `config.yaml`. Current values:

```yaml
pipeline:
  detector: yoloworld
  classifier: gcvit
  ood_gate: energy

classifier:
  backbone: gcvit_tiny
  weights: animal_recognition/models/weights/Initial_Experiments/gcvit_tiny_..._bitfit_....pt   # ⚠ path does not exist, see Known issues

ood:
  enabled: false        # detector already rejects 36/40 confounders on the provided set
  threshold: -3.9        # tuned for the energy gate
  temperature: 1.0
```

Scripts that read `config.yaml` for their defaults (`InferenceBackup.py`, `ood_gate_accuracy_comparison.py`, training scripts) pick these up automatically; CLI flags always override them.

---

<a name="known-issues-before-submission"></a>
## Known issues before submission

1. **Root `inference.py` is broken.** It still wires the retired `AnimalDetector`/`BaselineCNN`/`TransferClassifier` pipeline, and its weights path (from `config.yaml: classifier.weights`) points at `animal_recognition/models/weights/Initial_Experiments/...`, which does not exist in this checkout. Since the chair runs exactly this file, this is the top priority: either point it at the working `InferenceBackup.Model` pipeline, or fix its config path and confirm the old pipeline still trains/loads.
2. **`classifier.weights` in `config.yaml`** references a file that isn't present under `animal_recognition/models/weights/` — either the `Initial_Experiments/` folder needs to be recreated with that checkpoint, or the path should point at `animal_recognition/src/models/models_weights/` where the actual checkpoints live.
3. **`evaluation/inference.py`** has no OOD gate and crashes on CPU-only machines (missing `map_location` in `torch.load`) — low priority since `InferenceBackup.py` replaces it, but worth deleting or fixing to avoid confusion.
4. **`test_dataset.py` currently fails 6/6** because `animal_recognition/data/raw/` is empty in this checkout (mock placeholder images were removed). Not a code bug, but confirm the real training data is available wherever grading happens, or drop/update this sanity check.
5. **`inference_batch.py`** needs a Tk-enabled Python and only recognizes `gcvit`/`convnextv2` filenames — currently unusable in this venv and can't evaluate the plain `convnext_*` checkpoints at all.
6. Dead/legacy files that could be trimmed for a cleaner submission: `classifier_resnet50.py`, `yolo.py`, `detector.py`, `transfer_model.py`, `baseline_cnn.py` (only used by the retired pipeline and the XAI scripts), plus the near-duplicate `inference_batch.py`/`evaluation/inference.py`/`InferenceBackup.py` trio.

---

## Current status

| Component | Status |
|---|---|
| `YoloWorldDetector` | Done, deployed |
| `AnimalDataset` | Done |
| Augmentation presets | Done |
| `config.yaml` + `config.py` | Done |
| ConvNeXt / GCViT classifiers (TIMM) | Done, trained (see `models_weights/`) |
| Trainer(s) | Done |
| `OODGate` (softmax_threshold / energy / temperature_scaling) | Done, implemented and tuned; disabled by default (`ood.enabled: false`) |
| `InferenceBackup.py` evaluator | Done, tested |
| Root `inference.py` (chair's fixed interface) | **Broken — needs fixing before submission** |
| GradCAM / Occlusion / Layer-Activation XAI | Done |
| Download training data | Done |
| Download testing data | Done |

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
