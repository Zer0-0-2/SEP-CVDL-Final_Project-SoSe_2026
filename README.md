# Animal Recognition Challenge
**SEP: Computer Vision & Deep Learning — Group Project**
**Participants: Valerio, Tristan, Henrik**

> **Status: pipeline trained and working. Final submission due 2 August 2026, 23:59**

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
OODGate (optional, on by default — see config.yaml: ood.enabled)
      │  softmax_threshold:    max(softmax(logits)) < threshold        → reject
      │  energy / temperature_scaling:  -T·logsumexp(logits/T) > threshold → reject
      │  else → argmax
      ▼
{-1, 0, …, 19}
```

Swap detector/classifier/OOD settings by editing `config.yaml` — no code changes needed for the scripts that read it (`InferenceBackup.py`, training scripts) as well as the root `Inference.py`.

---

## Repository layout

```
.
├── config.yaml                         # pipeline routing + all hyperparameters (source of truth)
├── inference.py                        # chair's evaluation interface — wired to YoloWorld + ConvNeXt/GCViT + optional OOD gate, tested working
├── requirements.txt
├── README.md
├── evaluation_dashboard/               # Gradio web dashboard for browsing/comparing trained checkpoints, Dockerized
│   ├── app.py                          # entry point: precomputes results for every checkpoint, then launches the UI
│   ├── ui.py                           # Gradio layout — tables, confusion matrix plots, GPU monitor
│   ├── inference.py                    # batched evaluation of all checkpoints against the local dataset
│   ├── inference_provided.py           # same, against the chair-provided dataset (runs YoloWorld detection first)
│   ├── config.py                       # paths, class lists, logging setup (paths assume the Docker container layout)
│   ├── utils.py                        # checkpoint discovery, SHA256 hashing, name formatting
│   ├── Dockerfile / docker-compose.yml # container build for the dashboard
│   └── requirements.txt                # dashboard-only dependencies (gradio, seaborn, …)
└── animal_recognition/
    ├── models/
    │   ├── yolov8x-worldv2.pt          # YOLO-World detector weights (auto-downloaded, gitignored)
    │   └── weights/                    # misc weights (CLIP); NOT where classifier checkpoints live
    ├── data/
    │   ├── processed/                  # output of sanitize_scraped_data.py
    │   │   ├── rejected/               # images the detector/sanitizer rejected
    │   │   └── accepted/               # images actually used for training
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
        │   ├── classifier_resnet50.py   # trained once for comparison, not used further, not imported anywhere
        │   └── classifier_swin.py       # Swin classifier, has its own training script, not part of the main pipeline
        ├── training/
        │   ├── train_classifier.py             # main parameterized trainer (schedulers, augmentation presets, etc.)
        │   ├── train_classifier_top_5.py / _2nd_attempt.py   # experiments narrowing to the 5 best configs
        │   ├── train_bitfit_tiny.py / train_bitfit_base.py   # BitFit fine-tuning (bias/norm/head-only) for gcvit_tiny / gcvit_base
        │   └── train_swin_classifier.py        # trains classifier_swin.py
        ├── ood/
        │   └── gate.py                  # OODGate — softmax_threshold / energy/temperature_scaling
        ├── evaluation/
        │   ├── inference_batch.py       # multi-model leaderboard via Tkinter file picker — needs a Tk-enabled Python, only detects gcvit/convnextv2 checkpoints by filename
        │   ├── ood_gate_accuracy_comparison.py  # compares accuracy with vs. without the OOD gate, saves a bar chart to oodGate_stats/
        │   └── detector_eval_yoloworld.py       # sweeps YoloWorld confidence thresholds / prompt sets
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

### Recommended: `Inference.py`

```bash
python -m animal_recognition.src.evaluation.InferenceBackup \
    --image-folder animal_recognition/data/images \
    --classifier-type gcvit \
    --model-name gcvit_tiny \
    --weights-path animal_recognition/src/models/models_weights/<weights_file>.pt \
    --ood-gate softmax_threshold \
    --ood-threshold 0.5
```
`--classifier-type` is `convnext` or `gcvit`; `--model-name` must match the checkpoint's architecture (e.g. `convnext_tiny`, `convnext_small`, `gcvit_tiny`). `--ood-gate` is `softmax_threshold` | `energy` | `temperature_scaling`; defaults for `--ood-gate`/`--ood-threshold`/`--ood-temperature` come from `config.yaml` if omitted. Prints accuracy, a full classification report, and the confusion matrix.

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
  backbone: gcvit_base
  weights: /Users/h.hunstein/Desktop/SEP-CVDL-Final_Project-SoSe_2026/animal_recognition/src/models/models_weights/gcvit_base_gcvit_base_bitfit_experiment_base_0.01_preTrue_bs32_lr0.001_wd0_ls0.0_sz224_augstronger_schedCosineLRScheduler.pt

ood:
  enabled: true        
  threshold: -3.9        # tuned for the energy gate
  temperature: 1.0
```

Scripts that read `config.yaml` for their defaults (`InferenceBackup.py`, `ood_gate_accuracy_comparison.py`, training scripts) pick these up automatically; CLI flags always override them.

---