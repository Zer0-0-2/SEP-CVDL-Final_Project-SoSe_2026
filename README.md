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
├── config.yaml                         # pipeline routing + all hyperparameters
├── inference.py                        # fixed evaluation interface (do not change outer structure)
├── test_dataset.py                     # sanity checks for data pipeline
├── requirements.txt                    
├── README.md                           
└── animal_recognition/                 
    ├── models/                            
    │   └── weights/                    # Default Directory to put your .pt weights in
    ├── data/                           
    │   ├── raw/                        # training images — one subfolder per breed (not committed)
    │   ├── processed/                  # output folder of sanitize_scaped_data.py 
    |   │   ├── rejected/               # rejected images (not used for training)
    |   |   └── accepted/               # accepted images (used for training)
    │   └── confounders/                # OOD images labelled -1 (not committed)
    └── src/                            
        ├── config.py                   # load_config() → dot-accessible config namespace
        ├── data/                       
        │   ├── dataset.py              # AnimalDataset (torch Dataset)
        │   ├── augmentations_xxx.py    # Various augmenation files
        │   ├── downloader_reddit.py    # gallery-dl scraper for breed subreddits
        │   └── downloader_tiger_cat.py # ImageNet-1k streaming for Tiger Cat class
        ├── models/
        │   ├── baseline_cnn.py         # BaselineCNN - not used 
        │   ├── classifier_convnext.py  # ConvNextV2 Class used for training (TIMM)
        │   ├── classifier_gcvit.py     # GCViT Class used for training (TIMM)
        │   ├── classifier_resnet.py    # ResNet50 (trained just once, not used further)
        │   ├── yolo.py                 # YOLO26 detector (evaluated, not used in pipeline)
        │   ├── yoloworld.py            # YoloWorld detector (evaluated, used in pipeline)
        │   └── transfer_model.py       # TransferClassifier — timm backbone (TODO)
        ├── training/                   
        │   └── train_xxx.py            # Various training scripts all derived from train_classifier (Object oriented, parameterized)
        ├── ood/                        
        │   └── gate.py                 # OODGate — softmax threshold / energy (TODO)
        ├── evaluation/                  
        │   ├── detector_eval_yoloworld.py     # Script for evaluating yoloworld 
        │   ├── detector_eval.ipynb     # Ipynb file to evaluate yoloworld 
        |   └── inference_xxx  .py      # Various temporary inference scripts to test models against provided images
        └── xai/                        
            └── gradcam_wrapper.py      # GradCAMExplainer (TODO)
```

---

## Setup

Requires **a Tkinter-Version of Python**

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

---

## Getting the data

### Breed images (19 classes via Reddit)

Requires `gallery-dl` with a Firefox session cookie:
```bash

python animal_recognition/src/data/reddit_downloader.py
```
Downloads up to 200 images per class into `animal_recognition/data/raw/<ClassName>/`. Skips classes that already have 200+ images.

### Tiger Cat (ImageNet-1k)

Tiger Cat has no dedicated subreddit. Images are streamed from the gated ImageNet-1k dataset on HuggingFace. 
```bash
huggingface-cli login      # one-time: accept terms at huggingface.co/datasets/ILSVRC/imagenet-1k first
python animal_recognition/src/data/downloader_tiger_cat.py
```




## Configuration

All pipeline routing and hyperparameters live in `config.yaml`:

```yaml
pipeline:
  classifier: convnextv2_tiny   # adjust depending on the .pt file or whether you want to fine tune on your own
  ood_gate: softmax_threshold   # 'softmax_threshold' | 'temperature_scaling' | 'energy'

classifier:
  transfer:
    backbone: convnextv2_tiny   # adjust depending on the .pt file or whether you want to fine tune on your own
```

Change `classifier: baseline_cnn` to `classifier: transfer` to route through the transfer model. No other changes needed.

---



---

## Current status

| Component | Status |
|---|---|
| `AnimalDetector` (YOLOvWorld) | Done |
| `AnimalDataset` | Done |
| `Different Augmentations` | Done |
| `config.yaml` + `config.py` | TODO |
| `TIMM Models` | Done |
| `Trainer` | Done |
| `OODGate` | TODO |
| `Evaluator` | Done |
| `GradCAMExplainer` | TODO |
| Wire `inference.py::Model` | TODO |
| Download training data | Done |
| Download testing data  | Done | 

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
