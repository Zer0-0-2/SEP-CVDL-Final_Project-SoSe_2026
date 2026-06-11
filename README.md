# Animal Recognition Challenge
**SEP: Computer Vision & Deep Learning — Group Project**

> **Status: v1 — initial plan, open for team discussion**

---

## What we're building

A system that takes an image and returns a single integer: one of 20 cat/dog breed indices, or `−1` if the image contains no target-class animal. The evaluation runs on a private held-out test set through a fixed `inference.py` interface we cannot modify.

The tricky parts are not the classification itself — they are the **reject class** (confounders must return `−1`, wrong accepts are penalised equally to wrong breed predictions) and the **multi-animal rule** (largest target-class animal by bounding box area determines the label).

---

## Our planned pipeline

```
Input image
     │
     ▼
┌──────────────────────┐
│   Object Detector    │  YOLOv8 off-the-shelf, no fine-tuning
│                      │  → bounding boxes for cats & dogs
└─────────┬────────────┘
          │  crop largest detected region
          ▼
┌──────────────────────┐
│   Classifier         │  PyTorch, trained by us
│   20 breeds + OOD    │  → softmax logits over 21 outputs
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│   OOD Gate           │  confidence < τ  →  return −1
│                      │  else  →  return argmax
└─────────┬────────────┘
          │
          ▼
     {−1, 0, …, 19}
```

We are splitting the problem into three independent pieces so they can be developed and swapped out separately: the detector, the classifier, and the OOD gate. None of these interfaces are fixed in stone yet.

---

## Repository layout

```
animal-recognition/
├── data/
│   ├── raw/                  # original validation set (not committed to git)
│   ├── processed/            # train/val split, normalised
│   └── confounders/          # extra OOD images + sources.txt
├── models/
│   ├── baseline/             # CNN trained from scratch
│   ├── transfer/             # pretrained backbone experiments
│   └── checkpoints/          # saved .pt files
├── src/
│   ├── data/                 # Dataset, augmentations, split logic
│   ├── models/               # baseline CNN, transfer head, detector wrapper
│   ├── training/             # train loop, losses, LR schedulers
│   ├── evaluation/           # metrics, confusion matrix
│   ├── ood/                  # threshold calibration, temperature scaling
│   └── xai/                  # Grad-CAM, occlusion maps, visualisation
├── notebooks/                # EDA, per-experiment exploration
├── inference.py              # ← interface is fixed, pipeline lives behind it
├── requirements.txt
└── README.md
```

---

## Classifier — what we plan to train

### Baseline (mandatory)

We need at least one model trained from random initialisation. We plan a ResNet-style CNN:

```
Conv(3→64) → BN → ReLU → MaxPool
ResBlock(64→64)  × 2
ResBlock(64→128, stride=2) × 2
ResBlock(128→256, stride=2) × 2
GlobalAvgPool → FC(256→21)
```

This gives us a concrete lower bound to compare everything else against, and satisfies the course requirement.

### Main approach — transfer learning

We plan to fine-tune a pretrained backbone with a fresh 21-class head. The candidates we want to compare:

| Backbone | Params | Why it's interesting |
|---|---|---|
| `efficientnet_b3` | 12M | Fast, good accuracy/speed tradeoff |
| `convnext_tiny` | 28M | Strong on fine-grained tasks |
| `vit_base_patch16_224` | 86M | Best accuracy ceiling, slower |
| `resnet50` | 25M | Well-understood, easy to debug |

Fine-tuning strategy we're starting with:
1. Freeze backbone, train head only — 5 epochs
2. Unfreeze last two blocks, LR ÷ 10 — 10 epochs
3. Full network, cosine annealing — 10 epochs

We'll revisit this if it doesn't converge well.

---

## OOD / Reject class

This is the part we're most uncertain about. Our plan is to start simple and add complexity only if needed.

**Starting point — softmax threshold**
Predict `−1` when `max(softmax(logits)) < τ`. Sweep τ on the validation set. Straightforward to implement and already addresses the basic case.

**If threshold alone isn't enough — temperature scaling**
Learn a single scalar `T` on the val set to calibrate softmax confidence before thresholding. One extra parameter, often a significant improvement.

**Things we want to explore if time allows**

- **Energy-based OOD score** — use `−log Σ exp(logit_i)` instead of softmax max. Theoretically better separation between in- and out-of-distribution (Liu et al. 2020).
- **Explicit confounder class** — add confounders as a 21st training class so the model learns to route OOD images to a dedicated output node rather than relying on post-hoc thresholding.
- **OpenMax** — Weibull-based rejection at the activation level (Bendale & Boult 2016). Most complex option, only if earlier approaches fall short.

We'll decide how far to go based on what the validation F1 on the reject class looks like after the first round of training.

---

## Data

### What we have

The provided validation set is our only labelled data. It contains the 20 target breeds plus confounder images, with full bounding box and class label annotations.

### Augmentation plan

```python
A.RandomResizedCrop(224, scale=(0.6, 1.0))
A.HorizontalFlip(p=0.5)
A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
A.GaussianBlur(p=0.3)
A.CoarseDropout(max_holes=8, max_height=32, p=0.3)
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Extra confounder data

We plan to supplement with OOD images from:
- **ImageNet-1k** — non-target animal classes
- **OpenImages v7** — filtered for non-target species
- **iNaturalist** — wild animals, diverse backgrounds

Source URLs will be tracked in `data/confounders/sources.txt`.

### Additional techniques to test

- **Mixup / CutMix** — interpolation between training samples, especially useful for fine-grained inter-class boundaries
- **Test-time augmentation (TTA)** — average logits over multiple augmented versions at inference, essentially free accuracy gains

---

## Explainable AI

Mandatory for the final report. We plan to use **Grad-CAM** as the primary method — it hooks into the last convolutional layer and is straightforward to implement with `pytorch-grad-cam`.

What we want to show in the report:
1. Does the model look at the animal or at the background?
2. Are there breed-specific attention patterns (ears, fur texture, face shape)?
3. Side-by-side comparison between baseline CNN and fine-tuned model
4. Failure case analysis — wrong predictions, and what the saliency map reveals about why

We'll also run **occlusion sensitivity** as a cross-check since it requires no gradient access and works as a model-agnostic sanity check.

---

## Evaluation

Metrics are computed per class in one-vs-rest fashion, including the reject class. The provided script handles this — we just need to make sure `inference.py` produces correct output.

| Metric | Scope |
|---|---|
| Accuracy | Overall |
| Precision / Recall / F1 | Per class + macro + weighted avg |

Key constraint: a wrong prediction on a confounder is penalised the same as a wrong breed prediction. We need to monitor reject-class recall explicitly throughout training, not just overall accuracy.

---

## Reproducibility

Every experiment fixes seeds before training:

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

We'll track all runs in Weights & Biases. Failed experiments get logged and documented — the report needs them.

---

## Deadlines

| | |
|---|---|
| Preliminary report | 25 June 2026 |
| Final presentation | 16 July 2026 |
| Final submission | 2 August 2026 at 23:59 |

---

## References

Papers we're drawing on:

- He et al. (2016) — *Deep Residual Learning for Image Recognition* · [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Tan & Le (2019) — *EfficientNet* · [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Liu et al. (2022) — *A ConvNet for the 2020s (ConvNeXt)* · [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Dosovitskiy et al. (2021) — *An Image is Worth 16×16 Words (ViT)* · [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Selvaraju et al. (2017) — *Grad-CAM* · [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Zeiler & Fergus (2014) — *Visualizing and Understanding CNNs* · [arXiv:1311.1901](https://arxiv.org/abs/1311.1901)
- Guo et al. (2017) — *On Calibration of Modern Neural Networks* · [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Liu et al. (2020) — *Energy-based OOD Detection* · [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)
- Bendale & Boult (2016) — *Towards Open Set Deep Networks (OpenMax)* · [arXiv:1511.06233](https://arxiv.org/abs/1511.06233)
- Zhang et al. (2018) — *Mixup* · [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- Caron et al. (2021) — *DINO (self-supervised ViT)* · [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- Jocher et al. (2023) — *YOLOv8* · [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

Libraries:
- `timm` — [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- `pytorch-grad-cam` — [github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- `albumentations` — [albumentations.ai](https://albumentations.ai/docs)
- OpenOOD benchmark — [github.com/Jingkang50/OpenOOD](https://github.com/Jingkang50/OpenOOD)
- Oxford-IIIT Pet Dataset — [robots.ox.ac.uk/~vgg/data/pets](https://www.robots.ox.ac.uk/~vgg/data/pets/)