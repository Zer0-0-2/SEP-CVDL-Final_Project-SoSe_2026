"""Evaluation harness for the Fine-grained Animal Recognition project.

We run this script on the held-out test set, so do not change the interface.
Implement your solution as the `Model` below: an `nn.Module` whose `forward`
takes a PIL image and returns a predicted class index, an integer in
{-1, 0, ..., 19}, where -1 means "reject", i.e. no target species is present.
Inside `forward` you are free to do anything you like: run an off-the-shelf
detector, find bounding boxes, crop the largest animal, classify the crop,
decide when to return -1, and so on.

The script reads `labels.csv` from the image folder, with columns
`filename,label`, where `label` is the integer class index from CLASSES (or -1
for confounders / images with no target species). The images themselves are a
flat, numbered set (0001.jpg, 0002.jpg, ...) sitting next to `labels.csv`. The
script runs your model on every image and prints the standard classification
metrics.

    python inference.py --image-folder <folder>
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import transforms as T
from tqdm import tqdm

REJECT = -1

# Official class mapping fixed by the chair (index -> species). Train your
# classifier against this exact order so your labels match our evaluation.
CLASSES = [
    "Abyssinian",         #  0
    "Bengal",             #  1
    "Birman",             #  2
    "Bombay",             #  3
    "British_Shorthair",  #  4
    "Maine_Coon",         #  5
    "Ragdoll",            #  6
    "Sphynx",             #  7
    "Tabby",              #  8
    "Tiger_Cat",          #  9
    "Beagle",             # 10
    "Pug",                # 11
    "Boxer",              # 12
    "Shiba_Inu",          # 13
    "Samoyed",            # 14
    "Golden_Retriever",   # 15
    "German_Shepherd",    # 16
    "Siberian_Husky",     # 17
    "Dalmatian",          # 18
    "Rottweiler",         # 19
]
NUM_CLASSES = len(CLASSES)


class Model(nn.Module):
    """Full pipeline: Detector -> Classifier -> OODGate."""

    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(Path(__file__).parent))
        from animal_recognition.src.config import load_config
        from animal_recognition.src.models.detector import AnimalDetector
        from animal_recognition.src.models.baseline_cnn import BaselineCNN
        from animal_recognition.src.models.transfer_model import TransferClassifier
        from animal_recognition.src.ood.gate import OODGate

        self.cfg = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detector = AnimalDetector(weights=self.cfg.pipeline.detector)

        if self.cfg.pipeline.classifier == "transfer":
            tc = self.cfg.classifier.transfer
            self.classifier = TransferClassifier(
                backbone=tc.backbone,
                num_classes=self.cfg.classifier.num_classes,
                pretrained=False,
                weights=tc.weights,
            )
        else:
            self.classifier = BaselineCNN(num_classes=self.cfg.classifier.num_classes)
            w = self.cfg.classifier.baseline_cnn.weights
            if w is not None:
                self.classifier.load_state_dict(torch.load(w, map_location="cpu"))

        self.classifier = self.classifier.to(self.device).eval()
        self.gate = OODGate(self.cfg)

        self._transform = T.Compose([
            T.Resize((self.cfg.data.image_size, self.cfg.data.image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def forward(self, image: Image.Image) -> int:
        detections = self.detector.detect_pil(image)
        if not detections:
            return -1

        # pick largest bounding box by area
        best = max(detections, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
        x1, y1, x2, y2 = [int(v) for v in best["box"]]
        crop = image.crop((x1, y1, x2, y2))

        tensor = self._transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(tensor).squeeze(0)

        return self.gate(logits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, default="images")
    args = parser.parse_args()

    df = pd.read_csv(args.image_folder / "labels.csv")
    model = Model().eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for filename, label in tqdm(zip(df["filename"], df["label"]), total=len(df)):
            image = Image.open(args.image_folder / filename).convert("RGB")
            pred = model(image)
            y_true.append(int(label))
            y_pred.append(int(pred))

    labels = [REJECT] + list(range(NUM_CLASSES))
    target_names = ["reject(-1)"] + CLASSES
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=target_names, digits=3,
                                zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))
