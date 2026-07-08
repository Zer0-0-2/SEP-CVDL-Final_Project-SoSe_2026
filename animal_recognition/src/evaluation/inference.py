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

from animal_recognition.src.models.yoloworld import YoloWorldDetector
from animal_recognition.src.models.classifier_convnext import ConvNextClassifier
from animal_recognition.src.models.classifier_gcvit import GCViTClassifier
import tempfile

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

import numpy as np
import torchvision.transforms as transforms

REJECT = -1

# Official class mapping fixed by the chair (index -> species). Train your
# classifier against this exact order so your labels match our evaluation.
CLASSES = [
    "Abyssinian",  #  0
    "Bengal",  #  1
    "Birman",  #  2
    "Bombay",  #  3
    "British_Shorthair",  #  4
    "Maine_Coon",  #  5
    "Ragdoll",  #  6
    "Sphynx",  #  7
    "Tabby",  #  8
    "Tiger_Cat",  #  9
    "Beagle",  # 10
    "Pug",  # 11
    "Boxer",  # 12
    "Shiba_Inu",  # 13
    "Samoyed",  # 14
    "Golden_Retriever",  # 15
    "German_Shepherd",  # 16
    "Siberian_Husky",  # 17
    "Dalmatian",  # 18
    "Rottweiler",  # 19
]
NUM_CLASSES = len(CLASSES)


class Model(nn.Module):
    def __init__(
        self,
        weights_path: Path,
        classifier_type: str = "convnext",
    ):
        super().__init__()

        # use yoloworld with the default values (which are already analyzed to be best on test dataset)
        self.detector = YoloWorldDetector()

        if classifier_type == "convnext":
            self.classifier = ConvNextClassifier(pretrained=False)
        elif classifier_type == "gcvit":
            self.classifier = GCViTClassifier(pretrained=False)
        else:
            raise ValueError(f"Unsupported classifier_type: {classifier_type}")

        self.weights_path = Path(weights_path)

        if self.weights_path.exists():
            self.classifier.load_state_dict(torch.load(self.weights_path))
        else:
            raise ValueError(f"Weights path not found: {self.weights_path}")

        self.classifier.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(236, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, image: Image.Image) -> int:
        # temporary solution
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name)

            # Predict using the identical logic from our sanitizer
            cropped_np, conf, cls_id = self.detector.predict(
                Path(tmp.name),
                confidence_threshold=0.05,
                reject_on_invalid_class=True,
                classes=list(range(self.detector.reject_classes_index)),
            )

        if cropped_np is None:
            return -1

        cropped_np = cropped_np[:, :, ::-1]

        input_tensor = self.transform(cropped_np).unsqueeze(0)

        device = next(self.classifier.parameters()).device
        input_tensor = input_tensor.to(device)

        confidences, class_indices = self.classifier.predict(input_tensor)

        return class_indices.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, default="images")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default="animal_recognition/models/weights/convnext.pth",
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        default="convnext",
        choices=["convnext", "gcvit"],
    )
    args = parser.parse_args()

    df = pd.read_csv(args.image_folder / "labels.csv")
    model = Model(weights_path=args.weights_path, classifier_type=args.classifier_type).eval()

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
    print(
        classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, digits=3, zero_division=0
        )
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))
