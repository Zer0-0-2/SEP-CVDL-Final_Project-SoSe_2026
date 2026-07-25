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
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


PROJECT_ROOT = Path(__file__).resolve().parent


class Model(nn.Module):
    """Open-vocabulary detector -> crop -> fine-grained classifier.

    Everything is driven by config.yaml; see that file for the deployed values
    and why they were chosen. The detector also supplies the reject decision:
    an image in which no accept prompt clears the confidence threshold is
    answered -1 without the classifier running. The OOD gate in
    src/ood/gate.py can be re-enabled with `ood.enabled: true`.
    """

    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(PROJECT_ROOT))
        import animal_recognition.src.data.augmentations as augmentations
        from animal_recognition.src.config import load_config
        from animal_recognition.src.models.classifier_convnext import ConvNextClassifier
        from animal_recognition.src.models.classifier_gcvit import GCViTClassifier
        from animal_recognition.src.models.yoloworld import YoloWorldDetector

        self.cfg = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- stage 1: localisation ------------------------------------
        det_cfg = self.cfg.detector
        if self.cfg.pipeline.detector != "yoloworld":
            raise ValueError(
                f"pipeline.detector={self.cfg.pipeline.detector!r} is not wired into this "
                "script; the reported system uses 'yoloworld'."
            )
        self.detector = YoloWorldDetector(model_name=det_cfg.model)
        self.det_threshold = det_cfg.confidence_threshold
        self.reject_on_invalid_class = det_cfg.reject_on_invalid_class
        # Indices 0..reject_classes_index-1 are the accept prompts.
        self.accept_prompts = (
            list(range(self.detector.reject_classes_index))
            if det_cfg.restrict_to_accept_prompts
            else None
        )

        # ---- stage 2: classification ----------------------------------
        clf_cfg = self.cfg.classifier
        backbones = {"gcvit": GCViTClassifier, "convnext": ConvNextClassifier}
        if self.cfg.pipeline.classifier not in backbones:
            raise ValueError(f"Unknown pipeline.classifier: {self.cfg.pipeline.classifier!r}")
        self.classifier = backbones[self.cfg.pipeline.classifier](
            pretrained=False, model_name=clf_cfg.backbone
        )

        weights = Path(clf_cfg.weights)
        if not weights.is_absolute():
            weights = PROJECT_ROOT / weights
        if not weights.exists():
            raise FileNotFoundError(f"Classifier weights not found: {weights}")
        self.classifier.load_state_dict(torch.load(weights, map_location="cpu"))
        self.classifier = self.classifier.to(self.device).eval()

        # Same validation transform the checkpoint was evaluated with.
        self._transform = augmentations.get_val_transforms(image_size=clf_cfg.image_size)

        # ---- stage 3: optional reject gate ----------------------------
        self.gate = None
        if getattr(self.cfg.ood, "enabled", False):
            from animal_recognition.src.ood.gate import OODGate

            self.gate = OODGate(self.cfg)

    def forward(self, image: Image.Image) -> int:
        # YoloWorldDetector reads from disk, so the PIL image goes through a
        # temporary file. This is the exact path the reported numbers were
        # measured on, JPEG round-trip included.
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name)
            crop, _, _ = self.detector.predict(
                Path(tmp.name),
                confidence_threshold=self.det_threshold,
                reject_on_invalid_class=self.reject_on_invalid_class,
                # ultralytics caches predictor kwargs between calls, so this
                # has to be passed every time rather than once at setup.
                classes=self.accept_prompts,
            )

        if crop is None:
            return REJECT

        # The detector returns BGR (OpenCV); the classifier was trained on RGB.
        crop = np.ascontiguousarray(crop[:, :, ::-1])
        tensor = self._transform(image=crop)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(tensor).squeeze(0)

        if self.gate is not None:
            return self.gate(logits)
        return int(logits.argmax().item())


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
