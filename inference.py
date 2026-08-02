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

Every pipeline choice (classifier architecture, backbone/model name, weights
checkpoint, OOD gate on/off, OOD gate mode, threshold, temperature) defaults
to whatever is set in config.yaml.
Any of them can be overridden on the CLI without
touching config.yaml, e.g. to try a different checkpoint or gate:

    python inference.py --image-folder <folder> \
        --classifier-type gcvit --model-name gcvit_tiny \
        --weights-path <path/to/checkpoint>.pt \
        --ood-enabled --ood-gate energy --ood-threshold -3.9
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

    def __init__(
        self,
        classifier_type: str | None = None,
        model_name: str | None = None,
        weights_path: Path | str | None = None,
        ood_enabled: bool | None = None,
        ood_gate_mode: str | None = None,
        ood_threshold: float | None = None,
        ood_temperature: float | None = None,
    ):
        """All arguments default to config.yaml when omitted (None), so
        `Model()` reproduces the submitted configuration. Pass any of them
        to override just that one setting without touching config.yaml.
        """
        super().__init__()
        sys.path.insert(0, str(PROJECT_ROOT))
        import animal_recognition.src.data.augmentations_stronger as augmentations
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

        classifier_type = classifier_type or self.cfg.pipeline.classifier
        if classifier_type not in backbones:
            raise ValueError(f"Unknown classifier_type: {classifier_type!r}")

        if model_name is not None:
            pass  # explicit override wins
        elif classifier_type == self.cfg.pipeline.classifier:
            model_name = clf_cfg.backbone  # unchanged from config -> use its backbone
        else:
            # classifier_type was overridden but model_name wasn't: config's
            # backbone belongs to the *original* classifier and would build
            # the wrong architecture, so fall back to a safe per-type default.
            model_name = "convnext_tiny" if classifier_type == "convnext" else "gcvit_tiny"

        self.classifier = backbones[classifier_type](pretrained=False, model_name=model_name)

        weights = Path(weights_path) if weights_path is not None else Path(clf_cfg.weights)
        if not weights.is_absolute():
            weights = PROJECT_ROOT / weights
        if not weights.exists():
            raise FileNotFoundError(
                f"Classifier weights not found: {weights}\n"
                "Set classifier.weights in config.yaml, or pass --weights-path."
            )
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        self.classifier.load_state_dict(torch.load(weights, map_location=map_location))
        self.classifier = self.classifier.to(self.device).eval()

        # Same validation transform the checkpoint was evaluated with.
        self._transform = augmentations.get_val_transforms(image_size=clf_cfg.image_size)

        # ---- stage 3: optional reject gate ----------------------------
        ood_enabled = getattr(self.cfg.ood, "enabled", False) if ood_enabled is None else ood_enabled
        self.gate = None
        if ood_enabled:
            from animal_recognition.src.ood.gate import OODGate
            from types import SimpleNamespace

            gate_cfg = SimpleNamespace(
                pipeline=SimpleNamespace(ood_gate=ood_gate_mode or self.cfg.pipeline.ood_gate),
                ood=SimpleNamespace(
                    threshold=self.cfg.ood.threshold if ood_threshold is None else ood_threshold,
                    temperature=self.cfg.ood.temperature if ood_temperature is None else ood_temperature,
                ),
            )
            self.gate = OODGate(gate_cfg)

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
    # dataset selection
    parser.add_argument("--image-folder", type=Path, default="images")
    # model selection
    parser.add_argument(
        "--classifier-type", type=str, default=None, choices=["convnext", "gcvit"],
        help="Defaults to pipeline.classifier in config.yaml.",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help=(
            "Backbone/architecture name, must match the checkpoint (e.g. "
            "'convnext_tiny', 'convnext_small', 'gcvit_tiny'). Defaults to "
            "classifier.backbone in config.yaml."
        ),
    )
    parser.add_argument(
        "--weights-path", type=Path, default=None,
        help="Path to a .pt/.pth checkpoint. Defaults to classifier.weights in config.yaml.",
    )
    # OOD gate selection
    parser.add_argument(
        "--ood-enabled", action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable the OOD gate. Defaults to ood.enabled in config.yaml.",
    )
    parser.add_argument(
        "--ood-gate", type=str, default=None,
        choices=["softmax_threshold", "energy", "temperature_scaling"],
        help="OOD gate mode. Defaults to pipeline.ood_gate in config.yaml.",
    )
    parser.add_argument(
        "--ood-threshold", type=float, default=None,
        help="softmax_threshold: reject below this confidence. energy/temperature_scaling: reject above this energy. Defaults to ood.threshold in config.yaml.",
    )
    parser.add_argument(
        "--ood-temperature", type=float, default=None,
        help="Temperature T used by energy/temperature_scaling gates. Defaults to ood.temperature in config.yaml.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.image_folder / "labels.csv")
    model = Model(
        classifier_type=args.classifier_type,
        model_name=args.model_name,
        weights_path=args.weights_path,
        ood_enabled=args.ood_enabled,
        ood_gate_mode=args.ood_gate,
        ood_threshold=args.ood_threshold,
        ood_temperature=args.ood_temperature,
    ).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for filename, label in tqdm(zip(df["filename"], df["label"]), total=len(df)):
            image = Image.open(args.image_folder / filename).convert("RGB")
            pred = model(image)
            y_true.append(int(label))
            y_pred.append(int(pred))

    labels = [REJECT] + list(range(NUM_CLASSES))
    target_names = ["reject(-1)"] + CLASSES
    gate_desc = "disabled" if model.gate is None else model.gate.mode
    print(f"\nClassifier: {model.classifier.__class__.__name__} | OOD gate: {gate_desc}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=target_names, digits=3,
                                zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))
