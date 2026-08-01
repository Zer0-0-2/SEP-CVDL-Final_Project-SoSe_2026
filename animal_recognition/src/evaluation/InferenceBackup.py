"""Evaluation harness for the Fine-grained Animal Recognition project.

Same interface as inference.py, but with the OOD gate (src/ood/gate.py) wired
into the forward pass instead of a plain argmax. Switch between OOD gate
modes ('softmax_threshold' | 'energy' | 'temperature_scaling') via
--ood-gate; --ood-threshold / --ood-temperature override the gate's
sensitivity. Defaults are read from config.yaml so behavior stays in sync
with the rest of the pipeline unless you override them on the CLI.

    python -m animal_recognition.src.evaluation.InferenceBackup \
        --image-folder <folder> \
        --classifier-type convnext \
        --weights-path animal_recognition/src/models/models_weights/<weights_file>.pt \
        --ood-gate softmax_threshold
"""

from animal_recognition.src.models.yoloworld import YoloWorldDetector
from animal_recognition.src.models.classifier_convnext import ConvNextClassifier
from animal_recognition.src.models.classifier_gcvit import GCViTClassifier
from animal_recognition.src.ood.gate import OODGate
from animal_recognition.src.config import load_config
import tempfile

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

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
        model_name: str | None = None,
        ood_gate_mode: str = "softmax_threshold",
        ood_threshold: float = 0.5,
        ood_temperature: float = 1.0,
    ):
        super().__init__()

        # use yoloworld with the default values (which are already analyzed to be best on test dataset)
        self.detector = YoloWorldDetector()

        if model_name is None:
            # per-classifier defaults; a convnext default would silently build the
            # wrong architecture (and load garbage weights) if applied to gcvit.
            model_name = "convnext_tiny" if classifier_type == "convnext" else "gcvit_tiny"

        if classifier_type == "convnext":
            self.classifier = ConvNextClassifier(pretrained=False, model_name=model_name)
        elif classifier_type == "gcvit":
            self.classifier = GCViTClassifier(pretrained=False, model_name=model_name)
        else:
            raise ValueError(f"Unsupported classifier_type: {classifier_type}")

        self.weights_path = Path(weights_path)

        if self.weights_path.exists():
            map_location = None if torch.cuda.is_available() else torch.device("cpu")
            self.classifier.load_state_dict(torch.load(self.weights_path, map_location=map_location))
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

        # OODGate expects cfg.pipeline.ood_gate / cfg.ood.threshold / cfg.ood.temperature.
        # Build a minimal stand-in so the gate mode/threshold/temperature can be
        # switched per run (CLI) without touching config.yaml.
        ood_cfg = SimpleNamespace(
            pipeline=SimpleNamespace(ood_gate=ood_gate_mode),
            ood=SimpleNamespace(threshold=ood_threshold, temperature=ood_temperature),
        )
        self.ood_gate = OODGate(ood_cfg)

    def get_logits(self, image: Image.Image) -> torch.Tensor | None:
        """Runs detector + classifier and returns raw classifier logits for one
        image, or None if the detector found no valid crop. Shared by forward()
        and by other scripts (e.g. ood_gate_accuracy_comparison.py) that need
        the logits before any OOD-gate decision is applied.
        """
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
            return None

        cropped_np = cropped_np[:, :, ::-1]

        input_tensor = self.transform(cropped_np).unsqueeze(0)

        device = next(self.classifier.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            return self.classifier(input_tensor)

    def forward(self, image: Image.Image) -> int:
        logits = self.get_logits(image)
        if logits is None:
            return -1
        return self.ood_gate(logits)


if __name__ == "__main__":
    cfg = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, default="images")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default="animal_recognition/src/models/models_weights/convnext.pth",
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        default="convnext",
        choices=["convnext", "gcvit"],
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Backbone variant, must match the checkpoint's architecture "
            "(encoded in its filename). Defaults to 'convnext_tiny' "
            "(--classifier-type convnext) or 'gcvit_tiny' (--classifier-type gcvit) "
            "if omitted. For convnext: 'convnext_tiny' | 'convnext_small' | "
            "'convnext_base' | 'convnext_large'. For gcvit: any timm gcvit_* name."
        ),
    )
    parser.add_argument(
        "--ood-gate",
        type=str,
        default=cfg.pipeline.ood_gate,
        choices=["softmax_threshold", "energy", "temperature_scaling"],
        help="OOD gate mode to use for accept/reject decisions.",
    )
    parser.add_argument(
        "--ood-threshold",
        type=float,
        default=cfg.ood.threshold,
        help="softmax_threshold: reject below this confidence. energy/temperature_scaling: reject above this energy.",
    )
    parser.add_argument(
        "--ood-temperature",
        type=float,
        default=cfg.ood.temperature,
        help="Temperature T used by energy/temperature_scaling gates.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.image_folder / "labels.csv")
    model = Model(
        weights_path=args.weights_path,
        classifier_type=args.classifier_type,
        model_name=args.model_name,
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
    print(f"\nOOD gate: {args.ood_gate} (threshold={args.ood_threshold}, temperature={args.ood_temperature})")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(
        classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, digits=3, zero_division=0
        )
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=labels))
