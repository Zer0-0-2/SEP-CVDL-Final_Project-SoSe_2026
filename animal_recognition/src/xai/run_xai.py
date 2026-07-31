import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from animal_recognition.src.config import load_config
from animal_recognition.src.data.dataset import CLASSES
from animal_recognition.src.models.baseline_cnn import BaselineCNN
from animal_recognition.src.models.classifier_convnext import ConvNextClassifier
from animal_recognition.src.models.classifier_gcvit import GCViTClassifier
from animal_recognition.src.models.classifier_swin import SwinClassifier

from animal_recognition.src.xai.gradcam_xai import run_gradcam
from animal_recognition.src.xai.occlusion_xai import run_occlusion
from animal_recognition.src.xai.layer_activation_xai import run_layer_activation

CLASSIFIER_REGISTRY = {
    "baseline_cnn": lambda model_name, num_classes: BaselineCNN(num_classes=num_classes),
    "convnext": lambda model_name, num_classes: ConvNextClassifier(pretrained=False, model_name=model_name),
    "gcvit": lambda model_name, num_classes: GCViTClassifier(pretrained=False, model_name=model_name),
    "swin": lambda model_name, num_classes: SwinClassifier(pretrained=False, model_name=model_name),
}

XAI_REGISTRY = {
    "gradcam_xai": run_gradcam,
    "occlusion_xai": run_occlusion,
    "layer_activation_xai": run_layer_activation,
}

DEFAULT_WEIGHTS_DIR = Path("animal_recognition/models/weights")

def resolve_checkpoint_path(weights_arg: str, weights_dir: Path = DEFAULT_WEIGHTS_DIR) -> Path:
    direct_path = Path(weights_arg)
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    matches = list(weights_dir.rglob(f"*{weights_arg}*.pt"))
    if not matches:
        raise FileNotFoundError(
            f"No Checkpoint found"
        )
    if len(matches) > 1:
        print(f"WARNING: {len(matches)} found '{weights_arg}' take first:")
        for m in matches:
            print(f"   - {m}")
    return matches[0]

def build_classifier(cfg, classifier_override: str | None, classifier_type: str | None):
    classifier_name = classifier_override or cfg.pipeline.classifier

    if classifier_name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"No Classifier '{classifier_name}'. "
            f"Available: {list(CLASSIFIER_REGISTRY.keys())}"
        )

    num_classes = cfg.classifier.num_classes
    builder = CLASSIFIER_REGISTRY[classifier_name]

    model_name = classifier_type
    if classifier_name != "baseline_cnn" and model_name is None:
        raise ValueError(
            f"for classifier='{classifier_name}' --classifier-type is necessary "
            f"(e.g. 'convnext_tiny', 'gcvit_tiny', 'swin_tiny_patch4_window7_224')."
        )

    return builder(model_name, num_classes), classifier_name

def run_xai(model, image: Image.Image, cfg, method_override: str | None = None, target_class: int | None = None):
    method_name = method_override or cfg.xai.methode

    if method_name not in XAI_REGISTRY:
        raise ValueError(
            f"Unknown XAI-Methode '{method_name}'. "
            f"Available: {list(XAI_REGISTRY.keys())}"
        )

    xai_function = XAI_REGISTRY[method_name]

    if method_name == "occlusion_xai":
        occlusion_cfg = getattr(cfg.xai, "occlusion", None)
        patch_size = getattr(occlusion_cfg, "patch_size", 32) if occlusion_cfg else 32
        stride = getattr(occlusion_cfg, "stride", 16) if occlusion_cfg else 16
        visualization, predicted_class, used_target = xai_function(
            model, image, cfg, target_class=target_class, patch_size=patch_size, stride=stride
        )
    else:
        visualization, predicted_class, used_target = xai_function(
            model, image, cfg, target_class=target_class
        )

    return visualization, predicted_class, used_target, method_name

def class_name(idx: int) -> str:
    if idx == -1:
        return "reject(-1)"
    if 0 <= idx < len(CLASSES):
        return CLASSES[idx]
    return f"unknown({idx})"

def save_with_title(visualization, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(visualization)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="central XAI-script — Classifier & Methode from config.yaml")
    parser.add_argument("--config", type=Path, default=None,
                         help="optional path to config.yaml; Standard = <project_root>/config.yaml")
    parser.add_argument("--image", type=Path, required=True,
                         help="Path to the image")
    parser.add_argument("--weights-path", type=str, required=True,
                         help="Direct path to a .pt checkpoint OR a search pattern "
                         "(e.g. 'convnext_tiny_baseline') to search recursively in 'animal_recognition/models/weights/' (incl. subfolders).")
    parser.add_argument("--classifier-type", type=str, default=None,
                         help="concrete timm-Modellname, e.g. 'convnext_tiny', 'gcvit_tiny', "
                              "'swin_tiny_patch4_window7_224'. not necessary for classifier='baseline_cnn'.")
    parser.add_argument("--classifier", type=str, default=None, choices=list(CLASSIFIER_REGISTRY.keys()),
                         help="Overrides cfg.pipeline.classifier, if available")
    parser.add_argument("--method", type=str, default=None, choices=list(XAI_REGISTRY.keys()),
                         help="Overrides cfg.xai.methode, if available")
    parser.add_argument("--target-class", type=int, default=None,
                         help="Overrides cfg.xai.target_class, if available")
    parser.add_argument("--output", type=Path, default=None,
                         help="optional output dir")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()


    model, classifier_name = build_classifier(cfg, args.classifier, args.classifier_type)

    checkpoint_path = resolve_checkpoint_path(args.weights_path)
    print(f"Lade Checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()


    image = Image.open(args.image).convert("RGB")


    visualization, predicted_class, used_target, method_used = run_xai(
        model, image, cfg, method_override=args.method, target_class=args.target_class
    )

    predicted_name = class_name(predicted_class)
    target_name = class_name(used_target)

    print(f"Classifier: {classifier_name} ({args.classifier_type or 'default'})")
    print(f"XAI-Methode: {method_used}")
    print(f"Predicted class: {predicted_class} ({predicted_name})")
    print(f"Heatmap generated for class: {used_target} ({target_name})")


    if args.output is not None:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        xai_cfg = getattr(cfg, "xai", None)
        output_dir = Path(getattr(xai_cfg, "output_dir", "animal_recognition/data"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{classifier_name}_{method_used}_{predicted_name}_{args.image.stem}.png"


    if used_target == predicted_class:
        title = f"{classifier_name} | {method_used}\nPredicted: {predicted_name}"
    else:
        title = f"{classifier_name} | {method_used}\nPredicted: {predicted_name} | Heatmap for: {target_name}"

    save_with_title(visualization, output_path, title)
    print(f"Saved visualization to: {output_path}")