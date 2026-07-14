import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from animal_recognition.src.config import load_config
from animal_recognition.src.models.baseline_cnn import BaselineCNN


def run_layer_activation(model: torch.nn.Module, image: Image.Image, cfg, target_class: int | None = None):
    model.eval()

    image_size = cfg.data.image_size
    mean = cfg.data.normalize_mean
    std = cfg.data.normalize_std

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean= mean, std= std),])
    input_tensor = transform(image).unsqueeze(0)

    resized = image.convert("RGB").resize((image_size, image_size))
    rgb_float = np.array(resized, dtype= np.float32) / 255.0

    target_layer = model.get_last_conv_layer()

    activations = {}

    def hook(module, input, output):
        activations["value"] = output.detach()

    handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = int(logits.argmax(dim= 1).item())

    handle.remove()

    if target_class is None:
        xai_cfg = getattr(cfg, "xai", None)
        target_class = getattr(xai_cfg, "target_class", None) if xai_cfg else None
    used_target = predicted_class if target_class is None else target_class

    feature_map = activations["value"][0]
    heatmap = feature_map.mean(dim= 0).cpu().numpy()

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_resized = cv2.resize(heatmap, (image_size, image_size))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    alpha = 0.5
    visualization = alpha * heatmap_color + (1 - alpha) * rgb_float
    visualization = np.uint8(255 * visualization)

    return visualization, predicted_class, used_target

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Layer Activation")
    parser.add_argument("--config", type= Path, default= None)
    parser.add_argument("--image", type= Path, required= True)
    parser.add_argument("--checkpoint", type= Path, default= None)
    parser.add_argument("--num-classes", type= int, default= None)
    parser.add_argument("--target-class", type= int, default= None)
    parser.add_argument("--output", type= Path, default= None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()

    num_classes = args.num_classes or cfg.classifier.num_classes
    model = BaselineCNN(num_classes=num_classes)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    image = Image.open(args.image).convert("RGB")

    visualization, predicted_class, used_target = run_layer_activation(
        model, image, cfg, target_class=args.target_class
    )

    print(f"Predicted class: {predicted_class}")
    print(f"Layer activation shown for class context: {used_target}")

    if args.output is not None:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        xai_cfg = getattr(cfg, "xai", None)
        output_dir = Path(getattr(xai_cfg, "output_dir", "data/xai_output")) if xai_cfg else Path("data/xai_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"layer_activation_{args.image.stem}.png"

    plt.imsave(output_path, visualization)
    print(f"Saved visualization to: {output_path}")