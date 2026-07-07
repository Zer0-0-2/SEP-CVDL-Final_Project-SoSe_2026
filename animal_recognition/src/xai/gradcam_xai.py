import numpy as np

import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image
from torchvision import transforms

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from src.config import load_config
from src.models.baseline_cnn import BaselineCNN

def run_gradcam(model: torch.nn.Module, image: Image.Image, cfg, target_class: int | None = None):
    model.eval()

    image_size = cfg.data.image_size
    mean = cfg.data.normalize_mean
    std = cfg.data.normalize_std

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean= mean, std= std),])
    input_tensor = transform(image).unsqueeze(0)

    resized = image.convert("RGB").resize((image_size, image_size))
    rgb_float = np.array(resized, dtype= np.float32) / 255.0

    target_layers = [model.get_last_conv_layer()]

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = int(logits.argmax(dim= 1).item())

    if target_class is None:
        xai_cfg = getattr(cfg, "xai", None)
        target_class = getattr(xai_cfg, "target_class", None) if xai_cfg else None

    used_target = predicted_class if target_class is None else target_class
    targets = [ClassifierOutputTarget(used_target)]

    with GradCAM(model= model, target_layers= target_layers) as cam:
        grayscale_cam = cam(input_tensor= input_tensor, targets= targets)
        grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb= True)
    return visualization, predicted_class, used_target


if __name__ == "__main__":
    parser = argparse.ArugmentParser(description= "Grad-CAM for classifier")
    parser.add_argument("--config", type= Path, default = None)
    parser.add_argument("--image", type= Path, required= True)
    parser.add_argument("--checkpoint", type= Path, default= None)
    parser.add_argument("--num-classes", type= int, default= None)
    parser.add_argument("--target-class", type= int, default= None)
    parser.add_argument("--output", type= Path, default= Path("gradcam_output.png"))
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()

    num_classes = args.num_classes or cfg.classifier.num_classes
    model = BaselineCNN(num_classes= cfg.classifier.num_classes)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location= "cpu"))

    image = Image.open(args.image).convert("RGB")
    visualization, predicted_class, used_target = run_gradcam(model, image, cfg, args.target_class)

    print(f"Predicted class: {predicted_class}")
    print(f"Heatmap generated for class: {used_target}")
    plt.imsave(args.output, visualization)
    print(f"Saved Heatmap to: {args.output}")