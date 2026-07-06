import numpy as np

import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image
from torchvision import transforms

def run_gradcam(model: torch.nn.Module, image: Image.Image, cfg: dict, target_class: int | None = None):
    model.eval()

    image_size = cfg["data"]["image_size"]
    mean = cfg["data"]["normalize_mean"]
    std = cfg["data"]["normalize_std"]

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean= mean, std= std),])
    input_tensor = transform(image).unsqueeze(0)

    resized = image.convert("RGB").resize((image_size, image_size))
    rgb_float = np.array(resized, dtype= np.float32) / 255.0

    target_layers = [model.get_last_conv_layer()]

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = int(logits.argmax(dim= 1).item())

    if target_class is None:
        target_class = cfg.get("xai", {}).get("target_class", None)

    used_target = predicted_class if target_class is None else target_class
    targets = [ClassifierOutputTarget(used_target)]

    with GradCAM(model= model, target_layers= target_layers) as cam:
        grayscale_cam = cam(input_tensor= input_tensor, targets= targets)
        grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb= True)
    return visualization, predicted_class, used_target