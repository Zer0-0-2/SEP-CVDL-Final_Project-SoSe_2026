import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.config import load_config
from src.models.baseline_cnn import BaselineCNN
from captum.attr import Occlusion

def run_occlusion(model: torch.nn.Module, image: Image.Image, cfg, target_class: int | None = None, patch_size: int | None = None, stride: int | None = None):
    model.eval()

    image_size = cfg.data.image_size
    mean = cfg.data.normalize_mean
    std = cfg.data.normalize_std

    xai_cfg = getattr(cfg, "xai", None)
    occlusion_cfg = getattr(xai_cfg, "occlusion", None) if xai_cfg else None
    if patch_size is None:
        patch_size = getattr(occlusion_cfg, "patch_size", 32) if occlusion_cfg else 32
    if stride is None:
        stride = getattr(occlusion_cfg, "stride", 16) if occlusion_cfg else 16
    
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean= mean, std= std),])
    input_tensor = transform(image).unsqueeze(0)

    resized = image.convert("RGB").resize((image_size, image_size))
    rgb_float = np.array(resized, dtype= np.float32) / 255.0

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = int(logits.argmax(dim= 1).item())

    if target_class is None:
        target_class = getattr(xai_cfg, "target_class", None) if xai_cfg else None

    used_target = predicted_class if target_class is None else target_class

    occlusion = Occlusion(model)

    attributions = occlusion.attribute(input_tensor, target= used_target, sliding_window_shapes= (3, patch_size, patch_size),
                                        strides= (3, stride, stride), baselines= 0)
    
    heatmap = attributions.squeeze(0).mean(dim= 0).detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    alpha = 0.5
    visualization = alpha * heatmap_color + (1 - alpha) * rgb_float
    visualization = np.uint8(255 * visualization)

    return visualization, predicted_class, used_target