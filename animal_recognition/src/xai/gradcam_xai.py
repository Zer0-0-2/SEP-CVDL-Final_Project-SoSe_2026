import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image
from torchvision import transforms
import argparse
from pathlib import Path

XAI_METHODS = {
    "gradcam": GradCAM,
}

def build_cam_methode(method_name: str, model: torch.nn.Module, target_layers: list, cfg: dict):
    method_name = method_name.lower()
    if method_name not in XAI_METHODS:
        raise ValueError(f"'This is not a valide methode: {method_name}.")

    cam_class = XAI_METHODS[method_name]    

    return cam_class(model= model, target_layers= target_layers)