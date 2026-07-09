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