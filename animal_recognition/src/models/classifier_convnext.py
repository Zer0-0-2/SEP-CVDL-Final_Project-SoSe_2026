import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)
from pathlib import Path

# Import the dataset classes to determine the number of classes
import animal_recognition.src.data.dataset as animal_dataset


class ConvNextClassifier(nn.Module):
    def __init__(self, pretrained: bool = False, model_name: str = "convnext_tiny"):
        """
        Model names: "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
        the first three have 224x224 input size, the last one has 384x384 input size (probably too large for compute resources)
        """

        super().__init__()
        self.num_classes = len(animal_dataset.CLASSES)

        # for later
        if not pretrained:
            weights = None
        else:
            if model_name == "convnext_tiny":
                weights = ConvNeXt_Tiny_Weights.DEFAULT
            elif model_name == "convnext_small":
                weights = ConvNeXt_Small_Weights.DEFAULT
            elif model_name == "convnext_base":
                weights = ConvNeXt_Base_Weights.DEFAULT
            elif model_name == "convnext_large":
                weights = ConvNeXt_Large_Weights.DEFAULT
            else:
                raise ValueError(f"unknown model name: {model_name}")

        if model_name == "convnext_tiny":
            model = convnext_tiny(weights=weights)
        elif model_name == "convnext_small":
            model = convnext_small(weights=weights)
        elif model_name == "convnext_base":
            model = convnext_base(weights=weights)
        elif model_name == "convnext_large":
            model = convnext_large(weights=weights)
        else:
            raise ValueError(f"unknown model name: {model_name}")

        self.model: nn.Module = model

        # assert needed to fix some IDE issue, runs fine without it as well though
        last_layer = self.model.classifier[-1]
        assert isinstance(last_layer, nn.Linear)
        in_features: int = last_layer.in_features

        self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def predict(self, x: torch.Tensor):
        """
        returns the predicted class indices and the softmax probabilities
        """
        with torch.no_grad():
            self.model.eval()
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            confidences, class_indices = torch.max(probabilities, dim=1)

        return confidences, class_indices
