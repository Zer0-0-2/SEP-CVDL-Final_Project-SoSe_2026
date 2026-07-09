import torch
import torch.nn as nn
import timm
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

        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=self.num_classes
        )

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
