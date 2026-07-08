import torch
import torch.nn as nn
import timm
from pathlib import Path

# Import the dataset classes to determine the number of classes
import animal_recognition.src.data.dataset as animal_dataset


# https://github.com/NVlabs/GCVit
class GCViTClassifier(nn.Module):
    def __init__(self, pretrained: bool = False, model_name: str = "gcvit_tiny"):
        """
        Model names: "gcvit_tiny", "gcvit_small", "gcvit_base", etc.
        By default, gcvit takes 224x224 input sizes.
        """
        super().__init__()
        self.num_classes = len(animal_dataset.CLASSES)

        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=self.num_classes
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            self.model.eval()
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            confidences, class_indices = torch.max(probabilities, dim=1)

        return confidences, class_indices
