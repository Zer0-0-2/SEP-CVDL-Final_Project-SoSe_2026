import torch
import torch.nn as nn
import timm
from pathlib import Path

# Import the dataset classes to determine the number of classes
import animal_recognition.src.data.dataset as animal_dataset


class ConvNextClassifier(nn.Module):
    def __init__(self, pretrained: bool = False, model_name: str = "convnext_tiny"):
        """
        output of timm_list_models():
        ['convnext_atto',
        'convnext_atto_ols',
        'convnext_atto_rms',
        'convnext_base',
        'convnext_femto',
        'convnext_femto_ols',
        'convnext_large',
        'convnext_large_mlp',
        'convnext_nano',
        'convnext_nano_ols',
        'convnext_pico',
        'convnext_pico_ols',
        'convnext_small',
        'convnext_tiny',
        'convnext_tiny_hnf',
        'convnext_xlarge',
        'convnext_xxlarge',
        'convnext_zepto_rms',
        'convnext_zepto_rms_ols',
        'convnextv2_atto',
        'convnextv2_base',
        'convnextv2_femto',
        'convnextv2_huge',
        'convnextv2_large',
        'convnextv2_nano',
        'convnextv2_pico',
        'convnextv2_small',
        'convnextv2_tiny',
        'test_convnext',
        'test_convnext2',
        'test_convnext3']
        """

        super().__init__()
        self.num_classes = len(animal_dataset.CLASSES)
        self.architecture = "convnext"
        self.model_name = model_name
        self.pretrained = pretrained

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
    
    def get_last_conv_layer(self) -> nn.Module:
        return self.model.stages[-1].blocks[-1].conv_dw
