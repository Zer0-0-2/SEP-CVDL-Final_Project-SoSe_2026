import torch
import torch.nn as nn
import timm
from pathlib import Path

# Import the dataset classes to determine the number of classes
import animal_recognition.src.data.dataset as animal_dataset

class SwinClassifier(nn.Module):
    def __init__(self, pretrained: bool = False, model_name: str = "swin_tiny_patch4_window7_224"):
        """
        output of timm_list_models("swin*"):
        ['swin_base_patch4_window7_224',
        'swin_base_patch4_window12_384',
        'swin_large_patch4_window7_224',
        'swin_large_patch4_window12_384',
        'swin_s3_base_224',
        'swin_s3_small_224',
        'swin_s3_tiny_224',
        'swin_small_patch4_window7_224',
        'swin_tiny_patch4_window7_224',
        'swinv2_base_window8_256',
        'swinv2_base_window12_192',
        'swinv2_base_window12to16_192to256',
        'swinv2_base_window12to24_192to384',
        'swinv2_base_window16_256',
        'swinv2_cr_base_224',
        'swinv2_cr_base_384',
        'swinv2_cr_small_224',
        'swinv2_cr_small_384',
        'swinv2_cr_tiny_224',
        'swinv2_cr_tiny_384',
        'swinv2_large_window12to16_192to256',
        'swinv2_large_window12to24_192to384',
        'swinv2_small_window8_256',
        'swinv2_small_window16_256',
        'swinv2_tiny_window8_256',
        'swinv2_tiny_window16_256']
        """

        super().__init__()
        self.num_classes = len(animal_dataset.CLASSES)

        self.model = timm.create_model(
            model_name, pretrained= pretrained, num_classes= self.num_classes
        )

        self.architecture = "swin"
        self.model_name = model_name
        self.pretrained = pretrained

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            self.model.eval()
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            confidences, class_indices = torch.max(probabilities, dim=1)

        return confidences, class_indices
    
    def get_last_conv_layer(self) -> nn.Module:
        return self.model.patch_embed.proj 
