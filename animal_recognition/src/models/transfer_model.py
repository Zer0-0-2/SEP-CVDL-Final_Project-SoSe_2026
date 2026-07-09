
from __future__ import annotations
from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn


class TransferClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 20,
        pretrained: bool = True,
        weights: Optional[str | Path] = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        self.head = nn.Linear(feature_dim, num_classes)

        if weights is not None:
            self.load_state_dict(torch.load(weights, map_location="cpu"))

    def get_last_conv_layer(self) -> nn.Module:
        for module in reversed(list(self.backbone.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise RuntimeError("No Conv2d found — use a CNN backbone, not ViT")

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        if not hasattr(self.backbone, "blocks"):
            raise AttributeError("Backbone has no .blocks — try efficientnet_b3 or convnext_tiny")
        for block in self.backbone.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


if __name__ == "__main__":
    model = TransferClassifier(backbone="efficientnet_b3", pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")  # expect [4, 20]

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")

    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after freeze: {trainable:,}  (head only)")

    model.unfreeze_all()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after unfreeze: {trainable:,}  (full model)")

