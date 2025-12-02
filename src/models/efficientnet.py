"""
EfficientNet-B3 architecture for pneumonia detection
"""

import torch
import torch.nn as nn
from torchvision import models
from .base import BaseModel


class EfficientNetB3Model(BaseModel):
    """
    EfficientNet-B3 with custom classifier head for multi-class classification
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_batch_norm: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            freeze_batch_norm: Freeze batch normalization layers
            dropout: Dropout probability
        """
        super().__init__(num_classes=num_classes)

        # Load pretrained EfficientNet-B3
        if pretrained:
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b3(weights=None)

        # Freeze batch normalization layers
        if freeze_batch_norm:
            self._freeze_bn_layers()

        # Get number of features
        num_features = self.backbone.classifier[1].in_features

        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(512, num_classes)
        )

    def _freeze_bn_layers(self):
        """Freeze all batch normalization layers"""
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Logits [B, num_classes]
        """
        return self.backbone(x)

    def get_feature_extractor(self):
        """Get backbone feature extractor (without classifier)"""
        return self.backbone.features

    def train(self, mode: bool = True):
        """
        Override train mode to keep BN layers in eval mode

        Args:
            mode: Training mode
        """
        super().train(mode)
        # Keep batch norm layers in eval mode
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self
