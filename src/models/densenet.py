"""
DenseNet201 architecture for pneumonia detection
"""

import torch
import torch.nn as nn
from torchvision import models
from .base import BaseModel


class DenseNet201Model(BaseModel):
    """
    DenseNet201 with custom classifier head for multi-class classification
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze DenseNet feature layers
            dropout: Dropout probability
        """
        super().__init__(num_classes=num_classes)

        # Load pretrained DenseNet201
        if pretrained:
            self.backbone = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.densenet201(weights=None)

        # Freeze feature extraction layers
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Get number of features
        num_features = self.backbone.classifier.in_features

        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.3),
            nn.Linear(256, num_classes)
        )

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
