"""
ResNet152 architecture for pneumonia detection
"""

import torch
import torch.nn as nn
from torchvision import models
from .base import BaseModel


class ResNet152Model(BaseModel):
    """
    ResNet152 with custom classifier head for multi-class classification
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_layers: int = 3,
        dropout: float = 0.5
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            freeze_layers: Number of residual blocks to freeze (0-4)
            dropout: Dropout probability
        """
        super().__init__(num_classes=num_classes)

        # Load pretrained ResNet152
        if pretrained:
            self.backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet152(weights=None)

        # Freeze early layers for transfer learning
        if freeze_layers > 0:
            self._freeze_backbone_layers(freeze_layers)

        # Get number of features from last layer
        num_features = self.backbone.fc.in_features

        # Replace classifier head with custom layers
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            nn.Linear(256, num_classes)
        )

    def _freeze_backbone_layers(self, num_layers: int):
        """
        Freeze early ResNet layers

        Args:
            num_layers: Number of layers to freeze (1-4)
                1: Freeze conv1 + bn1
                2: Freeze + layer1
                3: Freeze + layer2
                4: Freeze + layer3
        """
        # Always freeze initial conv and bn
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False

        # Freeze layer blocks
        layers_to_freeze = []
        if num_layers >= 2:
            layers_to_freeze.append(self.backbone.layer1)
        if num_layers >= 3:
            layers_to_freeze.append(self.backbone.layer2)
        if num_layers >= 4:
            layers_to_freeze.append(self.backbone.layer3)

        for layer in layers_to_freeze:
            for param in layer.parameters():
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
        return nn.Sequential(*list(self.backbone.children())[:-1])
