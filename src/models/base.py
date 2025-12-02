"""
Base model class for all architectures
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all model architectures
    """

    def __init__(self, num_classes: int = 4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass

    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_layers(self, num_layers: int = None):
        """
        Freeze layers for transfer learning

        Args:
            num_layers: Number of layers to freeze. If None, freeze all.
        """
        if num_layers is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Architecture-specific implementation in subclasses
            pass

    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': self.get_num_parameters(),
            'trainable_parameters': self.get_trainable_parameters()
        }

    def __repr__(self) -> str:
        info = self.get_model_info()
        return (f"{info['name']}("
                f"classes={info['num_classes']}, "
                f"params={info['total_parameters']:,}, "
                f"trainable={info['trainable_parameters']:,})")
