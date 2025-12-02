"""
Ensemble model combining ResNet152, DenseNet201, and EfficientNet-B3
Critical for achieving >94% accuracy target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from .base import BaseModel
from .resnet import ResNet152Model
from .densenet import DenseNet201Model
from .efficientnet import EfficientNetB3Model


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models with weighted voting
    Combines predictions from ResNet152, DenseNet201, and EfficientNet-B3
    """

    def __init__(
        self,
        num_classes: int = 4,
        voting: str = 'soft',
        weights: Optional[List[float]] = None,
        pretrained: bool = True
    ):
        """
        Args:
            num_classes: Number of output classes
            voting: 'soft' (probability averaging) or 'hard' (majority voting)
            weights: Weights for each model [resnet, densenet, efficientnet]
                     If None, uses equal weights [0.33, 0.33, 0.34]
            pretrained: Use pretrained backbones
        """
        super().__init__(num_classes=num_classes)

        self.voting = voting
        self.num_models = 3

        # Initialize individual models
        self.resnet152 = ResNet152Model(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_layers=3,
            dropout=0.5
        )

        self.densenet201 = DenseNet201Model(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=True,
            dropout=0.5
        )

        self.efficientnet_b3 = EfficientNetB3Model(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_batch_norm=True,
            dropout=0.3
        )

        # Set ensemble weights
        if weights is None:
            self.weights = torch.tensor([0.33, 0.33, 0.34])
        else:
            if len(weights) != 3:
                raise ValueError("Weights must have 3 values for 3 models")
            if abs(sum(weights) - 1.0) > 1e-5:
                raise ValueError("Weights must sum to 1.0")
            self.weights = torch.tensor(weights)

        self.models = [self.resnet152, self.densenet201, self.efficientnet_b3]
        self.model_names = ['ResNet152', 'DenseNet201', 'EfficientNet-B3']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Ensemble predictions [B, num_classes]
        """
        # Get predictions from all models
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Stack outputs [num_models, B, num_classes]
        outputs = torch.stack(outputs)

        if self.voting == 'soft':
            # Soft voting: Average probabilities
            probabilities = F.softmax(outputs, dim=2)

            # Apply weights
            weights = self.weights.view(-1, 1, 1).to(x.device)
            weighted_probs = probabilities * weights

            # Sum weighted probabilities
            ensemble_probs = weighted_probs.sum(dim=0)

            # Convert back to logits for loss computation
            ensemble_logits = torch.log(ensemble_probs + 1e-10)

            return ensemble_logits

        else:
            # Hard voting: Majority vote
            predictions = torch.argmax(outputs, dim=2)

            # Count votes for each class
            batch_size = x.size(0)
            ensemble_preds = torch.zeros(batch_size, self.num_classes, device=x.device)

            for i in range(self.num_models):
                for b in range(batch_size):
                    pred_class = predictions[i, b]
                    ensemble_preds[b, pred_class] += self.weights[i]

            return ensemble_preds

    def predict_with_details(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed predictions from each model

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Dictionary with:
                - ensemble: Ensemble predictions
                - individual: Individual model predictions
                - agreement: Agreement score across models
        """
        # Get predictions from all models
        outputs = {}
        with torch.no_grad():
            for name, model in zip(self.model_names, self.models):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                outputs[name] = {
                    'logits': logits,
                    'probabilities': probs,
                    'predictions': torch.argmax(probs, dim=1)
                }

        # Ensemble prediction
        ensemble_logits = self.forward(x)
        ensemble_probs = F.softmax(ensemble_logits, dim=1)
        ensemble_preds = torch.argmax(ensemble_probs, dim=1)

        # Calculate agreement (how many models agree with ensemble)
        agreements = []
        for name in self.model_names:
            agreement = (outputs[name]['predictions'] == ensemble_preds).float()
            agreements.append(agreement)

        agreement_score = torch.stack(agreements).mean(dim=0)

        return {
            'ensemble_logits': ensemble_logits,
            'ensemble_probabilities': ensemble_probs,
            'ensemble_predictions': ensemble_preds,
            'individual_outputs': outputs,
            'agreement_score': agreement_score
        }

    def load_individual_models(
        self,
        resnet_path: Optional[str] = None,
        densenet_path: Optional[str] = None,
        efficientnet_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Load pre-trained individual models

        Args:
            resnet_path: Path to ResNet152 checkpoint
            densenet_path: Path to DenseNet201 checkpoint
            efficientnet_path: Path to EfficientNet-B3 checkpoint
            device: Device to load models on
        """
        if resnet_path:
            checkpoint = torch.load(resnet_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.resnet152.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.resnet152.load_state_dict(checkpoint)
            print(f"✓ Loaded ResNet152 from {resnet_path}")

        if densenet_path:
            checkpoint = torch.load(densenet_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.densenet201.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.densenet201.load_state_dict(checkpoint)
            print(f"✓ Loaded DenseNet201 from {densenet_path}")

        if efficientnet_path:
            checkpoint = torch.load(efficientnet_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.efficientnet_b3.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.efficientnet_b3.load_state_dict(checkpoint)
            print(f"✓ Loaded EfficientNet-B3 from {efficientnet_path}")

    def get_model_info(self) -> Dict:
        """Get information about ensemble and individual models"""
        info = {
            'name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': self.get_num_parameters(),
            'trainable_parameters': self.get_trainable_parameters(),
            'ensemble': {
                'num_models': self.num_models,
                'voting': self.voting,
                'weights': self.weights.tolist()
            },
            'individual_models': {}
        }

        for name, model in zip(self.model_names, self.models):
            info['individual_models'][name] = model.get_model_info()

        return info
