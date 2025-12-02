"""
Production-grade predictor with uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from ..data.preprocessing import get_inference_transforms


class Predictor:
    """
    Single image predictor with uncertainty quantification
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: torch.device = None,
        confidence_threshold: float = 0.8,
        uncertainty_quantification: bool = True,
        mc_dropout_iterations: int = 20
    ):
        """
        Args:
            model: Trained PyTorch model
            class_names: List of class names
            device: Device to run inference on
            confidence_threshold: Threshold for high confidence predictions
            uncertainty_quantification: Enable Monte Carlo dropout
            mc_dropout_iterations: Number of MC dropout iterations
        """
        self.model = model
        self.class_names = class_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.uncertainty_quantification = uncertainty_quantification
        self.mc_dropout_iterations = mc_dropout_iterations

        self.model.to(self.device)
        self.model.eval()

        # Get image transforms
        self.transform = get_inference_transforms()

    def predict(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        return_probabilities: bool = True,
        return_uncertainty: bool = True
    ) -> Dict:
        """
        Predict on single image

        Args:
            image: Input image (path, PIL Image, or tensor)
            return_probabilities: Return class probabilities
            return_uncertainty: Return uncertainty estimates

        Returns:
            Dictionary with:
                - predicted_class: Class name
                - predicted_class_idx: Class index
                - confidence: Confidence score
                - probabilities: Class probabilities (if requested)
                - uncertainty: Uncertainty metrics (if requested)
                - risk_level: Risk stratification
        """
        # Prepare image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('L')

        if isinstance(image, Image.Image):
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be path, PIL Image, or tensor")

        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Get prediction
        with torch.no_grad():
            if self.uncertainty_quantification and return_uncertainty:
                result = self._predict_with_uncertainty(image)
            else:
                result = self._predict_deterministic(image)

        # Add risk stratification
        result['risk_level'] = self._stratify_risk(result['confidence'])

        # Add interpretable confidence
        result['confidence_interpretation'] = self._interpret_confidence(result['confidence'])

        if not return_probabilities and 'probabilities' in result:
            del result['probabilities']

        if not return_uncertainty and 'uncertainty' in result:
            del result['uncertainty']

        return result

    def _predict_deterministic(self, image: torch.Tensor) -> Dict:
        """Standard deterministic prediction"""
        outputs = self.model(image)
        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_idx = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()

        return {
            'predicted_class': self.class_names[predicted_idx],
            'predicted_class_idx': predicted_idx,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }

    def _predict_with_uncertainty(self, image: torch.Tensor) -> Dict:
        """
        Predict with Monte Carlo dropout for uncertainty estimation
        """
        # Enable dropout during inference
        self._enable_dropout()

        all_outputs = []
        for _ in range(self.mc_dropout_iterations):
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = F.softmax(outputs, dim=1)
                all_outputs.append(probabilities.cpu().numpy())

        # Disable dropout
        self.model.eval()

        # Stack all predictions
        all_outputs = np.array(all_outputs)  # [iterations, batch, num_classes]

        # Mean prediction
        mean_probs = all_outputs.mean(axis=0)[0]
        predicted_idx = mean_probs.argmax()
        confidence = mean_probs[predicted_idx]

        # Uncertainty metrics
        std_probs = all_outputs.std(axis=0)[0]
        epistemic_uncertainty = std_probs.mean()

        # Predictive entropy
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))

        return {
            'predicted_class': self.class_names[predicted_idx],
            'predicted_class_idx': int(predicted_idx),
            'confidence': float(confidence),
            'probabilities': mean_probs.tolist(),
            'uncertainty': {
                'epistemic': float(epistemic_uncertainty),
                'entropy': float(entropy),
                'std_per_class': std_probs.tolist()
            }
        }

    def _enable_dropout(self):
        """Enable dropout layers during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _stratify_risk(self, confidence: float) -> str:
        """
        Stratify risk level based on confidence

        Args:
            confidence: Confidence score

        Returns:
            Risk level: 'low', 'medium', or 'high'
        """
        if confidence < 0.3:
            return 'low'
        elif confidence < 0.7:
            return 'medium'
        else:
            return 'high'

    def _interpret_confidence(self, confidence: float) -> str:
        """
        Human-readable confidence interpretation

        Args:
            confidence: Confidence score

        Returns:
            Interpretation string
        """
        if confidence >= 0.9:
            return 'Very High Confidence'
        elif confidence >= 0.7:
            return 'High Confidence'
        elif confidence >= 0.5:
            return 'Moderate Confidence'
        elif confidence >= 0.3:
            return 'Low Confidence'
        else:
            return 'Very Low Confidence'


class BatchPredictor:
    """
    Batch prediction for multiple images
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: torch.device = None,
        batch_size: int = 32
    ):
        """
        Args:
            model: Trained PyTorch model
            class_names: List of class names
            device: Device to run inference on
            batch_size: Batch size for inference
        """
        self.predictor = Predictor(
            model=model,
            class_names=class_names,
            device=device,
            uncertainty_quantification=False  # Disable for batch processing
        )
        self.batch_size = batch_size

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict on batch of images

        Args:
            images: List of images (paths or PIL Images)
            show_progress: Show progress bar

        Returns:
            List of prediction dictionaries
        """
        results = []

        if show_progress:
            from tqdm import tqdm
            images = tqdm(images, desc="Predicting")

        for image in images:
            result = self.predictor.predict(
                image,
                return_probabilities=True,
                return_uncertainty=False
            )
            results.append(result)

        return results

    def predict_directory(
        self,
        directory: Union[str, Path],
        extensions: List[str] = None
    ) -> Dict[str, Dict]:
        """
        Predict on all images in directory

        Args:
            directory: Directory containing images
            extensions: List of file extensions to process

        Returns:
            Dictionary mapping filenames to predictions
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png']

        directory = Path(directory)
        image_paths = []

        for ext in extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))

        print(f"Found {len(image_paths)} images in {directory}")

        results = {}
        for img_path in image_paths:
            result = self.predictor.predict(img_path)
            results[img_path.name] = result

        return results
