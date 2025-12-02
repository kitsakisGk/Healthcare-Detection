"""
Explainability module: Grad-CAM, SHAP, LIME
CRITICAL for medical AI - shows what the model is looking at
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMExplainer:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) explainer
    Shows which regions of the image the model focuses on
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[nn.Module]] = None,
        device: torch.device = None
    ):
        """
        Args:
            model: PyTorch model
            target_layers: Layers to compute Grad-CAM on (auto-detect if None)
            device: Device to run on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Auto-detect target layers if not provided
        if target_layers is None:
            target_layers = self._auto_detect_target_layers()

        self.target_layers = target_layers
        self.cam = GradCAM(model=model, target_layers=target_layers)

    def _auto_detect_target_layers(self) -> List[nn.Module]:
        """
        Auto-detect appropriate layers for Grad-CAM based on model architecture
        """
        model_name = self.model.__class__.__name__.lower()

        # Check if it's an ensemble
        if hasattr(self.model, 'resnet152'):
            # Ensemble model - use ResNet's last layer
            return [self.model.resnet152.backbone.layer4[-1]]

        # ResNet
        if 'resnet' in model_name or hasattr(self.model, 'layer4'):
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layer4'):
                return [self.model.backbone.layer4[-1]]
            elif hasattr(self.model, 'layer4'):
                return [self.model.layer4[-1]]

        # DenseNet
        if 'densenet' in model_name or hasattr(self.model, 'features'):
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'features'):
                return [self.model.backbone.features[-1]]
            elif hasattr(self.model, 'features'):
                return [self.model.features[-1]]

        # EfficientNet
        if 'efficientnet' in model_name:
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'features'):
                return [self.model.backbone.features[-1]]

        # Fallback: try to find the last convolutional layer
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Conv2d):
                return [module]

        raise ValueError("Could not auto-detect target layers. Please specify manually.")

    def generate_heatmap(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap

        Args:
            image: Input image tensor [1, C, H, W] or [C, H, W]
            target_class: Target class index (if None, uses predicted class)
            normalize: Normalize heatmap to [0, 1]

        Returns:
            Heatmap as numpy array [H, W]
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Get target class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()

        # Generate Grad-CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=image, targets=targets)

        # Get first image in batch
        heatmap = grayscale_cam[0]

        if normalize:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    def overlay_heatmap(
        self,
        image: Union[np.ndarray, Image.Image],
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image

        Args:
            image: Original image (numpy array or PIL Image)
            heatmap: Grad-CAM heatmap
            alpha: Transparency of overlay
            colormap: OpenCV colormap

        Returns:
            Overlayed image as numpy array [H, W, 3]
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))

        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize heatmap to match image size
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # Generate visualization
        visualization = show_cam_on_image(image, heatmap, use_rgb=True)

        return visualization

    def explain_prediction(
        self,
        image_tensor: torch.Tensor,
        original_image: Union[np.ndarray, Image.Image],
        target_class: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete explanation with heatmap and overlay

        Args:
            image_tensor: Preprocessed image tensor
            original_image: Original image for overlay
            target_class: Target class (if None, uses predicted)

        Returns:
            Dictionary with 'heatmap' and 'overlay'
        """
        heatmap = self.generate_heatmap(image_tensor, target_class)
        overlay = self.overlay_heatmap(original_image, heatmap)

        return {
            'heatmap': heatmap,
            'overlay': overlay
        }


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) explainer
    Provides feature importance explanations
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        device: torch.device = None
    ):
        """
        Args:
            model: PyTorch model
            background_data: Background dataset for SHAP (optional)
            device: Device to run on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        try:
            import shap
            self.shap = shap
            self.available = True
        except ImportError:
            print("Warning: SHAP not installed. Run: pip install shap")
            self.available = False
            return

        # Prepare background data
        if background_data is not None:
            self.background = background_data.to(self.device)
        else:
            # Use a small random background
            self.background = torch.randn(10, 3, 224, 224).to(self.device)

    def explain(
        self,
        image: torch.Tensor,
        num_samples: int = 100
    ) -> Optional[np.ndarray]:
        """
        Generate SHAP explanation

        Args:
            image: Input image tensor
            num_samples: Number of samples for SHAP

        Returns:
            SHAP values as numpy array
        """
        if not self.available:
            return None

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Create explainer
        def model_predict(x):
            x = torch.from_numpy(x).float().to(self.device)
            with torch.no_grad():
                outputs = self.model(x)
                return F.softmax(outputs, dim=1).cpu().numpy()

        explainer = self.shap.KernelExplainer(
            model_predict,
            self.background.cpu().numpy()
        )

        # Get SHAP values
        shap_values = explainer.shap_values(
            image.cpu().numpy(),
            nsamples=num_samples
        )

        return shap_values


class ExplainabilityManager:
    """
    Unified manager for all explainability methods
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        enable_gradcam: bool = True,
        enable_shap: bool = False
    ):
        """
        Args:
            model: PyTorch model
            device: Device to run on
            enable_gradcam: Enable Grad-CAM
            enable_shap: Enable SHAP
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize explainers
        self.gradcam = None
        self.shap_explainer = None

        if enable_gradcam:
            self.gradcam = GradCAMExplainer(model, device=device)

        if enable_shap:
            self.shap_explainer = SHAPExplainer(model, device=device)

    def explain_prediction(
        self,
        image_tensor: torch.Tensor,
        original_image: Union[np.ndarray, Image.Image],
        target_class: Optional[int] = None,
        methods: List[str] = None
    ) -> Dict[str, any]:
        """
        Generate explanations using multiple methods

        Args:
            image_tensor: Preprocessed image tensor
            original_image: Original image
            target_class: Target class
            methods: List of methods to use ['gradcam', 'shap']

        Returns:
            Dictionary with explanations from each method
        """
        if methods is None:
            methods = ['gradcam']

        results = {}

        if 'gradcam' in methods and self.gradcam:
            results['gradcam'] = self.gradcam.explain_prediction(
                image_tensor,
                original_image,
                target_class
            )

        if 'shap' in methods and self.shap_explainer:
            results['shap'] = self.shap_explainer.explain(image_tensor)

        return results

    def generate_report(
        self,
        image_tensor: torch.Tensor,
        original_image: Union[np.ndarray, Image.Image],
        prediction: Dict,
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Generate comprehensive explainability report

        Args:
            image_tensor: Preprocessed image
            original_image: Original image
            prediction: Prediction dictionary
            save_path: Optional path to save visualizations

        Returns:
            Complete report dictionary
        """
        report = {
            'prediction': prediction,
            'explanations': {}
        }

        # Grad-CAM
        if self.gradcam:
            gradcam_results = self.gradcam.explain_prediction(
                image_tensor,
                original_image,
                target_class=prediction.get('predicted_class_idx')
            )
            report['explanations']['gradcam'] = gradcam_results

            # Save if path provided
            if save_path:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)

                overlay_path = save_path / "gradcam_overlay.png"
                Image.fromarray(gradcam_results['overlay']).save(overlay_path)
                report['saved_files'] = {'gradcam_overlay': str(overlay_path)}

        return report
