"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.
Shows which regions of an image the model focuses on for its predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for CNN models.

    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate_cam(input_tensor, target_class)
        overlay = gradcam.overlay_heatmap(original_image, heatmap)
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for an input image.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)

        Returns:
            numpy array: Heatmap (H, W) with values in [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image.

        Args:
            image: PIL Image or numpy array (H, W, 3)
            heatmap: Heatmap array (H, W) with values in [0, 1]
            alpha: Transparency of overlay (0-1)
            colormap: OpenCV colormap

        Returns:
            PIL Image: Overlay image
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)

        return Image.fromarray(overlay)


def get_target_layer(model, model_type='efficientnet'):
    """
    Get the last convolutional layer for a given model type.

    Args:
        model: PyTorch model
        model_type: Type of model ('efficientnet', 'resnet', 'densenet')

    Returns:
        nn.Module: Target layer for Grad-CAM
    """
    if model_type.lower() == 'efficientnet':
        # EfficientNet: last conv layer before avg pool
        return model.model.features[-1]
    elif model_type.lower() == 'resnet':
        # ResNet: layer4
        return model.model.layer4[-1]
    elif model_type.lower() == 'densenet':
        # DenseNet: last dense block
        return model.model.features[-1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_gradcam_overlay(model, image, transform, class_idx=None, model_type='efficientnet'):
    """
    Convenience function to generate Grad-CAM overlay.

    Args:
        model: Trained PyTorch model
        image: PIL Image
        transform: Preprocessing transform
        class_idx: Target class (if None, uses prediction)
        model_type: Type of model

    Returns:
        tuple: (overlay_image, predicted_class, confidence, heatmap)
    """
    # Get target layer
    target_layer = get_target_layer(model, model_type)

    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)

    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[predicted_class].item()

    # Generate CAM
    heatmap = gradcam.generate_cam(input_tensor, class_idx or predicted_class)

    # Create overlay
    overlay = gradcam.overlay_heatmap(image, heatmap)

    return overlay, predicted_class, confidence, heatmap
