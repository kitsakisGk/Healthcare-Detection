"""
Quality Control checks for medical images
Ensures images meet minimum standards before inference
"""

import numpy as np
from PIL import Image
import cv2
from typing import Dict, Tuple, Union, List
from pathlib import Path
import warnings


class QualityController:
    """
    Perform quality control checks on medical images
    """

    def __init__(
        self,
        min_size: Tuple[int, int] = (128, 128),
        max_size: Tuple[int, int] = (4096, 4096),
        min_contrast: float = 0.1,
        max_brightness: float = 0.95,
        min_brightness: float = 0.05
    ):
        """
        Args:
            min_size: Minimum image dimensions (width, height)
            max_size: Maximum image dimensions
            min_contrast: Minimum contrast ratio
            max_brightness: Maximum average brightness
            min_brightness: Minimum average brightness
        """
        self.min_size = min_size
        self.max_size = max_size
        self.min_contrast = min_contrast
        self.max_brightness = max_brightness
        self.min_brightness = min_brightness

    def check_image_size(self, image: Union[np.ndarray, Image.Image]) -> Tuple[bool, str]:
        """
        Check if image size is within acceptable range

        Args:
            image: Input image

        Returns:
            Tuple of (is_valid, message)
        """
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]

        if width < self.min_size[0] or height < self.min_size[1]:
            return False, f"Image too small: {width}x{height} (min: {self.min_size[0]}x{self.min_size[1]})"

        if width > self.max_size[0] or height > self.max_size[1]:
            return False, f"Image too large: {width}x{height} (max: {self.max_size[0]}x{self.max_size[1]})"

        return True, "Image size OK"

    def check_contrast(self, image: Union[np.ndarray, Image.Image]) -> Tuple[bool, str, float]:
        """
        Check image contrast

        Args:
            image: Input image

        Returns:
            Tuple of (is_valid, message, contrast_value)
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate contrast (standard deviation)
        contrast = image.std() / 255.0

        if contrast < self.min_contrast:
            return False, f"Low contrast: {contrast:.3f} (min: {self.min_contrast})", contrast

        return True, "Contrast OK", contrast

    def check_brightness(self, image: Union[np.ndarray, Image.Image]) -> Tuple[bool, str, float]:
        """
        Check image brightness

        Args:
            image: Input image

        Returns:
            Tuple of (is_valid, message, brightness_value)
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate average brightness
        brightness = image.mean() / 255.0

        if brightness < self.min_brightness:
            return False, f"Too dark: {brightness:.3f} (min: {self.min_brightness})", brightness

        if brightness > self.max_brightness:
            return False, f"Too bright: {brightness:.3f} (max: {self.max_brightness})", brightness

        return True, "Brightness OK", brightness

    def detect_artifacts(self, image: Union[np.ndarray, Image.Image]) -> Tuple[bool, str, Dict]:
        """
        Detect common image artifacts

        Args:
            image: Input image

        Returns:
            Tuple of (is_clean, message, artifact_metrics)
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        artifacts = {}

        # Check for black borders
        artifacts['black_border_top'] = (image[0, :].mean() / 255.0) < 0.05
        artifacts['black_border_bottom'] = (image[-1, :].mean() / 255.0) < 0.05
        artifacts['black_border_left'] = (image[:, 0].mean() / 255.0) < 0.05
        artifacts['black_border_right'] = (image[:, -1].mean() / 255.0) < 0.05

        # Check for saturation (too many pure black or white pixels)
        black_pixels = (image == 0).sum()
        white_pixels = (image == 255).sum()
        total_pixels = image.size

        artifacts['black_saturation'] = black_pixels / total_pixels
        artifacts['white_saturation'] = white_pixels / total_pixels

        # Check for extreme saturation
        if artifacts['black_saturation'] > 0.3:
            return False, f"High black saturation: {artifacts['black_saturation']:.2%}", artifacts

        if artifacts['white_saturation'] > 0.3:
            return False, f"High white saturation: {artifacts['white_saturation']:.2%}", artifacts

        return True, "No significant artifacts detected", artifacts

    def check_image_quality(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> Dict[str, any]:
        """
        Perform comprehensive quality check

        Args:
            image: Input image (array, PIL Image, or path)

        Returns:
            Dictionary with all quality metrics and pass/fail status
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        results = {
            'overall_pass': True,
            'checks': {}
        }

        # Size check
        size_pass, size_msg = self.check_image_size(image)
        results['checks']['size'] = {
            'pass': size_pass,
            'message': size_msg
        }
        if not size_pass:
            results['overall_pass'] = False

        # Contrast check
        contrast_pass, contrast_msg, contrast_value = self.check_contrast(image)
        results['checks']['contrast'] = {
            'pass': contrast_pass,
            'message': contrast_msg,
            'value': contrast_value
        }
        if not contrast_pass:
            results['overall_pass'] = False

        # Brightness check
        brightness_pass, brightness_msg, brightness_value = self.check_brightness(image)
        results['checks']['brightness'] = {
            'pass': brightness_pass,
            'message': brightness_msg,
            'value': brightness_value
        }
        if not brightness_pass:
            results['overall_pass'] = False

        # Artifact check
        artifact_pass, artifact_msg, artifact_metrics = self.detect_artifacts(image)
        results['checks']['artifacts'] = {
            'pass': artifact_pass,
            'message': artifact_msg,
            'metrics': artifact_metrics
        }
        if not artifact_pass:
            results['overall_pass'] = False

        return results

    def get_quality_report(
        self,
        image: Union[np.ndarray, Image.Image, str, Path]
    ) -> str:
        """
        Get human-readable quality report

        Args:
            image: Input image

        Returns:
            Formatted quality report string
        """
        results = self.check_image_quality(image)

        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("IMAGE QUALITY CONTROL REPORT")
        report_lines.append("=" * 50)

        status = "✓ PASS" if results['overall_pass'] else "✗ FAIL"
        report_lines.append(f"\nOverall Status: {status}\n")

        for check_name, check_result in results['checks'].items():
            status_icon = "✓" if check_result['pass'] else "✗"
            report_lines.append(f"{status_icon} {check_name.upper()}: {check_result['message']}")

            # Add value if available
            if 'value' in check_result:
                report_lines.append(f"   Value: {check_result['value']:.3f}")

        report_lines.append("=" * 50)

        return "\n".join(report_lines)

    def batch_quality_check(
        self,
        image_paths: List[Union[str, Path]]
    ) -> Dict[str, Dict]:
        """
        Perform quality checks on multiple images

        Args:
            image_paths: List of image paths

        Returns:
            Dictionary mapping paths to quality results
        """
        results = {}

        for img_path in image_paths:
            try:
                qc_result = self.check_image_quality(img_path)
                results[str(img_path)] = qc_result
            except Exception as e:
                results[str(img_path)] = {
                    'overall_pass': False,
                    'error': str(e)
                }

        return results
