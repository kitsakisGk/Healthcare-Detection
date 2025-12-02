"""
Advanced augmentation strategies for medical images
"""

import torchvision.transforms as T
from typing import Optional, Dict, Any


class MedicalImageAugmentation:
    """
    Advanced augmentation pipeline for medical images
    Includes domain-specific transformations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config or self._default_config()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default augmentation configuration"""
        return {
            'geometric': {
                'horizontal_flip': 0.5,
                'rotation': 15,
                'affine_translate': [0.1, 0.1],
                'affine_scale': [0.9, 1.1]
            },
            'intensity': {
                'brightness': 0.2,
                'contrast': 0.2,
                'gamma': [0.8, 1.2]
            },
            'noise': {
                'enabled': False,
                'gaussian_std': 0.01
            }
        }

    def get_pipeline(self, mode: str = 'train'):
        """
        Get augmentation pipeline

        Args:
            mode: 'train' or 'val'/'test'

        Returns:
            List of transforms
        """
        if mode != 'train':
            return []

        transforms = []
        geo_config = self.config.get('geometric', {})
        intensity_config = self.config.get('intensity', {})

        # Geometric transforms
        if geo_config.get('horizontal_flip', 0) > 0:
            transforms.append(
                T.RandomHorizontalFlip(p=geo_config['horizontal_flip'])
            )

        if geo_config.get('rotation', 0) > 0:
            transforms.append(
                T.RandomRotation(degrees=geo_config['rotation'])
            )

        if 'affine_translate' in geo_config or 'affine_scale' in geo_config:
            transforms.append(
                T.RandomAffine(
                    degrees=geo_config.get('rotation', 0),
                    translate=tuple(geo_config.get('affine_translate', [0.1, 0.1])),
                    scale=tuple(geo_config.get('affine_scale', [0.9, 1.1]))
                )
            )

        # Intensity transforms
        if intensity_config.get('brightness') or intensity_config.get('contrast'):
            transforms.append(
                T.ColorJitter(
                    brightness=intensity_config.get('brightness', 0),
                    contrast=intensity_config.get('contrast', 0)
                )
            )

        return transforms


def get_augmentation_pipeline(config: Optional[Dict[str, Any]] = None):
    """
    Factory function to get augmentation pipeline

    Args:
        config: Augmentation configuration

    Returns:
        MedicalImageAugmentation instance
    """
    return MedicalImageAugmentation(config)
