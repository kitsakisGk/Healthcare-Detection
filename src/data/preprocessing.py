"""
Image preprocessing and transformation pipelines
"""

from torchvision import transforms
from typing import Optional


def get_transforms(
    img_size: int = 224,
    mode: str = 'train',
    augmentation_config: Optional[dict] = None
):
    """
    Get transformation pipeline for chest X-rays

    Args:
        img_size: Target image size
        mode: 'train', 'val', or 'test'
        augmentation_config: Configuration for augmentation (from config.yaml)

    Returns:
        torchvision.transforms.Compose object
    """
    if mode == 'train':
        # Training with augmentation
        if augmentation_config is None:
            augmentation_config = {
                'random_horizontal_flip': 0.5,
                'random_rotation': 15,
                'color_jitter': {
                    'brightness': 0.2,
                    'contrast': 0.2
                }
            }

        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=augmentation_config.get('random_horizontal_flip', 0.5)),
            transforms.RandomRotation(degrees=augmentation_config.get('random_rotation', 15)),
        ]

        # Add color jitter if specified
        jitter_config = augmentation_config.get('color_jitter', {})
        if jitter_config:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=jitter_config.get('brightness', 0.2),
                    contrast=jitter_config.get('contrast', 0.2)
                )
            )

        # Add random affine if specified
        affine_config = augmentation_config.get('random_affine', {})
        if affine_config:
            transform_list.append(
                transforms.RandomAffine(
                    degrees=affine_config.get('degrees', 10),
                    translate=tuple(affine_config.get('translate', [0.1, 0.1]))
                )
            )

        # Convert to tensor and normalize
        transform_list.extend([
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for pretrained models
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transforms.Compose(transform_list)

    else:
        # Validation/Test without augmentation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_inference_transforms(img_size: int = 224):
    """
    Get transforms for inference on single images

    Args:
        img_size: Target image size

    Returns:
        torchvision.transforms.Compose object
    """
    return get_transforms(img_size=img_size, mode='test')


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization

    Args:
        tensor: Normalized image tensor
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized tensor
    """
    import torch

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    return tensor * std + mean
