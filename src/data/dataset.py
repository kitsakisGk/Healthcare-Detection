"""
Dataset classes for multi-class chest X-ray classification
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class MultiClassDataset(Dataset):
    """
    Multi-class chest X-ray dataset for production use
    Supports: NORMAL, BACTERIAL, VIRAL, COVID19
    """

    def __init__(
        self,
        root_dir: str,
        class_names: List[str] = None,
        transform: Optional[Callable] = None,
        return_path: bool = False
    ):
        """
        Args:
            root_dir: Root directory containing class folders
            class_names: List of class names (folder names)
            transform: Torchvision transforms
            return_path: If True, also return image path
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_path = return_path

        if class_names is None:
            class_names = ["NORMAL", "BACTERIAL", "VIRAL", "COVID19"]

        self.class_names = class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        # Load all image paths and labels
        self.samples = []
        self._load_samples()

        print(f"Loaded {len(self.samples)} images from {root_dir}")
        self._print_class_distribution()

    def _load_samples(self):
        """Load all image paths and their labels"""
        for class_name in self.class_names:
            class_path = self.root_dir / class_name

            if not class_path.exists():
                print(f"Warning: Class folder not found: {class_path}")
                continue

            for img_path in class_path.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    label = self.class_to_idx[class_name]
                    self.samples.append((str(img_path), label))

    def _print_class_distribution(self):
        """Print class distribution"""
        class_counts = {cls: 0 for cls in self.class_names}

        for _, label in self.samples:
            class_name = self.class_names[label]
            class_counts[class_name] += 1

        print("Class distribution:")
        for cls, count in class_counts.items():
            percentage = (count / len(self.samples)) * 100
            print(f"  {cls}: {count} ({percentage:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get item by index

        Returns:
            If return_path=False: (image_tensor, label)
            If return_path=True: (image_tensor, label, path)
        """
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('L')  # Grayscale
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            image = Image.new('L', (224, 224), color=0)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, label, img_path
        else:
            return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset

        Returns:
            Tensor of class weights
        """
        class_counts = [0] * len(self.class_names)

        for _, label in self.samples:
            class_counts[label] += 1

        total = sum(class_counts)
        weights = [total / (len(class_counts) * count) if count > 0 else 0
                   for count in class_counts]

        return torch.FloatTensor(weights)


class ChestXRayDataset(Dataset):
    """
    Legacy binary classification dataset (for backward compatibility)
    Classes: NORMAL, PNEUMONIA
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        return_path: bool = False
    ):
        """
        Args:
            root_dir: Root directory containing NORMAL and PNEUMONIA folders
            transform: Torchvision transforms
            return_path: If True, also return image path
        """
        self.dataset = MultiClassDataset(
            root_dir=root_dir,
            class_names=["NORMAL", "PNEUMONIA"],
            transform=transform,
            return_path=return_path
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_class_weights(self):
        return self.dataset.get_class_weights()
