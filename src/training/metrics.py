"""
Metrics calculation for model evaluation
Critical for medical AI: precision, recall, F1, AUC-ROC
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple


class MetricsCalculator:
    """
    Calculate comprehensive metrics for multi-class classification
    """

    def __init__(self, num_classes: int = 4, class_names: List[str] = None):
        """
        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics

        Args:
            y_true: True labels [N]
            y_pred: Predicted labels [N]
            y_proba: Predicted probabilities [N, num_classes] (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 (macro and weighted)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precisions[i]
            metrics[f'recall_{class_name}'] = recalls[i]
            metrics[f'f1_{class_name}'] = f1s[i]

        # AUC-ROC (if probabilities provided)
        if y_proba is not None:
            try:
                if self.num_classes == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['auc_roc_macro'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average='macro'
                    )
                    metrics['auc_roc_weighted'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                print(f"Warning: Could not calculate AUC-ROC: {e}")

        return metrics

    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Get confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        return confusion_matrix(y_true, y_pred)

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Get detailed classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report string
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0
        )


def calculate_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_proba: torch.Tensor = None,
    num_classes: int = 4,
    class_names: List[str] = None
) -> Dict[str, float]:
    """
    Convenience function to calculate metrics from tensors

    Args:
        y_true: True labels tensor
        y_pred: Predicted labels tensor
        y_proba: Predicted probabilities tensor (optional)
        num_classes: Number of classes
        class_names: Names of classes

    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_proba_np = y_proba.cpu().numpy() if y_proba is not None else None

    calculator = MetricsCalculator(num_classes=num_classes, class_names=class_names)
    return calculator.calculate(y_true_np, y_pred_np, y_proba_np)
