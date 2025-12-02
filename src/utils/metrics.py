"""
Advanced metrics and visualization utilities for model evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd


def compute_confusion_matrix(model, dataloader, class_names, device='cpu'):
    """
    Compute confusion matrix and per-class metrics.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for test data
        class_names: List of class names
        device: Device to run on

    Returns:
        dict: Contains confusion matrix and per-class metrics
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # Create per-class metrics DataFrame
    per_class_metrics = []
    for i, class_name in enumerate(class_names):
        per_class_metrics.append({
            'class': class_name,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        })

    # Overall metrics
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return {
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'per_class_metrics': per_class_metrics,
        'overall_metrics': {
            'precision': float(precision_avg),
            'recall': float(recall_avg),
            'f1_score': float(f1_avg),
            'accuracy': float(np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels))
        }
    }


def get_top_k_predictions(model, image_tensor, class_names, k=3, device='cpu'):
    """
    Get top-k predictions with confidence scores.

    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        class_names: List of class names
        k: Number of top predictions to return
        device: Device to run on

    Returns:
        list: Top-k predictions with class names and confidences
    """
    model.eval()
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, min(k, len(class_names)))

        predictions = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            predictions.append({
                'class': class_names[idx.item()],
                'class_idx': idx.item(),
                'confidence': float(prob.item())
            })

    return predictions


def calculate_model_metrics(model, dataloader, device='cpu'):
    """
    Calculate comprehensive metrics for a model.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run on

    Returns:
        dict: Metrics including accuracy, loss, precision, recall, f1
    """
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return {
        'accuracy': float(accuracy),
        'loss': float(avg_loss),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'num_samples': total
    }


def format_metrics_table(metrics_dict):
    """
    Format metrics dictionary as a pretty table.

    Args:
        metrics_dict: Dictionary with metrics

    Returns:
        pandas.DataFrame: Formatted metrics table
    """
    df = pd.DataFrame([metrics_dict])
    return df.T.rename(columns={0: 'Value'})
