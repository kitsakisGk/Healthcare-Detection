"""
Evaluation script for trained models
Usage: python scripts/evaluate.py --model-path models/best_model.pth --data-dir data/test
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data import MultiClassDataset, get_transforms
from src.models import ResNet152Model, DenseNet201Model, EfficientNetB3Model, EnsembleModel
from src.training.metrics import MetricsCalculator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate trained pneumonia detection model"
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['resnet152', 'densenet201', 'efficientnet', 'ensemble'],
        help='Type of model architecture'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/test',
        help='Path to test data directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='reports',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'cuda', 'auto'],
        help='Device to run on'
    )

    return parser.parse_args()


def load_model(model_type: str, model_path: str, num_classes: int, device: torch.device):
    """Load trained model"""
    print(f"Loading {model_type} model from {model_path}...")

    # Create model
    if model_type == 'resnet152':
        model = ResNet152Model(num_classes=num_classes)
    elif model_type == 'densenet201':
        model = DenseNet201Model(num_classes=num_classes)
    elif model_type == 'efficientnet':
        model = EfficientNetB3Model(num_classes=num_classes)
    elif model_type == 'ensemble':
        model = EnsembleModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model


def evaluate_model(model, data_loader, device, class_names):
    """Evaluate model on test set"""
    print("\nEvaluating model...")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics_calc = MetricsCalculator(
        num_classes=len(class_names),
        class_names=class_names
    )

    metrics = metrics_calc.calculate(all_labels, all_preds, all_probs)
    cm = metrics_calc.get_confusion_matrix(all_labels, all_preds)
    report = metrics_calc.get_classification_report(all_labels, all_preds)

    return metrics, cm, report, all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )

    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()


def plot_metrics_summary(metrics, class_names, save_path):
    """Plot metrics summary"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-class metrics
    classes = class_names
    precisions = [metrics.get(f'precision_{cls}', 0) for cls in classes]
    recalls = [metrics.get(f'recall_{cls}', 0) for cls in classes]
    f1s = [metrics.get(f'f1_{cls}', 0) for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    axes[0].bar(x - width, precisions, width, label='Precision', alpha=0.8)
    axes[0].bar(x, recalls, width, label='Recall', alpha=0.8)
    axes[0].bar(x + width, f1s, width, label='F1-Score', alpha=0.8)

    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3)

    # Overall metrics
    metric_names = ['Accuracy', 'Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1\n(Weighted)']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('precision_weighted', 0),
        metrics.get('recall_weighted', 0),
        metrics.get('f1_weighted', 0)
    ]

    bars = axes[1].bar(metric_names, metric_values, alpha=0.8, color='steelblue')
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Overall Metrics', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics summary saved: {save_path}")
    plt.close()


def main():
    """Main evaluation function"""
    args = parse_args()

    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = get_config(args.config)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_type}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Device: {device}")
    print(f"Test Data: {args.data_dir}")
    print(f"{'='*60}\n")

    # Load test dataset
    test_transform = get_transforms(img_size=config.img_size, mode='test')

    test_dataset = MultiClassDataset(
        root_dir=args.data_dir,
        class_names=config.class_names,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Load model
    model = load_model(args.model_type, args.model_path, config.num_classes, device)

    # Evaluate
    metrics, cm, report, all_labels, all_preds, all_probs = evaluate_model(
        model, test_loader, device, config.class_names
    )

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}\n")

    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
    print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")

    if 'auc_roc_weighted' in metrics:
        print(f"AUC-ROC (Weighted): {metrics['auc_roc_weighted']:.4f}")

    print("\nPer-Class Metrics:")
    for cls in config.class_names:
        print(f"\n{cls}:")
        print(f"  Precision: {metrics.get(f'precision_{cls}', 0):.4f}")
        print(f"  Recall: {metrics.get(f'recall_{cls}', 0):.4f}")
        print(f"  F1-Score: {metrics.get(f'f1_{cls}', 0):.4f}")

    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}\n")
    print(report)

    # Save results
    results_file = save_dir / f"{args.model_type}_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved: {results_file}")

    # Plot confusion matrix
    cm_path = save_dir / f"{args.model_type}_confusion_matrix.png"
    plot_confusion_matrix(cm, config.class_names, cm_path)

    # Plot metrics summary
    metrics_path = save_dir / f"{args.model_type}_metrics_summary.png"
    plot_metrics_summary(metrics, config.class_names, metrics_path)

    print(f"\n{'='*60}")
    print("✓ Evaluation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
