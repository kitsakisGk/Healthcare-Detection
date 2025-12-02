"""
Training script for pneumonia detection models
Usage: python scripts/train.py --model ensemble --config config/config.yaml
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data import MultiClassDataset, get_transforms
from src.models import ResNet152Model, DenseNet201Model, EfficientNetB3Model, EnsembleModel
from src.training import Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train pneumonia detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train EfficientNet-B3
  python scripts/train.py --model efficientnet --epochs 30

  # Train ensemble model
  python scripts/train.py --model ensemble --epochs 20 --batch-size 16

  # Resume training from checkpoint
  python scripts/train.py --model resnet152 --resume models/checkpoint_epoch_10.pth
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['resnet152', 'densenet201', 'efficientnet', 'ensemble'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'auto'],
        help='Device to train on'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader workers'
    )

    return parser.parse_args()


def create_model(model_name: str, num_classes: int, config: dict):
    """Create model based on name"""
    print(f"\n{'='*60}")
    print(f"Creating model: {model_name.upper()}")
    print(f"{'='*60}")

    if model_name == 'resnet152':
        model = ResNet152Model(
            num_classes=num_classes,
            pretrained=True,
            freeze_layers=config.get('models', {}).get('resnet152', {}).get('freeze_layers', 3),
            dropout=config.get('models', {}).get('resnet152', {}).get('dropout', 0.5)
        )
    elif model_name == 'densenet201':
        model = DenseNet201Model(
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=config.get('models', {}).get('densenet201', {}).get('freeze_backbone', True),
            dropout=config.get('models', {}).get('densenet201', {}).get('dropout', 0.5)
        )
    elif model_name == 'efficientnet':
        model = EfficientNetB3Model(
            num_classes=num_classes,
            pretrained=True,
            freeze_batch_norm=config.get('models', {}).get('efficientnet_b3', {}).get('freeze_batch_norm', True),
            dropout=config.get('models', {}).get('efficientnet_b3', {}).get('dropout', 0.3)
        )
    elif model_name == 'ensemble':
        model = EnsembleModel(
            num_classes=num_classes,
            voting=config.get('models', {}).get('ensemble', {}).get('voting', 'soft'),
            pretrained=True
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\n{model}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")

    return model


def main():
    """Main training function"""
    args = parse_args()

    # Load config
    print("Loading configuration...")
    config = get_config(args.config)

    # Override config with CLI args
    if args.epochs is not None:
        config.config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config.config['data']['batch_size'] = args.batch_size
    if args.lr is not None:
        config.config['training']['optimizer']['lr'] = args.lr

    # Setup device
    if args.device is None or args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Number of Classes: {config.num_classes}")
    print(f"Class Names: {config.class_names}")
    print(f"{'='*60}\n")

    # Create datasets
    print("Loading datasets...")
    train_transform = get_transforms(
        img_size=config.img_size,
        mode='train',
        augmentation_config=config.get('data.augmentation.train')
    )
    val_transform = get_transforms(
        img_size=config.img_size,
        mode='val'
    )

    train_dataset = MultiClassDataset(
        root_dir=Path(args.data_dir) / 'train',
        class_names=config.class_names,
        transform=train_transform
    )

    val_dataset = MultiClassDataset(
        root_dir=Path(args.data_dir) / 'val',
        class_names=config.class_names,
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    model = create_model(args.model, config.num_classes, config.config)

    # Get class weights for imbalanced data
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"\nClass weights for imbalanced data: {class_weights.tolist()}")

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer_config = config.get('training.optimizer', {})
    optimizer = optim.Adam(
        model.parameters(),
        lr=optimizer_config.get('lr', 0.0001),
        weight_decay=optimizer_config.get('weight_decay', 0.0001)
    )

    # Learning rate scheduler
    scheduler_config = config.get('training.scheduler', {})
    if scheduler_config.get('type') == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 3),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None

    # Create trainer
    training_config = config.config.get('training', {})
    training_config['num_classes'] = config.num_classes
    training_config['class_names'] = config.class_names

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=training_config,
        scheduler=scheduler,
        save_dir=args.save_dir
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train!
    trainer.train(epochs=config.epochs)

    print("\n✓ Training complete!")
    print(f"Best model saved in: {args.save_dir}/best_model.pth")
    print(f"Training history saved in: {args.save_dir}/training_history.json")


if __name__ == "__main__":
    main()
