"""
Production-grade trainer for medical image classification
Includes: early stopping, checkpointing, mixed precision, logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import time
from typing import Dict, Optional, Callable
from tqdm import tqdm
import json

from .metrics import MetricsCalculator


class Trainer:
    """
    Production-grade trainer with all bells and whistles
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        config: Dict,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        save_dir: str = "models"
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            config: Training configuration dict
            scheduler: Learning rate scheduler (optional)
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision training
        self.use_amp = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.grad_clip = config.get('gradient_clipping', None)

        # Early stopping
        self.early_stopping = config.get('early_stopping', {})
        self.patience = self.early_stopping.get('patience', 5)
        self.min_delta = self.early_stopping.get('min_delta', 0.001)

        # Training timeout (in minutes)
        self.timeout_minutes = config.get('timeout_minutes', None)
        self.training_start_time = None

        # Metrics calculator
        num_classes = config.get('num_classes', 4)
        class_names = config.get('class_names', [f"Class_{i}" for i in range(num_classes)])
        self.metrics_calculator = MetricsCalculator(num_classes, class_names)

        # Training state
        self.current_epoch = 0
        self.best_metric = -float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        metrics = self.metrics_calculator.calculate(
            y_true=all_labels,
            y_pred=all_preds
        )

        metrics['loss'] = avg_loss
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        # Calculate epoch metrics
        avg_loss = running_loss / len(self.val_loader)
        metrics = self.metrics_calculator.calculate(
            y_true=all_labels,
            y_pred=all_preds,
            y_proba=all_probs
        )

        metrics['loss'] = avg_loss
        return metrics

    def train(self, epochs: int):
        """
        Full training loop

        Args:
            epochs: Number of epochs to train
        """
        print(f"\n{'='*60}")
        print(f"Starting Training: {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        if self.timeout_minutes:
            print(f"Timeout: {self.timeout_minutes} minutes")
        print(f"{'='*60}\n")

        start_time = time.time()
        self.training_start_time = start_time

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Check timeout
            if self.timeout_minutes:
                elapsed_minutes = (time.time() - self.training_start_time) / 60
                if elapsed_minutes >= self.timeout_minutes:
                    print(f"\n⏱ Training timeout reached ({self.timeout_minutes} minutes)")
                    print(f"Stopping at epoch {epoch + 1}/{epochs}")
                    break

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_weighted'])
            self.history['learning_rate'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_weighted']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Check if best model
            monitor_metric = val_metrics.get('f1_weighted', val_metrics['accuracy'])
            if monitor_metric > self.best_metric + self.min_delta:
                self.best_metric = monitor_metric
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
                print(f"  ✓ New best model! F1: {monitor_metric:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_frequency', 5) == 0:
                self.save_checkpoint(is_best=False)

            # Early stopping check
            if self.early_stopping.get('enabled', True):
                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total Time: {total_time/60:.2f} minutes")
        print(f"Best F1 Score: {self.best_metric:.4f}")
        print(f"{'='*60}\n")

        # Save training history
        self.save_history()

    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'history': self.history
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            save_path = self.save_dir / "best_model.pth"
        else:
            save_path = self.save_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"

        torch.save(checkpoint, save_path)

    def save_history(self):
        """Save training history to JSON"""
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint.get('history', self.history)

        print(f"✓ Loaded checkpoint from epoch {self.current_epoch + 1}")
