"""
Training utilities for dental 3D reconstruction pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import os
import json
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from datetime import datetime


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Validation loss
            model: Model to save weights from
            
        Returns:
            True if training should be stopped
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class ModelCheckpoint:
    """Save model checkpoints during training."""
    
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_loss = float('inf')
        
    def save_checkpoint(self, 
                       model: Dict[str, nn.Module], 
                       optimizer: Dict[str, optim.Optimizer],
                       scheduler: Optional[Dict[str, Any]],
                       epoch: int,
                       loss: float,
                       metrics: Dict[str, float]) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Dictionary of models (e.g., {'generator': gen, 'discriminator': disc})
            optimizer: Dictionary of optimizers
            scheduler: Dictionary of schedulers
            epoch: Current epoch
            loss: Current loss
            metrics: Current metrics
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics,
            'model_state_dict': {name: model[name].state_dict() for name in model},
            'optimizer_state_dict': {name: optimizer[name].state_dict() for name in optimizer}
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = {
                name: scheduler[name].state_dict() for name in scheduler
            }
        
        # Save current checkpoint
        current_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, current_path)
        
        # Save best checkpoint
        if not self.save_best_only or loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            
            # Save individual model files for easy loading
            for name, model_state in checkpoint['model_state_dict'].items():
                model_path = self.checkpoint_dir / f"best_{name}.pth"
                torch.save(model_state, model_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        return str(current_path)
    
    def load_checkpoint(self, 
                       model: Dict[str, nn.Module],
                       optimizer: Optional[Dict[str, optim.Optimizer]] = None,
                       scheduler: Optional[Dict[str, Any]] = None,
                       checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Dictionary of models to load state into
            optimizer: Dictionary of optimizers to load state into
            scheduler: Dictionary of schedulers to load state into
            checkpoint_path: Path to checkpoint file (uses best if None)
            
        Returns:
            Checkpoint information
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pth"
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model states
        for name in model:
            if name in checkpoint['model_state_dict']:
                model[name].load_state_dict(checkpoint['model_state_dict'][name])
        
        # Load optimizer states
        if optimizer and 'optimizer_state_dict' in checkpoint:
            for name in optimizer:
                if name in checkpoint['optimizer_state_dict']:
                    optimizer[name].load_state_dict(checkpoint['optimizer_state_dict'][name])
        
        # Load scheduler states
        if scheduler and 'scheduler_state_dict' in checkpoint:
            for name in scheduler:
                if name in checkpoint['scheduler_state_dict']:
                    scheduler[name].load_state_dict(checkpoint['scheduler_state_dict'][name])
        
        return checkpoint


class LearningRateScheduler:
    """Custom learning rate scheduler for GAN training."""
    
    def __init__(self, 
                 optimizer: Dict[str, optim.Optimizer],
                 scheduler_type: str = 'plateau',
                 **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.schedulers = {}
        
        for name, opt in optimizer.items():
            if scheduler_type == 'plateau':
                self.schedulers[name] = ReduceLROnPlateau(
                    opt, mode='min', factor=0.5, patience=5, **kwargs
                )
            elif scheduler_type == 'cosine':
                self.schedulers[name] = CosineAnnealingLR(
                    opt, T_max=kwargs.get('T_max', 100), **kwargs
                )
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metrics: Optional[float] = None):
        """Step all schedulers."""
        for scheduler in self.schedulers.values():
            if self.scheduler_type == 'plateau' and metrics is not None:
                scheduler.step(metrics)
            else:
                scheduler.step()
    
    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates."""
        return {
            name: scheduler.optimizer.param_groups[0]['lr']
            for name, scheduler in self.schedulers.items()
        }


class TrainingLogger:
    """Training progress logger."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}_training.log"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics storage
        self.train_metrics = []
        self.val_metrics = []
        
    def log_epoch(self, 
                  epoch: int,
                  train_losses: Dict[str, float],
                  val_losses: Dict[str, float],
                  learning_rates: Dict[str, float],
                  epoch_time: float):
        """Log epoch results."""
        # Log to console/file
        self.logger.info(f"Epoch {epoch:04d} - Time: {epoch_time:.2f}s")
        
        # Log training losses
        train_str = " | ".join([f"Train_{k}: {v:.6f}" for k, v in train_losses.items()])
        self.logger.info(f"Training - {train_str}")
        
        # Log validation losses
        val_str = " | ".join([f"Val_{k}: {v:.6f}" for k, v in val_losses.items()])
        self.logger.info(f"Validation - {val_str}")
        
        # Log learning rates
        lr_str = " | ".join([f"LR_{k}: {v:.8f}" for k, v in learning_rates.items()])
        self.logger.info(f"Learning Rates - {lr_str}")
        
        # Store metrics
        self.train_metrics.append({
            'epoch': epoch,
            'time': epoch_time,
            **train_losses,
            **{f"lr_{k}": v for k, v in learning_rates.items()}
        })
        
        self.val_metrics.append({
            'epoch': epoch,
            **val_losses
        })
        
        # Save metrics to JSON
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to JSON files."""
        train_metrics_file = self.log_dir / f"{self.experiment_name}_train_metrics.json"
        val_metrics_file = self.log_dir / f"{self.experiment_name}_val_metrics.json"
        
        with open(train_metrics_file, 'w') as f:
            json.dump(self.train_metrics, f, indent=2)
        
        with open(val_metrics_file, 'w') as f:
            json.dump(self.val_metrics, f, indent=2)


class GradientClipping:
    """Gradient clipping utility for stable training."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients and return gradient norm.
        
        Args:
            model: Model to clip gradients for
            
        Returns:
            Gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm, 
            self.norm_type
        )
        return total_norm.item()


class MovingAverage:
    """Exponential moving average for metrics."""
    
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.value = None
        
    def update(self, new_value: float) -> float:
        """Update moving average with new value."""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value
        return self.value
    
    def get(self) -> Optional[float]:
        """Get current moving average value."""
        return self.value


def setup_training_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup complete training environment.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary with training components
    """
    # Create directories
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    log_dir = Path(config['paths']['log_dir'])
    output_dir = Path(config['paths']['output_dir'])
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"dental_3d_reconstruction_{timestamp}"
    
    # Create training utilities
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience']
    )
    
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir=checkpoint_dir / experiment_name,
        save_best_only=config['training']['save_best_only']
    )
    
    training_logger = TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name
    )
    
    gradient_clipper = GradientClipping(max_norm=1.0)
    
    # Moving averages for metrics
    loss_averages = {
        'train_total': MovingAverage(),
        'val_total': MovingAverage(),
        'train_generator': MovingAverage(),
        'train_discriminator': MovingAverage()
    }
    
    return {
        'experiment_name': experiment_name,
        'early_stopping': early_stopping,
        'model_checkpoint': model_checkpoint,
        'training_logger': training_logger,
        'gradient_clipper': gradient_clipper,
        'loss_averages': loss_averages,
        'checkpoint_dir': checkpoint_dir / experiment_name,
        'log_dir': log_dir,
        'output_dir': output_dir / experiment_name
    }
