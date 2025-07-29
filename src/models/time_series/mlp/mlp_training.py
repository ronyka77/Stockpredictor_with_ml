"""
MLP Training Module

This module contains advanced training methods for the MLP model.
Includes optimizers, schedulers, checkpointing, and training utilities.
"""

import torch
import torch.nn as nn
import os
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader

from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLPTrainingMixin:
    """
    Mixin class providing advanced training functionality for MLPPredictor.
    """
    
    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        learning_rate = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.0)
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        scheduler_type = self.config.get('lr_scheduler')
        if scheduler_type is None:
            return None
        
        scheduler_params = self.config.get('lr_scheduler_params', {})
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=scheduler_params.get('T_max', self.config.get('epochs', 50))
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.1),
                patience=scheduler_params.get('patience', 5),
                min_lr=scheduler_params.get('min_lr', 1e-6)
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}")
            return None

    def _save_checkpoint(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, 
                        val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping_counter
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'{self.model_name}_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if specified
        if is_best and self.config.get('save_best_model', True):
            best_path = os.path.join(checkpoint_dir, f'{self.model_name}_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with validation loss: {val_loss:.6f}")

    def _load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                        checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'epoch': []
        })
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        """Calculate the gradient norm for monitoring."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def _create_dataloader(self, X: pd.DataFrame, y: pd.Series, shuffle: bool = True):
        """
        Create a DataLoader from DataFrame and Series
        
        Args:
            X: Features DataFrame
            y: Targets Series
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader instance
        """
        from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
        
        batch_size = self.config.get('batch_size', 32)
        return MLPDataUtils.create_dataloader_from_dataframe(X, y, batch_size, shuffle)

    def _training_step(self, batch_X: torch.Tensor, batch_y: torch.Tensor, 
                      optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                      scaler: Optional[torch.amp.GradScaler], gradient_clip: Optional[float]) -> Tuple[Optional[float], float]:
        """
        Execute one training step with mixed precision support.
        
        Returns:
            tuple: (loss_value, gradient_norm) or (None, 0) if NaN/Inf detected
        """
        optimizer.zero_grad()
        
        if scaler is not None:
            return self._mixed_precision_step(batch_X, batch_y, optimizer, criterion, scaler, gradient_clip)
        else:
            return self._standard_step(batch_X, batch_y, optimizer, criterion, gradient_clip)

    def _mixed_precision_step(self, batch_X: torch.Tensor, batch_y: torch.Tensor,
                             optimizer: torch.optim.Optimizer, criterion: nn.Module,
                             scaler: torch.amp.GradScaler, gradient_clip: Optional[float]) -> Tuple[Optional[float], float]:
        """Execute training step with mixed precision."""
        with torch.amp.autocast('cuda'):
            outputs = self.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
        
        # NaN/Inf loss check
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf loss detected! Skipping batch.")
            return None, 0.0
        
        scaler.scale(loss).backward()
        
        if gradient_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate gradient norm
        grad_norm = self._calculate_gradient_norm(self.model)
        
        return loss.item(), grad_norm

    def _standard_step(self, batch_X: torch.Tensor, batch_y: torch.Tensor,
                      optimizer: torch.optim.Optimizer, criterion: nn.Module,
                      gradient_clip: Optional[float]) -> Tuple[Optional[float], float]:
        """Execute training step with standard precision."""
        outputs = self.model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        
        # NaN/Inf loss check
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf loss detected! Skipping batch.")
            return None, 0.0
        
        loss.backward()
        
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Calculate gradient norm
        grad_norm = self._calculate_gradient_norm(self.model)
        
        return loss.item(), grad_norm

    def _training_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                       criterion: nn.Module, scaler: Optional[torch.amp.GradScaler], 
                       gradient_clip: Optional[float]) -> Dict[str, float]:
        """
        Execute one training epoch.
        
        Returns:
            dict: {'avg_loss', 'avg_gradient_norm', 'num_batches'}
        """
        self.model.train()
        total_loss = 0.0
        total_gradient_norm = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            # Training step
            loss, grad_norm = self._training_step(batch_X, batch_y, optimizer, criterion, scaler, gradient_clip)
            
            if loss is not None:  # Skip if NaN/Inf detected
                total_loss += loss
                total_gradient_norm += grad_norm
                num_batches += 1
        
        return {
            'avg_loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'avg_gradient_norm': total_gradient_norm / num_batches if num_batches > 0 else 0.0,
            'num_batches': num_batches
        }

    def _validation_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Execute one validation epoch.
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                val_outputs = self.model(batch_X_val)
                val_loss = criterion(val_outputs.squeeze(), batch_y_val)
                total_val_loss += val_loss.item()
        
        return total_val_loss / len(val_loader)

    def _update_learning_rate(self, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], 
                             val_loss: Optional[float]) -> float:
        """
        Update learning rate based on scheduler type.
        
        Args:
            scheduler: Learning rate scheduler
            val_loss: Validation loss (for ReduceLROnPlateau)
            
        Returns:
            float: Current learning rate
        """
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        return self.optimizer.param_groups[0]['lr']

    def _check_early_stopping(self, val_loss: float, epoch: int, 
                             early_stopping_patience: int, early_stopping_min_delta: float) -> Dict[str, Any]:
        """
        Check if early stopping should be triggered.
        
        Returns:
            dict: {'should_stop', 'is_best', 'counter'}
        """
        is_best = val_loss < self.best_val_loss - early_stopping_min_delta
        
        if is_best:
            self.best_val_loss = val_loss
            self.best_model_state = self.model.state_dict().copy()
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        should_stop = self.early_stopping_counter >= early_stopping_patience
        
        return {
            'should_stop': should_stop,
            'is_best': is_best,
            'counter': self.early_stopping_counter
        }

    def _log_epoch_metrics(self, epoch: int, epochs: int, train_metrics: Dict[str, float], 
                            val_loss: Optional[float], learning_rate: float):
        """
        Log epoch metrics and update training history.
        """
        # Update training history
        self.training_history['train_loss'].append(train_metrics['avg_loss'])
        self.training_history['gradient_norm'].append(train_metrics['avg_gradient_norm'])
        self.training_history['epoch'].append(epoch)
        
        if val_loss is not None:
            self.training_history['val_loss'].append(val_loss)
        
        self.training_history['learning_rate'].append(learning_rate)
        
        # Log progress
        if val_loss is not None:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_metrics['avg_loss']:.6f}, "
                        f"Val Loss: {val_loss:.6f}, LR: {learning_rate:.2e}, "
                        f"Grad Norm: {train_metrics['avg_gradient_norm']:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_metrics['avg_loss']:.6f}, "
                        f"LR: {learning_rate:.2e}, Grad Norm: {train_metrics['avg_gradient_norm']:.4f}")

    def fit(self, train_data, val_data=None, feature_names=None, resume_from_checkpoint: str = None):
        """
        Train the MLP model with advanced features including early stopping, 
        learning rate scheduling, and model checkpointing.
        
        Args:
            train_data: Either DataLoader for training data or (X_train, y_train) tuple
            val_data: Either DataLoader for validation data or (X_val, y_val) tuple (optional)
            feature_names: List of feature names (optional)
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
        """
        # Handle DataFrame inputs by converting to DataLoaders
        if isinstance(train_data, tuple) and len(train_data) == 2:
            # train_data is (X_train, y_train) tuple
            X_train, y_train = train_data
            train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
            
            if val_data is not None and isinstance(val_data, tuple) and len(val_data) == 2:
                # val_data is (X_val, y_val) tuple
                X_val, y_val = val_data
                val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
            else:
                val_loader = val_data  # val_data is already a DataLoader or None
        else:
            # train_data is already a DataLoader
            train_loader = train_data
            val_loader = val_data
        
        # Initialize model and training state
        self.model = self._create_model()
        self.model.to(self.device)
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Initialize training history if not exists
        if not hasattr(self, 'training_history'):
            self.training_history = {
                'train_loss': [],
                'val_loss': [],
                'learning_rate': [],
                'gradient_norm': [],
                'epoch': []
            }
        
        # Initialize early stopping variables if not exists
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
        if not hasattr(self, 'best_model_state'):
            self.best_model_state = None
        if not hasattr(self, 'early_stopping_counter'):
            self.early_stopping_counter = 0

        # Training configuration
        epochs = self.config.get('epochs', 50)
        gradient_clip = self.config.get('gradient_clip', None)
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        early_stopping_min_delta = self.config.get('early_stopping_min_delta', 1e-4)
        save_checkpoint_frequency = self.config.get('save_checkpoint_frequency', 5)

        # Loss function
        if self.config.get('task', 'regression') == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss()

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer(self.model)
        scheduler = self._create_scheduler(self.optimizer)

        # Mixed precision training if CUDA available
        scaler = None
        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            logger.info("Using mixed precision training")

        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self._load_checkpoint(self.model, self.optimizer, resume_from_checkpoint)

        logger.info(f"Starting training for {epochs} epochs from epoch {start_epoch}...")
        logger.info(f"Advanced features: Early stopping patience={early_stopping_patience}, "
                    f"Gradient clip={gradient_clip}, Scheduler={self.config.get('lr_scheduler')}")

        # Main training loop - now much cleaner
        for epoch in range(start_epoch, epochs):
            # Training phase
            train_metrics = self._training_epoch(train_loader, self.optimizer, criterion, scaler, gradient_clip)
            
            # Validation phase
            val_loss = None
            if val_loader:
                val_loss = self._validation_epoch(val_loader, criterion)
                
                # Early stopping check
                early_stop_result = self._check_early_stopping(
                    val_loss, epoch, early_stopping_patience, early_stopping_min_delta
                )
                
                if early_stop_result['is_best']:
                    self._save_checkpoint(epoch, self.model, self.optimizer, val_loss, is_best=True)
                
                if early_stop_result['should_stop']:
                    logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                    break
            
            # Learning rate scheduling
            learning_rate = self._update_learning_rate(scheduler, val_loss)
            
            # Logging
            self._log_epoch_metrics(epoch, epochs, train_metrics, val_loss, learning_rate)
            
            # Regular checkpointing
            if (epoch + 1) % save_checkpoint_frequency == 0:
                self._save_checkpoint(epoch, self.model, self.optimizer, val_loss)

        # Restore best model if early stopping was used
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model based on validation loss")

        self.is_trained = True
        logger.info("Training completed!")


# Extend MLPPredictor with training mixin
class MLPPredictorWithTraining(MLPPredictor, MLPTrainingMixin):
    """
    MLPPredictor with advanced training capabilities.
    """
    pass 