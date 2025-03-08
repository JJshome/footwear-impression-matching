"""
Model Trainer Module

This module implements the training and evaluation code for footwear impression matching models.
It provides a Trainer class with functionality for:
- Training and validation loops
- Learning rate scheduling
- Metrics tracking
- Model checkpointing
- Visualization
- Mixed precision training
- EMA model maintenance

The trainer is designed to work with various model architectures and loss functions.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from utils.common import (
    save_checkpoint, load_checkpoint, calculate_metrics,
    plot_precision_recall_curve, plot_roc_curve, plot_training_history,
    AverageMeter, EarlyStopping, TicToc
)


class FootwearMatchingTrainer:
    """
    Trainer class for footwear impression matching models.
    
    Handles the entire training process including:
    - Model training and validation
    - Metric computation and tracking
    - Checkpointing
    - Learning rate scheduling
    - Visualization
    """
    
    def __init__(
        self,
        model,
        criterion,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device='cuda',
        output_dir='./results',
        fp16=True,
        use_ema=True,
        ema_decay=0.999,
        clip_grad_norm=1.0
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            criterion: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to use for training ('cuda' or 'cpu')
            output_dir: Directory to save results
            fp16: Whether to use mixed precision training
            use_ema: Whether to use Exponential Moving Average for model weights
            ema_decay: EMA decay rate
            clip_grad_norm: Gradient clipping norm
        """
        # Basic components
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.fp16 = fp16 and torch.cuda.is_available()
        self.clip_grad_norm = clip_grad_norm
        
        # Create output directories
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        self.plot_dir = os.path.join(output_dir, 'plots')
        self.log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup mixed precision training
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Setup EMA model
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        if use_ema:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_ap': [],
            'val_auc': [],
            'learning_rate': []
        }
        
        # Best metric tracking
        self.best_ap = 0.0
        self.best_epoch = 0
    
    def _create_ema_model(self):
        """Create EMA model by cloning the current model."""
        ema_model = type(self.model)(**self.model.module.__dict__ if isinstance(self.model, nn.DataParallel) else self.model.__dict__)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()  # Set to evaluation mode
        
        # Disable gradient computation for EMA model
        for param in ema_model.parameters():
            param.requires_grad = False
            
        return ema_model
    
    def _update_ema_model(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
            
            # Update batch norm statistics if present
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), self.model.buffers()):
                if ema_buffer.data.dtype in [torch.float32, torch.float16, torch.float64]:
                    ema_buffer.data.mul_(self.ema_decay).add_(model_buffer.data, alpha=1 - self.ema_decay)
                else:
                    ema_buffer.copy_(model_buffer)
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            epoch_loss: Average loss for the epoch
            epoch_acc: Average accuracy for the epoch
        """
        # Set model to training mode
        self.model.train()
        
        # Metrics tracking
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Acc')
        batch_time = AverageMeter('Batch Time')
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        end_time = time.time()
        
        # Iterate through batches
        for batch_idx, data in enumerate(pbar):
            # Handle different data formats (triplet vs standard)
            if len(data) == 6:  # Triplet mode
                track_imgs, ref_imgs, pos_imgs, neg_imgs, labels, _ = data
                track_imgs = track_imgs.to(self.device)
                ref_imgs = ref_imgs.to(self.device)
                pos_imgs = pos_imgs.to(self.device)
                neg_imgs = neg_imgs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass with mixed precision
                self.optimizer.zero_grad()
                
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        logits, similarities, track_features, ref_features = self.model(track_imgs, ref_imgs)
                        pos_features = self.model(pos_imgs, None, mode='track')
                        neg_features = self.model(None, neg_imgs, mode='ref')
                        
                        # Compute loss
                        loss = self.criterion(
                            logits, similarities, labels,
                            track_features, ref_features,
                            pos_features, neg_features
                        )
                    
                    # Backward pass with mixed precision
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.clip_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                else:
                    # Standard forward and backward pass
                    logits, similarities, track_features, ref_features = self.model(track_imgs, ref_imgs)
                    pos_features = self.model(pos_imgs, None, mode='track')
                    neg_features = self.model(None, neg_imgs, mode='ref')
                    
                    # Compute loss
                    loss = self.criterion(
                        logits, similarities, labels,
                        track_features, ref_features,
                        pos_features, neg_features
                    )
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Update weights
                    self.optimizer.step()
                
            else:  # Standard mode
                track_imgs, ref_imgs, labels, _ = data
                track_imgs = track_imgs.to(self.device)
                ref_imgs = ref_imgs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass with mixed precision
                self.optimizer.zero_grad()
                
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        logits, similarities, track_features, ref_features = self.model(track_imgs, ref_imgs)
                        
                        # Compute loss
                        loss = self.criterion(
                            logits, similarities, labels,
                            track_features, ref_features
                        )
                    
                    # Backward pass with mixed precision
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.clip_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                else:
                    # Standard forward and backward pass
                    logits, similarities, track_features, ref_features = self.model(track_imgs, ref_imgs)
                    
                    # Compute loss
                    loss = self.criterion(
                        logits, similarities, labels,
                        track_features, ref_features
                    )
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Update weights
                    self.optimizer.step()
            
            # Update EMA model if enabled
            if self.use_ema:
                self._update_ema_model()
            
            # Update step-based scheduler if applicable
            if self.scheduler is not None and isinstance(self.scheduler, (
                optim.lr_scheduler.OneCycleLR,
                optim.lr_scheduler.CyclicLR
            )):
                self.scheduler.step()
            
            # Calculate accuracy
            with torch.no_grad():
                probas = torch.sigmoid(logits.view(-1))
                preds = (probas > 0.5).float()
                accuracy = (preds == labels.float()).float().mean().item()
            
            # Update metrics
            loss_meter.update(loss.item())
            acc_meter.update(accuracy)
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'acc': f"{acc_meter.avg:.4f}",
                'time': f"{batch_time.avg:.3f}s"
            })
        
        # Update epoch-based scheduler if applicable
        if self.scheduler is not None and not isinstance(self.scheduler, (
            optim.lr_scheduler.OneCycleLR,
            optim.lr_scheduler.CyclicLR,
            optim.lr_scheduler.ReduceLROnPlateau
        )):
            self.scheduler.step()
        
        # Store current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.history['learning_rate'].append(current_lr)
        
        # Store epoch metrics
        self.history['train_loss'].append(loss_meter.avg)
        self.history['train_acc'].append(acc_meter.avg)
        
        return loss_meter.avg, acc_meter.avg
    
    def validate(self, epoch, use_ema=False):
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            use_ema: Whether to use the EMA model for validation
            
        Returns:
            val_loss: Average validation loss
            metrics: Dictionary of validation metrics
        """
        # Choose model for validation
        eval_model = self.ema_model if use_ema and self.ema_model is not None else self.model
        
        # Set model to evaluation mode
        eval_model.eval()
        
        # Metrics tracking
        loss_meter = AverageMeter('Val Loss')
        
        # Collect all outputs and targets
        all_logits = []
        all_similarities = []
        all_labels = []
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val" + (" EMA]" if use_ema else "]"))
        
        # Iterate through batches
        with torch.no_grad():
            for data in pbar:
                # Handle different data formats
                if len(data) == 6:  # Triplet mode
                    track_imgs, ref_imgs, pos_imgs, neg_imgs, labels, _ = data
                    track_imgs = track_imgs.to(self.device)
                    ref_imgs = ref_imgs.to(self.device)
                    pos_imgs = pos_imgs.to(self.device)
                    neg_imgs = neg_imgs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    logits, similarities, track_features, ref_features = eval_model(track_imgs, ref_imgs)
                    pos_features = eval_model(pos_imgs, None, mode='track')
                    neg_features = eval_model(None, neg_imgs, mode='ref')
                    
                    # Compute loss
                    loss = self.criterion(
                        logits, similarities, labels,
                        track_features, ref_features,
                        pos_features, neg_features
                    )
                    
                else:  # Standard mode
                    track_imgs, ref_imgs, labels, _ = data
                    track_imgs = track_imgs.to(self.device)
                    ref_imgs = ref_imgs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    logits, similarities, track_features, ref_features = eval_model(track_imgs, ref_imgs)
                    
                    # Compute loss
                    loss = self.criterion(
                        logits, similarities, labels,
                        track_features, ref_features
                    )
                
                # Update loss meter
                loss_meter.update(loss.item())
                
                # Collect outputs and targets
                all_logits.append(logits.detach().cpu())
                all_similarities.append(similarities.detach().cpu())
                all_labels.append(labels.detach().cpu())
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})
        
        # Concatenate all outputs and targets
        all_logits = torch.cat(all_logits, dim=0).squeeze().numpy()
        all_similarities = torch.cat(all_similarities, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # Calculate metrics
        metrics = calculate_metrics(all_logits, all_similarities, all_labels)
        
        # Store metrics
        self.history['val_loss'].append(loss_meter.avg)
        self.history['val_acc'].append(metrics['accuracy'])
        self.history['val_ap'].append(metrics['ap'])
        self.history['val_auc'].append(metrics['roc_auc'])
        
        # Plot curves
        model_type = "ema" if use_ema else "regular"
        plot_precision_recall_curve(
            metrics['precision'], metrics['recall'], metrics['ap'],
            os.path.join(self.plot_dir, f"pr_curve_epoch_{epoch}_{model_type}.png"),
            title=f"Precision-Recall Curve (Epoch {epoch}, {model_type.capitalize()} Model)"
        )
        plot_roc_curve(
            metrics['fpr'], metrics['tpr'], metrics['roc_auc'],
            os.path.join(self.plot_dir, f"roc_curve_epoch_{epoch}_{model_type}.png"),
            title=f"ROC Curve (Epoch {epoch}, {model_type.capitalize()} Model)"
        )
        
        # Update plateau scheduler if applicable
        if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics['ap'])
        
        # Print summary
        print(f"\nValidation Summary (Epoch {epoch}, {model_type.capitalize()} Model):")
        print(f"  Loss: {loss_meter.avg:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AP: {metrics['ap']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  Similarity AP: {metrics['sim_ap']:.4f}")
        
        return loss_meter.avg, metrics
    
    def train(self, num_epochs, validate_every=1, early_stopping_patience=10):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            validate_every: Validate every N epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            best_ap: Best validation AP achieved
            history: Training history
        """
        # Setup early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        
        # Start time
        start_time = time.time()
        print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training for {num_epochs} epochs with validation every {validate_every} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Using device: {self.device}")
        print(f"Mixed precision training: {self.fp16}")
        print(f"EMA model: {self.use_ema}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*30} Epoch {epoch}/{num_epochs} {'='*30}")
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate if needed
            if epoch % validate_every == 0:
                # Validate with regular model
                val_loss, val_metrics = self.validate(epoch, use_ema=False)
                
                # Validate with EMA model if available
                if self.use_ema:
                    ema_val_loss, ema_val_metrics = self.validate(epoch, use_ema=True)
                    
                    # Use EMA metrics for checkpoint saving and early stopping
                    val_metrics = ema_val_metrics
                
                # Plot training history
                plot_training_history(
                    self.history,
                    os.path.join(self.plot_dir, f"training_history_epoch_{epoch}.png")
                )
                
                # Check for best model
                is_best = early_stopping(val_metrics['ap'])
                if is_best:
                    self.best_ap = val_metrics['ap']
                    self.best_epoch = epoch
                    print(f"✓ New best model with AP: {self.best_ap:.4f}")
                
                # Save checkpoint
                model_to_save = self.ema_model if self.use_ema else self.model
                save_checkpoint(
                    model_to_save,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics,
                    self.checkpoint_dir,
                    is_best=is_best
                )
                
                # Check early stopping
                if early_stopping.early_stop:
                    print(f"\n! Early stopping triggered after {epoch} epochs")
                    print(f"Best AP: {self.best_ap:.4f} at epoch {self.best_epoch}")
                    break
            else:
                # Save checkpoint without validation
                model_to_save = self.ema_model if self.use_ema else self.model
                save_checkpoint(
                    model_to_save,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    {'train_loss': train_loss, 'train_acc': train_acc},
                    self.checkpoint_dir,
                    is_best=False
                )
        
        # Calculate total training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"Best AP: {self.best_ap:.4f} at epoch {self.best_epoch}")
        
        # Final history plot
        plot_training_history(
            self.history,
            os.path.join(self.plot_dir, "final_training_history.png")
        )
        
        return self.best_ap, self.history


# Test the trainer
if __name__ == "__main__":
    import torch.nn as nn
    import torchvision.models as models
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.backbone = models.resnet18(pretrained=False, num_classes=128)
            self.fc = nn.Linear(128, 1)
        
        def forward(self, x1, x2, mode='full'):
            if mode == 'track':
                return self.backbone(x1)
            if mode == 'ref':
                return self.backbone(x2)
            
            feat1 = self.backbone(x1)
            feat2 = self.backbone(x2)
            similarity = (feat1 * feat2).sum(dim=1)
            logits = self.fc(torch.cat([feat1, feat2], dim=1))
            
            return logits, similarity, feat1, feat2
    
    # Create dummy data
    x1 = torch.randn(100, 3, 224, 224)
    x2 = torch.randn(100, 3, 224, 224)
    y = torch.randint(0, 2, (100, 1)).float()
    
    dataset = TensorDataset(x1, x2, y)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=10)
    
    # Create model and optimizer
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Create trainer
    trainer = FootwearMatchingTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device='cpu',  # Use CPU for testing
        output_dir='./test_results',
        fp16=False,
        use_ema=False
    )
    
    # Train for a few epochs
    trainer.train(num_epochs=2, validate_every=1)
