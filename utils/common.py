"""
Common Utility Functions

This module contains utility functions used across the project for:
- Metric calculation
- Visualization
- Model checkpointing
- Logging
- General helper functions
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import random
import time
from datetime import datetime
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logger(name, log_file, level=logging.INFO, console_output=True):
    """Setup logger with file and console outputs."""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        metrics: Dictionary of performance metrics
        save_path: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    # Regular checkpoint
    torch.save(checkpoint, os.path.join(save_path, f"checkpoint_epoch_{epoch}.pth"))
    
    # Save best model separately
    if is_best:
        torch.save(checkpoint, os.path.join(save_path, "best_model.pth"))


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Learning rate scheduler to load state into (optional)
        device: Device to load model onto
        
    Returns:
        epoch: Epoch of the checkpoint
        metrics: Performance metrics from the checkpoint
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Extract metadata
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return epoch, metrics


def calculate_metrics(logits, similarities, labels):
    """
    Calculate performance metrics for footwear impression matching.
    
    Args:
        logits: Model logits
        similarities: Similarity scores
        labels: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(similarities, torch.Tensor):
        similarities = similarities.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Convert logits to probabilities
    probas = 1 / (1 + np.exp(-logits))  # Sigmoid
    
    # Classification metrics based on logits
    predictions = (probas > 0.5).astype(int)
    accuracy = np.mean(predictions == labels)
    
    # Precision-Recall curve and Average Precision (logits)
    precision, recall, thresholds_pr = precision_recall_curve(labels, probas)
    ap = average_precision_score(labels, probas)
    
    # ROC curve and AUC (logits)
    fpr, tpr, thresholds_roc = roc_curve(labels, probas)
    roc_auc = auc(fpr, tpr)
    
    # Metrics based on similarity scores
    sim_precision, sim_recall, sim_thresholds_pr = precision_recall_curve(labels, similarities)
    sim_ap = average_precision_score(labels, similarities)
    
    # Compile metrics
    metrics = {
        'accuracy': accuracy,
        'ap': ap,
        'roc_auc': roc_auc,
        'sim_ap': sim_ap,
        'precision': precision,
        'recall': recall,
        'sim_precision': sim_precision,
        'sim_recall': sim_recall,
        'fpr': fpr,
        'tpr': tpr
    }
    
    return metrics


def plot_precision_recall_curve(precision, recall, ap, filename, title=None):
    """
    Plot precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        ap: Average precision score
        filename: Output file path
        title: Plot title (optional)
    """
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, lw=2, marker='.', markersize=3, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title or f'Precision-Recall Curve (AP = {ap:.4f})')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, filename, title=None):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under ROC curve
        filename: Output file path
        title: Plot title (optional)
    """
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, lw=2, marker='.', markersize=3, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title or f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_training_history(history, save_path):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary of training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Extract epoch numbers
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', marker='.', label='Train Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', marker='.', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', marker='.', label='Train Acc')
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], 'r-', marker='.', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot AP and AUC
    plt.subplot(2, 2, 3)
    if 'val_ap' in history:
        plt.plot(epochs, history['val_ap'], 'g-', marker='.', label='Val AP')
    if 'val_auc' in history:
        plt.plot(epochs, history['val_auc'], 'm-', marker='.', label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('AP and AUC vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    if 'learning_rate' in history:
        plt.plot(epochs, history['learning_rate'], 'k-', marker='.')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate vs. Epoch')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_batch(track_imgs, ref_imgs, labels, predictions, similarities, filename, num_samples=10):
    """
    Visualize a batch of footwear impression pairs with predictions.
    
    Args:
        track_imgs: Batch of track images
        ref_imgs: Batch of reference images
        labels: Ground truth labels
        predictions: Model predictions
        similarities: Similarity scores
        filename: Output file path
        num_samples: Number of samples to visualize
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(track_imgs, torch.Tensor):
        track_imgs = track_imgs.detach().cpu().numpy()
    if isinstance(ref_imgs, torch.Tensor):
        ref_imgs = ref_imgs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(similarities, torch.Tensor):
        similarities = similarities.detach().cpu().numpy()
    
    # Limit the number of samples
    n = min(num_samples, len(track_imgs))
    
    # Create figure
    plt.figure(figsize=(15, 3 * n))
    
    # Normalize images if needed
    if track_imgs.max() > 1.0:
        track_imgs = track_imgs / 255.0
    if ref_imgs.max() > 1.0:
        ref_imgs = ref_imgs / 255.0
    
    # Iterate through samples
    for i in range(n):
        # Track image
        plt.subplot(n, 3, i*3 + 1)
        if track_imgs.shape[1] == 3:  # RGB image
            plt.imshow(np.transpose(track_imgs[i], (1, 2, 0)))
        else:  # Grayscale image
            plt.imshow(track_imgs[i, 0], cmap='gray')
        plt.title('Track Image')
        plt.axis('off')
        
        # Reference image
        plt.subplot(n, 3, i*3 + 2)
        if ref_imgs.shape[1] == 3:  # RGB image
            plt.imshow(np.transpose(ref_imgs[i], (1, 2, 0)))
        else:  # Grayscale image
            plt.imshow(ref_imgs[i, 0], cmap='gray')
        plt.title('Reference Image')
        plt.axis('off')
        
        # Prediction and similarity
        plt.subplot(n, 3, i*3 + 3)
        plt.text(0.5, 0.5, 
                f"Label: {labels[i]}\n"
                f"Prediction: {predictions[i]:.2f}\n"
                f"Similarity: {similarities[i]:.2f}",
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


class TicToc:
    """Class for timing code execution."""
    
    def __init__(self, name=None):
        self.name = name
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args, **kwargs):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        name = f" [{self.name}]" if self.name else ""
        print(f"Elapsed time{name}: {self.elapsed:.4f} seconds")


class AverageMeter:
    """Class to track average values during training."""
    
    def __init__(self, name=None):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors a validation metric and stops training when it doesn't improve
    for a specified number of epochs.
    """
    
    def __init__(self, patience=10, delta=0, mode='max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement before stopping
            delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that decrease (e.g. loss) or 'max' for metrics that increase (e.g. AP)
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score):
        """
        Check if training should be stopped.
        
        Args:
            score: Current score to check
            
        Returns:
            is_best: Whether the current score is the best so far
        """
        is_best = False
        
        if self.best_score is None:
            # First epoch
            self.best_score = score
            is_best = True
        elif self.mode == 'min':
            # Score decreases (e.g. loss)
            if score < self.best_score - self.delta:
                self.best_score = score
                self.counter = 0
                is_best = True
            else:
                self.counter += 1
        else:  # mode == 'max'
            # Score increases (e.g. accuracy, AP)
            if score > self.best_score + self.delta:
                self.best_score = score
                self.counter = 0
                is_best = True
            else:
                self.counter += 1
        
        # Check if we should stop
        if self.counter >= self.patience:
            self.early_stop = True
        
        return is_best


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


# Test the utilities
if __name__ == "__main__":
    # Test seed setting
    set_seed(42)
    
    # Test timing
    with TicToc("Test operation"):
        time.sleep(1)
    
    # Test average meter
    meter = AverageMeter("Test")
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.65, 0.63, 0.62, 0.61]
    for epoch, score in enumerate(scores):
        is_best = early_stopping(score)
        print(f"Epoch {epoch}, Score: {score}, Best: {is_best}, Counter: {early_stopping.counter}")
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
