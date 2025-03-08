"""
Main Training Script for Footwear Impression Matching

This script handles the entire training pipeline:
1. Loading configuration
2. Setting up data loaders
3. Building model
4. Setting up optimizer and scheduler
5. Training and validation
6. Saving results

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/fast.yaml --resume path/to/checkpoint.pth
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime

# Import project modules
from data.dataloader import get_transforms, ShoeImpressDataset, get_dataloaders
from models.network import FootwearMatchingNetwork, EnhancedSiameseNetwork
from models.losses import FocalLoss, EnhancedContrastiveLoss
from utils.trainer import FootwearMatchingTrainer
from utils.common import set_seed, setup_logger, load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train footwear impression matching model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory from config')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model(config):
    """
    Create model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Model instance
    """
    model_name = config['name']
    
    if model_name == 'FootwearMatchingNetwork':
        model = FootwearMatchingNetwork(
            backbone=config['backbone'],
            pretrained=config['pretrained'],
            feature_dim=config['feature_dim'],
            dropout_rate=config['dropout_rate'],
            use_attention=config['use_attention'],
            use_self_attention=config['use_self_attention'],
            correlation_temp=config['correlation_temp']
        )
    elif model_name == 'EnhancedSiameseNetwork':
        model = EnhancedSiameseNetwork(
            backbone=config['backbone'],
            pretrained=config['pretrained'],
            feature_dim=config['feature_dim'],
            dropout_rate=config['dropout_rate']
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def get_loss_function(config):
    """
    Create loss function based on configuration.
    
    Args:
        config: Loss configuration
        
    Returns:
        Loss function instance
    """
    loss_name = config['name']
    
    if loss_name == 'FocalLoss':
        criterion = FocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma']
        )
    elif loss_name == 'EnhancedContrastiveLoss':
        criterion = EnhancedContrastiveLoss(
            margin=config['margin'],
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            focal_alpha=config['focal_alpha'],
            focal_gamma=config['focal_gamma'],
            use_focal=config['use_focal']
        )
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    
    return criterion


def get_optimizer(config, model):
    """
    Create optimizer based on configuration.
    
    Args:
        config: Optimizer configuration
        model: Model to optimize
        
    Returns:
        Optimizer instance
    """
    optim_name = config['name']
    
    # Split parameters into backbone and head
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'feature_extractor' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    # Set different learning rates
    backbone_lr = config['lr'] * config['backbone_lr_factor']
    head_lr = config['lr']
    
    if optim_name == 'Adam':
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr}
        ], weight_decay=config['weight_decay'])
    elif optim_name == 'AdamW':
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr}
        ], weight_decay=config['weight_decay'])
    elif optim_name == 'SGD':
        optimizer = optim.SGD([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr}
        ], momentum=0.9, weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")
    
    return optimizer


def get_scheduler(config, optimizer, steps_per_epoch=None, num_epochs=None):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        config: Scheduler configuration
        optimizer: Optimizer to schedule
        steps_per_epoch: Number of training steps per epoch (needed for some schedulers)
        num_epochs: Total number of epochs (needed for some schedulers)
        
    Returns:
        Scheduler instance or None
    """
    scheduler_name = config['name']
    
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    
    elif scheduler_name == 'OneCycleLR':
        if steps_per_epoch is None or num_epochs is None:
            raise ValueError("steps_per_epoch and num_epochs are required for OneCycleLR")
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config['max_lr'] * config['backbone_lr_factor'], config['max_lr']],
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=config['pct_start'],
            div_factor=config['div_factor'],
            final_div_factor=config['final_div_factor']
        )
    
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=config['lr'] / 100
        )
    
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return scheduler


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Create output directory
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Set timestamp for logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(
        'train',
        os.path.join(log_dir, f'train_{timestamp}.log')
    )
    
    # Save configuration
    config_save_path = os.path.join(output_dir, f'config_{timestamp}.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Log configuration summary
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {config['model']['name']} with {config['model']['backbone']} backbone")
    logger.info(f"Loss function: {config['loss']['name']}")
    logger.info(f"Optimizer: {config['optimizer']['name']} with lr={config['optimizer']['lr']}")
    logger.info(f"Scheduler: {config['scheduler']['name']}")
    logger.info(f"Training for {config['training']['num_epochs']} epochs")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    train_transform = get_transforms(mode='train', img_size=config['data']['img_size'])
    val_transform = get_transforms(mode='val', img_size=config['data']['img_size'])
    
    # Create datasets
    train_dataset = ShoeImpressDataset(
        config['data']['train_csv'],
        transform=train_transform,
        triplet_mode=config['data']['triplet_mode'],
        online_augment=config['data']['online_augment']
    )
    
    val_dataset = ShoeImpressDataset(
        config['data']['val_csv'],
        transform=val_transform,
        triplet_mode=False,
        online_augment=False
    )
    
    # Create data loaders
    data_config = argparse.Namespace(
        train_csv=config['data']['train_csv'],
        val_csv=config['data']['val_csv'],
        batch_size=config['data']['batch_size'],
        img_size=config['data']['img_size'],
        num_workers=config['data']['num_workers'],
        triplet_mode=config['data']['triplet_mode'],
        online_augment=config['data']['online_augment'],
        seed=config['seed']
    )
    
    train_loader, val_loader = get_dataloaders(data_config)
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create model
    logger.info("Creating model...")
    model = get_model(config['model'])
    
    # Move model to device
    model = model.to(device)
    
    # Create loss function
    logger.info("Creating loss function...")
    criterion = get_loss_function(config['loss'])
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = get_optimizer(config['optimizer'], model)
    
    # Create scheduler
    steps_per_epoch = len(train_loader)
    num_epochs = config['training']['num_epochs']
    
    scheduler = get_scheduler(
        config['scheduler'],
        optimizer,
        steps_per_epoch=steps_per_epoch,
        num_epochs=num_epochs
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, metrics = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        logger.info(f"Resuming from epoch {start_epoch}")
        if metrics:
            logger.info(f"Previous metrics: {metrics}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = FootwearMatchingTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        fp16=config['training']['fp16'],
        use_ema=config['training']['use_ema'],
        ema_decay=config['training']['ema_decay'],
        clip_grad_norm=config['training']['clip_grad_norm']
    )
    
    # Train model
    logger.info("Starting training...")
    best_ap, history = trainer.train(
        num_epochs=num_epochs,
        validate_every=config['training']['validate_every'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    logger.info(f"Training completed. Best AP: {best_ap:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f'final_model_{timestamp}.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()
