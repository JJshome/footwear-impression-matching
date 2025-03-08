"""
Data Loading Module for Footwear Impression Matching

This module implements dataset and dataloader classes for the footwear impression matching task.
It supports:
1. Loading pairs of impression images and their references
2. Online augmentation
3. Triplet sampling for metric learning
4. Batch processing with proper handling of variable-length data

Usage:
    from data.dataloader import ShoeImpressDataset, get_dataloaders
    train_loader, val_loader = get_dataloaders(args)
"""

import os
import torch
import numpy as np
import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from .augmentation import FootwearAugmenter

class ShoeImpressDataset(Dataset):
    """
    Dataset for footwear impression matching task.
    
    Supports:
    - Standard pair loading (track/probe + reference)
    - Online augmentation
    - Triplet mode for metric learning
    """
    
    def __init__(self, pairs_csv, transform=None, triplet_mode=False, 
                 online_augment=False, augmenter=None):
        """
        Initialize the dataset.
        
        Args:
            pairs_csv: Path to CSV file containing matching pairs
            transform: Transformation pipeline for preprocessing
            triplet_mode: Enable triplet sampling for metric learning
            online_augment: Enable online augmentation
            augmenter: Custom augmenter instance (if None, create a new one)
        """
        self.pairs_df = pd.read_csv(pairs_csv)
        self.transform = transform
        self.triplet_mode = triplet_mode
        self.online_augment = online_augment
        self.augmenter = augmenter if augmenter else FootwearAugmenter()
        
        # Analyze class balance
        pos_count = (self.pairs_df['label'] == 1).sum()
        neg_count = (self.pairs_df['label'] == 0).sum()
        total_count = len(self.pairs_df)
        
        # Calculate positive weight for loss function (to handle class imbalance)
        self.pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"Dataset Stats - Positive: {pos_count} ({pos_count/total_count:.2f}), "
              f"Negative: {neg_count} ({neg_count/total_count:.2f}), "
              f"Total: {total_count}")
        print(f"Positive weight (for loss function): {self.pos_weight:.2f}")
        
        # For triplet mode, organize samples by classes
        if triplet_mode:
            self._setup_triplet_sampling()
    
    def _setup_triplet_sampling(self):
        """Prepare data structures for triplet sampling."""
        # Group positive pairs by reference ID (for positive sampling)
        self.pos_groups = {}
        pos_pairs = self.pairs_df[self.pairs_df['label'] == 1]
        
        for _, row in pos_pairs.iterrows():
            ref_id = row['ref_id']
            if ref_id not in self.pos_groups:
                self.pos_groups[ref_id] = []
            self.pos_groups[ref_id].append(row.to_dict())
        
        # Group negative pairs by track ID (for negative sampling)
        self.neg_groups = {}
        neg_pairs = self.pairs_df[self.pairs_df['label'] == 0]
        
        for _, row in neg_pairs.iterrows():
            track_id = row['track_id']
            if track_id not in self.neg_groups:
                self.neg_groups[track_id] = []
            self.neg_groups[track_id].append(row.to_dict())
    
    def __len__(self):
        """Return the number of pairs in the dataset."""
        return len(self.pairs_df)
    
    def _load_image(self, img_path):
        """
        Load and preprocess an image.
        
        Args:
            img_path: Path to the image
            
        Returns:
            Preprocessed image (PIL or tensor depending on transform)
        """
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Apply online augmentation (if enabled)
            if self.online_augment and random.random() > 0.5:
                # Convert to numpy for augmentation
                img_np = np.array(img)
                # Apply augmentation
                aug_img, _ = self.augmenter.augment(img_np)
                # Convert back to PIL
                img = Image.fromarray(aug_img)
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            
            return img
        
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a placeholder (black image)
            if self.transform:
                return torch.zeros(3, 512, 512)
            else:
                return Image.new('RGB', (512, 512), (0, 0, 0))
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            In standard mode: (track_img, ref_img, label, metadata)
            In triplet mode: (track_img, ref_img, pos_img, neg_img, label, metadata)
        """
        try:
            row = self.pairs_df.iloc[idx]
            
            # Load track and reference images
            track_img = self._load_image(row['track_path'])
            ref_img = self._load_image(row['ref_path'])
            
            # Get label
            label = int(row['label'])
            
            # Metadata for debugging and visualization
            meta = {
                'track_id': row['track_id'],
                'ref_id': row['ref_id'],
                'track_path': row['track_path'],
                'ref_path': row['ref_path']
            }
            
            # Standard mode (pair-based)
            if not self.triplet_mode or label == 0:
                return track_img, ref_img, label, meta
            
            # Triplet mode (only for positive pairs)
            try:
                # Find additional positive sample (same reference, different track)
                same_ref_samples = [p for p in self.pos_groups.get(row['ref_id'], [])
                                   if p['track_id'] != row['track_id']]
                
                # Find negative sample (same track, different reference)
                diff_ref_samples = self.neg_groups.get(row['track_id'], [])
                
                # Skip triplet if we don't have enough samples
                if not same_ref_samples or not diff_ref_samples:
                    return track_img, ref_img, label, meta
                
                # Sample positive and negative examples
                pos_sample = random.choice(same_ref_samples)
                neg_sample = random.choice(diff_ref_samples)
                
                # Load images
                pos_img = self._load_image(pos_sample['track_path'])
                neg_img = self._load_image(neg_sample['ref_path'])
                
                # Return triplet format
                return track_img, ref_img, pos_img, neg_img, label, meta
                
            except Exception as e:
                print(f"Error creating triplet for index {idx}: {str(e)}")
                # Fall back to standard format if triplet creation fails
                return track_img, ref_img, label, meta
            
        except Exception as e:
            print(f"General error for index {idx}: {str(e)}")
            # Return a placeholder in case of errors
            if self.transform:
                dummy_img = torch.zeros(3, 512, 512)
            else:
                dummy_img = Image.new('RGB', (512, 512), (0, 0, 0))
            
            return dummy_img, dummy_img, 0, {'track_id': '0', 'ref_id': '0'}


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length batch items.
    
    Args:
        batch: List of samples returned by __getitem__
        
    Returns:
        Batched tensors and metadata
    """
    # Filter out None items (if any)
    batch = [b for b in batch if b is not None]
    
    # Check if batch is empty
    if not batch:
        return None
    
    # Determine if we're in triplet mode by checking the length of the first item
    num_elements = len(batch[0])
    
    # Filter items with incorrect length
    filtered_batch = [b for b in batch if len(b) == num_elements]
    
    if num_elements == 4:  # Standard mode
        track_imgs, ref_imgs, labels, metas = zip(*filtered_batch)
        return torch.stack(track_imgs), torch.stack(ref_imgs), torch.tensor(labels), metas
    
    elif num_elements == 6:  # Triplet mode
        track_imgs, ref_imgs, pos_imgs, neg_imgs, labels, metas = zip(*filtered_batch)
        return (torch.stack(track_imgs), torch.stack(ref_imgs), 
                torch.stack(pos_imgs), torch.stack(neg_imgs), 
                torch.tensor(labels), metas)
    
    else:
        raise ValueError(f"Unexpected number of elements in batch item: {num_elements}")


def get_transforms(mode='train', img_size=512, normalize=True):
    """
    Get image transformation pipeline.
    
    Args:
        mode: 'train' or 'val'
        img_size: Target image size
        normalize: Whether to apply normalization
        
    Returns:
        Transformation pipeline
    """
    # Normalization parameters (ImageNet statistics)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=255  # White background
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ]
    else:
        # Validation transforms (no augmentation)
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    
    # Add normalization if required
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_dataloaders(args):
    """
    Create dataloaders for training and validation.
    
    Args:
        args: Configuration object with attributes:
            - train_csv: Path to training pairs CSV
            - val_csv: Path to validation pairs CSV
            - batch_size: Batch size
            - img_size: Image size
            - num_workers: Number of workers for data loading
            - triplet_mode: Whether to use triplet sampling
            - online_augment: Whether to use online augmentation
            
    Returns:
        Training and validation dataloaders
    """
    # Get transforms
    train_transform = get_transforms(mode='train', img_size=args.img_size)
    val_transform = get_transforms(mode='val', img_size=args.img_size)
    
    # Create augmenter
    augmenter = FootwearAugmenter(seed=args.seed)
    
    # Create datasets
    train_dataset = ShoeImpressDataset(
        args.train_csv,
        transform=train_transform,
        triplet_mode=args.triplet_mode,
        online_augment=args.online_augment,
        augmenter=augmenter
    )
    
    val_dataset = ShoeImpressDataset(
        args.val_csv,
        transform=val_transform,
        triplet_mode=False,  # No triplet mode for validation
        online_augment=False  # No online augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader


# Usage example
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Footwear impression dataloader test')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training pairs CSV')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to validation pairs CSV')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--triplet_mode', action='store_true', help='Use triplet mode')
    parser.add_argument('--online_augment', action='store_true', help='Use online augmentation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(args)
    
    # Print information
    print(f"Training dataloader: {len(train_loader)} batches")
    print(f"Validation dataloader: {len(val_loader)} batches")
    
    # Test a batch
    for batch in train_loader:
        if args.triplet_mode and len(batch) == 6:
            track_imgs, ref_imgs, pos_imgs, neg_imgs, labels, metas = batch
            print(f"Triplet batch shapes: {track_imgs.shape}, {ref_imgs.shape}, "
                  f"{pos_imgs.shape}, {neg_imgs.shape}")
        else:
            track_imgs, ref_imgs, labels, metas = batch
            print(f"Standard batch shapes: {track_imgs.shape}, {ref_imgs.shape}")
        
        print(f"Labels: {labels}")
        print(f"First metadata: {metas[0]}")
        break
