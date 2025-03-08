"""
FID-300 Dataset Processing Module

This script processes the FID-300 dataset for footwear impression matching.
It handles:
1. Image preprocessing (background removal, normalization)
2. Dataset organization
3. Train/validation splitting

Usage:
    python process_dataset.py --data_dir /path/to/fid300 --output_dir /path/to/output --split_ratio 0.8
"""

import os
import cv2
import numpy as np
import pandas as pd
import argparse
import random
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process FID-300 dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to FID-300 dataset')
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                        help='Output directory for processed data')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='Train/validation split ratio')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Output image size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def remove_background(img, is_probe=False, threshold=230):
    """
    Remove background from footwear impression image.
    
    Args:
        img: Input image
        is_probe: Whether the image is a probe (crime scene impression)
        threshold: Threshold for binarization
        
    Returns:
        Processed image with white background
    """
    # Ensure image is loaded
    if img is None:
        raise ValueError("Image could not be loaded")
    
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Preprocessing for better segmentation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarization with Otsu's method
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean noise
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Additional processing for probe images (which may have more noise)
    if is_probe:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.medianBlur(binary, 5)
    
    # Create output image with white background
    result = np.ones_like(img) * 255
    
    # Apply the binary mask to keep the footwear impression
    if len(img.shape) == 3:
        # For color images
        binary_3ch = cv2.merge([binary, binary, binary])
        result = np.where(binary_3ch > 0, img, result)
    else:
        # For grayscale images
        result = np.where(binary > 0, img, result)
    
    return result

def preprocess_image(image, target_size=512):
    """
    Preprocess image: resize with white padding to maintain aspect ratio.
    
    Args:
        image: Input image
        target_size: Target size for the output image
        
    Returns:
        Preprocessed image
    """
    if image is None:
        return None
    
    # Get dimensions
    h, w = image.shape[:2]
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a white canvas of target size
    if len(image.shape) == 3:
        result = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    else:
        result = np.ones((target_size, target_size), dtype=np.uint8) * 255
    
    # Calculate offsets to center the image
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    # Place the resized image on the white canvas
    if len(image.shape) == 3:
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized
    else:
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result

def process_reference_images(ref_dir, output_dir, target_size=512):
    """
    Process reference footwear impressions.
    
    Args:
        ref_dir: Directory containing reference images
        output_dir: Output directory for processed images
        target_size: Target image size
        
    Returns:
        Dictionary mapping reference IDs to processed image paths
    """
    processed_refs = {}
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in tqdm(image_files, desc="Processing reference impressions"):
        ref_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(ref_dir, img_name)
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            # Process the image
            processed_img = remove_background(img, is_probe=False)
            processed_img = preprocess_image(processed_img, target_size)
            
            # Save processed image
            output_path = os.path.join(output_dir, f"{ref_id}.png")
            cv2.imwrite(output_path, processed_img)
            
            processed_refs[ref_id] = output_path
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Processed {len(processed_refs)} reference impressions")
    return processed_refs

def process_probe_images(probes_dir, output_dir, target_size=512):
    """
    Process probe (crime scene) footwear impressions.
    
    Args:
        probes_dir: Directory containing probe images
        output_dir: Output directory for processed images
        target_size: Target image size
        
    Returns:
        Dictionary mapping probe IDs to processed image paths
    """
    processed_probes = {}
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(probes_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in tqdm(image_files, desc="Processing probe impressions"):
        probe_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(probes_dir, img_name)
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            # Process the image (probes need more aggressive processing)
            processed_img = remove_background(img, is_probe=True)
            processed_img = preprocess_image(processed_img, target_size)
            
            # Save processed image
            output_path = os.path.join(output_dir, f"{probe_id}.png")
            cv2.imwrite(output_path, processed_img)
            
            processed_probes[probe_id] = output_path
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Processed {len(processed_probes)} probe impressions")
    return processed_probes

def create_pairs(probes, references, label_data, output_dir):
    """
    Create positive and negative pairs for training.
    
    Args:
        probes: Dictionary mapping probe IDs to image paths
        references: Dictionary mapping reference IDs to image paths
        label_data: DataFrame containing label mappings
        output_dir: Output directory for pair files
        
    Returns:
        DataFrame containing pair information
    """
    pairs = []
    
    # Extract probe-to-reference mapping from label data
    probe_to_ref = {}
    for _, row in label_data.iterrows():
        # Assuming label_data has 'probe_id' and 'reference_id' columns
        probe_id = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else None
        ref_id = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else None
        
        if probe_id and ref_id:
            probe_to_ref[probe_id] = ref_id
    
    # Create positive pairs (matching impressions)
    for probe_id, probe_path in probes.items():
        if probe_id in probe_to_ref:
            ref_id = probe_to_ref[probe_id]
            if ref_id in references:
                pairs.append({
                    'track_id': probe_id,
                    'ref_id': ref_id,
                    'track_path': probe_path,
                    'ref_path': references[ref_id],
                    'label': 1  # Positive match
                })
    
    # Create negative pairs (non-matching impressions)
    all_ref_ids = list(references.keys())
    for pair in pairs.copy():  # Use copy to avoid modifying while iterating
        probe_id = pair['track_id']
        correct_ref = pair['ref_id']
        
        # Create 5 negative samples for each positive sample
        neg_sample_count = min(5, len(all_ref_ids) - 1)
        negative_refs = random.sample([r for r in all_ref_ids if r != correct_ref], neg_sample_count)
        
        for neg_ref in negative_refs:
            pairs.append({
                'track_id': probe_id,
                'ref_id': neg_ref,
                'track_path': probes[probe_id],
                'ref_path': references[neg_ref],
                'label': 0  # Negative match
            })
    
    # Create DataFrame from pairs
    pairs_df = pd.DataFrame(pairs)
    
    # Save pairs information
    pairs_df.to_csv(os.path.join(output_dir, 'all_pairs.csv'), index=False)
    
    print(f"Created {len(pairs_df)} pairs ({pairs_df['label'].sum()} positive, {len(pairs_df) - pairs_df['label'].sum()} negative)")
    return pairs_df

def split_train_val(pairs_df, output_dir, split_ratio=0.8, stratify_by_track=True):
    """
    Split pairs into training and validation sets.
    
    Args:
        pairs_df: DataFrame containing pair information
        output_dir: Output directory for split files
        split_ratio: Train/validation split ratio
        stratify_by_track: Whether to stratify split by track ID
        
    Returns:
        Training and validation DataFrames
    """
    if stratify_by_track:
        # Get unique track IDs
        unique_tracks = pairs_df['track_id'].unique()
        
        # Split track IDs
        train_tracks, val_tracks = train_test_split(
            unique_tracks, 
            train_size=split_ratio,
            random_state=42
        )
        
        # Split pairs based on track IDs
        train_df = pairs_df[pairs_df['track_id'].isin(train_tracks)]
        val_df = pairs_df[pairs_df['track_id'].isin(val_tracks)]
    else:
        # Split directly on pairs (not recommended for this task)
        train_df, val_df = train_test_split(
            pairs_df,
            train_size=split_ratio,
            stratify=pairs_df['label'],
            random_state=42
        )
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train_pairs.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_pairs.csv'), index=False)
    
    # Print statistics
    print(f"Train set: {len(train_df)} pairs ({train_df['label'].sum()} positive, {len(train_df) - train_df['label'].sum()} negative)")
    print(f"Validation set: {len(val_df)} pairs ({val_df['label'].sum()} positive, {len(val_df) - val_df['label'].sum()} negative)")
    
    return train_df, val_df

def main():
    """Main function to process the FID-300 dataset."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    refs_output_dir = os.path.join(args.output_dir, 'references')
    probes_output_dir = os.path.join(args.output_dir, 'tracks')
    
    # Process reference impressions
    print("\nProcessing reference impressions...")
    references_dir = os.path.join(args.data_dir, 'references')
    processed_refs = process_reference_images(references_dir, refs_output_dir, args.img_size)
    
    # Process probe impressions
    print("\nProcessing probe impressions...")
    probes_dir = os.path.join(args.data_dir, 'probes_cropped')  # Using pre-cropped probes
    processed_probes = process_probe_images(probes_dir, probes_output_dir, args.img_size)
    
    # Load label information
    print("\nLoading label information...")
    label_path = os.path.join(args.data_dir, 'label_table.csv')
    label_data = pd.read_csv(label_path, header=None)
    
    # Create pairs
    print("\nCreating impression pairs...")
    pairs_df = create_pairs(processed_probes, processed_refs, label_data, args.output_dir)
    
    # Split into train/validation sets
    print("\nSplitting into train/validation sets...")
    train_df, val_df = split_train_val(pairs_df, args.output_dir, args.split_ratio)
    
    print(f"\nDataset processing complete! Processed data saved to {args.output_dir}")

if __name__ == '__main__':
    main()
