"""
Inference and Evaluation Script for Footwear Impression Matching

This script handles:
1. Loading a trained model
2. Running inference on test data
3. Calculating evaluation metrics
4. Visualizing results

It supports individual image matching as well as database retrieval evaluation.

Usage:
    # Evaluate on test set
    python inference.py --model path/to/model.pth --test_csv data/test_pairs.csv
    
    # Match a single impression to reference database
    python inference.py --model path/to/model.pth --track_img path/to/track.jpg --ref_dir path/to/references
"""

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import cv2
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

# Import project modules
from data.dataloader import get_transforms, ShoeImpressDataset
from models.network import FootwearMatchingNetwork, EnhancedSiameseNetwork, FootwearRetrievalNetwork
from utils.common import calculate_metrics, plot_precision_recall_curve, plot_roc_curve


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inference and evaluation for footwear impression matching')
    
    # Model parameters
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model if config not provided')
    parser.add_argument('--feature_dim', type=int, default=256, help='Feature dimension if config not provided')
    
    # Inference mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test_csv', type=str, help='Path to test pairs CSV for evaluation')
    group.add_argument('--track_img', type=str, help='Path to track image for single match')
    
    # Additional parameters
    parser.add_argument('--ref_dir', type=str, default=None, help='Directory with reference images (for single match)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Output directory')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top matches to display')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def load_model(model_path, config_path=None, backbone='resnet50', feature_dim=256, device='cuda'):
    """
    Load a trained model.
    
    Args:
        model_path: Path to model weights
        config_path: Path to model configuration (optional)
        backbone: Backbone architecture if config not provided
        feature_dim: Feature dimension if config not provided
        device: Device to load model onto
        
    Returns:
        Loaded model
    """
    # Load configuration if provided
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config['model']
        backbone = model_config['backbone']
        feature_dim = model_config['feature_dim']
        model_name = model_config.get('name', 'FootwearMatchingNetwork')
    else:
        model_name = 'FootwearMatchingNetwork'
    
    # Create model
    print(f"Creating model {model_name} with {backbone} backbone and feature_dim={feature_dim}")
    if model_name == 'FootwearMatchingNetwork':
        model = FootwearMatchingNetwork(
            backbone=backbone,
            pretrained=False,
            feature_dim=feature_dim
        )
    elif model_name == 'EnhancedSiameseNetwork':
        model = EnhancedSiameseNetwork(
            backbone=backbone,
            pretrained=False,
            feature_dim=feature_dim
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load weights
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_on_test_set(model, test_csv, batch_size=32, img_size=512, output_dir='./inference_results', device='cuda'):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_csv: Path to test pairs CSV
        batch_size: Batch size for evaluation
        img_size: Image size
        output_dir: Output directory
        device: Device to use
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup transforms
    transform = get_transforms(mode='val', img_size=img_size)
    
    # Create dataset
    test_dataset = ShoeImpressDataset(
        test_csv,
        transform=transform,
        triplet_mode=False,
        online_augment=False
    )
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Evaluating on {len(test_dataset)} test pairs")
    
    # Collect all outputs and targets
    all_logits = []
    all_similarities = []
    all_labels = []
    
    # Evaluate
    with torch.no_grad():
        for batch_idx, (track_imgs, ref_imgs, labels, _) in enumerate(tqdm(test_loader)):
            # Move to device
            track_imgs = track_imgs.to(device)
            ref_imgs = ref_imgs.to(device)
            
            # Forward pass
            logits, similarities, _, _ = model(track_imgs, ref_imgs)
            
            # Collect outputs
            all_logits.append(logits.cpu().squeeze())
            all_similarities.append(similarities.cpu())
            all_labels.append(labels)
    
    # Concatenate all outputs and targets
    all_logits = torch.cat(all_logits).numpy()
    all_similarities = torch.cat(all_similarities).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(all_logits, all_similarities, all_labels)
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average Precision: {metrics['ap']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"AP from Similarity: {metrics['sim_ap']:.4f}")
    
    # Plot curves
    plot_precision_recall_curve(
        metrics['precision'], metrics['recall'], metrics['ap'],
        os.path.join(output_dir, "pr_curve.png"),
        title="Precision-Recall Curve"
    )
    plot_roc_curve(
        metrics['fpr'], metrics['tpr'], metrics['roc_auc'],
        os.path.join(output_dir, "roc_curve.png"),
        title="ROC Curve"
    )
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.csv")
    metrics_df = pd.DataFrame({
        'accuracy': [metrics['accuracy']],
        'ap': [metrics['ap']],
        'roc_auc': [metrics['roc_auc']],
        'sim_ap': [metrics['sim_ap']]
    })
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    # Analysis of failure cases
    analyze_failure_cases(all_logits, all_similarities, all_labels, test_dataset, output_dir)


def analyze_failure_cases(logits, similarities, labels, dataset, output_dir):
    """
    Analyze and visualize failure cases.
    
    Args:
        logits: Model logits
        similarities: Similarity scores
        labels: Ground truth labels
        dataset: Test dataset
        output_dir: Output directory
    """
    # Create directory for failure cases
    failure_dir = os.path.join(output_dir, "failure_cases")
    os.makedirs(failure_dir, exist_ok=True)
    
    # Get probabilities
    probas = 1 / (1 + np.exp(-logits))  # Sigmoid
    predictions = (probas > 0.5).astype(int)
    
    # Find failure cases
    failure_indices = np.where(predictions != labels)[0]
    
    print(f"\nFound {len(failure_indices)} failure cases out of {len(labels)} samples ({len(failure_indices)/len(labels)*100:.2f}%)")
    
    # Analyze false positives and false negatives
    false_positives = [(i, probas[i]) for i in failure_indices if predictions[i] == 1 and labels[i] == 0]
    false_negatives = [(i, probas[i]) for i in failure_indices if predictions[i] == 0 and labels[i] == 1]
    
    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")
    
    # Sort by confidence
    false_positives.sort(key=lambda x: x[1], reverse=True)
    false_negatives.sort(key=lambda x: x[1])
    
    # Visualize top failure cases
    num_to_visualize = min(10, len(failure_indices))
    
    # False positives
    if false_positives:
        visualize_failure_cases(
            false_positives[:num_to_visualize],
            dataset,
            os.path.join(failure_dir, "false_positives.png"),
            "False Positives (Predicted Match, Actually Different)"
        )
    
    # False negatives
    if false_negatives:
        visualize_failure_cases(
            false_negatives[:num_to_visualize],
            dataset,
            os.path.join(failure_dir, "false_negatives.png"),
            "False Negatives (Predicted Different, Actually Match)"
        )


def visualize_failure_cases(failure_cases, dataset, output_path, title):
    """
    Create visualization of failure cases.
    
    Args:
        failure_cases: List of (index, confidence) tuples
        dataset: Test dataset
        output_path: Path to save visualization
        title: Plot title
    """
    n = len(failure_cases)
    plt.figure(figsize=(15, 4 * n))
    plt.suptitle(title, fontsize=16)
    
    for i, (idx, conf) in enumerate(failure_cases):
        # Get images and metadata
        track_img, ref_img, label, meta = dataset[idx]
        
        # Convert from tensor to numpy
        if isinstance(track_img, torch.Tensor):
            track_img = track_img.permute(1, 2, 0).numpy()
            # Denormalize
            track_img = (track_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            track_img = track_img.astype(np.uint8)
        
        if isinstance(ref_img, torch.Tensor):
            ref_img = ref_img.permute(1, 2, 0).numpy()
            # Denormalize
            ref_img = (ref_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            ref_img = ref_img.astype(np.uint8)
        
        # Track image
        plt.subplot(n, 3, i*3 + 1)
        plt.imshow(track_img)
        plt.title(f"Track {meta['track_id']}")
        plt.axis('off')
        
        # Reference image
        plt.subplot(n, 3, i*3 + 2)
        plt.imshow(ref_img)
        plt.title(f"Reference {meta['ref_id']}")
        plt.axis('off')
        
        # Prediction details
        plt.subplot(n, 3, i*3 + 3)
        plt.text(0.5, 0.5, 
                f"True Label: {label}\n"
                f"Confidence: {conf:.4f}",
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")


def match_single_impression(model, track_img_path, ref_dir, img_size=512, output_dir='./inference_results', top_k=5, device='cuda'):
    """
    Match a single impression against a reference database.
    
    Args:
        model: Trained model
        track_img_path: Path to track image
        ref_dir: Directory with reference images
        img_size: Image size
        output_dir: Output directory
        top_k: Number of top matches to display
        device: Device to use
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup transforms
    transform = get_transforms(mode='val', img_size=img_size)
    
    # Load track image
    track_img = Image.open(track_img_path).convert('RGB')
    track_tensor = transform(track_img).unsqueeze(0).to(device)
    
    # Find all reference images
    ref_paths = []
    ref_ids = []
    for filename in os.listdir(ref_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            ref_path = os.path.join(ref_dir, filename)
            ref_id = os.path.splitext(filename)[0]
            ref_paths.append(ref_path)
            ref_ids.append(ref_id)
    
    print(f"Matching track image against {len(ref_paths)} reference images")
    
    # Process in batches
    batch_size = 32
    similarities = []
    
    with torch.no_grad():
        # Extract track features once
        track_features = model(track_tensor, None, mode='track')
        
        # Process reference images in batches
        for i in range(0, len(ref_paths), batch_size):
            batch_paths = ref_paths[i:i+batch_size]
            
            # Load and preprocess reference images
            ref_tensors = []
            for ref_path in batch_paths:
                ref_img = Image.open(ref_path).convert('RGB')
                ref_tensor = transform(ref_img)
                ref_tensors.append(ref_tensor)
            
            # Stack tensors and move to device
            ref_batch = torch.stack(ref_tensors).to(device)
            
            # Extract reference features
            ref_features = model(None, ref_batch, mode='ref')
            
            # Compute similarities
            for j in range(len(batch_paths)):
                # Extract single reference features
                single_ref_features = ref_features[j:j+1]
                
                # Compute similarity
                similarity = model.compute_mcncc(track_features, single_ref_features)
                
                # Store result
                similarities.append((ref_ids[i+j], similarity.item()))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Display top matches
    top_matches = similarities[:top_k]
    
    print("\nTop Matches:")
    for i, (ref_id, score) in enumerate(top_matches):
        print(f"{i+1}. Reference ID: {ref_id}, Similarity: {score:.4f}")
    
    # Create visualization
    visualize_matches(track_img_path, ref_dir, top_matches, output_dir)


def visualize_matches(track_img_path, ref_dir, matches, output_dir):
    """
    Visualize top matches.
    
    Args:
        track_img_path: Path to track image
        ref_dir: Directory with reference images
        matches: List of (ref_id, score) tuples
        output_dir: Output directory
    """
    n = len(matches)
    plt.figure(figsize=(15, 4 * (n+1)))
    plt.suptitle("Track Image and Top Matches", fontsize=16)
    
    # Load track image
    track_img = Image.open(track_img_path).convert('RGB')
    
    # Display track image
    plt.subplot(n+1, 2, 1)
    plt.imshow(track_img)
    plt.title("Query Track Image")
    plt.axis('off')
    
    # Display info
    plt.subplot(n+1, 2, 2)
    plt.text(0.5, 0.5, 
            f"File: {os.path.basename(track_img_path)}\n"
            f"Top {n} matches shown",
            ha='center', va='center', fontsize=12)
    plt.axis('off')
    
    # Display matches
    for i, (ref_id, score) in enumerate(matches):
        # Load reference image
        ref_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            path = os.path.join(ref_dir, f"{ref_id}{ext}")
            if os.path.exists(path):
                ref_path = path
                break
        
        if ref_path:
            ref_img = Image.open(ref_path).convert('RGB')
            
            plt.subplot(n+1, 2, 3 + i*2)
            plt.imshow(ref_img)
            plt.title(f"Match #{i+1}: Reference {ref_id}")
            plt.axis('off')
            
            plt.subplot(n+1, 2, 4 + i*2)
            plt.text(0.5, 0.5, 
                    f"Similarity Score: {score:.4f}",
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "top_matches.png"))
    plt.close()
    print(f"Visualization saved to {os.path.join(output_dir, 'top_matches.png')}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(
        args.model,
        args.config,
        args.backbone,
        args.feature_dim,
        device
    )
    
    # Run in appropriate mode
    if args.test_csv:
        evaluate_on_test_set(
            model,
            args.test_csv,
            args.batch_size,
            args.img_size,
            args.output_dir,
            device
        )
    elif args.track_img:
        if not args.ref_dir:
            raise ValueError("--ref_dir must be specified when using --track_img")
        
        match_single_impression(
            model,
            args.track_img,
            args.ref_dir,
            args.img_size,
            args.output_dir,
            args.top_k,
            device
        )


if __name__ == '__main__':
    main()
