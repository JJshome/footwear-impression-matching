"""
Loss Functions for Footwear Impression Matching

This module implements specialized loss functions for the footwear impression matching task,
including:
1. Focal Loss for handling class imbalance
2. Contrastive Loss for similarity learning
3. Triplet Loss for embedding learning
4. Combined losses for multi-task learning

These losses are designed to work together to optimize the model for both
classification and metric learning objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces the relative loss for well-classified examples, focusing more on hard, 
    misclassified examples.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Compute focal loss.
        
        Args:
            logits: Raw model outputs (before sigmoid) [batch_size, 1]
            targets: Target values [batch_size]
            
        Returns:
            Focal loss value
        """
        # Flatten and ensure proper shapes
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-bce_loss)
        
        # Compute focal weights
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        # Apply weights to BCE loss
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for similarity learning.
    
    Pushes similar items closer together and dissimilar items further apart
    in the embedding space.
    """
    
    def __init__(self, margin=1.0, reduction='mean'):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Minimum distance for negative pairs
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, similarity, labels):
        """
        Compute contrastive loss.
        
        Args:
            similarity: Similarity scores between pairs [-1, 1]
            labels: Binary labels (1 for similar, 0 for dissimilar)
            
        Returns:
            Contrastive loss value
        """
        # Convert similarity to distance (1 - similarity)
        distance = 1.0 - similarity
        
        # Compute loss for positive and negative pairs
        pos_loss = labels * distance.pow(2)
        neg_loss = (1.0 - labels) * F.relu(self.margin - distance).pow(2)
        
        # Combine losses
        losses = pos_loss + neg_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses


class TripletLoss(nn.Module):
    """
    Triplet Loss for embedding learning.
    
    Minimizes the distance between an anchor and a positive example, while
    maximizing the distance between the anchor and a negative example.
    """
    
    def __init__(self, margin=0.3, reduction='mean', distance='euclidean'):
        """
        Initialize triplet loss.
        
        Args:
            margin: Minimum difference between positive and negative distances
            reduction: Reduction method ('mean', 'sum', 'none')
            distance: Distance metric ('euclidean', 'cosine')
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.distance = distance
    
    def forward(self, anchors, positives, negatives):
        """
        Compute triplet loss.
        
        Args:
            anchors: Anchor embeddings [batch_size, embedding_dim]
            positives: Positive embeddings [batch_size, embedding_dim]
            negatives: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            Triplet loss value
        """
        if self.distance == 'euclidean':
            # Compute Euclidean distances
            pos_dist = F.pairwise_distance(anchors, positives)
            neg_dist = F.pairwise_distance(anchors, negatives)
        elif self.distance == 'cosine':
            # Compute cosine similarity and convert to distance
            pos_sim = F.cosine_similarity(anchors, positives)
            neg_sim = F.cosine_similarity(anchors, negatives)
            pos_dist = 1.0 - pos_sim
            neg_dist = 1.0 - neg_sim
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance}")
        
        # Compute triplet loss with margin
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses


def batch_hard_triplet_loss(embeddings, labels, margin=0.3, distance='euclidean'):
    """
    Batch Hard Triplet Loss with online mining.
    
    For each anchor, mines the hardest positive and hardest negative within the batch.
    
    Args:
        embeddings: Feature embeddings [batch_size, embedding_dim]
        labels: Class labels [batch_size]
        margin: Margin for triplet loss
        distance: Distance metric ('euclidean', 'cosine')
        
    Returns:
        Triplet loss value
    """
    # Get the number of samples
    batch_size = embeddings.size(0)
    
    if distance == 'euclidean':
        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings)
    elif distance == 'cosine':
        # Compute pairwise cosine similarity
        sim_matrix = torch.matmul(F.normalize(embeddings, p=2, dim=1), 
                                  F.normalize(embeddings, p=2, dim=1).t())
        # Convert to distance
        dist_matrix = 1.0 - sim_matrix
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")
    
    # Create a mask for positive pairs (same label)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float()
    neg_mask = (labels != labels.t()).float()
    
    # For each anchor, find the hardest positive
    pos_dist = dist_matrix * pos_mask
    # Replace zeros (from mask) with large value to avoid selecting them
    pos_dist[pos_dist == 0] = 1e9
    # Find the hardest positive
    hardest_pos_dist, _ = torch.min(pos_dist, dim=1)
    
    # For each anchor, find the hardest negative
    neg_dist = dist_matrix * neg_mask
    # Replace zeros (from mask) with large value to avoid selecting them
    neg_dist[neg_dist == 0] = 1e9
    # Find the hardest negative
    hardest_neg_dist, _ = torch.min(neg_dist, dim=1)
    
    # Compute triplet loss
    triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
    
    # Exclude invalid triplets (no positives or negatives)
    valid_mask = (hardest_pos_dist < 1e8) & (hardest_neg_dist < 1e8)
    if valid_mask.sum() > 0:
        triplet_loss = triplet_loss[valid_mask].mean()
    else:
        triplet_loss = torch.tensor(0.0, device=embeddings.device)
    
    return triplet_loss


class EnhancedContrastiveLoss(nn.Module):
    """
    Enhanced Contrastive Loss combining multiple loss functions.
    
    Combines classification loss, contrastive loss, and triplet loss
    for multi-task learning.
    """
    
    def __init__(self, margin=0.5, alpha=0.5, beta=0.3, gamma=0.2, 
                 focal_alpha=0.25, focal_gamma=2.0, use_focal=True):
        """
        Initialize enhanced contrastive loss.
        
        Args:
            margin: Margin for contrastive and triplet losses
            alpha: Weight for contrastive loss
            beta: Weight for triplet loss
            gamma: Weight for L2 regularization
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            use_focal: Whether to use focal loss for classification
        """
        super(EnhancedContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Classification loss
        if use_focal:
            self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.cls_loss = nn.BCEWithLogitsLoss()
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(margin=margin)
        
        # Triplet loss
        self.triplet_loss = TripletLoss(margin=margin)
    
    def forward(self, logits, similarity, labels, track_features=None, ref_features=None,
               pos_features=None, neg_features=None):
        """
        Compute combined loss.
        
        Args:
            logits: Classification logits [batch_size, 1]
            similarity: Similarity scores [batch_size]
            labels: Binary labels [batch_size]
            track_features: Track features for regularization (optional)
            ref_features: Reference features for regularization (optional)
            pos_features: Positive features for triplet loss (optional)
            neg_features: Negative features for triplet loss (optional)
            
        Returns:
            Combined loss value
        """
        # Basic classification loss
        cls_loss = self.cls_loss(logits.squeeze(), labels.float())
        
        # Convert to binary masks
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        # Start with classification loss
        total_loss = cls_loss
        
        # Add contrastive loss if we have positive and negative samples
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            # Contrastive loss
            contrastive_loss = self.contrastive_loss(similarity, labels.float())
            total_loss = total_loss + self.alpha * contrastive_loss
            
            # Triplet loss with batch hard mining (if features are provided)
            if track_features is not None and ref_features is not None and self.beta > 0:
                # Global average pooling to get embeddings
                track_embeddings = F.adaptive_avg_pool2d(track_features, 1).squeeze(-1).squeeze(-1)
                ref_embeddings = F.adaptive_avg_pool2d(ref_features, 1).squeeze(-1).squeeze(-1)
                
                # Combine track and reference embeddings for batch hard triplet mining
                all_embeddings = torch.cat([track_embeddings, ref_embeddings], dim=0)
                all_labels = torch.cat([labels, labels], dim=0)
                
                # Compute batch hard triplet loss
                triplet_loss = batch_hard_triplet_loss(all_embeddings, all_labels, self.margin)
                total_loss = total_loss + self.beta * triplet_loss
            
            # Direct triplet loss if positive and negative samples are provided
            if pos_features is not None and neg_features is not None and self.beta > 0:
                # Create embeddings
                anchor_embeddings = F.adaptive_avg_pool2d(track_features, 1).squeeze(-1).squeeze(-1)
                positive_embeddings = F.adaptive_avg_pool2d(pos_features, 1).squeeze(-1).squeeze(-1)
                negative_embeddings = F.adaptive_avg_pool2d(neg_features, 1).squeeze(-1).squeeze(-1)
                
                # Compute triplet loss
                direct_triplet_loss = self.triplet_loss(
                    anchor_embeddings, positive_embeddings, negative_embeddings)
                total_loss = total_loss + self.beta * direct_triplet_loss
            
            # L2 regularization on features
            if self.gamma > 0 and track_features is not None and ref_features is not None:
                l2_reg = (track_features.pow(2).mean() + ref_features.pow(2).mean()) / 2
                total_loss = total_loss + self.gamma * l2_reg
            
        return total_loss


# For testing
if __name__ == "__main__":
    # Test focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8,)).float()
    loss = focal_loss(logits, targets)
    print(f"Focal Loss: {loss.item()}")
    
    # Test contrastive loss
    contrastive_loss = ContrastiveLoss(margin=0.5)
    similarity = torch.rand(8)  # Similarity scores between 0 and 1
    labels = torch.randint(0, 2, (8,)).float()
    loss = contrastive_loss(similarity, labels)
    print(f"Contrastive Loss: {loss.item()}")
    
    # Test triplet loss
    triplet_loss = TripletLoss(margin=0.3)
    anchors = torch.randn(8, 128)
    positives = torch.randn(8, 128)
    negatives = torch.randn(8, 128)
    loss = triplet_loss(anchors, positives, negatives)
    print(f"Triplet Loss: {loss.item()}")
    
    # Test batch hard triplet loss
    embeddings = torch.randn(16, 128)
    labels = torch.randint(0, 4, (16,))  # 4 classes
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.3)
    print(f"Batch Hard Triplet Loss: {loss.item()}")
    
    # Test enhanced contrastive loss
    enhanced_loss = EnhancedContrastiveLoss(
        margin=0.5, alpha=0.5, beta=0.3, gamma=0.2,
        focal_alpha=0.25, focal_gamma=2.0, use_focal=True
    )
    logits = torch.randn(8, 1)
    similarity = torch.rand(8)
    labels = torch.randint(0, 2, (8,))
    track_features = torch.randn(8, 256, 16, 16)
    ref_features = torch.randn(8, 256, 16, 16)
    loss = enhanced_loss(logits, similarity, labels, track_features, ref_features)
    print(f"Enhanced Contrastive Loss: {loss.item()}")
