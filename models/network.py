"""
Main Network Architecture for Footwear Impression Matching

This module implements the core network architecture for matching footwear impressions
from crime scenes to reference database entries. The model uses a Siamese-style
architecture with domain-specific feature extraction and specialized correlation mechanisms.

Key features:
- CNN backbone with pre-trained weights
- Attention mechanisms for focusing on relevant patterns
- Domain-specific feature projections for track and reference impressions
- Multi-channel normalized cross-correlation for robust pattern matching
- Multi-task learning with classification and metric learning objectives

Usage:
    from models.network import FootwearMatchingNetwork
    model = FootwearMatchingNetwork(backbone='resnet50')
    output = model(track_image, reference_image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .modules import (
    CBAM, SEBlock, DomainProjection, FeatureCorrelation, 
    AdaptiveFeatureFusion, SelfAttention
)


class FootwearMatchingNetwork(nn.Module):
    """
    Footwear Impression Matching Network
    
    A Siamese-based network for matching footwear impressions from crime scenes
    to reference impressions in a database.
    """
    
    def __init__(
        self, 
        backbone='resnet50', 
        pretrained=True, 
        feature_dim=256, 
        dropout_rate=0.4,
        use_attention=True,
        use_self_attention=True,
        correlation_temp=10.0
    ):
        """
        Initialize the network.
        
        Args:
            backbone: Backbone CNN architecture ('resnet18', 'resnet50', 'resnext50_32x4d')
            pretrained: Whether to use pre-trained weights
            feature_dim: Output feature dimension
            dropout_rate: Dropout rate for regularization
            use_attention: Whether to use attention mechanisms
            use_self_attention: Whether to use self-attention
            correlation_temp: Temperature parameter for correlation scaling
        """
        super(FootwearMatchingNetwork, self).__init__()
        
        # Save configuration (for model reconstruction)
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_self_attention = use_self_attention
        
        # Load backbone network
        self.in_features, self.feature_extractor = self._get_backbone(backbone, pretrained)
        
        # Attention modules
        if use_attention:
            self.cbam = CBAM(self.in_features)
            self.se_block = SEBlock(self.in_features)
        
        # Domain-specific projection layers
        self.track_projection = DomainProjection(self.in_features, feature_dim, dropout_rate=dropout_rate/2)
        self.ref_projection = DomainProjection(self.in_features, feature_dim, dropout_rate=dropout_rate/2)
        
        # Feature normalization
        self.bn_track = nn.BatchNorm2d(feature_dim)
        self.bn_ref = nn.BatchNorm2d(feature_dim)
        
        # Self-attention for feature enhancement (optional)
        if use_self_attention:
            self.track_self_attn = SelfAttention(feature_dim)
            self.ref_self_attn = SelfAttention(feature_dim)
        
        # Feature correlation module
        self.correlation = FeatureCorrelation(feature_dim, learn_weights=True, temperature=correlation_temp)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 512),  # +1 for correlation score
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, 1)
        )
    
    def _get_backbone(self, backbone_name, pretrained):
        """
        Load and configure the backbone feature extractor.
        
        Args:
            backbone_name: Name of the backbone architecture
            pretrained: Whether to use pre-trained weights
            
        Returns:
            in_features: Number of output features from the backbone
            feature_extractor: The backbone module
        """
        if backbone_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            layers = list(base_model.children())[:-2]  # Remove average pooling and FC
            in_features = 2048
        elif backbone_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            layers = list(base_model.children())[:-2]
            in_features = 512
        elif backbone_name == 'resnext50_32x4d':
            base_model = models.resnext50_32x4d(pretrained=pretrained)
            layers = list(base_model.children())[:-2]
            in_features = 2048
        elif backbone_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            layers = list(base_model.children())[:-1]  # Keep features, remove classifier
            in_features = 1280
        elif backbone_name == 'densenet121':
            base_model = models.densenet121(pretrained=pretrained)
            layers = list(base_model.children())[:-1]
            in_features = 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Create sequential module
        feature_extractor = nn.Sequential(*layers)
        
        return in_features, feature_extractor
    
    def forward_one(self, x, domain='track'):
        """
        Process a single image through the network.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            domain: Domain type ('track' or 'ref')
            
        Returns:
            features: Extracted features
        """
        if x is None:
            return None
        
        # Extract features through backbone
        features = self.feature_extractor(x)
        
        # Apply attention (if enabled)
        if self.use_attention:
            # Combine CBAM and SE attention
            features_cbam = self.cbam(features)
            features_se = self.se_block(features)
            features = 0.5 * features_cbam + 0.5 * features_se
        
        # Apply domain-specific projection
        if domain == 'track':
            features = self.track_projection(features)
            features = self.bn_track(features)
            
            # Apply self-attention for track domain (if enabled)
            if self.use_self_attention:
                features = self.track_self_attn(features)
        else:
            features = self.ref_projection(features)
            features = self.bn_ref(features)
            
            # Apply self-attention for reference domain (if enabled)
            if self.use_self_attention:
                features = self.ref_self_attn(features)
        
        return features
    
    def forward(self, track_img, ref_img, mode='full'):
        """
        Forward pass through the network.
        
        Args:
            track_img: Track (crime scene) image tensor
            ref_img: Reference image tensor
            mode: Operation mode:
                  - 'full': Complete forward pass
                  - 'track': Process only track image
                  - 'ref': Process only reference image
                  - 'features': Return features only
                  - 'similarity': Return only similarity score
            
        Returns:
            Based on mode, different outputs:
            - 'full': (logits, similarity, track_features, ref_features)
            - 'track' or 'ref': Corresponding features
            - 'features': (track_features, ref_features)
            - 'similarity': similarity score only
        """
        # Process individual images based on mode
        if mode == 'track':
            return self.forward_one(track_img, domain='track')
        
        if mode == 'ref':
            return self.forward_one(ref_img, domain='ref')
        
        # Extract features for both images
        track_features = self.forward_one(track_img, domain='track')
        ref_features = self.forward_one(ref_img, domain='ref')
        
        if mode == 'features':
            return track_features, ref_features
        
        # Compute correlation-based similarity
        similarity = self.correlation(track_features, ref_features)
        
        if mode == 'similarity':
            return similarity
        
        # Global pooling to get feature vectors
        # Combine average and max pooling for better representation
        track_avg_pool = self.global_avg_pool(track_features).squeeze(-1).squeeze(-1)
        track_max_pool = self.global_max_pool(track_features).squeeze(-1).squeeze(-1)
        track_vector = 0.5 * (track_avg_pool + track_max_pool)
        
        ref_avg_pool = self.global_avg_pool(ref_features).squeeze(-1).squeeze(-1)
        ref_max_pool = self.global_max_pool(ref_features).squeeze(-1).squeeze(-1)
        ref_vector = 0.5 * (ref_avg_pool + ref_max_pool)
        
        # Combine features with similarity score
        similarity_expanded = similarity.unsqueeze(1)  # Add channel dimension
        combined = torch.cat([track_vector, ref_vector, similarity_expanded], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        # Return full output
        return logits, similarity, track_features, ref_features


class EnhancedSiameseNetwork(nn.Module):
    """
    Enhanced Siamese Network for Footwear Impression Matching
    
    An alternative implementation with additional features for research experimentation.
    This version includes more options and flexibility.
    """
    
    def __init__(
        self, 
        backbone='resnet50', 
        pretrained=True, 
        feature_dim=256, 
        dropout_rate=0.4,
        use_cbam=True,
        use_se=True,
        use_self_attention=True
    ):
        """
        Initialize the enhanced network.
        
        Args:
            backbone: Backbone architecture
            pretrained: Whether to use pre-trained weights
            feature_dim: Feature dimension
            dropout_rate: Dropout rate
            use_cbam: Whether to use CBAM attention
            use_se: Whether to use Squeeze-and-Excitation blocks
            use_self_attention: Whether to use self-attention
        """
        super(EnhancedSiameseNetwork, self).__init__()
        
        # Save configuration
        self.backbone = backbone
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        
        # Load backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            backbone_layers = list(base_model.children())[:-2]
            self.feature_extractor = nn.Sequential(*backbone_layers)
            in_features = 2048
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            backbone_layers = list(base_model.children())[:-2]
            self.feature_extractor = nn.Sequential(*backbone_layers)
            in_features = 512
        elif backbone == 'resnext50_32x4d':
            base_model = models.resnext50_32x4d(pretrained=pretrained)
            backbone_layers = list(base_model.children())[:-2]
            self.feature_extractor = nn.Sequential(*backbone_layers)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Attention modules
        if use_cbam:
            self.cbam = CBAM(in_features)
        else:
            self.cbam = None
            
        if use_se:
            self.se_block = SEBlock(in_features)
        else:
            self.se_block = None
        
        # Domain projection
        self.track_projection = DomainProjection(in_features, feature_dim, dropout_rate=dropout_rate/2)
        self.ref_projection = DomainProjection(in_features, feature_dim, dropout_rate=dropout_rate/2)
        
        # Normalization
        self.bn_track = nn.BatchNorm2d(feature_dim)
        self.bn_ref = nn.BatchNorm2d(feature_dim)
        
        # Self-attention
        if use_self_attention:
            self.self_attention = SelfAttention(feature_dim)
        else:
            self.self_attention = None
        
        # Channel weights for MCNCC
        self.channel_weights = nn.Parameter(torch.ones(feature_dim))
        
        # Layer normalization
        feature_map_size = 16  # Assuming 512x512 input, feature map is 16x16
        self.layer_norm = nn.LayerNorm([feature_dim, feature_map_size, feature_map_size])
        
        # Temperature scaling
        self.temperature = nn.Parameter(torch.tensor(10.0))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 512),  # +1 for correlation score
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, 1)
        )
    
    def forward_one(self, x, domain='track'):
        """Extract features from one image."""
        if x is None:
            return None
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        if self.cbam is not None:
            features_cbam = self.cbam(features)
            if self.se_block is not None:
                features_se = self.se_block(features)
                features = 0.5 * features_cbam + 0.5 * features_se
            else:
                features = features_cbam
        elif self.se_block is not None:
            features = self.se_block(features)
        
        # Domain projection
        if domain == 'track':
            features = self.track_projection(features)
            features = self.bn_track(features)
        else:
            features = self.ref_projection(features)
            features = self.bn_ref(features)
        
        # Apply self-attention if enabled
        if self.self_attention is not None:
            features = self.self_attention(features)
        
        return features
    
    def compute_mcncc(self, x, y):
        """
        Compute Multi-Channel Normalized Cross-Correlation.
        
        Args:
            x: First feature map [batch_size, channels, height, width]
            y: Second feature map [batch_size, channels, height, width]
            
        Returns:
            Similarity scores [batch_size]
        """
        # L2 normalize along channel dimension
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        
        # Compute correlation for each spatial position and average
        correlation = (x_norm * y_norm).sum(dim=[2, 3]) / (x.size(2) * x.size(3))
        
        # Apply channel weights with softmax normalization
        normalized_weights = F.softmax(self.channel_weights, dim=0)
        weighted_corr = (correlation * normalized_weights).sum(dim=1)
        
        # Apply temperature scaling
        scaled_similarity = weighted_corr * self.temperature
        
        return scaled_similarity
    
    def forward(self, track_img, ref_img, mode='full'):
        """Network forward pass."""
        if mode == 'track':
            return self.forward_one(track_img, domain='track')
        
        if mode == 'ref':
            return self.forward_one(ref_img, domain='ref')
        
        # Extract features
        track_features = self.forward_one(track_img, domain='track')
        ref_features = self.forward_one(ref_img, domain='ref')
        
        # Compute similarity
        similarity = self.compute_mcncc(track_features, ref_features)
        
        if mode == 'mcncc_only':
            return similarity
        
        # Global average pooling
        track_vector = F.adaptive_avg_pool2d(track_features, 1).squeeze(-1).squeeze(-1)
        ref_vector = F.adaptive_avg_pool2d(ref_features, 1).squeeze(-1).squeeze(-1)
        
        # Combine features with similarity score
        similarity_expanded = similarity.unsqueeze(1)
        combined = torch.cat([track_vector, ref_vector, similarity_expanded], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits, similarity, track_features, ref_features


# Retrieval model for database search
class FootwearRetrievalNetwork(nn.Module):
    """
    Footwear Impression Retrieval Network
    
    A specialized version of the network for efficient database retrieval.
    Optimized for embedding extraction and fast similarity computation.
    """
    
    def __init__(
        self, 
        backbone='resnet50', 
        pretrained=True, 
        embedding_dim=256,
        pooling='gem'  # 'avg', 'max', 'gem'
    ):
        """
        Initialize the retrieval network.
        
        Args:
            backbone: Backbone architecture
            pretrained: Whether to use pre-trained weights
            embedding_dim: Embedding dimension
            pooling: Feature pooling method ('avg', 'max', 'gem')
        """
        super(FootwearRetrievalNetwork, self).__init__()
        
        # Load backbone and get feature dimension
        in_features, self.feature_extractor = self._get_backbone(backbone, pretrained)
        
        # Attention mechanism
        self.attention = CBAM(in_features)
        
        # Projection to embedding space
        self.embedding_projection = nn.Sequential(
            nn.Conv2d(in_features, in_features // 2, kernel_size=1),
            nn.BatchNorm2d(in_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 2, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim)
        )
        
        # Pooling method
        self.pooling = pooling
        if pooling == 'gem':
            # Generalized mean pooling (learnable p-norm)
            self.gem_p = nn.Parameter(torch.ones(1) * 3)
        
        # L2 normalization for embeddings
        self.normalize_output = True
    
    def _get_backbone(self, backbone_name, pretrained):
        """Load backbone architecture."""
        if backbone_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            layers = list(base_model.children())[:-2]
            in_features = 2048
        elif backbone_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            layers = list(base_model.children())[:-2]
            in_features = 512
        elif backbone_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            layers = list(base_model.children())[:-1]
            in_features = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        feature_extractor = nn.Sequential(*layers)
        return in_features, feature_extractor
    
    def gem_pooling(self, x):
        """
        Generalized Mean Pooling.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Pooled features [batch_size, channels]
        """
        # Apply p-norm pooling with learnable p
        p = F.softplus(self.gem_p)  # Ensure p > 0
        x = x.clamp(min=1e-6).pow(p)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.pow(1.0 / p)
        return x.squeeze(-1).squeeze(-1)
    
    def forward(self, x):
        """
        Extract embedding from image.
        
        Args:
            x: Input image tensor
            
        Returns:
            Normalized embedding vector
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Project to embedding space
        embeddings = self.embedding_projection(features)
        
        # Apply appropriate pooling
        if self.pooling == 'avg':
            pooled = F.adaptive_avg_pool2d(embeddings, 1).squeeze(-1).squeeze(-1)
        elif self.pooling == 'max':
            pooled = F.adaptive_max_pool2d(embeddings, 1).squeeze(-1).squeeze(-1)
        elif self.pooling == 'gem':
            pooled = self.gem_pooling(embeddings)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        
        # Apply L2 normalization if required
        if self.normalize_output:
            pooled = F.normalize(pooled, p=2, dim=1)
        
        return pooled


if __name__ == "__main__":
    # Simple test code for models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy input
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 512, 512).to(device)
    x2 = torch.randn(batch_size, 3, 512, 512).to(device)
    
    # Test standard model
    print("\nTesting FootwearMatchingNetwork...")
    model = FootwearMatchingNetwork(backbone='resnet18', pretrained=False).to(device)
    with torch.no_grad():
        logits, similarity, track_feats, ref_feats = model(x1, x2)
        print(f"Logits shape: {logits.shape}")
        print(f"Similarity shape: {similarity.shape}")
        print(f"Feature shapes: {track_feats.shape}, {ref_feats.shape}")
    
    # Test enhanced model
    print("\nTesting EnhancedSiameseNetwork...")
    model = EnhancedSiameseNetwork(backbone='resnet18', pretrained=False).to(device)
    with torch.no_grad():
        logits, similarity, track_feats, ref_feats = model(x1, x2)
        print(f"Logits shape: {logits.shape}")
        print(f"Similarity shape: {similarity.shape}")
        print(f"Feature shapes: {track_feats.shape}, {ref_feats.shape}")
    
    # Test retrieval model
    print("\nTesting FootwearRetrievalNetwork...")
    model = FootwearRetrievalNetwork(backbone='resnet18', pretrained=False, pooling='gem').to(device)
    with torch.no_grad():
        embeddings = model(x1)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Test similarity computation
        emb1 = model(x1)
        emb2 = model(x2)
        sim = torch.matmul(emb1, emb2.T)
        print(f"Similarity matrix shape: {sim.shape}")
