"""
Neural Network Building Blocks for Footwear Impression Matching

This module implements specialized neural network components used in the
footwear impression matching architecture, including:
- Attention mechanisms (Channel, Spatial, CBAM)
- Squeeze-and-Excitation blocks
- Domain-specific projection modules
- Other specialized components

These components are used to build the main network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (from CBAM)
    
    Applies attention along the channel dimension using both max and average pooling
    to capture channel dependencies.
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Initialize the channel attention module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Channel reduction ratio for the bottleneck
        """
        super(ChannelAttention, self).__init__()
        
        # Pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Channel attention map [batch_size, channels, 1, 1]
        """
        # Process through pooling and MLP
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        
        # Combine outputs
        out = avg_out + max_out
        
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (from CBAM)
    
    Applies attention along the spatial dimensions to focus on relevant regions.
    """
    
    def __init__(self, kernel_size=7):
        """
        Initialize the spatial attention module.
        
        Args:
            kernel_size: Size of the convolutional kernel (3 or 7)
        """
        super(SpatialAttention, self).__init__()
        
        # Validate kernel size
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # Convolutional layer for generating attention map
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Spatial attention map [batch_size, 1, height, width]
        """
        # Generate channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        x = torch.cat([avg_out, max_out], dim=1)
        
        # Generate spatial attention map
        x = self.conv(x)
        
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    
    Combines channel and spatial attention mechanisms for comprehensive feature refinement.
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Initialize the CBAM module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Channel reduction ratio for bottleneck
        """
        super(CBAM, self).__init__()
        
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Refined feature map [batch_size, channels, height, width]
        """
        # Apply channel attention
        x = x * self.channel_att(x)
        
        # Apply spatial attention
        x = x * self.spatial_att(x)
        
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Models channel-wise interdependencies to recalibrate channel features.
    """
    
    def __init__(self, channels, reduction=16):
        """
        Initialize the SE block.
        
        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio for bottleneck
        """
        super(SEBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Recalibrated feature map [batch_size, channels, height, width]
        """
        b, c, _, _ = x.size()
        
        # Squeeze operation
        y = self.avg_pool(x).view(b, c)
        
        # Excitation operation
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale the input
        return x * y.expand_as(x)


class DomainProjection(nn.Module):
    """
    Domain-Specific Projection Module
    
    Projects features into a domain-specific embedding space with residual connections.
    Used to create specialized representations for different image domains (track/reference).
    """
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        """
        Initialize the domain projection module.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout_rate: Dropout rate for regularization
        """
        super(DomainProjection, self).__init__()
        
        # Main projection path
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Skip connection (identity or 1x1 conv for dimension matching)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            
        Returns:
            Projected features [batch_size, out_channels, height, width]
        """
        return self.relu(self.proj(x) + self.skip(x))


class FeatureCorrelation(nn.Module):
    """
    Multi-Channel Normalized Cross-Correlation (MCNCC) Module
    
    Computes the normalized cross-correlation between feature maps from different domains.
    """
    
    def __init__(self, in_channels, learn_weights=True, temperature=10.0):
        """
        Initialize the feature correlation module.
        
        Args:
            in_channels: Number of input channels
            learn_weights: Whether to learn channel weights
            temperature: Temperature parameter for scaling correlation
        """
        super(FeatureCorrelation, self).__init__()
        
        # Learnable channel weights for weighted correlation
        if learn_weights:
            self.channel_weights = nn.Parameter(torch.ones(in_channels))
        else:
            self.register_buffer('channel_weights', torch.ones(in_channels))
        
        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, x, y):
        """
        Compute normalized cross-correlation.
        
        Args:
            x: First feature map [batch_size, channels, height, width]
            y: Second feature map [batch_size, channels, height, width]
            
        Returns:
            Per-sample correlation scores [batch_size]
        """
        # L2 normalization along channel dimension
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        
        # Compute correlation for each spatial position
        corr = (x_norm * y_norm).sum(dim=[2, 3]) / (x.size(2) * x.size(3))
        
        # Normalize channel weights with softmax
        weights = F.softmax(self.channel_weights, dim=0)
        
        # Apply weighted sum across channels
        weighted_corr = (corr * weights).sum(dim=1)
        
        # Apply temperature scaling
        return weighted_corr * self.temperature


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion Module
    
    Fuses features from different sources with adaptive weighting.
    """
    
    def __init__(self, in_channels):
        """
        Initialize feature fusion module.
        
        Args:
            in_channels: Number of input channels
        """
        super(AdaptiveFeatureFusion, self).__init__()
        
        # Attention mechanism for fusion
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x1, x2):
        """
        Forward pass.
        
        Args:
            x1: First feature map [batch_size, channels, height, width]
            x2: Second feature map [batch_size, channels, height, width]
            
        Returns:
            Fused features [batch_size, channels, height, width]
        """
        # Concatenate features
        concat = torch.cat([x1, x2], dim=1)
        
        # Generate attention weights
        weights = self.attention(concat)
        
        # Split weights for each input
        w1, w2 = weights[:, 0:1, :, :], weights[:, 1:2, :, :]
        
        # Apply weighted combination
        return w1 * x1 + w2 * x2


# Residual Block for deeper networks
class ResidualBlock(nn.Module):
    """
    Residual Block
    
    Standard residual block with bottleneck design.
    """
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution
            downsample: Downsampling function for skip connection
        """
        super(ResidualBlock, self).__init__()
        
        # Expansion factor for bottleneck
        self.expansion = 4
        mid_channels = out_channels // self.expansion
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.downsample = downsample
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual connection
        out += identity
        out = self.relu(out)
        
        return out


# Self-Attention for feature maps
class SelfAttention(nn.Module):
    """
    Self-Attention Module
    
    Applies self-attention mechanism to feature maps.
    """
    
    def __init__(self, in_channels, key_channels=None, value_channels=None, head_count=1):
        """
        Initialize self-attention module.
        
        Args:
            in_channels: Number of input channels
            key_channels: Number of key channels (if None, in_channels // 2)
            value_channels: Number of value channels (if None, in_channels)
            head_count: Number of attention heads
        """
        super(SelfAttention, self).__init__()
        
        self.in_channels = in_channels
        self.head_count = head_count
        
        self.key_channels = key_channels or in_channels // 2
        self.value_channels = value_channels or in_channels
        
        self.keys = nn.Conv2d(in_channels, self.key_channels, 1)
        self.queries = nn.Conv2d(in_channels, self.key_channels, 1)
        self.values = nn.Conv2d(in_channels, self.value_channels, 1)
        self.reprojection = nn.Conv2d(self.value_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input feature map [batch_size, channels, height, width]
            
        Returns:
            Attended feature map [batch_size, channels, height, width]
        """
        batch_size, _, h, w = x.size()
        
        # Reshape for attention computation
        keys = self.keys(x).view(batch_size, self.key_channels, -1)
        queries = self.queries(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        values = self.values(x).view(batch_size, self.value_channels, -1)
        
        # Compute attention scores
        attention = torch.bmm(queries, keys)  # [batch_size, h*w, h*w]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(values, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.value_channels, h, w)
        
        # Residual connection with learnable weight
        out = self.reprojection(out)
        out = self.gamma * out + x
        
        return out


if __name__ == "__main__":
    # Test code for modules
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test input
    x = torch.randn(4, 256, 32, 32).to(device)
    y = torch.randn(4, 256, 32, 32).to(device)
    
    # Test CBAM
    cbam = CBAM(256).to(device)
    start_time = time.time()
    cbam_out = cbam(x)
    print(f"CBAM output shape: {cbam_out.shape}, time: {time.time() - start_time:.4f}s")
    
    # Test SE Block
    se = SEBlock(256).to(device)
    start_time = time.time()
    se_out = se(x)
    print(f"SE Block output shape: {se_out.shape}, time: {time.time() - start_time:.4f}s")
    
    # Test Domain Projection
    dp = DomainProjection(256, 128).to(device)
    start_time = time.time()
    dp_out = dp(x)
    print(f"Domain Projection output shape: {dp_out.shape}, time: {time.time() - start_time:.4f}s")
    
    # Test Feature Correlation
    fc = FeatureCorrelation(128).to(device)
    x_proj = dp(x)
    y_proj = dp(y)
    start_time = time.time()
    corr = fc(x_proj, y_proj)
    print(f"Feature Correlation output shape: {corr.shape}, time: {time.time() - start_time:.4f}s")
    
    # Test Self-Attention
    sa = SelfAttention(128).to(device)
    start_time = time.time()
    sa_out = sa(x_proj)
    print(f"Self-Attention output shape: {sa_out.shape}, time: {time.time() - start_time:.4f}s")
