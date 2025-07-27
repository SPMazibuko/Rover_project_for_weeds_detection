"""
ResUNet3D: 3D Residual U-Net for Dental Volume Segmentation
Advanced 3D architecture with residual connections and attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .layers import ResidualBlock3D, AttentionGate3D, SelfAttention3D, PixelShuffle3D


class ResUNet3D(nn.Module):
    """
    3D Residual U-Net for volumetric dental segmentation.
    Features residual connections, attention gates, and anatomical-aware processing.
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 32,  # Number of tooth classes
                 base_filters: int = 32,
                 depth: int = 4,
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 use_self_attention: bool = True):
        super(ResUNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_self_attention = use_self_attention
        
        # Calculate filter sizes for each level
        self.filters = [base_filters * (2 ** i) for i in range(depth + 1)]
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build bottleneck
        self.bottleneck = self._build_bottleneck()
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Final classification layer
        self.classifier = self._build_classifier()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_encoder(self) -> nn.ModuleList:
        """Build encoder with residual blocks."""
        encoder_blocks = nn.ModuleList()
        
        # Initial convolution
        initial_conv = nn.Sequential(
            nn.Conv3d(self.in_channels, self.filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.filters[0]),
            nn.ReLU(inplace=True)
        )
        encoder_blocks.append(initial_conv)
        
        # Encoder levels
        for i in range(self.depth):
            in_filters = self.filters[i]
            out_filters = self.filters[i + 1]
            
            # Residual block with downsampling
            res_block = nn.Sequential(
                ResidualBlock3D(in_filters, in_filters),
                ResidualBlock3D(in_filters, out_filters, stride=2),
                nn.Dropout3d(self.dropout) if i > 0 else nn.Identity()
            )
            encoder_blocks.append(res_block)
        
        return encoder_blocks
    
    def _build_bottleneck(self) -> nn.Module:
        """Build bottleneck with self-attention."""
        bottleneck_filters = self.filters[-1]
        
        layers = [
            ResidualBlock3D(bottleneck_filters, bottleneck_filters),
            ResidualBlock3D(bottleneck_filters, bottleneck_filters)
        ]
        
        if self.use_self_attention:
            layers.append(SelfAttention3D(bottleneck_filters))
        
        layers.append(nn.Dropout3d(self.dropout))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.ModuleList:
        """Build decoder with attention gates."""
        decoder_blocks = nn.ModuleList()
        attention_gates = nn.ModuleList() if self.use_attention else None
        
        # Decoder levels
        for i in range(self.depth):
            in_filters = self.filters[self.depth - i]
            out_filters = self.filters[self.depth - i - 1]
            
            # Upsampling
            upsample = nn.ConvTranspose3d(in_filters, out_filters, kernel_size=2, stride=2)
            
            # Attention gate
            if self.use_attention:
                attention = AttentionGate3D(out_filters, out_filters, out_filters // 2)
                attention_gates.append(attention)
            
            # Residual blocks after concatenation
            combined_filters = out_filters * 2  # After concatenation with skip connection
            decoder_block = nn.Sequential(
                nn.Conv3d(combined_filters, out_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_filters),
                nn.ReLU(inplace=True),
                ResidualBlock3D(out_filters, out_filters),
                ResidualBlock3D(out_filters, out_filters),
                nn.Dropout3d(self.dropout)
            )
            
            decoder_blocks.append(nn.ModuleDict({
                'upsample': upsample,
                'decoder_block': decoder_block
            }))
        
        if self.use_attention:
            self.attention_gates = attention_gates
        
        return decoder_blocks
    
    def _build_classifier(self) -> nn.Module:
        """Build final classification layers."""
        return nn.Sequential(
            nn.Conv3d(self.filters[0], self.filters[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.filters[0] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(self.filters[0] // 2, self.out_channels, kernel_size=1),
            nn.Softmax(dim=1)  # Multi-class segmentation
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through ResUNet3D.
        
        Args:
            x: Input 3D volume [batch_size, 1, D, H, W]
            
        Returns:
            Tuple of (segmentation output, intermediate features for loss computation)
        """
        # Store encoder features for skip connections
        encoder_features = []
        
        # Encoder forward pass
        current = x
        for i, encoder_block in enumerate(self.encoder):
            current = encoder_block(current)
            if i > 0:  # Skip initial conv for skip connections
                encoder_features.append(current)
        
        # Bottleneck
        current = self.bottleneck(current)
        bottleneck_features = current
        
        # Decoder forward pass
        decoder_features = []
        for i, decoder_block in enumerate(self.decoder):
            # Upsampling
            current = decoder_block['upsample'](current)
            
            # Get corresponding encoder feature for skip connection
            skip_connection = encoder_features[self.depth - 1 - i]
            
            # Apply attention gate if enabled
            if self.use_attention:
                skip_connection = self.attention_gates[i](current, skip_connection)
            
            # Concatenate with skip connection
            current = torch.cat([current, skip_connection], dim=1)
            
            # Decoder block
            current = decoder_block['decoder_block'](current)
            decoder_features.append(current)
        
        # Final classification
        segmentation = self.classifier(current)
        
        # Collect intermediate features for loss computation
        intermediate_features = {
            'encoder_features': encoder_features,
            'bottleneck_features': bottleneck_features,
            'decoder_features': decoder_features
        }
        
        return segmentation, intermediate_features
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)


class DentalVolumeProcessor(nn.Module):
    """
    Specialized processor for dental volume data.
    Converts depth maps to 3D volumes for ResUNet3D input.
    """
    
    def __init__(self, 
                 depth_range: Tuple[float, float] = (0.0, 100.0),
                 volume_size: Tuple[int, int, int] = (128, 128, 128),
                 interpolation_method: str = 'trilinear'):
        super(DentalVolumeProcessor, self).__init__()
        
        self.depth_range = depth_range
        self.volume_size = volume_size
        self.interpolation_method = interpolation_method
    
    def depth_to_volume(self, depth_map: torch.Tensor, x_ray: torch.Tensor) -> torch.Tensor:
        """
        Convert 2D depth map to 3D volume using back-projection.
        
        Args:
            depth_map: Depth map [batch_size, 1, H, W]
            x_ray: Original X-ray [batch_size, 1, H, W]
            
        Returns:
            3D volume [batch_size, 1, D, H, W]
        """
        batch_size, _, height, width = depth_map.shape
        D, H, W = self.volume_size
        
        # Initialize volume
        volume = torch.zeros(batch_size, 1, D, H, W, device=depth_map.device)
        
        # Resize inputs to match volume spatial dimensions
        depth_resized = F.interpolate(depth_map, size=(H, W), mode='bilinear', align_corners=False)
        xray_resized = F.interpolate(x_ray, size=(H, W), mode='bilinear', align_corners=False)
        
        # Normalize depth to volume depth dimension
        depth_min, depth_max = self.depth_range
        normalized_depth = (depth_resized - depth_min) / (depth_max - depth_min)
        depth_indices = (normalized_depth * (D - 1)).long()
        depth_indices = torch.clamp(depth_indices, 0, D - 1)
        
        # Back-project to 3D volume
        for b in range(batch_size):
            for h in range(H):
                for w in range(W):
                    d_idx = depth_indices[b, 0, h, w]
                    intensity = xray_resized[b, 0, h, w]
                    
                    # Distribute intensity around depth index with Gaussian falloff
                    for d_offset in range(-2, 3):
                        d_target = d_idx + d_offset
                        if 0 <= d_target < D:
                            weight = torch.exp(-0.5 * (d_offset ** 2))
                            volume[b, 0, d_target, h, w] += intensity * weight
        
        return volume
    
    def forward(self, depth_map: torch.Tensor, x_ray: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to create 3D volume.
        
        Args:
            depth_map: Input depth map
            x_ray: Original X-ray image
            
        Returns:
            3D volume for ResUNet3D processing
        """
        volume = self.depth_to_volume(depth_map, x_ray)
        
        # Apply 3D smoothing
        volume = F.avg_pool3d(volume, kernel_size=3, stride=1, padding=1)
        
        return volume


class HybridDental3DNet(nn.Module):
    """
    Complete hybrid network combining volume processing and ResUNet3D.
    """
    
    def __init__(self, 
                 depth_range: Tuple[float, float] = (0.0, 100.0),
                 volume_size: Tuple[int, int, int] = (128, 128, 128),
                 num_classes: int = 32):
        super(HybridDental3DNet, self).__init__()
        
        # Volume processor
        self.volume_processor = DentalVolumeProcessor(
            depth_range=depth_range,
            volume_size=volume_size
        )
        
        # 3D segmentation network
        self.resunet3d = ResUNet3D(
            in_channels=1,
            out_channels=num_classes,
            base_filters=32,
            depth=4,
            use_attention=True,
            use_self_attention=True
        )
    
    def forward(self, depth_map: torch.Tensor, x_ray: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Complete forward pass from depth map to 3D segmentation.
        
        Args:
            depth_map: Generated depth map [batch_size, 1, H, W]
            x_ray: Original X-ray [batch_size, 1, H, W]
            
        Returns:
            Tuple of (3D segmentation, intermediate features)
        """
        # Convert to 3D volume
        volume = self.volume_processor(depth_map, x_ray)
        
        # 3D segmentation
        segmentation, intermediate_features = self.resunet3d(volume)
        
        return segmentation, intermediate_features
