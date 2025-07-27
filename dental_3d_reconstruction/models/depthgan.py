"""
DepthGAN: Generative Adversarial Network for Depth Estimation from 2D Dental X-rays
Novel architecture for realistic depth map generation from panoramic dental images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from .layers import SpectralNormalization, SelfAttention3D


class DepthGenerator(nn.Module):
    """
    Generator network for DepthGAN.
    Converts 2D panoramic dental X-ray to realistic depth maps.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 latent_dim: int = 100,
                 filters: list = [64, 128, 256, 512, 1024],
                 output_size: Tuple[int, int] = (512, 512)):
        super(DepthGenerator, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.filters = filters
        self.output_size = output_size
        
        # Encoder: 2D X-ray to feature maps
        self.encoder = self._build_encoder()
        
        # Latent space processing
        self.latent_processor = self._build_latent_processor()
        
        # Decoder: Feature maps to depth map
        self.decoder = self._build_decoder()
        
        # Final depth refinement
        self.depth_refiner = self._build_depth_refiner()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_encoder(self) -> nn.ModuleList:
        """Build encoder layers for feature extraction."""
        layers = nn.ModuleList()
        
        in_channels = self.input_channels
        for i, out_channels in enumerate(self.filters):
            # Convolution block
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True) if i > 0 else nn.ReLU(inplace=True)
            )
            layers.append(conv_block)
            in_channels = out_channels
        
        return layers
    
    def _build_latent_processor(self) -> nn.Module:
        """Build latent space processing module."""
        # Calculate encoded feature size
        encoded_size = self.output_size[0] // (2 ** len(self.filters))
        encoded_features = self.filters[-1] * encoded_size * encoded_size
        
        return nn.Sequential(
            nn.Linear(encoded_features, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.latent_dim, encoded_features),
            nn.ReLU(inplace=True)
        )
    
    def _build_decoder(self) -> nn.ModuleList:
        """Build decoder layers for depth map generation."""
        layers = nn.ModuleList()
        
        # Reverse filter order for decoder
        reversed_filters = self.filters[::-1]
        
        for i in range(len(reversed_filters) - 1):
            in_channels = reversed_filters[i]
            out_channels = reversed_filters[i + 1]
            
            # Transposed convolution block
            decoder_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1) if i < 2 else nn.Identity()
            )
            layers.append(decoder_block)
        
        return layers
    
    def _build_depth_refiner(self) -> nn.Module:
        """Build final depth refinement layers."""
        return nn.Sequential(
            nn.Conv2d(self.filters[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Normalize depth to [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.
        
        Args:
            x: Input 2D dental X-ray [batch_size, 1, H, W]
            
        Returns:
            Generated depth map [batch_size, 1, H, W]
        """
        batch_size = x.size(0)
        
        # Encoder forward pass
        features = []
        current = x
        for encoder_layer in self.encoder:
            current = encoder_layer(current)
            features.append(current)
        
        # Latent space processing
        encoded_flat = current.view(batch_size, -1)
        latent = self.latent_processor(encoded_flat)
        
        # Reshape back to spatial dimensions
        spatial_size = self.output_size[0] // (2 ** len(self.filters))
        current = latent.view(batch_size, self.filters[-1], spatial_size, spatial_size)
        
        # Decoder forward pass with skip connections
        for i, decoder_layer in enumerate(self.decoder):
            current = decoder_layer(current)
            # Add skip connection from encoder
            if i < len(features) - 1:
                skip_feature = features[-(i + 2)]  # Corresponding encoder feature
                current = current + skip_feature
        
        # Final depth refinement
        depth_map = self.depth_refiner(current)
        
        return depth_map
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)


class DepthDiscriminator(nn.Module):
    """
    Discriminator network for DepthGAN.
    Distinguishes between real and generated depth maps.
    """
    
    def __init__(self, 
                 input_channels: int = 2,  # X-ray + depth
                 filters: list = [64, 128, 256, 512],
                 use_spectral_norm: bool = True):
        super(DepthDiscriminator, self).__init__()
        
        self.input_channels = input_channels
        self.filters = filters
        self.use_spectral_norm = use_spectral_norm
        
        # Build discriminator layers
        self.layers = self._build_layers()
        
        # Final classification layer
        self.classifier = self._build_classifier()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_layers(self) -> nn.ModuleList:
        """Build discriminator layers."""
        layers = nn.ModuleList()
        
        in_channels = self.input_channels
        for i, out_channels in enumerate(self.filters):
            # Convolution layer
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            
            # Apply spectral normalization if requested
            if self.use_spectral_norm and i > 0:
                conv = SpectralNormalization(conv)
            
            # Build block
            if i == 0:
                # First layer: no batch norm
                block = nn.Sequential(
                    conv,
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                block = nn.Sequential(
                    conv,
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.2)
                )
            
            layers.append(block)
            in_channels = out_channels
        
        return layers
    
    def _build_classifier(self) -> nn.Module:
        """Build final classification layers."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.filters[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_ray: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.
        
        Args:
            x_ray: Input 2D dental X-ray [batch_size, 1, H, W]
            depth: Input depth map [batch_size, 1, H, W]
            
        Returns:
            Discrimination score [batch_size, 1]
        """
        # Concatenate X-ray and depth
        combined = torch.cat([x_ray, depth], dim=1)
        
        # Forward through layers
        current = combined
        for layer in self.layers:
            current = layer(current)
        
        # Final classification
        output = self.classifier(current)
        
        return output
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)


class DepthGAN(nn.Module):
    """
    Complete DepthGAN model combining generator and discriminator.
    Novel architecture for depth estimation from dental X-rays.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 latent_dim: int = 100,
                 gen_filters: list = [64, 128, 256, 512, 1024],
                 disc_filters: list = [64, 128, 256, 512],
                 output_size: Tuple[int, int] = (512, 512),
                 depth_range: Tuple[float, float] = (0.0, 100.0)):
        super(DepthGAN, self).__init__()
        
        self.depth_range = depth_range
        
        # Generator and Discriminator
        self.generator = DepthGenerator(
            input_channels=input_channels,
            latent_dim=latent_dim,
            filters=gen_filters,
            output_size=output_size
        )
        
        self.discriminator = DepthDiscriminator(
            input_channels=input_channels + 1,  # X-ray + depth
            filters=disc_filters
        )
    
    def generate_depth(self, x_ray: torch.Tensor) -> torch.Tensor:
        """
        Generate depth map from X-ray image.
        
        Args:
            x_ray: Input 2D dental X-ray [batch_size, 1, H, W]
            
        Returns:
            Generated depth map in real units [batch_size, 1, H, W]
        """
        # Generate normalized depth [0, 1]
        normalized_depth = self.generator(x_ray)
        
        # Convert to real depth values
        depth_min, depth_max = self.depth_range
        real_depth = normalized_depth * (depth_max - depth_min) + depth_min
        
        return real_depth
    
    def discriminate(self, x_ray: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Discriminate between real and fake depth maps.
        
        Args:
            x_ray: Input 2D dental X-ray [batch_size, 1, H, W]
            depth: Depth map [batch_size, 1, H, W]
            
        Returns:
            Discrimination score [batch_size, 1]
        """
        # Normalize depth to [0, 1] for discriminator
        depth_min, depth_max = self.depth_range
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
        normalized_depth = torch.clamp(normalized_depth, 0, 1)
        
        return self.discriminator(x_ray, normalized_depth)
    
    def forward(self, x_ray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.
        
        Args:
            x_ray: Input 2D dental X-ray [batch_size, 1, H, W]
            
        Returns:
            Tuple of (generated depth, discrimination score)
        """
        # Generate depth
        generated_depth = self.generate_depth(x_ray)
        
        # Discriminate generated depth
        disc_score = self.discriminate(x_ray, generated_depth)
        
        return generated_depth, disc_score
