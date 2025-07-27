"""
Adversarial Loss Functions for DepthGAN Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training with improved stability.
    Supports multiple GAN loss types.
    """
    
    def __init__(self, loss_type: str = 'lsgan', target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super(AdversarialLoss, self).__init__()
        
        self.loss_type = loss_type.lower()
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        
        if self.loss_type == 'vanilla':
            self.loss_fn = nn.BCELoss()
        elif self.loss_type == 'lsgan':
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == 'wgan':
            self.loss_fn = None  # Uses direct loss calculation
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create target tensor with same size as prediction."""
        if target_is_real:
            target_tensor = torch.full_like(prediction, self.target_real_label)
        else:
            target_tensor = torch.full_like(prediction, self.target_fake_label)
        return target_tensor
    
    def discriminator_loss(self, 
                          real_pred: torch.Tensor, 
                          fake_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate discriminator loss.
        
        Args:
            real_pred: Discriminator output for real samples
            fake_pred: Discriminator output for fake samples
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        if self.loss_type == 'wgan':
            # Wasserstein loss
            losses['d_real'] = -real_pred.mean()
            losses['d_fake'] = fake_pred.mean()
            losses['d_total'] = losses['d_real'] + losses['d_fake']
        else:
            # BCE or LSGAN
            real_target = self.get_target_tensor(real_pred, True)
            fake_target = self.get_target_tensor(fake_pred, False)
            
            losses['d_real'] = self.loss_fn(real_pred, real_target)
            losses['d_fake'] = self.loss_fn(fake_pred, fake_target)
            losses['d_total'] = (losses['d_real'] + losses['d_fake']) * 0.5
        
        return losses
    
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate generator loss.
        
        Args:
            fake_pred: Discriminator output for generated samples
            
        Returns:
            Generator loss
        """
        if self.loss_type == 'wgan':
            return -fake_pred.mean()
        else:
            target = self.get_target_tensor(fake_pred, True)
            return self.loss_fn(fake_pred, target)


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for improved GAN training stability.
    """
    
    def __init__(self, num_layers: int = 3):
        super(FeatureMatchingLoss, self).__init__()
        self.num_layers = num_layers
        self.l1_loss = nn.L1Loss()
    
    def forward(self, 
                real_features: list, 
                fake_features: list) -> torch.Tensor:
        """
        Calculate feature matching loss.
        
        Args:
            real_features: List of feature maps from real samples
            fake_features: List of feature maps from fake samples
            
        Returns:
            Feature matching loss
        """
        loss = 0.0
        num_discriminators = len(real_features)
        
        for i in range(num_discriminators):
            for j in range(min(len(real_features[i]), len(fake_features[i]), self.num_layers)):
                loss += self.l1_loss(real_features[i][j], fake_features[i][j])
        
        return loss / (num_discriminators * self.num_layers)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    Adapted for medical imaging.
    """
    
    def __init__(self, feature_layers: list = [2, 7, 12, 21, 30]):
        super(PerceptualLoss, self).__init__()
        
        self.feature_layers = feature_layers
        self.l1_loss = nn.L1Loss()
        
        # Use a simple CNN as feature extractor for medical images
        self.feature_extractor = self._build_feature_extractor()
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extractor for medical images."""
        layers = []
        in_channels = 1
        channels = [64, 128, 256, 512, 512]
        
        for i, out_channels in enumerate(channels):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def extract_features(self, x: torch.Tensor) -> list:
        """Extract features from multiple layers."""
        features = []
        current = x
        
        for i, layer in enumerate(self.feature_extractor):
            current = layer(current)
            if i in self.feature_layers:
                features.append(current)
        
        return features
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss.
        
        Args:
            generated: Generated images
            target: Target images
            
        Returns:
            Perceptual loss
        """
        gen_features = self.extract_features(generated)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for gen_feat, target_feat in zip(gen_features, target_features):
            loss += self.l1_loss(gen_feat, target_feat)
        
        return loss / len(gen_features)


class GradientPenalty(nn.Module):
    """
    Gradient penalty for WGAN-GP training.
    """
    
    def __init__(self, lambda_gp: float = 10.0):
        super(GradientPenalty, self).__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, 
                discriminator: nn.Module,
                real_samples: torch.Tensor,
                fake_samples: torch.Tensor,
                device: torch.device) -> torch.Tensor:
        """
        Calculate gradient penalty.
        
        Args:
            discriminator: Discriminator model
            real_samples: Real sample batch
            fake_samples: Fake sample batch
            device: Device to run on
            
        Returns:
            Gradient penalty loss
        """
        batch_size = real_samples.size(0)
        
        # Generate random interpolation points
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Calculate discriminator output for interpolates
        d_interpolates = discriminator(interpolates)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty
