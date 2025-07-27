"""
Combined Loss Functions for End-to-End Dental 3D Reconstruction
Integrates all loss components for optimal training
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .tooth_landmark_loss import ToothLandmarkLoss, DentalAnatomyLoss
from .adversarial_loss import AdversarialLoss, PerceptualLoss


class CombinedDentalLoss(nn.Module):
    """
    Combined loss function for end-to-end dental 3D reconstruction.
    Integrates adversarial, landmark, anatomical, and reconstruction losses.
    """
    
    def __init__(self,
                 adversarial_weight: float = 0.1,
                 landmark_weight: float = 10.0,
                 anatomy_weight: float = 5.0,
                 reconstruction_weight: float = 1.0,
                 perceptual_weight: float = 2.0,
                 depth_consistency_weight: float = 3.0,
                 num_teeth: int = 32):
        super(CombinedDentalLoss, self).__init__()
        
        # Loss weights
        self.adversarial_weight = adversarial_weight
        self.landmark_weight = landmark_weight
        self.anatomy_weight = anatomy_weight
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.depth_consistency_weight = depth_consistency_weight
        
        # Initialize individual loss functions
        self.adversarial_loss = AdversarialLoss(loss_type='lsgan')
        self.tooth_landmark_loss = ToothLandmarkLoss(num_teeth=num_teeth)
        self.dental_anatomy_loss = DentalAnatomyLoss()
        self.perceptual_loss = PerceptualLoss()
        
        # Basic losses
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def depth_consistency_loss(self, 
                             depth_map: torch.Tensor, 
                             x_ray: torch.Tensor) -> torch.Tensor:
        """
        Loss ensuring depth map consistency with X-ray intensities.
        Bright regions in X-ray should correspond to reasonable depths.
        
        Args:
            depth_map: Generated depth map [batch_size, 1, H, W]
            x_ray: Original X-ray [batch_size, 1, H, W]
            
        Returns:
            Depth consistency loss
        """
        # Normalize inputs
        x_ray_norm = (x_ray - x_ray.min()) / (x_ray.max() - x_ray.min() + 1e-8)
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Bright X-ray regions should have consistent depth relationships
        # This is a simplified heuristic - in practice, would use more sophisticated analysis
        
        # Calculate gradients to ensure smooth transitions
        x_ray_grad_x = x_ray_norm[:, :, :, 1:] - x_ray_norm[:, :, :, :-1]
        x_ray_grad_y = x_ray_norm[:, :, 1:, :] - x_ray_norm[:, :, :-1, :]
        
        depth_grad_x = depth_norm[:, :, :, 1:] - depth_norm[:, :, :, :-1]
        depth_grad_y = depth_norm[:, :, 1:, :] - depth_norm[:, :, :-1, :]
        
        # Correlation between X-ray and depth gradients
        grad_consistency = (
            self.l1_loss(x_ray_grad_x.sign(), depth_grad_x.sign()) +
            self.l1_loss(x_ray_grad_y.sign(), depth_grad_y.sign())
        )
        
        # Smooth depth transitions in homogeneous X-ray regions
        x_ray_variance = torch.var(x_ray_norm, dim=(2, 3), keepdim=True)
        depth_smoothness = torch.var(depth_norm, dim=(2, 3), keepdim=True)
        
        # Where X-ray is uniform, depth should also be relatively smooth
        smoothness_loss = self.mse_loss(
            depth_smoothness * (1 - x_ray_variance), 
            torch.zeros_like(depth_smoothness)
        )
        
        return grad_consistency + smoothness_loss
    
    def reconstruction_loss(self, 
                          predicted: torch.Tensor, 
                          target: torch.Tensor) -> torch.Tensor:
        """
        Basic reconstruction loss combining L1 and L2.
        
        Args:
            predicted: Predicted output
            target: Ground truth target
            
        Returns:
            Reconstruction loss
        """
        l1_loss = self.l1_loss(predicted, target)
        l2_loss = self.mse_loss(predicted, target)
        
        return l1_loss + 0.5 * l2_loss
    
    def compute_generator_loss(self,
                             x_ray: torch.Tensor,
                             generated_depth: torch.Tensor,
                             segmentation: torch.Tensor,
                             discriminator_output: torch.Tensor,
                             target_depth: Optional[torch.Tensor] = None,
                             target_segmentation: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute complete generator loss.
        
        Args:
            x_ray: Original X-ray image
            generated_depth: Generated depth map
            segmentation: 3D segmentation output
            discriminator_output: Discriminator score for generated sample
            target_depth: Ground truth depth (optional)
            target_segmentation: Ground truth segmentation (optional)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Adversarial loss
        adv_loss = self.adversarial_loss.generator_loss(discriminator_output)
        losses['adversarial'] = adv_loss * self.adversarial_weight
        
        # Tooth landmark loss
        landmark_losses = self.tooth_landmark_loss(
            segmentation, target_segmentation
        )
        for key, value in landmark_losses.items():
            losses[f'landmark_{key}'] = value * self.landmark_weight
        
        # Dental anatomy loss
        anatomy_losses = self.dental_anatomy_loss(segmentation)
        for key, value in anatomy_losses.items():
            losses[f'anatomy_{key}'] = value * self.anatomy_weight
        
        # Depth consistency loss
        depth_consistency = self.depth_consistency_loss(generated_depth, x_ray)
        losses['depth_consistency'] = depth_consistency * self.depth_consistency_weight
        
        # Reconstruction losses (if targets available)
        if target_depth is not None:
            depth_recon = self.reconstruction_loss(generated_depth, target_depth)
            losses['depth_reconstruction'] = depth_recon * self.reconstruction_weight
            
            # Perceptual loss for depth maps
            perceptual = self.perceptual_loss(generated_depth, target_depth)
            losses['perceptual'] = perceptual * self.perceptual_weight
        
        if target_segmentation is not None:
            seg_recon = self.ce_loss(segmentation, target_segmentation.long())
            losses['seg_reconstruction'] = seg_recon * self.reconstruction_weight
        
        # Total generator loss
        losses['total_generator'] = sum(losses.values())
        
        return losses
    
    def compute_discriminator_loss(self,
                                 real_x_ray: torch.Tensor,
                                 real_depth: torch.Tensor,
                                 fake_depth: torch.Tensor,
                                 real_pred: torch.Tensor,
                                 fake_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator loss.
        
        Args:
            real_x_ray: Real X-ray images
            real_depth: Real depth maps
            fake_depth: Generated depth maps
            real_pred: Discriminator output for real samples
            fake_pred: Discriminator output for fake samples
            
        Returns:
            Dictionary of discriminator loss components
        """
        losses = self.adversarial_loss.discriminator_loss(real_pred, fake_pred)
        
        # Scale by adversarial weight
        for key in losses:
            losses[key] *= self.adversarial_weight
        
        return losses
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: Optional[Dict[str, torch.Tensor]] = None,
                mode: str = 'generator') -> Dict[str, torch.Tensor]:
        """
        Forward pass computing appropriate losses.
        
        Args:
            outputs: Dictionary of model outputs
            targets: Dictionary of ground truth targets (optional)
            mode: 'generator' or 'discriminator'
            
        Returns:
            Dictionary of computed losses
        """
        if mode == 'generator':
            return self.compute_generator_loss(
                x_ray=outputs['x_ray'],
                generated_depth=outputs['depth'],
                segmentation=outputs['segmentation'],
                discriminator_output=outputs['discriminator_score'],
                target_depth=targets.get('depth') if targets else None,
                target_segmentation=targets.get('segmentation') if targets else None
            )
        
        elif mode == 'discriminator':
            return self.compute_discriminator_loss(
                real_x_ray=outputs['real_x_ray'],
                real_depth=outputs['real_depth'],
                fake_depth=outputs['fake_depth'],
                real_pred=outputs['real_pred'],
                fake_pred=outputs['fake_pred']
            )
        
        else:
            raise ValueError(f"Unknown mode: {mode}")


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for better detail preservation at different resolutions.
    """
    
    def __init__(self, scales: list = [1.0, 0.5, 0.25], weights: list = [1.0, 0.5, 0.25]):
        super(MultiScaleLoss, self).__init__()
        
        self.scales = scales
        self.weights = weights
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale loss.
        
        Args:
            predicted: Predicted tensor
            target: Target tensor
            
        Returns:
            Multi-scale loss
        """
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale != 1.0:
                # Downsample both tensors
                size = [int(s * scale) for s in predicted.shape[2:]]
                pred_scaled = F.interpolate(predicted, size=size, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
            else:
                pred_scaled = predicted
                target_scaled = target
            
            scale_loss = self.l1_loss(pred_scaled, target_scaled)
            total_loss += weight * scale_loss
        
        return total_loss
