"""
Tooth-Landmark Loss Functions for Dental 3D Reconstruction
Novel loss functions incorporating dental anatomical knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class ToothLandmarkLoss(nn.Module):
    """
    Custom loss function incorporating dental landmark constraints.
    Enforces anatomically correct tooth positioning and relationships.
    """
    
    def __init__(self,
                 num_teeth: int = 32,
                 landmark_weight: float = 10.0,
                 spatial_weight: float = 5.0,
                 symmetry_weight: float = 3.0,
                 anatomy_weight: float = 7.0):
        super(ToothLandmarkLoss, self).__init__()
        
        self.num_teeth = num_teeth
        self.landmark_weight = landmark_weight
        self.spatial_weight = spatial_weight
        self.symmetry_weight = symmetry_weight
        self.anatomy_weight = anatomy_weight
        
        # Define tooth anatomy relationships
        self.tooth_relationships = self._define_tooth_relationships()
        
        # Define expected tooth positions (normalized coordinates)
        self.expected_positions = self._define_expected_positions()
        
        # Base losses
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def _define_tooth_relationships(self) -> Dict:
        """Define anatomical relationships between teeth."""
        # Standard dental numbering (1-32 for permanent teeth)
        relationships = {
            'opposing_pairs': [
                (1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9),
                (17, 32), (18, 31), (19, 30), (20, 29), (21, 28), (22, 27), (23, 26), (24, 25)
            ],
            'adjacent_pairs': [
                (i, i+1) for i in range(1, 16) if i != 8
            ] + [
                (i, i+1) for i in range(17, 32) if i != 24
            ],
            'arch_groups': {
                'upper_right': list(range(1, 9)),
                'upper_left': list(range(9, 17)),
                'lower_left': list(range(17, 25)),
                'lower_right': list(range(25, 33))
            },
            'tooth_types': {
                'incisors': [7, 8, 9, 10, 23, 24, 25, 26],
                'canines': [6, 11, 22, 27],
                'premolars': [4, 5, 12, 13, 20, 21, 28, 29],
                'molars': [1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32]
            }
        }
        return relationships
    
    def _define_expected_positions(self) -> torch.Tensor:
        """Define expected 3D positions for each tooth (normalized)."""
        # Simplified dental arch coordinates (would be refined with real data)
        positions = torch.zeros(32, 3)  # 32 teeth, 3D coordinates
        
        # Upper arch (teeth 1-16)
        upper_angles = torch.linspace(-np.pi/2, np.pi/2, 16)
        upper_radius = 0.6
        for i in range(16):
            angle = upper_angles[i]
            positions[i] = torch.tensor([
                upper_radius * torch.cos(angle),
                upper_radius * torch.sin(angle),
                0.7  # Upper arch height
            ])
        
        # Lower arch (teeth 17-32)
        lower_angles = torch.linspace(np.pi/2, -np.pi/2, 16)
        lower_radius = 0.55
        for i in range(16):
            angle = lower_angles[i]
            positions[i + 16] = torch.tensor([
                lower_radius * torch.cos(angle),
                lower_radius * torch.sin(angle),
                0.3  # Lower arch height
            ])
        
        return positions
    
    def extract_tooth_centers(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Extract center of mass for each tooth from segmentation.
        
        Args:
            segmentation: 3D segmentation [batch_size, num_classes, D, H, W]
            
        Returns:
            Tooth centers [batch_size, num_classes, 3]
        """
        batch_size, num_classes, D, H, W = segmentation.shape
        centers = torch.zeros(batch_size, num_classes, 3, device=segmentation.device)
        
        # Create coordinate grids
        z_coords = torch.arange(D, device=segmentation.device).float().view(D, 1, 1)
        y_coords = torch.arange(H, device=segmentation.device).float().view(1, H, 1)
        x_coords = torch.arange(W, device=segmentation.device).float().view(1, 1, W)
        
        for b in range(batch_size):
            for c in range(num_classes):
                mask = segmentation[b, c]
                total_mass = mask.sum()
                
                if total_mass > 0:
                    # Center of mass calculation
                    center_z = (mask * z_coords).sum() / total_mass
                    center_y = (mask * y_coords).sum() / total_mass
                    center_x = (mask * x_coords).sum() / total_mass
                    
                    # Normalize to [0, 1]
                    centers[b, c] = torch.tensor([
                        center_x / W, center_y / H, center_z / D
                    ], device=segmentation.device)
        
        return centers
    
    def landmark_positioning_loss(self, predicted_centers: torch.Tensor) -> torch.Tensor:
        """
        Loss based on expected anatomical positions.
        
        Args:
            predicted_centers: Predicted tooth centers [batch_size, num_teeth, 3]
            
        Returns:
            Positioning loss
        """
        expected = self.expected_positions.to(predicted_centers.device)
        expected = expected.unsqueeze(0).expand(predicted_centers.size(0), -1, -1)
        
        # L2 distance to expected positions
        position_loss = self.mse_loss(predicted_centers, expected)
        
        return position_loss
    
    def spatial_relationship_loss(self, predicted_centers: torch.Tensor) -> torch.Tensor:
        """
        Loss enforcing spatial relationships between teeth.
        
        Args:
            predicted_centers: Predicted tooth centers [batch_size, num_teeth, 3]
            
        Returns:
            Spatial relationship loss
        """
        batch_size = predicted_centers.size(0)
        total_loss = 0.0
        
        # Adjacent teeth should be close
        for tooth1, tooth2 in self.tooth_relationships['adjacent_pairs']:
            if tooth1 <= self.num_teeth and tooth2 <= self.num_teeth:
                dist = torch.norm(
                    predicted_centers[:, tooth1-1] - predicted_centers[:, tooth2-1], 
                    dim=1
                )
                # Penalize if distance is too large (should be ~0.1 in normalized coords)
                target_dist = 0.1
                loss = F.relu(dist - target_dist).mean()
                total_loss += loss
        
        # Opposing teeth should have similar x,y but different z
        for tooth1, tooth2 in self.tooth_relationships['opposing_pairs']:
            if tooth1 <= self.num_teeth and tooth2 <= self.num_teeth:
                pos1 = predicted_centers[:, tooth1-1]
                pos2 = predicted_centers[:, tooth2-1]
                
                # X,Y should be similar
                xy_loss = self.mse_loss(pos1[:, :2], pos2[:, :2])
                
                # Z should be different (upper vs lower arch)
                z_diff = torch.abs(pos1[:, 2] - pos2[:, 2])
                z_loss = F.relu(0.3 - z_diff).mean()  # Minimum separation
                
                total_loss += xy_loss + z_loss
        
        return total_loss
    
    def symmetry_loss(self, predicted_centers: torch.Tensor) -> torch.Tensor:
        """
        Loss enforcing bilateral symmetry of dental arches.
        
        Args:
            predicted_centers: Predicted tooth centers [batch_size, num_teeth, 3]
            
        Returns:
            Symmetry loss
        """
        batch_size = predicted_centers.size(0)
        total_loss = 0.0
        
        # Define symmetric pairs (left-right)
        symmetric_pairs = [
            # Upper arch
            (1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9),
            # Lower arch
            (17, 32), (18, 31), (19, 30), (20, 29), (21, 28), (22, 27), (23, 26), (24, 25)
        ]
        
        for tooth1, tooth2 in symmetric_pairs:
            if tooth1 <= self.num_teeth and tooth2 <= self.num_teeth:
                pos1 = predicted_centers[:, tooth1-1]
                pos2 = predicted_centers[:, tooth2-1]
                
                # Y and Z should be similar, X should be mirrored
                yz_loss = self.mse_loss(pos1[:, 1:], pos2[:, 1:])
                x_loss = self.mse_loss(pos1[:, 0], -pos2[:, 0])  # Mirror in X
                
                total_loss += yz_loss + x_loss
        
        return total_loss
    
    def anatomy_consistency_loss(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Loss enforcing anatomical consistency (size, shape).
        
        Args:
            segmentation: 3D segmentation [batch_size, num_classes, D, H, W]
            
        Returns:
            Anatomy consistency loss
        """
        batch_size, num_classes = segmentation.shape[:2]
        total_loss = 0.0
        
        # Calculate volumes for each tooth
        volumes = segmentation.sum(dim=(2, 3, 4))  # [batch_size, num_classes]
        
        # Teeth of same type should have similar volumes
        for tooth_type, tooth_indices in self.tooth_relationships['tooth_types'].items():
            valid_indices = [i-1 for i in tooth_indices if i <= num_classes]
            
            if len(valid_indices) > 1:
                tooth_volumes = volumes[:, valid_indices]
                # Variance loss - similar teeth should have similar volumes
                variance_loss = torch.var(tooth_volumes, dim=1).mean()
                total_loss += variance_loss
        
        return total_loss
    
    def forward(self, 
                segmentation: torch.Tensor,
                target_segmentation: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute complete tooth-landmark loss.
        
        Args:
            segmentation: Predicted 3D segmentation [batch_size, num_classes, D, H, W]
            target_segmentation: Ground truth segmentation (optional)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Extract tooth centers
        predicted_centers = self.extract_tooth_centers(segmentation)
        
        # Landmark positioning loss
        landmark_loss = self.landmark_positioning_loss(predicted_centers)
        losses['landmark'] = landmark_loss * self.landmark_weight
        
        # Spatial relationship loss
        spatial_loss = self.spatial_relationship_loss(predicted_centers)
        losses['spatial'] = spatial_loss * self.spatial_weight
        
        # Symmetry loss
        symmetry_loss = self.symmetry_loss(predicted_centers)
        losses['symmetry'] = symmetry_loss * self.symmetry_weight
        
        # Anatomy consistency loss
        anatomy_loss = self.anatomy_consistency_loss(segmentation)
        losses['anatomy'] = anatomy_loss * self.anatomy_weight
        
        # Segmentation loss (if target provided)
        if target_segmentation is not None:
            seg_loss = self.ce_loss(segmentation, target_segmentation.long())
            losses['segmentation'] = seg_loss
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class DentalAnatomyLoss(nn.Module):
    """
    Advanced anatomical loss incorporating dental morphology knowledge.
    """
    
    def __init__(self,
                 crown_root_ratio_weight: float = 2.0,
                 surface_smoothness_weight: float = 1.5,
                 cusp_detection_weight: float = 3.0):
        super(DentalAnatomyLoss, self).__init__()
        
        self.crown_root_ratio_weight = crown_root_ratio_weight
        self.surface_smoothness_weight = surface_smoothness_weight
        self.cusp_detection_weight = cusp_detection_weight
    
    def crown_root_ratio_loss(self, segmentation: torch.Tensor) -> torch.Tensor:
        """Loss based on expected crown-to-root ratios."""
        # Expected ratios for different tooth types
        expected_ratios = {
            'incisors': 1.0,  # Crown ≈ Root
            'canines': 0.8,   # Longer roots
            'premolars': 0.9,
            'molars': 1.2     # Larger crowns
        }
        
        # Simplified implementation - would need more sophisticated analysis
        total_loss = 0.0
        batch_size, num_classes = segmentation.shape[:2]
        
        for b in range(batch_size):
            for c in range(num_classes):
                tooth_mask = segmentation[b, c]
                if tooth_mask.sum() > 0:
                    # Estimate crown (top half) vs root (bottom half) ratio
                    depth = tooth_mask.shape[0]
                    crown_volume = tooth_mask[:depth//2].sum()
                    root_volume = tooth_mask[depth//2:].sum()
                    
                    if root_volume > 0:
                        ratio = crown_volume / root_volume
                        # Use molar ratio as default
                        expected_ratio = expected_ratios['molars']
                        ratio_loss = (ratio - expected_ratio) ** 2
                        total_loss += ratio_loss
        
        return total_loss / (batch_size * num_classes)
    
    def surface_smoothness_loss(self, segmentation: torch.Tensor) -> torch.Tensor:
        """Loss encouraging smooth tooth surfaces."""
        # Calculate gradients in 3D
        grad_x = segmentation[:, :, :, :, 1:] - segmentation[:, :, :, :, :-1]
        grad_y = segmentation[:, :, :, 1:, :] - segmentation[:, :, :, :-1, :]
        grad_z = segmentation[:, :, 1:, :, :] - segmentation[:, :, :-1, :, :]
        
        # Total variation loss
        tv_loss = (
            grad_x.abs().mean() + 
            grad_y.abs().mean() + 
            grad_z.abs().mean()
        )
        
        return tv_loss
    
    def forward(self, segmentation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute dental anatomy loss.
        
        Args:
            segmentation: Predicted 3D segmentation
            
        Returns:
            Dictionary of anatomy loss components
        """
        losses = {}
        
        # Crown-root ratio loss
        cr_loss = self.crown_root_ratio_loss(segmentation)
        losses['crown_root'] = cr_loss * self.crown_root_ratio_weight
        
        # Surface smoothness loss
        smooth_loss = self.surface_smoothness_loss(segmentation)
        losses['smoothness'] = smooth_loss * self.surface_smoothness_weight
        
        # Total anatomy loss
        losses['total_anatomy'] = sum(losses.values())
        
        return losses
