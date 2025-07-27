"""
Evaluation metrics for dental 3D reconstruction
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, jaccard_score
import scipy.ndimage as ndimage


class DentalMetrics:
    """
    Comprehensive metrics for evaluating dental 3D reconstruction.
    """
    
    def __init__(self, num_classes: int = 32, ignore_background: bool = True):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
    
    def dice_coefficient(self, 
                        pred: torch.Tensor, 
                        target: torch.Tensor, 
                        smooth: float = 1e-8) -> torch.Tensor:
        """
        Calculate Dice coefficient for segmentation.
        
        Args:
            pred: Predicted segmentation [B, C, D, H, W]
            target: Ground truth segmentation [B, D, H, W]
            smooth: Smoothing factor
            
        Returns:
            Dice coefficient per class
        """
        pred_probs = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3, 4))
        union = pred_probs.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        
        if self.ignore_background:
            dice = dice[:, 1:]  # Exclude background class
        
        return dice.mean(dim=0)  # Mean across batch
    
    def iou_score(self, 
                  pred: torch.Tensor, 
                  target: torch.Tensor, 
                  smooth: float = 1e-8) -> torch.Tensor:
        """
        Calculate IoU (Intersection over Union) score.
        
        Args:
            pred: Predicted segmentation [B, C, D, H, W]
            target: Ground truth segmentation [B, D, H, W]
            smooth: Smoothing factor
            
        Returns:
            IoU score per class
        """
        pred_labels = torch.argmax(pred, dim=1)
        
        iou_per_class = []
        start_class = 1 if self.ignore_background else 0
        
        for class_id in range(start_class, self.num_classes):
            pred_mask = (pred_labels == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).float().sum(dim=(1, 2, 3))
            union = (pred_mask | target_mask).float().sum(dim=(1, 2, 3))
            
            iou = (intersection + smooth) / (union + smooth)
            iou_per_class.append(iou.mean())
        
        return torch.stack(iou_per_class)
    
    def hausdorff_distance_3d(self, 
                             pred: np.ndarray, 
                             target: np.ndarray,
                             voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """
        Calculate 3D Hausdorff distance between predicted and target surfaces.
        
        Args:
            pred: Predicted binary mask [D, H, W]
            target: Target binary mask [D, H, W]
            voxel_spacing: Voxel spacing in mm
            
        Returns:
            Hausdorff distance in mm
        """
        from scipy.spatial.distance import directed_hausdorff
        
        # Extract surface points
        pred_surface = self._extract_surface_points(pred, voxel_spacing)
        target_surface = self._extract_surface_points(target, voxel_spacing)
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        h1 = directed_hausdorff(pred_surface, target_surface)[0]
        h2 = directed_hausdorff(target_surface, pred_surface)[0]
        
        return max(h1, h2)
    
    def _extract_surface_points(self, 
                               mask: np.ndarray, 
                               voxel_spacing: Tuple[float, float, float]) -> np.ndarray:
        """Extract surface points from 3D binary mask."""
        # Find edges using morphological operations
        from scipy import ndimage
        
        # Create structure element for 6-connectivity
        struct = ndimage.generate_binary_structure(3, 1)
        
        # Erode and find boundary
        eroded = ndimage.binary_erosion(mask, struct)
        boundary = mask & ~eroded
        
        # Get coordinates of boundary points
        coords = np.where(boundary)
        if len(coords[0]) == 0:
            return np.array([])
        
        # Apply voxel spacing
        surface_points = np.column_stack([
            coords[0] * voxel_spacing[0],
            coords[1] * voxel_spacing[1], 
            coords[2] * voxel_spacing[2]
        ])
        
        return surface_points
    
    def volume_similarity(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate volume similarity between predicted and target segmentations.
        
        Args:
            pred: Predicted segmentation [B, C, D, H, W]
            target: Ground truth segmentation [B, D, H, W]
            
        Returns:
            Volume similarity coefficient
        """
        pred_labels = torch.argmax(pred, dim=1)
        
        pred_volume = (pred_labels > 0).float().sum()
        target_volume = (target > 0).float().sum()
        
        if pred_volume == 0 and target_volume == 0:
            return 1.0
        
        volume_diff = torch.abs(pred_volume - target_volume)
        volume_sim = 1.0 - (volume_diff / (pred_volume + target_volume))
        
        return volume_sim.item()
    
    def depth_accuracy(self, 
                      pred_depth: torch.Tensor, 
                      target_depth: torch.Tensor,
                      threshold: float = 5.0) -> Dict[str, float]:
        """
        Calculate depth estimation accuracy metrics.
        
        Args:
            pred_depth: Predicted depth map [B, 1, H, W]
            target_depth: Ground truth depth map [B, 1, H, W]
            threshold: Threshold for accuracy calculation (mm)
            
        Returns:
            Dictionary of depth metrics
        """
        # Flatten tensors
        pred_flat = pred_depth.view(-1)
        target_flat = target_depth.view(-1)
        
        # Remove invalid depth values (assuming 0 means invalid)
        valid_mask = target_flat > 0
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        if len(pred_valid) == 0:
            return {'mae': float('inf'), 'rmse': float('inf'), 'accuracy': 0.0}
        
        # Mean Absolute Error
        mae = torch.abs(pred_valid - target_valid).mean()
        
        # Root Mean Square Error
        rmse = torch.sqrt(((pred_valid - target_valid) ** 2).mean())
        
        # Accuracy within threshold
        accurate = torch.abs(pred_valid - target_valid) < threshold
        accuracy = accurate.float().mean()
        
        return {
            'mae': mae.item(),
            'rmse': rmse.item(), 
            'accuracy': accuracy.item()
        }
    
    def tooth_detection_metrics(self, 
                               pred: torch.Tensor, 
                               target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate tooth detection metrics (precision, recall, F1).
        
        Args:
            pred: Predicted segmentation [B, C, D, H, W]
            target: Ground truth segmentation [B, D, H, W]
            
        Returns:
            Dictionary of detection metrics
        """
        pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
        target_labels = target.cpu().numpy()
        
        # Flatten for sklearn metrics
        pred_flat = pred_labels.flatten()
        target_flat = target_labels.flatten()
        
        # Calculate confusion matrix
        cm = confusion_matrix(target_flat, pred_flat, labels=range(self.num_classes))
        
        # Calculate per-class metrics
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        start_class = 1 if self.ignore_background else 0
        
        for i in range(start_class, self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
        
        return {
            'precision': np.mean(precision_per_class),
            'recall': np.mean(recall_per_class),
            'f1_score': np.mean(f1_per_class),
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
    
    def landmark_accuracy(self, 
                         pred_centers: torch.Tensor, 
                         target_centers: torch.Tensor,
                         threshold: float = 2.0) -> Dict[str, float]:
        """
        Calculate tooth landmark positioning accuracy.
        
        Args:
            pred_centers: Predicted tooth centers [B, N, 3]
            target_centers: Ground truth tooth centers [B, N, 3]
            threshold: Distance threshold for correct detection (mm)
            
        Returns:
            Dictionary of landmark metrics
        """
        # Calculate Euclidean distances
        distances = torch.norm(pred_centers - target_centers, dim=2)  # [B, N]
        
        # Accuracy within threshold
        accurate = distances < threshold
        accuracy = accurate.float().mean()
        
        # Mean distance error
        mean_error = distances.mean()
        
        # Standard deviation of errors
        std_error = distances.std()
        
        return {
            'landmark_accuracy': accuracy.item(),
            'mean_distance_error': mean_error.item(),
            'std_distance_error': std_error.item(),
            'max_distance_error': distances.max().item(),
            'distances_per_tooth': distances.mean(dim=0).tolist()
        }


def evaluate_reconstruction(results: Dict[str, torch.Tensor],
                          targets: Optional[Dict[str, torch.Tensor]] = None,
                          num_classes: int = 32) -> Dict[str, float]:
    """
    Comprehensive evaluation of reconstruction results.
    
    Args:
        results: Dictionary of reconstruction results
        targets: Dictionary of ground truth targets (optional)
        num_classes: Number of classes for segmentation
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    evaluator = DentalMetrics(num_classes=num_classes)
    
    # Extract results
    pred_segmentation = results.get('segmentation')  # [B, C, D, H, W]
    pred_depth = results.get('depth_map')  # [B, 1, H, W]
    
    if targets is None:
        # If no ground truth, calculate internal consistency metrics
        if pred_segmentation is not None:
            # Calculate segmentation statistics
            pred_labels = torch.argmax(pred_segmentation, dim=1)
            unique_classes = torch.unique(pred_labels)
            
            metrics['detected_classes'] = len(unique_classes.tolist())
            metrics['class_distribution'] = {
                int(cls): (pred_labels == cls).float().mean().item() 
                for cls in unique_classes
            }
        
        if pred_depth is not None:
            # Calculate depth statistics
            depth_values = pred_depth[pred_depth > 0]  # Exclude background
            if len(depth_values) > 0:
                metrics['depth_mean'] = depth_values.mean().item()
                metrics['depth_std'] = depth_values.std().item()
                metrics['depth_range'] = (depth_values.min().item(), depth_values.max().item())
        
        return metrics
    
    # Evaluation with ground truth
    target_segmentation = targets.get('segmentation')
    target_depth = targets.get('depth')
    
    # Segmentation metrics
    if pred_segmentation is not None and target_segmentation is not None:
        # Dice coefficient
        dice_scores = evaluator.dice_coefficient(pred_segmentation, target_segmentation)
        metrics['dice_mean'] = dice_scores.mean().item()
        metrics['dice_per_class'] = dice_scores.tolist()
        
        # IoU scores
        iou_scores = evaluator.iou_score(pred_segmentation, target_segmentation)
        metrics['iou_mean'] = iou_scores.mean().item()
        metrics['iou_per_class'] = iou_scores.tolist()
        
        # Volume similarity
        vol_sim = evaluator.volume_similarity(pred_segmentation, target_segmentation)
        metrics['volume_similarity'] = vol_sim
        
        # Tooth detection metrics
        detection_metrics = evaluator.tooth_detection_metrics(pred_segmentation, target_segmentation)
        metrics.update(detection_metrics)
    
    # Depth estimation metrics  
    if pred_depth is not None and target_depth is not None:
        depth_metrics = evaluator.depth_accuracy(pred_depth, target_depth)
        metrics.update({f'depth_{k}': v for k, v in depth_metrics.items()})
    
    # Calculate composite score
    if 'dice_mean' in metrics and 'depth_accuracy' in metrics:
        metrics['composite_score'] = (metrics['dice_mean'] + metrics['depth_accuracy']) / 2
    elif 'dice_mean' in metrics:
        metrics['composite_score'] = metrics['dice_mean']
    elif 'depth_accuracy' in metrics:
        metrics['composite_score'] = metrics['depth_accuracy']
    
    return metrics


class MetricsTracker:
    """Track metrics during training and validation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {
            'dice_scores': [],
            'iou_scores': [],
            'depth_errors': [],
            'landmark_errors': []
        }
    
    def update(self, batch_metrics: Dict[str, float]):
        """Update metrics with batch results."""
        for key, value in batch_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def compute_averages(self) -> Dict[str, float]:
        """Compute average metrics."""
        averages = {}
        for key, values in self.metrics.items():
            if values:
                averages[f'avg_{key}'] = np.mean(values)
                averages[f'std_{key}'] = np.std(values)
        return averages
    
    def get_best_metric(self, metric_name: str, mode: str = 'max') -> float:
        """Get best value for a specific metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return float('-inf') if mode == 'max' else float('inf')
        
        values = self.metrics[metric_name]
        return max(values) if mode == 'max' else min(values)
