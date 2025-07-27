"""
Visualization utilities for dental 3D reconstruction results
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.patches as patches


class Visualizer3D:
    """3D visualization utilities for dental reconstruction results."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_2d_results(self, 
                       x_ray: np.ndarray,
                       depth_map: np.ndarray,
                       roi_mask: Optional[np.ndarray] = None,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 2D results: X-ray, depth map, and ROI.
        
        Args:
            x_ray: Original X-ray image
            depth_map: Generated depth map
            roi_mask: ROI mask (optional)
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        num_plots = 3 if roi_mask is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=self.figsize)
        
        if num_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Original X-ray
        axes[0].imshow(x_ray, cmap='gray')
        axes[0].set_title('Original X-ray', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Depth map
        depth_plot = axes[1].imshow(depth_map, cmap='viridis')
        axes[1].set_title('Generated Depth Map', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04, label='Depth (mm)')
        
        # ROI mask if provided
        if roi_mask is not None:
            axes[2].imshow(x_ray, cmap='gray', alpha=0.7)
            axes[2].imshow(roi_mask, cmap='Reds', alpha=0.3)
            axes[2].set_title('ROI Detection', fontsize=14, fontweight='bold')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_3d_volume(self, 
                      volume: np.ndarray,
                      slice_indices: Optional[List[int]] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 3D volume as multiple slices.
        
        Args:
            volume: 3D volume array [D, H, W]
            slice_indices: Specific slice indices to show
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        D, H, W = volume.shape
        
        if slice_indices is None:
            # Show slices at 25%, 50%, 75% depth
            slice_indices = [D//4, D//2, 3*D//4]
        
        num_slices = len(slice_indices)
        fig, axes = plt.subplots(1, num_slices, figsize=(5*num_slices, 5))
        
        if num_slices == 1:
            axes = [axes]
        
        for i, slice_idx in enumerate(slice_indices):
            if slice_idx < D:
                axes[i].imshow(volume[slice_idx], cmap='gray')
                axes[i].set_title(f'Axial Slice {slice_idx}/{D}', fontsize=12, fontweight='bold')
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_segmentation_results(self,
                                volume: np.ndarray,
                                segmentation: np.ndarray,
                                class_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot segmentation results overlaid on volume.
        
        Args:
            volume: Original 3D volume [D, H, W]
            segmentation: Segmentation mask [D, H, W]
            class_names: Names for segmentation classes
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        D, H, W = volume.shape
        middle_slice = D // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Axial view
        axes[0, 0].imshow(volume[middle_slice], cmap='gray')
        axes[0, 0].set_title('Axial - Original', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(volume[middle_slice], cmap='gray', alpha=0.7)
        axes[1, 0].imshow(segmentation[middle_slice], cmap='jet', alpha=0.5)
        axes[1, 0].set_title('Axial - Segmented', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Coronal view
        coronal_slice = H // 2
        axes[0, 1].imshow(volume[:, coronal_slice, :], cmap='gray')
        axes[0, 1].set_title('Coronal - Original', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 1].imshow(volume[:, coronal_slice, :], cmap='gray', alpha=0.7)
        axes[1, 1].imshow(segmentation[:, coronal_slice, :], cmap='jet', alpha=0.5)
        axes[1, 1].set_title('Coronal - Segmented', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Sagittal view
        sagittal_slice = W // 2
        axes[0, 2].imshow(volume[:, :, sagittal_slice], cmap='gray')
        axes[0, 2].set_title('Sagittal - Original', fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 2].imshow(volume[:, :, sagittal_slice], cmap='gray', alpha=0.7)
        axes[1, 2].imshow(segmentation[:, :, sagittal_slice], cmap='jet', alpha=0.5)
        axes[1, 2].set_title('Sagittal - Segmented', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_curves(self,
                           train_metrics: List[Dict],
                           val_metrics: List[Dict],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training curves.
        
        Args:
            train_metrics: List of training metrics per epoch
            val_metrics: List of validation metrics per epoch
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        epochs = [m['epoch'] for m in train_metrics]
        
        # Find common loss keys
        train_keys = set(train_metrics[0].keys()) - {'epoch', 'time'}
        val_keys = set(val_metrics[0].keys()) - {'epoch'}
        common_keys = train_keys.intersection(val_keys)
        
        if not common_keys:
            print("No common metrics found between training and validation")
            return None
        
        num_plots = len(common_keys)
        fig, axes = plt.subplots(2, (num_plots + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if num_plots > 1 else [axes]
        
        for i, key in enumerate(sorted(common_keys)):
            if i >= len(axes):
                break
                
            train_values = [m.get(key, 0) for m in train_metrics]
            val_values = [m.get(key, 0) for m in val_metrics]
            
            axes[i].plot(epochs, train_values, label=f'Train {key}', marker='o', markersize=3)
            axes[i].plot(epochs, val_values, label=f'Val {key}', marker='s', markersize=3)
            axes[i].set_title(f'{key.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(common_keys), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_3d_plot(self,
                                 volume: np.ndarray,
                                 segmentation: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None):
        """
        Create interactive 3D plot using Plotly.
        
        Args:
            volume: 3D volume array
            segmentation: 3D segmentation array (optional)
            save_path: Path to save HTML file
        """
        # Create isosurface from volume
        X, Y, Z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
        
        fig = go.Figure()
        
        # Add volume isosurface
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=volume.flatten(),
            isomin=volume.mean() - volume.std(),
            isomax=volume.mean() + volume.std(), 
            surface_count=3,
            colorscale='Viridis',
            showscale=True,
            name='Volume'
        ))
        
        # Add segmentation if provided
        if segmentation is not None:
            # Show only non-zero segmentation values
            mask = segmentation > 0
            if mask.any():
                fig.add_trace(go.Isosurface(
                    x=X[mask],
                    y=Y[mask],
                    z=Z[mask],
                    value=segmentation[mask],
                    isomin=1,
                    isomax=segmentation.max(),
                    surface_count=2,
                    colorscale='Reds',
                    opacity=0.7,
                    name='Segmentation'
                ))
        
        fig.update_layout(
            title='3D Dental Reconstruction',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


def plot_reconstruction_results(results: Dict[str, torch.Tensor], 
                              output_dir: str,
                              sample_id: str = "sample"):
    """
    Plot complete reconstruction results.
    
    Args:
        results: Dictionary of reconstruction results
        output_dir: Output directory for saving plots
        sample_id: Sample identifier
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = Visualizer3D()
    
    # Convert tensors to numpy
    x_ray = results['input_xray'][0, 0].cpu().numpy()
    depth_map = results['depth_map'][0, 0].cpu().numpy()
    segmentation = results['segmentation'][0].argmax(0).cpu().numpy()
    
    # Plot 2D results
    fig_2d = visualizer.plot_2d_results(
        x_ray=x_ray,
        depth_map=depth_map,
        save_path=str(output_path / f"{sample_id}_2d_results.png")
    )
    plt.close(fig_2d)
    
    # Create simple 3D volume from depth map for visualization
    D, H, W = 64, depth_map.shape[0], depth_map.shape[1]
    volume = np.zeros((D, H, W))
    
    # Back-project depth map to create volume
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_indices = (depth_normalized * (D - 1)).astype(int)
    
    for h in range(H):
        for w in range(W):
            d_idx = depth_indices[h, w]
            if 0 <= d_idx < D:
                volume[d_idx, h, w] = x_ray[h, w]
    
    # Plot 3D volume slices
    fig_3d = visualizer.plot_3d_volume(
        volume=volume,
        save_path=str(output_path / f"{sample_id}_3d_volume.png")
    )
    plt.close(fig_3d)
    
    # Create segmentation volume for visualization
    seg_volume = np.zeros((D, H, W))
    for h in range(H):
        for w in range(W):
            d_idx = depth_indices[h, w]
            if 0 <= d_idx < D:
                seg_volume[d_idx, h, w] = segmentation[h, w]
    
    # Plot segmentation results
    fig_seg = visualizer.plot_segmentation_results(
        volume=volume,
        segmentation=seg_volume,
        save_path=str(output_path / f"{sample_id}_segmentation.png")
    )
    plt.close(fig_seg)
    
    # Create interactive 3D plot
    visualizer.create_interactive_3d_plot(
        volume=volume,
        segmentation=seg_volume,
        save_path=str(output_path / f"{sample_id}_interactive_3d.html")
    )
    
    print(f"Visualization results saved to {output_path}")


def create_summary_report(results: Dict[str, torch.Tensor],
                        metrics: Dict[str, float],
                        output_path: str):
    """
    Create a summary report with key statistics.
    
    Args:
        results: Reconstruction results
        metrics: Evaluation metrics
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("Dental 3D Reconstruction Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Input statistics
        x_ray = results['input_xray'][0, 0].cpu().numpy()
        f.write(f"Input X-ray statistics:\n")
        f.write(f"  - Shape: {x_ray.shape}\n")
        f.write(f"  - Mean intensity: {x_ray.mean():.4f}\n")
        f.write(f"  - Std intensity: {x_ray.std():.4f}\n")
        f.write(f"  - Min/Max: {x_ray.min():.4f}/{x_ray.max():.4f}\n\n")
        
        # Depth map statistics
        depth_map = results['depth_map'][0, 0].cpu().numpy()
        f.write(f"Generated depth map statistics:\n")
        f.write(f"  - Shape: {depth_map.shape}\n")
        f.write(f"  - Mean depth: {depth_map.mean():.4f} mm\n")
        f.write(f"  - Std depth: {depth_map.std():.4f} mm\n")
        f.write(f"  - Depth range: {depth_map.min():.4f} - {depth_map.max():.4f} mm\n\n")
        
        # Segmentation statistics
        segmentation = results['segmentation'][0].cpu().numpy()
        unique_classes = np.unique(segmentation.argmax(0))
        f.write(f"Segmentation statistics:\n")
        f.write(f"  - Shape: {segmentation.shape}\n")
        f.write(f"  - Number of detected classes: {len(unique_classes)}\n")
        f.write(f"  - Detected class IDs: {unique_classes.tolist()}\n\n")
        
        # Evaluation metrics
        if metrics:
            f.write("Evaluation metrics:\n")
            for key, value in metrics.items():
                f.write(f"  - {key}: {value:.4f}\n")
        
        f.write(f"\nReport generated on: {np.datetime64('now')}\n")
    
    print(f"Summary report saved to {output_path}")
