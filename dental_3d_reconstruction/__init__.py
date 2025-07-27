"""
Dental 3D Reconstruction Pipeline: DepthGAN + ResUNet3D
=======================================================

A novel pipeline for reconstructing 3D dental structures from 2D panoramic X-ray images.

Pipeline Components:
- DepthGAN: Generative adversarial network for depth estimation
- ResUNet3D: 3D residual U-Net for volumetric segmentation
- Custom preprocessing with CLAHE and ROI detection
- Tooth-landmark loss for anatomically-aware training
- Comprehensive evaluation metrics

Usage:
    from dental_3d_reconstruction import DentalReconstructionPipeline
    
    pipeline = DentalReconstructionPipeline('configs/config.yaml')
    results = pipeline.predict(x_ray_image)
"""

from .pipeline import DentalReconstructionPipeline
from .models import DepthGAN, ResUNet3D, HybridDental3DNet
from .preprocessing import DentalImagePreprocessor, ROIDetector
from .losses import ToothLandmarkLoss, CombinedDentalLoss
from .utils import (
    DentalDataLoader, Visualizer3D, DentalMetrics,
    plot_reconstruction_results, evaluate_reconstruction
)

__version__ = "1.0.0"
__author__ = "AI Engine"

__all__ = [
    # Main pipeline
    'DentalReconstructionPipeline',
    
    # Models
    'DepthGAN', 'ResUNet3D', 'HybridDental3DNet',
    
    # Preprocessing
    'DentalImagePreprocessor', 'ROIDetector',
    
    # Losses
    'ToothLandmarkLoss', 'CombinedDentalLoss',
    
    # Utilities
    'DentalDataLoader', 'Visualizer3D', 'DentalMetrics',
    'plot_reconstruction_results', 'evaluate_reconstruction'
]
