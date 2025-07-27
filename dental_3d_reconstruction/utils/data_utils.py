"""
Data utilities for dental 3D reconstruction pipeline
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import cv2
from pathlib import Path
import json
import nibabel as nib
from PIL import Image


class DentalDataset(Dataset):
    """
    Dataset class for dental X-ray and 3D volume data.
    Supports various data formats and augmentations.
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 target_size: Tuple[int, int] = (512, 512),
                 volume_size: Tuple[int, int, int] = (128, 128, 128),
                 load_3d: bool = True):
        """
        Initialize dental dataset.
        
        Args:
            data_dir: Path to data directory
            split: Data split ('train', 'val', 'test')
            transform: Data transformations
            target_size: Target 2D image size
            volume_size: Target 3D volume size
            load_3d: Whether to load 3D volumes
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.volume_size = volume_size
        self.load_3d = load_3d
        
        # Load data index
        self.data_index = self._load_data_index()
        
    def _load_data_index(self) -> List[Dict]:
        """Load data index file containing sample information."""
        index_file = self.data_dir / f"{self.split}_index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        else:
            # Create index from directory structure
            return self._create_data_index()
    
    def _create_data_index(self) -> List[Dict]:
        """Create data index from directory structure."""
        data_index = []
        
        # Expected directory structure:
        # data_dir/
        #   train/
        #     xrays/
        #     depths/ (optional)
        #     volumes/ (optional)
        #     segmentations/ (optional)
        
        split_dir = self.data_dir / self.split
        xray_dir = split_dir / 'xrays'
        
        if not xray_dir.exists():
            raise ValueError(f"X-ray directory not found: {xray_dir}")
        
        for xray_file in xray_dir.glob('*.png'):
            sample_id = xray_file.stem
            
            sample_info = {
                'id': sample_id,
                'xray_path': str(xray_file),
                'depth_path': None,
                'volume_path': None,
                'segmentation_path': None
            }
            
            # Check for corresponding files
            depth_file = split_dir / 'depths' / f"{sample_id}.png"
            if depth_file.exists():
                sample_info['depth_path'] = str(depth_file)
            
            volume_file = split_dir / 'volumes' / f"{sample_id}.nii.gz"
            if volume_file.exists():
                sample_info['volume_path'] = str(volume_file)
            
            seg_file = split_dir / 'segmentations' / f"{sample_id}.nii.gz"
            if seg_file.exists():
                sample_info['segmentation_path'] = str(seg_file)
            
            data_index.append(sample_info)
        
        # Save index for future use
        index_file = self.data_dir / f"{self.split}_index.json"
        with open(index_file, 'w') as f:
            json.dump(data_index, f, indent=2)
        
        return data_index
    
    def _load_image(self, path: str, is_depth: bool = False) -> np.ndarray:
        """Load and preprocess 2D image."""
        if is_depth:
            # Load depth map (assuming 16-bit PNG)
            image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            if image is None:
                raise ValueError(f"Could not load depth image: {path}")
            image = image.astype(np.float32)
        else:
            # Load X-ray image
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load X-ray image: {path}")
            image = image.astype(np.float32) / 255.0
        
        # Resize to target size
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        return image
    
    def _load_volume(self, path: str) -> np.ndarray:
        """Load and preprocess 3D volume."""
        try:
            # Load NIfTI volume
            nii_img = nib.load(path)
            volume = nii_img.get_fdata().astype(np.float32)
            
            # Resize to target volume size if needed
            if volume.shape != self.volume_size:
                from scipy import ndimage
                zoom_factors = [
                    self.volume_size[i] / volume.shape[i] 
                    for i in range(3)
                ]
                volume = ndimage.zoom(volume, zoom_factors, order=1)
            
            return volume
            
        except Exception as e:
            raise ValueError(f"Could not load volume {path}: {e}")
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample_info = self.data_index[idx]
        sample = {}
        
        # Load X-ray image
        xray = self._load_image(sample_info['xray_path'])
        sample['xray'] = torch.FloatTensor(xray).unsqueeze(0)  # Add channel dim
        
        # Load depth if available
        if sample_info['depth_path']:
            depth = self._load_image(sample_info['depth_path'], is_depth=True)
            sample['depth'] = torch.FloatTensor(depth).unsqueeze(0)
        
        # Load 3D volume if available and requested
        if self.load_3d and sample_info['volume_path']:
            volume = self._load_volume(sample_info['volume_path'])
            sample['volume'] = torch.FloatTensor(volume).unsqueeze(0)
        
        # Load segmentation if available
        if self.load_3d and sample_info['segmentation_path']:
            segmentation = self._load_volume(sample_info['segmentation_path'])
            sample['segmentation'] = torch.LongTensor(segmentation.astype(np.int64))
        
        # Apply transforms if specified
        if self.transform:
            sample = self.transform(sample)
        
        sample['id'] = sample_info['id']
        
        return sample


class DentalDataLoader:
    """
    Data loader factory for dental reconstruction pipeline.
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Create train, validation, and optionally test data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = DentalDataset(
            data_dir=self.config['data']['data_dir'],
            split='train',
            target_size=tuple(self.config['data']['input_size']),
            volume_size=tuple(self.config['data']['output_size']),
            load_3d=True
        )
        
        val_dataset = DentalDataset(
            data_dir=self.config['data']['data_dir'],
            split='val',
            target_size=tuple(self.config['data']['input_size']),
            volume_size=tuple(self.config['data']['output_size']),
            load_3d=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Optional test loader
        test_loader = None
        test_data_dir = Path(self.config['data']['data_dir']) / 'test'
        if test_data_dir.exists():
            test_dataset = DentalDataset(
                data_dir=self.config['data']['data_dir'],
                split='test',
                target_size=tuple(self.config['data']['input_size']),
                volume_size=tuple(self.config['data']['output_size']),
                load_3d=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,  # Process one at a time for testing
                shuffle=False,
                num_workers=1,
                pin_memory=True
            )
        
        return train_loader, val_loader, test_loader


class SyntheticDentalDataGenerator:
    """
    Generate synthetic dental data for testing and development.
    """
    
    def __init__(self, 
                 output_dir: str,
                 num_samples: int = 100,
                 image_size: Tuple[int, int] = (512, 512),
                 volume_size: Tuple[int, int, int] = (128, 128, 128)):
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.image_size = image_size
        self.volume_size = volume_size
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            for subdir in ['xrays', 'depths', 'volumes', 'segmentations']:
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    def generate_synthetic_xray(self) -> np.ndarray:
        """Generate synthetic dental X-ray image."""
        H, W = self.image_size
        
        # Create background
        xray = np.random.normal(0.2, 0.1, (H, W))
        
        # Add dental arch
        center_x, center_y = W // 2, H // 2
        arch_radius = min(H, W) // 3
        
        # Upper arch
        for angle in np.linspace(-np.pi/2, np.pi/2, 16):
            tooth_x = int(center_x + arch_radius * np.cos(angle))
            tooth_y = int(center_y - arch_radius * 0.3 + arch_radius * 0.2 * np.sin(angle))
            
            # Add tooth-like structure
            tooth_size = 15
            cv2.rectangle(xray, 
                         (tooth_x - tooth_size//2, tooth_y - tooth_size),
                         (tooth_x + tooth_size//2, tooth_y + tooth_size//2),
                         0.8, -1)
        
        # Lower arch
        for angle in np.linspace(np.pi/2, -np.pi/2, 16):
            tooth_x = int(center_x + arch_radius * 0.9 * np.cos(angle))
            tooth_y = int(center_y + arch_radius * 0.3 + arch_radius * 0.2 * np.sin(angle))
            
            # Add tooth-like structure
            tooth_size = 15
            cv2.rectangle(xray,
                         (tooth_x - tooth_size//2, tooth_y - tooth_size//2),
                         (tooth_x + tooth_size//2, tooth_y + tooth_size),
                         0.8, -1)
        
        # Add noise and normalize
        xray = np.clip(xray + np.random.normal(0, 0.05, xray.shape), 0, 1)
        
        return xray
    
    def generate_synthetic_depth(self, xray: np.ndarray) -> np.ndarray:
        """Generate synthetic depth map from X-ray."""
        # Simple depth generation based on intensity
        depth = (1 - xray) * 50  # Invert and scale to mm
        
        # Add some variation
        depth += np.random.normal(0, 2, depth.shape)
        depth = np.clip(depth, 0, 100)
        
        return depth
    
    def generate_data(self):
        """Generate complete synthetic dataset."""
        splits = {
            'train': int(0.7 * self.num_samples),
            'val': int(0.2 * self.num_samples),
            'test': int(0.1 * self.num_samples)
        }
        
        sample_id = 0
        
        for split, count in splits.items():
            print(f"Generating {count} samples for {split} split...")
            
            for i in range(count):
                # Generate synthetic X-ray
                xray = self.generate_synthetic_xray()
                
                # Generate corresponding depth
                depth = self.generate_synthetic_depth(xray)
                
                # Save X-ray
                xray_path = self.output_dir / split / 'xrays' / f"sample_{sample_id:04d}.png"
                cv2.imwrite(str(xray_path), (xray * 255).astype(np.uint8))
                
                # Save depth
                depth_path = self.output_dir / split / 'depths' / f"sample_{sample_id:04d}.png"
                cv2.imwrite(str(depth_path), depth.astype(np.uint16))
                
                sample_id += 1
        
        print(f"Generated {self.num_samples} synthetic samples in {self.output_dir}")


def create_sample_data(output_dir: str = "dental_3d_reconstruction/data"):
    """Create sample synthetic data for testing."""
    generator = SyntheticDentalDataGenerator(output_dir, num_samples=50)
    generator.generate_data()
    print(f"Sample data created in {output_dir}")
