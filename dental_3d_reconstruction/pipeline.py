"""
Dental 3D Reconstruction Pipeline: DepthGAN + ResUNet3D
=======================================================

This script implements a novel pipeline for reconstructing 3D dental structures 
from 2D panoramic X-ray images using DepthGAN + ResUNet3D with Tooth-Landmark Loss.

Pipeline Overview:
2D Teeth Image -> Preprocessing (CLAHE, ROI) -> DepthGAN -> 3D Volume -> ResUNet3D -> Analysis

Novel Contributions:
- DepthGAN for realistic depth estimation
- ResUNet3D with anatomical-aware segmentation
- Custom tooth-landmark loss function
- End-to-end 3D dental reconstruction

Author: AI Engine
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
from tqdm import tqdm

# Import custom modules
from models import DepthGAN, ResUNet3D, HybridDental3DNet
from preprocessing import DentalImagePreprocessor, ROIDetector
from losses import CombinedDentalLoss
from utils import (
    DentalDataLoader, setup_training_environment, 
    LearningRateScheduler, create_sample_data
)


class DentalReconstructionPipeline:
    """
    Complete pipeline for dental 3D reconstruction from panoramic X-rays.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the reconstruction pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self._initialize_models()
        self._initialize_preprocessing()
        self._initialize_loss_functions()
        self._initialize_optimizers()
        
        # Setup training environment
        self.training_env = setup_training_environment(self.config)
        
        print("Dental 3D Reconstruction Pipeline initialized successfully!")
    
    def _initialize_models(self):
        """Initialize all neural network models."""
        print("Initializing models...")
        
        # DepthGAN for depth estimation
        self.depthgan = DepthGAN(
            input_channels=self.config['data']['input_size'][0] if len(self.config['data']['input_size']) > 2 else 1,
            latent_dim=self.config['depthgan']['latent_dim'],
            gen_filters=self.config['depthgan']['gen_filters'],
            disc_filters=self.config['depthgan']['disc_filters'],
            output_size=tuple(self.config['data']['input_size']),
            depth_range=tuple(self.config['depthgan']['depth_range'])
        ).to(self.device)
        
        # ResUNet3D for 3D segmentation
        self.resunet3d = ResUNet3D(
            in_channels=self.config['resunet3d']['in_channels'],
            out_channels=self.config['data']['num_classes'],
            base_filters=self.config['resunet3d']['base_filters'],
            depth=self.config['resunet3d']['depth'],
            dropout=self.config['resunet3d']['dropout']
        ).to(self.device)
        
        # Hybrid network combining volume processing and segmentation
        self.hybrid_net = HybridDental3DNet(
            depth_range=tuple(self.config['depthgan']['depth_range']),
            volume_size=tuple(self.config['data']['output_size']),
            num_classes=self.config['data']['num_classes']
        ).to(self.device)
        
        print(f"Models initialized - Parameters:")
        print(f"  DepthGAN: {sum(p.numel() for p in self.depthgan.parameters()):,}")
        print(f"  ResUNet3D: {sum(p.numel() for p in self.resunet3d.parameters()):,}")
        print(f"  Hybrid Net: {sum(p.numel() for p in self.hybrid_net.parameters()):,}")
    
    def _initialize_preprocessing(self):
        """Initialize preprocessing components."""
        print("Initializing preprocessing...")
        
        self.preprocessor = DentalImagePreprocessor(
            clahe_clip_limit=self.config['preprocessing']['clahe_clip_limit'],
            clahe_tile_grid_size=tuple(self.config['preprocessing']['clahe_tile_grid_size']),
            target_size=tuple(self.config['data']['input_size']),
            gaussian_sigma=self.config['preprocessing']['gaussian_blur_sigma']
        )
        
        self.roi_detector = ROIDetector()
        
        print("Preprocessing components initialized.")
    
    def _initialize_loss_functions(self):
        """Initialize loss functions."""
        print("Initializing loss functions...")
        
        self.combined_loss = CombinedDentalLoss(
            adversarial_weight=self.config['loss']['adversarial_weight'],
            landmark_weight=self.config['loss']['landmark_weight'],
            anatomy_weight=self.config['loss']['anatomy_weight'],
            reconstruction_weight=self.config['loss']['segmentation_weight'],
            depth_consistency_weight=self.config['loss']['depth_weight'],
            num_teeth=self.config['data']['num_classes']
        ).to(self.device)
        
        print("Loss functions initialized.")
    
    def _initialize_optimizers(self):
        """Initialize optimizers and schedulers."""
        print("Initializing optimizers...")
        
        # Generator optimizer (DepthGAN generator + ResUNet3D)
        generator_params = list(self.depthgan.generator.parameters()) + list(self.resunet3d.parameters())
        self.optimizer_G = optim.Adam(
            generator_params,
            lr=self.config['depthgan']['learning_rate'],
            betas=(self.config['depthgan']['beta1'], self.config['depthgan']['beta2'])
        )
        
        # Discriminator optimizer
        self.optimizer_D = optim.Adam(
            self.depthgan.discriminator.parameters(),
            lr=self.config['depthgan']['learning_rate'],
            betas=(self.config['depthgan']['beta1'], self.config['depthgan']['beta2'])
        )
        
        # Setup schedulers
        self.scheduler = LearningRateScheduler(
            optimizer={'generator': self.optimizer_G, 'discriminator': self.optimizer_D},
            scheduler_type='plateau'
        )
        
        print("Optimizers and schedulers initialized.")
    
    def preprocess_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of dental X-ray images.
        
        Args:
            batch: Batch of input data
            
        Returns:
            Preprocessed batch
        """
        xray_batch = batch['xray'].cpu().numpy()
        processed_batch = []
        
        for i in range(xray_batch.shape[0]):
            xray = xray_batch[i, 0]  # Remove channel dimension
            
            # Apply preprocessing
            processed_xray, _ = self.preprocessor.preprocess_image(
                xray, extract_roi=self.config['preprocessing']['roi_detection']
            )
            
            processed_batch.append(processed_xray)
        
        # Convert back to tensor
        processed_tensor = torch.FloatTensor(np.stack(processed_batch)).unsqueeze(1)
        batch['xray'] = processed_tensor.to(self.device)
        
        return batch
    
    def forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through the pipeline.
        
        Args:
            batch: Input batch
            
        Returns:
            Pipeline outputs
        """
        # Preprocess input
        batch = self.preprocess_batch(batch)
        xray = batch['xray']
        
        # Generate depth map using DepthGAN
        generated_depth = self.depthgan.generate_depth(xray)
        
        # Convert depth to 3D volume and segment using ResUNet3D
        segmentation, intermediate_features = self.hybrid_net(generated_depth, xray)
        
        # Get discriminator score for adversarial loss
        discriminator_score = self.depthgan.discriminate(xray, generated_depth)
        
        outputs = {
            'xray': xray,
            'depth': generated_depth,
            'segmentation': segmentation,
            'discriminator_score': discriminator_score,
            'intermediate_features': intermediate_features
        }
        
        return outputs
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of losses
        """
        # Forward pass
        outputs = self.forward_pass(batch)
        
        # Prepare targets if available
        targets = {}
        if 'depth' in batch:
            targets['depth'] = batch['depth'].to(self.device)
        if 'segmentation' in batch:
            targets['segmentation'] = batch['segmentation'].to(self.device)
        
        # === Train Discriminator ===
        self.optimizer_D.zero_grad()
        
        # Real samples
        if 'depth' in targets:
            real_pred = self.depthgan.discriminate(outputs['xray'], targets['depth'])
            fake_pred = self.depthgan.discriminate(outputs['xray'], outputs['depth'].detach())
            
            disc_outputs = {
                'real_x_ray': outputs['xray'],
                'real_depth': targets['depth'],
                'fake_depth': outputs['depth'].detach(),
                'real_pred': real_pred,
                'fake_pred': fake_pred
            }
            
            disc_losses = self.combined_loss(disc_outputs, mode='discriminator')
            disc_losses['total_discriminator'].backward()
            self.optimizer_D.step()
        else:
            disc_losses = {'total_discriminator': torch.tensor(0.0)}
        
        # === Train Generator ===
        self.optimizer_G.zero_grad()
        
        gen_losses = self.combined_loss(outputs, targets, mode='generator')
        gen_losses['total_generator'].backward()
        self.optimizer_G.step()
        
        # Combine all losses
        all_losses = {**gen_losses, **disc_losses}
        
        # Convert to float for logging
        return {k: v.item() if torch.is_tensor(v) else v for k, v in all_losses.items()}
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single validation step.
        
        Args:
            batch: Validation batch
            
        Returns:
            Dictionary of validation metrics
        """
        with torch.no_grad():
            outputs = self.forward_pass(batch)
            
            # Prepare targets
            targets = {}
            if 'depth' in batch:
                targets['depth'] = batch['depth'].to(self.device)
            if 'segmentation' in batch:
                targets['segmentation'] = batch['segmentation'].to(self.device)
            
            # Calculate losses
            gen_losses = self.combined_loss(outputs, targets, mode='generator')
            
            return {k: v.item() if torch.is_tensor(v) else v for k, v in gen_losses.items()}
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.depthgan.train()
        self.resunet3d.train()
        self.hybrid_net.train()
        
        epoch_losses = {}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            try:
                losses = self.train_step(batch)
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value
                
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'G_loss': losses.get('total_generator', 0.0),
                    'D_loss': losses.get('total_discriminator', 0.0)
                })
                
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        # Average losses
        if num_batches > 0:
            epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        return epoch_losses
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.depthgan.eval()
        self.resunet3d.eval()
        self.hybrid_net.eval()
        
        epoch_losses = {}
        num_batches = 0
        
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            try:
                losses = self.validate_step(batch)
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value
                
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'Val_loss': losses.get('total_generator', 0.0)})
                
            except Exception as e:
                print(f"Error in validation step: {e}")
                continue
        
        # Average losses
        if num_batches > 0:
            epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        return epoch_losses
    
    def train(self, train_loader, val_loader):
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("\nStarting training...")
        print(f"Training for {self.config['training']['epochs']} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            epoch_start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate_epoch(val_loader)
            
            # Update learning rate
            val_loss = val_losses.get('total_generator', float('inf'))
            self.scheduler.step(val_loss)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Get current learning rates
            learning_rates = self.scheduler.get_lr()
            
            # Log epoch results
            self.training_env['training_logger'].log_epoch(
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                learning_rates=learning_rates,
                epoch_time=epoch_time
            )
            
            # Save checkpoint
            models = {
                'depthgan': self.depthgan,
                'resunet3d': self.resunet3d,
                'hybrid_net': self.hybrid_net
            }
            
            optimizers = {
                'generator': self.optimizer_G,
                'discriminator': self.optimizer_D
            }
            
            schedulers = {
                'scheduler': self.scheduler
            }
            
            metrics = {**train_losses, **val_losses}
            
            self.training_env['model_checkpoint'].save_checkpoint(
                model=models,
                optimizer=optimizers,
                scheduler=schedulers,
                epoch=epoch,
                loss=val_loss,
                metrics=metrics
            )
            
            # Early stopping check
            if self.training_env['early_stopping'](val_loss, self.depthgan):
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.6f}")
        
        print("Training completed!")
    
    def predict(self, x_ray: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction on a single X-ray image.
        
        Args:
            x_ray: Input X-ray image [1, H, W] or [H, W]
            
        Returns:
            Dictionary with reconstruction results
        """
        self.depthgan.eval()
        self.resunet3d.eval()
        self.hybrid_net.eval()
        
        with torch.no_grad():
            # Ensure correct shape
            if len(x_ray.shape) == 2:
                x_ray = x_ray.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif len(x_ray.shape) == 3:
                x_ray = x_ray.unsqueeze(0)  # Add batch dim
            
            x_ray = x_ray.to(self.device)
            
            # Preprocess
            batch = {'xray': x_ray}
            batch = self.preprocess_batch(batch)
            
            # Forward pass
            outputs = self.forward_pass(batch)
            
            return {
                'input_xray': outputs['xray'],
                'depth_map': outputs['depth'],
                'segmentation': outputs['segmentation'],
                'intermediate_features': outputs['intermediate_features']
            }
    
    def save_model(self, path: str):
        """Save the complete pipeline."""
        models = {
            'depthgan': self.depthgan.state_dict(),
            'resunet3d': self.resunet3d.state_dict(),
            'hybrid_net': self.hybrid_net.state_dict()
        }
        
        torch.save({
            'models': models,
            'config': self.config
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the complete pipeline."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.depthgan.load_state_dict(checkpoint['models']['depthgan'])
        self.resunet3d.load_state_dict(checkpoint['models']['resunet3d'])
        self.hybrid_net.load_state_dict(checkpoint['models']['hybrid_net'])
        
        print(f"Model loaded from {path}")


def main():
    """Main function for training or inference."""
    parser = argparse.ArgumentParser(description='Dental 3D Reconstruction Pipeline')
    parser.add_argument('--config', type=str, default='dental_3d_reconstruction/configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'create_data'], default='train',
                       help='Mode: train, predict, or create_data')
    parser.add_argument('--input', type=str, help='Input X-ray image for prediction')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'create_data':
        # Create sample synthetic data
        print("Creating sample synthetic data...")
        create_sample_data("dental_3d_reconstruction/data")
        return
    
    # Initialize pipeline
    pipeline = DentalReconstructionPipeline(args.config)
    
    if args.mode == 'train':
        # Create data loaders
        data_loader_factory = DentalDataLoader(pipeline.config)
        train_loader, val_loader, _ = data_loader_factory.create_dataloaders()
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Start training
        pipeline.train(train_loader, val_loader)
        
    elif args.mode == 'predict':
        if not args.input:
            raise ValueError("Input X-ray path required for prediction mode")
        
        if args.checkpoint:
            pipeline.load_model(args.checkpoint)
        
        # Load and predict
        import cv2
        x_ray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if x_ray is None:
            raise ValueError(f"Could not load image: {args.input}")
        
        x_ray = torch.FloatTensor(x_ray / 255.0)
        
        # Make prediction
        results = pipeline.predict(x_ray)
        
        # Save results
        output_dir = Path(args.output) if args.output else Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save depth map
        depth_map = results['depth_map'][0, 0].cpu().numpy()
        depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "depth_map.png"), depth_normalized)
        
        # Save segmentation (simplified)
        segmentation = results['segmentation'][0].argmax(0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(str(output_dir / "segmentation.png"), segmentation * 8)  # Scale for visibility
        
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
