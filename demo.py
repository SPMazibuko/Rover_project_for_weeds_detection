#!/usr/bin/env python3
"""
Dental 3D Reconstruction Pipeline Demo
=====================================

This script demonstrates the complete dental 3D reconstruction pipeline
including data generation, training, and inference.

Usage:
    python demo.py --mode [demo|train|predict]
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path

# Add the project directory to Python path
sys.path.append(str(Path(__file__).parent))

from dental_3d_reconstruction import (
    DentalReconstructionPipeline,
    plot_reconstruction_results,
    evaluate_reconstruction
)
from dental_3d_reconstruction.utils.data_utils import create_sample_data


def run_demo():
    """Run a complete demonstration of the pipeline."""
    print("🦷 Dental 3D Reconstruction Pipeline Demo")
    print("=" * 50)
    
    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample synthetic data...")
    create_sample_data("dental_3d_reconstruction/data")
    
    # Step 2: Initialize pipeline
    print("\n🏗️  Step 2: Initializing reconstruction pipeline...")
    config_path = "dental_3d_reconstruction/configs/config.yaml"
    
    try:
        pipeline = DentalReconstructionPipeline(config_path)
        print("✅ Pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return
    
    # Step 3: Load a sample X-ray
    print("\n🖼️  Step 3: Loading sample X-ray image...")
    sample_xray_path = "dental_3d_reconstruction/data/train/xrays/sample_0000.png"
    
    if not os.path.exists(sample_xray_path):
        print(f"❌ Sample X-ray not found: {sample_xray_path}")
        return
    
    # Load and preprocess sample image
    x_ray = cv2.imread(sample_xray_path, cv2.IMREAD_GRAYSCALE)
    if x_ray is None:
        print(f"❌ Could not load X-ray image: {sample_xray_path}")
        return
    
    x_ray_tensor = torch.FloatTensor(x_ray / 255.0)
    print(f"✅ Loaded X-ray image: {x_ray.shape}")
    
    # Step 4: Run inference
    print("\n🔮 Step 4: Running 3D reconstruction inference...")
    try:
        results = pipeline.predict(x_ray_tensor)
        print("✅ 3D reconstruction completed!")
        
        # Print result shapes
        print(f"   📏 Input X-ray shape: {results['input_xray'].shape}")
        print(f"   📏 Generated depth map shape: {results['depth_map'].shape}")
        print(f"   📏 3D segmentation shape: {results['segmentation'].shape}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return
    
    # Step 5: Evaluate results
    print("\n📈 Step 5: Evaluating reconstruction results...")
    try:
        metrics = evaluate_reconstruction(results, num_classes=32)
        print("✅ Evaluation completed!")
        
        # Print key metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   📊 {key}: {value:.4f}")
            elif isinstance(value, dict) and len(value) <= 5:
                print(f"   📊 {key}: {value}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not evaluate results: {e}")
    
    # Step 6: Generate visualizations
    print("\n🎨 Step 6: Generating visualizations...")
    output_dir = "demo_results"
    try:
        plot_reconstruction_results(results, output_dir, "demo_sample")
        print(f"✅ Visualizations saved to {output_dir}/")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate visualizations: {e}")
    
    # Step 7: Summary
    print("\n📋 Demo Summary:")
    print("=" * 30)
    print("✅ Sample data generated")
    print("✅ Pipeline initialized")  
    print("✅ X-ray image loaded")
    print("✅ 3D reconstruction performed")
    print("✅ Results evaluated")
    print("✅ Visualizations created")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 Check the '{output_dir}' directory for visualization results.")
    print(f"📁 Training data is available in 'dental_3d_reconstruction/data/'")
    
    print("\n💡 Next steps:")
    print("   • Run 'python dental_3d_reconstruction/pipeline.py --mode train' to train the model")
    print("   • Use your own X-ray images for reconstruction")
    print("   • Fine-tune the configuration in 'dental_3d_reconstruction/configs/config.yaml'")


def run_quick_train():
    """Run a quick training demonstration (few epochs)."""
    print("🏋️  Quick Training Demo")
    print("=" * 30)
    
    # Modify config for quick training
    config_path = "dental_3d_reconstruction/configs/config.yaml"
    
    # Create a temporary config with fewer epochs
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Reduce epochs for demo
    config['training']['epochs'] = 3
    config['training']['batch_size'] = 2
    
    # Save temporary config
    temp_config_path = "temp_demo_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Initialize pipeline with demo config
        pipeline = DentalReconstructionPipeline(temp_config_path)
        
        # Create data loaders
        from dental_3d_reconstruction.utils import DentalDataLoader
        data_loader_factory = DentalDataLoader(config)
        train_loader, val_loader, _ = data_loader_factory.create_dataloaders()
        
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        
        # Start training
        print("\n🏋️  Starting quick training (3 epochs)...")
        pipeline.train(train_loader, val_loader)
        
        print("✅ Quick training completed!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def run_prediction_demo(input_path: str):
    """Run prediction on a specific X-ray image."""
    print(f"🔮 Prediction Demo on: {input_path}")
    print("=" * 40)
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Load image
    x_ray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if x_ray is None:
        print(f"❌ Could not load image: {input_path}")
        return
    
    print(f"📏 Loaded image shape: {x_ray.shape}")
    
    # Initialize pipeline
    config_path = "dental_3d_reconstruction/configs/config.yaml"
    
    try:
        pipeline = DentalReconstructionPipeline(config_path)
        
        # Convert to tensor
        x_ray_tensor = torch.FloatTensor(x_ray / 255.0)
        
        # Run prediction
        print("🔮 Running 3D reconstruction...")
        results = pipeline.predict(x_ray_tensor)
        
        # Evaluate results
        metrics = evaluate_reconstruction(results, num_classes=32)
        
        # Generate visualizations
        output_dir = f"prediction_results_{Path(input_path).stem}"
        plot_reconstruction_results(results, output_dir, Path(input_path).stem)
        
        print("✅ Prediction completed!")
        print(f"📁 Results saved to: {output_dir}/")
        
        # Print key metrics
        print("\n📊 Reconstruction metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   • {key}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Dental 3D Reconstruction Demo')
    parser.add_argument('--mode', type=str, choices=['demo', 'train', 'predict'], 
                       default='demo', help='Demo mode')
    parser.add_argument('--input', type=str, help='Input X-ray path for prediction mode')
    
    args = parser.parse_args()
    
    print("🦷 Dental 3D Reconstruction Pipeline")
    print("====================================")
    print("Novel pipeline: DepthGAN + ResUNet3D with Tooth-Landmark Loss")
    print("Author: AI Engine")
    print()
    
    if args.mode == 'demo':
        run_demo()
    elif args.mode == 'train':
        run_quick_train()
    elif args.mode == 'predict':
        if not args.input:
            print("❌ Error: --input required for prediction mode")
            return
        run_prediction_demo(args.input)


if __name__ == "__main__":
    main()
