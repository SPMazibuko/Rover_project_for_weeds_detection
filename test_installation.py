#!/usr/bin/env python3
"""
Test script to verify the dental 3D reconstruction pipeline installation
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import torch
        import numpy as np
        import cv2
        import yaml
        print("✅ Basic dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Error importing basic dependencies: {e}")
        return False
    
    try:
        # Test pipeline imports
        from dental_3d_reconstruction.models import DepthGAN, ResUNet3D
        from dental_3d_reconstruction.preprocessing import DentalImagePreprocessor
        from dental_3d_reconstruction.losses import ToothLandmarkLoss
        print("✅ Pipeline components imported successfully")
    except ImportError as e:
        print(f"❌ Error importing pipeline components: {e}")
        return False
    
    try:
        # Test main pipeline
        from dental_3d_reconstruction import DentalReconstructionPipeline
        print("✅ Main pipeline imported successfully")
    except ImportError as e:
        print(f"❌ Error importing main pipeline: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration file loading."""
    print("\n⚙️  Testing configuration...")
    
    config_path = Path("dental_3d_reconstruction/configs/config.yaml")
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['data', 'depthgan', 'resunet3d', 'training', 'loss']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing configuration section: {section}")
                return False
        
        print("✅ Configuration file loaded and validated")
        return True
    
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def test_model_creation():
    """Test model instantiation."""
    print("\n🏗️  Testing model creation...")
    
    try:
        import torch
        from dental_3d_reconstruction.models import DepthGAN, ResUNet3D
        
        # Test DepthGAN creation
        depthgan = DepthGAN(
            input_channels=1,
            latent_dim=100,
            gen_filters=[64, 128, 256],
            disc_filters=[64, 128],
            output_size=(256, 256),
            depth_range=(0.0, 100.0)
        )
        print("✅ DepthGAN model created successfully")
        
        # Test ResUNet3D creation
        resunet = ResUNet3D(
            in_channels=1,
            out_channels=32,
            base_filters=16,
            depth=3
        )
        print("✅ ResUNet3D model created successfully")
        
        return True
    
    except Exception as e:
        print(f"❌ Error creating models: {e}")
        traceback.print_exc()
        return False

def test_preprocessing():
    """Test preprocessing components."""
    print("\n🔧 Testing preprocessing...")
    
    try:
        import numpy as np
        from dental_3d_reconstruction.preprocessing import DentalImagePreprocessor
        
        # Create preprocessor
        preprocessor = DentalImagePreprocessor(
            target_size=(256, 256)
        )
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        processed, bbox = preprocessor.preprocess_image(dummy_image)
        
        if processed.shape == (256, 256):
            print("✅ Image preprocessing working correctly")
            return True
        else:
            print(f"❌ Incorrect processed image shape: {processed.shape}")
            return False
    
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        traceback.print_exc()
        return False

def test_loss_functions():
    """Test loss function creation."""
    print("\n📊 Testing loss functions...")
    
    try:
        import torch
        from dental_3d_reconstruction.losses import ToothLandmarkLoss, CombinedDentalLoss
        
        # Test ToothLandmarkLoss
        landmark_loss = ToothLandmarkLoss(num_teeth=32)
        
        # Test CombinedDentalLoss
        combined_loss = CombinedDentalLoss(num_teeth=32)
        
        print("✅ Loss functions created successfully")
        return True
    
    except Exception as e:
        print(f"❌ Error creating loss functions: {e}")
        traceback.print_exc()
        return False

def test_data_utils():
    """Test data utilities."""
    print("\n📁 Testing data utilities...")
    
    try:
        from dental_3d_reconstruction.utils.data_utils import create_sample_data
        
        # Test data creation (without actually creating files)
        print("✅ Data utilities imported successfully")
        return True
    
    except Exception as e:
        print(f"❌ Error with data utilities: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🦷 Dental 3D Reconstruction Pipeline - Installation Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_model_creation,
        test_preprocessing,
        test_loss_functions,
        test_data_utils
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\n💡 Next steps:")
        print("   • Run 'python demo.py --mode demo' for a complete demo")
        print("   • Run 'python dental_3d_reconstruction/pipeline.py --mode create_data' to generate sample data")
        print("   • Run 'python dental_3d_reconstruction/pipeline.py --mode train' to start training")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the installation.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
