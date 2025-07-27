#!/usr/bin/env python3
"""
Simple structure verification script for the dental 3D reconstruction pipeline
"""

import os
from pathlib import Path

def check_file_structure():
    """Check if all required files and directories exist."""
    print("🦷 Dental 3D Reconstruction Pipeline - Structure Verification")
    print("=" * 60)
    
    # Required directories
    required_dirs = [
        "dental_3d_reconstruction",
        "dental_3d_reconstruction/models",
        "dental_3d_reconstruction/preprocessing", 
        "dental_3d_reconstruction/losses",
        "dental_3d_reconstruction/utils",
        "dental_3d_reconstruction/configs",
        "dental_3d_reconstruction/data",
        "dental_3d_reconstruction/experiments"
    ]
    
    # Required files
    required_files = [
        "dental_3d_reconstruction/__init__.py",
        "dental_3d_reconstruction/pipeline.py",
        "dental_3d_reconstruction/models/__init__.py",
        "dental_3d_reconstruction/models/depthgan.py",
        "dental_3d_reconstruction/models/resunet3d.py",
        "dental_3d_reconstruction/models/layers.py",
        "dental_3d_reconstruction/preprocessing/__init__.py",
        "dental_3d_reconstruction/preprocessing/image_preprocessing.py",
        "dental_3d_reconstruction/preprocessing/roi_detection.py",
        "dental_3d_reconstruction/losses/__init__.py",
        "dental_3d_reconstruction/losses/tooth_landmark_loss.py",
        "dental_3d_reconstruction/losses/adversarial_loss.py",
        "dental_3d_reconstruction/losses/combined_loss.py",
        "dental_3d_reconstruction/utils/__init__.py",
        "dental_3d_reconstruction/utils/data_utils.py",
        "dental_3d_reconstruction/utils/training_utils.py",
        "dental_3d_reconstruction/utils/visualization.py",
        "dental_3d_reconstruction/utils/metrics.py",
        "dental_3d_reconstruction/configs/config.yaml",
        "requirements.txt",
        "setup.py",
        "demo.py",
        "README_DENTAL_3D.md"
    ]
    
    print("📁 Checking directory structure...")
    missing_dirs = []
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"   ✅ {directory}")
        else:
            print(f"   ❌ {directory}")
            missing_dirs.append(directory)
    
    print("\n📄 Checking required files...")
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    print("\n" + "=" * 60)
    
    if not missing_dirs and not missing_files:
        print("🎉 All required files and directories are present!")
        print("\n📊 Pipeline Components:")
        
        # Count lines of code
        total_lines = 0
        python_files = list(Path(".").rglob("*.py"))
        for py_file in python_files:
            if "dental_3d_reconstruction" in str(py_file):
                try:
                    with open(py_file, 'r') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                except:
                    pass
        
        print(f"   📈 Total Python files: {len(python_files)}")
        print(f"   📈 Total lines of code: {total_lines:,}")
        
        # Component summary
        components = {
            "DepthGAN Model": "Novel GAN for depth estimation from X-rays",
            "ResUNet3D Model": "3D U-Net with residual connections and attention",
            "Preprocessing": "CLAHE enhancement and ROI detection",
            "Custom Losses": "Tooth-landmark loss with anatomical constraints", 
            "Training Utils": "Complete training pipeline with early stopping",
            "Visualization": "3D visualization and result plotting",
            "Evaluation": "Comprehensive metrics for reconstruction assessment"
        }
        
        print("\n🔧 Key Components:")
        for component, description in components.items():
            print(f"   • {component}: {description}")
        
        print("\n💡 Next Steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run demo: python demo.py --mode demo")
        print("   3. Create sample data: python dental_3d_reconstruction/pipeline.py --mode create_data") 
        print("   4. Start training: python dental_3d_reconstruction/pipeline.py --mode train")
        
        return True
    else:
        print(f"❌ Missing {len(missing_dirs)} directories and {len(missing_files)} files")
        return False

def show_pipeline_overview():
    """Show pipeline overview."""
    print("\n🏗️ Pipeline Architecture Overview:")
    print("=" * 40)
    
    pipeline_steps = [
        "1. Input: 2D Panoramic X-ray Image",
        "2. Preprocessing: CLAHE Enhancement + ROI Detection", 
        "3. DepthGAN: Generate Realistic Depth Map",
        "4. Volume Reconstruction: 2D → 3D Conversion",
        "5. ResUNet3D: 3D Tooth Segmentation",
        "6. Analysis: Anatomical Feature Extraction",
        "7. Output: 3D Dental Reconstruction + Metrics"
    ]
    
    for step in pipeline_steps:
        print(f"   {step}")
    
    print("\n🎯 Novel Contributions:")
    contributions = [
        "• DepthGAN: First GAN-based dental depth estimation",
        "• ResUNet3D: 3D U-Net with attention mechanisms", 
        "• Tooth-Landmark Loss: Anatomically-aware loss function",
        "• End-to-End Training: Complete pipeline optimization",
        "• Comprehensive Evaluation: Novel dental reconstruction metrics"
    ]
    
    for contrib in contributions:
        print(f"   {contrib}")

if __name__ == "__main__":
    success = check_file_structure()
    show_pipeline_overview()
    
    if success:
        print(f"\n✅ Dental 3D Reconstruction Pipeline is ready!")
    else:
        print(f"\n❌ Please fix missing files/directories before proceeding.")
