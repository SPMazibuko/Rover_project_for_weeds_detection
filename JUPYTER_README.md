# 🦷 Dental 3D Reconstruction Pipeline - Jupyter Notebook Guide

## Overview

This guide helps you run the **Dental 3D Reconstruction Pipeline** in Jupyter notebooks with full interactive capabilities.

The pipeline uses **DepthGAN + ResUNet3D** with novel tooth-landmark loss to reconstruct 3D dental structures from 2D panoramic X-rays.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install requirements
pip install -r requirements.txt

# Setup Jupyter environment
python setup_jupyter.py --install-extensions
```

### 2. Launch Notebook

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### 3. Open Demo

Open `dental_3d_reconstruction_demo.ipynb` and run all cells!

## 📚 Notebook Features

### 🎨 Interactive Visualizations

- **Slice Navigation**: Browse through 3D volumes with interactive sliders
- **Multi-View Display**: Axial, sagittal, and coronal views
- **Colormap Selection**: Choose from various colormaps for depth and segmentation
- **Real-time Updates**: Instant visualization updates as you adjust parameters

### 🔧 Interactive Configuration

- **Model Parameters**: Adjust learning rates, batch sizes, and architecture settings
- **Training Demo**: Run quick training demonstrations (3 epochs)
- **Preprocessing Pipeline**: See step-by-step image enhancement

### 📊 Comprehensive Analysis

- **Metrics Visualization**: Interactive charts for reconstruction quality
- **Statistical Analysis**: Detailed evaluation of results
- **Progress Tracking**: Real-time training progress with `tqdm`

## 🧠 Pipeline Components

### Architecture Overview

```
2D X-ray → Preprocessing → DepthGAN → 3D Volume → ResUNet3D → Segmentation
```

### Key Components

1. **DepthGAN**: Generative adversarial network for depth estimation
2. **ResUNet3D**: 3D residual U-Net with attention mechanisms
3. **Tooth-Landmark Loss**: Anatomically-aware loss function
4. **Multi-scale Training**: Progressive training strategy

## 📖 Notebook Sections

### Section 1: Setup and Imports
- Environment configuration
- Package imports
- GPU/CPU detection

### Section 2: Data Creation
- Synthetic dental data generation
- Sample visualization
- Data statistics

### Section 3: Pipeline Initialization
- Model loading
- Configuration setup
- Architecture summary

### Section 4: Reconstruction Demo
- X-ray loading
- 3D reconstruction
- Result visualization

### Section 5: Interactive Analysis
- Slice-by-slice exploration
- Multi-view analysis
- Parameter adjustment

### Section 6: Evaluation
- Quality metrics
- Performance analysis
- Comparative visualization

### Section 7: Advanced Features
- Preprocessing pipeline
- Training demonstration
- Configuration tuning

## 🎯 Usage Examples

### Basic Reconstruction

```python
from dental_3d_reconstruction import DentalReconstructionPipeline

# Initialize pipeline
pipeline = DentalReconstructionPipeline('configs/config.yaml')

# Load X-ray
x_ray = load_xray_image('path/to/xray.png')

# Run reconstruction
results = pipeline.predict(x_ray)

# Interactive visualization
from dental_3d_reconstruction import create_interactive_notebook_viewer
interactive_viewer = create_interactive_notebook_viewer(results)
display(interactive_viewer)
```

### Custom Configuration

```python
# Interactive parameter tuning
import ipywidgets as widgets

def update_config(learning_rate=0.0002, batch_size=4):
    # Update configuration
    config['training']['learning_rate'] = learning_rate
    config['training']['batch_size'] = batch_size
    
# Create interactive widgets
widgets.interactive(update_config,
    learning_rate=widgets.FloatSlider(min=0.0001, max=0.01, step=0.0001),
    batch_size=widgets.IntSlider(min=1, max=16, step=1)
)
```

### Training Demo

```python
# Quick training demonstration
demo = NotebookDentalDemo()
demo.setup_data()
demo.initialize_pipeline()

# Run training with progress tracking
from tqdm.notebook import tqdm
for epoch in tqdm(range(3), desc="Training"):
    # Training code here
    pass
```

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```python
# If imports fail, try:
import sys
sys.path.append('/path/to/project')

# Or restart kernel
%load_ext autoreload
%autoreload 2
```

#### Visualization Issues
```python
# Enable inline plotting
%matplotlib inline

# Or for interactive plots
%matplotlib widget
```

#### Widget Problems
```bash
# Install widget extensions
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

#### Memory Issues
```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Reduce batch size
config['training']['batch_size'] = 2
```

### Performance Tips

1. **GPU Usage**: Enable CUDA for faster processing
2. **Memory Management**: Use smaller batch sizes for large volumes
3. **Visualization**: Use static plots for large datasets
4. **Caching**: Save intermediate results to avoid recomputation

## 📊 Expected Results

### Reconstruction Quality
- **Dice Score**: 0.75-0.85 for tooth segmentation
- **IoU**: 0.65-0.80 for anatomical structures
- **Hausdorff Distance**: <2mm for boundary accuracy

### Performance Metrics
- **Inference Time**: ~2-5 seconds per X-ray
- **Memory Usage**: ~2-4GB GPU memory
- **Training Time**: ~1-2 hours for full training

## 🎨 Visualization Gallery

The notebook provides rich visualizations including:

- **2D X-ray Display**: Original panoramic images
- **Depth Maps**: Generated depth estimations with color coding
- **3D Segmentations**: Multi-class tooth segmentation
- **Interactive Slicing**: Browse through 3D volumes
- **Metric Charts**: Performance evaluation graphs
- **Training Progress**: Real-time loss curves

## 🚀 Advanced Usage

### Custom Data

```python
# Load your own X-ray data
custom_xray = cv2.imread('your_xray.png', cv2.IMREAD_GRAYSCALE)
results = pipeline.predict(torch.FloatTensor(custom_xray / 255.0))
```

### Batch Processing

```python
# Process multiple X-rays
xray_paths = ['xray1.png', 'xray2.png', 'xray3.png']
all_results = []

for path in tqdm(xray_paths, desc="Processing"):
    xray = load_and_preprocess(path)
    result = pipeline.predict(xray)
    all_results.append(result)
```

### Export Results

```python
# Save reconstruction results
import numpy as np
np.save('reconstruction_results.npy', results['segmentation'].cpu().numpy())

# Export visualizations
plot_reconstruction_results(results, output_dir='results', sample_name='my_case')
```

## 📱 Applications

### Clinical Use Cases

1. **Orthodontic Planning**: 3D tooth movement analysis
2. **Implant Design**: Precise placement planning
3. **Pathology Detection**: Abnormality identification
4. **Surgical Planning**: 3D anatomical guidance
5. **Education**: Interactive dental anatomy learning

### Research Applications

1. **Algorithm Development**: Novel reconstruction methods
2. **Benchmarking**: Comparative studies
3. **Dataset Creation**: Synthetic data generation
4. **Validation Studies**: Clinical accuracy assessment

## 🎉 Get Started Now!

1. **Install**: `pip install -r requirements.txt`
2. **Setup**: `python setup_jupyter.py --install-extensions`
3. **Launch**: `jupyter notebook`
4. **Open**: `dental_3d_reconstruction_demo.ipynb`
5. **Run**: Execute all cells and explore!

---

**Happy Reconstructing! 🦷✨**

For questions or issues, check the main README.md or create an issue in the repository.
