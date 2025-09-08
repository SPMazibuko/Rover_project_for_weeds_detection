#!/usr/bin/env python3
"""
Jupyter Notebook Setup Script for Dental 3D Reconstruction Pipeline
==================================================================

This script helps set up the environment for running the dental 3D reconstruction
pipeline in Jupyter notebooks.

Usage:
    python setup_jupyter.py [--install-extensions]
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse


def check_jupyter_installation():
    """Check if Jupyter is installed."""
    try:
        import jupyter
        print("✅ Jupyter is installed")
        return True
    except ImportError:
        print("❌ Jupyter is not installed")
        return False


def install_jupyter_requirements():
    """Install Jupyter-specific requirements."""
    print("📦 Installing Jupyter requirements...")
    
    requirements = [
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.0", 
        "plotly>=5.0.0",
        "jupyterlab>=3.0.0"
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"   ✅ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {req}")


def install_jupyter_extensions():
    """Install and enable Jupyter extensions."""
    print("🔧 Installing Jupyter extensions...")
    
    extensions = [
        "jupyter nbextension enable --py widgetsnbextension",
        "jupyter labextension install @jupyter-widgets/jupyterlab-manager",
        "jupyter labextension install plotlywidget"
    ]
    
    for ext_cmd in extensions:
        try:
            subprocess.check_call(ext_cmd.split())
            print(f"   ✅ Installed: {ext_cmd}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️ Warning: Could not install {ext_cmd}")
            

def setup_notebook_kernel():
    """Set up a custom kernel for the dental reconstruction environment."""
    print("🧠 Setting up notebook kernel...")
    
    kernel_name = "dental-3d-reconstruction"
    display_name = "Dental 3D Reconstruction"
    
    try:
        # Install the current environment as a kernel
        subprocess.check_call([
            sys.executable, "-m", "ipykernel", "install", 
            "--user", "--name", kernel_name, "--display-name", display_name
        ])
        print(f"   ✅ Kernel '{display_name}' installed successfully")
    except subprocess.CalledProcessError:
        print(f"   ❌ Failed to install kernel")


def create_jupyter_config():
    """Create Jupyter configuration for optimal visualization."""
    print("⚙️ Creating Jupyter configuration...")
    
    config_dir = Path.home() / ".jupyter"
    config_dir.mkdir(exist_ok=True)
    
    config_content = '''
# Jupyter configuration for Dental 3D Reconstruction Pipeline
c = get_config()

# Enable inline plotting
c.IPKernelApp.matplotlib = 'inline'

# Increase output limit for large visualizations
c.NotebookApp.iopub_data_rate_limit = 10000000

# Allow widgets
c.NotebookApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self'; report-uri /api/security/csp-report; default-src 'none'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' ws: wss:; base-uri 'self'; form-action 'self';"
    }
}

# Notebook extensions
c.NotebookApp.nbserver_extensions = {
    'jupyter_nbextensions_configurator': True,
    'widgetsnbextension': True
}
'''
    
    config_file = config_dir / "jupyter_notebook_config.py"
    try:
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"   ✅ Configuration saved to {config_file}")
    except Exception as e:
        print(f"   ❌ Failed to create config: {e}")


def verify_setup():
    """Verify that the setup is working correctly."""
    print("🔍 Verifying setup...")
    
    # Check if key modules can be imported
    test_imports = [
        "jupyter",
        "ipywidgets", 
        "plotly",
        "matplotlib",
        "numpy",
        "torch"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            failed_imports.append(module)
            print(f"   ❌ {module}")
    
    if failed_imports:
        print(f"\n⚠️ Warning: Failed to import {failed_imports}")
        print("   Please install missing packages manually.")
    else:
        print("\n🎉 All required modules are available!")


def print_usage_instructions():
    """Print instructions for using the notebook."""
    print("\n📚 Usage Instructions:")
    print("=" * 50)
    print("1. Start Jupyter Notebook or JupyterLab:")
    print("   jupyter notebook")
    print("   # or")
    print("   jupyter lab")
    print()
    print("2. Open the demo notebook:")
    print("   dental_3d_reconstruction_demo.ipynb")
    print()
    print("3. Select the 'Dental 3D Reconstruction' kernel if available")
    print()
    print("4. Run the cells to see the interactive demo!")
    print()
    print("💡 Tips:")
    print("   • Use %matplotlib inline for static plots")
    print("   • Use %matplotlib widget for interactive plots")
    print("   • Restart kernel if you encounter import issues")
    print("   • Enable widget extensions for interactive visualizations")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup Jupyter for Dental 3D Reconstruction')
    parser.add_argument('--install-extensions', action='store_true',
                       help='Install Jupyter extensions (may require admin rights)')
    
    args = parser.parse_args()
    
    print("🦷 Dental 3D Reconstruction - Jupyter Setup")
    print("=" * 50)
    
    # Check current environment
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Install Jupyter if needed
    if not check_jupyter_installation():
        print("📦 Installing Jupyter...")
        install_jupyter_requirements()
    
    # Install requirements
    install_jupyter_requirements()
    
    # Install extensions if requested
    if args.install_extensions:
        install_jupyter_extensions()
    
    # Setup kernel
    setup_notebook_kernel()
    
    # Create configuration
    create_jupyter_config()
    
    # Verify setup
    verify_setup()
    
    # Print usage instructions
    print_usage_instructions()
    
    print(f"\n🎉 Jupyter setup complete!")
    print(f"🚀 Ready to run dental_3d_reconstruction_demo.ipynb")


if __name__ == "__main__":
    main()
