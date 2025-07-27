"""
Setup script for Dental 3D Reconstruction Pipeline
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README_DENTAL_3D.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dental-3d-reconstruction",
    version="1.0.0",
    author="AI Engine",
    author_email="contact@dental3d.ai",
    description="Novel pipeline for 3D dental reconstruction from 2D panoramic X-rays using DepthGAN + ResUNet3D",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/dental-3d-reconstruction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "isort>=5.0",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dental-3d-train=dental_3d_reconstruction.pipeline:main",
            "dental-3d-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dental_3d_reconstruction": [
            "configs/*.yaml",
            "configs/*.yml",
        ],
    },
    keywords="dental, 3d-reconstruction, medical-imaging, deep-learning, gan, unet, pytorch",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/dental-3d-reconstruction/issues",
        "Source": "https://github.com/your-repo/dental-3d-reconstruction",
        "Documentation": "https://dental-3d-reconstruction.readthedocs.io/",
    },
)
