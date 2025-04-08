#!/bin/bash

# Exit on error
set -e

echo "Setting up Planktoscope Object Classifier environment..."

# Remove existing environment if it exists
conda deactivate 2>/dev/null || true
conda env remove -n planktoscope 2>/dev/null || true

# Create a new conda environment
echo "Creating conda environment with Python 3.10..."
conda create -n planktoscope python=3.10 -y

# Activate the environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate planktoscope

# Install PyQt through conda
echo "Installing PyQt via conda..."
conda install -c conda-forge pyqt=5.15.7 -y

# Install TensorFlow system dependencies
echo "Installing TensorFlow dependencies..."
conda install -c apple tensorflow-deps -y

# Install NumPy with force-reinstall to ensure correct version
echo "Installing NumPy..."
pip install "numpy>=1.24.0,<1.25.0" --force-reinstall --no-cache-dir

# Install remaining packages
echo "Installing remaining packages..."
pip install scipy==1.10.1 tensorflow-macos==2.10.0 tensorflow-metal==0.6.0 tensorflow-hub==0.12.0 
pip install scikit-learn==1.2.2 scikit-image==0.20.0 pillow==9.5.0 matplotlib==3.7.1 pandas==2.0.1
pip install opencv-python

echo ""
echo "Setup complete! To activate the environment, run:"
echo "conda activate planktoscope"
echo ""
echo "To run the application:"
echo "python main.py"