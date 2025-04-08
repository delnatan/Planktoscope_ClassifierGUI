# Detailed Installation Guide

This guide provides comprehensive instructions for setting up the Microscopy Classifier application on different operating systems.

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- 2GB free disk space

### Recommended Requirements
- Python 3.9 or higher
- 8GB RAM
- NVIDIA GPU with at least 4GB VRAM
- CUDA 11.2 or higher
- 5GB free disk space

## Installation Steps

### Windows

1. **Install Python**
   - Download Python from [python.org](https://www.python.org/downloads/windows/)
   - During installation, check "Add Python to PATH"
   - Verify installation by opening Command Prompt and typing:
     ```
     python --version
     ```

2. **Install Git** (if not already installed)
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Follow default installation options

3. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/microscopy-classifier.git
   cd microscopy-classifier
   ```

4. **Create a Virtual Environment**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

5. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

6. **For GPU Support** (optional)
   - Install NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)
   - Install CUDA Toolkit from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Install cuDNN from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
   - Uncomment the GPU-related lines in requirements.txt and run:
     ```
     pip install -r requirements.txt
     ```

### macOS

1. **Install Homebrew** (if not already installed)
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python**
   ```
   brew install python
   ```

3. **Install Git** (if not already installed)
   ```
   brew install git
   ```

4. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/microscopy-classifier.git
   cd microscopy-classifier
   ```

5. **Create a Virtual Environment**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

6. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

### Linux (Ubuntu/Debian)

1. **Update Package Manager**
   ```
   sudo apt update
   sudo apt upgrade
   ```

2. **Install Python and Dependencies**
   ```
   sudo apt install python3 python3-pip python3-venv git
   ```

3. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/microscopy-classifier.git
   cd microscopy-classifier
   ```

4. **Create a Virtual Environment**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

6. **For GPU Support** (optional)
   - Follow the [TensorFlow GPU support guide](https://www.tensorflow.org/install/gpu) for Linux
   - Uncomment the GPU-related lines in requirements.txt and run:
     ```
     pip install -r requirements.txt
     ```

## Verifying Installation

After installation, verify that everything is working correctly:

1. **Check TensorFlow Installation**
   ```python
   python -c "import tensorflow as tf; print(tf.__version__); print('GPU Available: ', len(tf.config.list_physical_devices('GPU'))>0)"
   ```
   This should print the TensorFlow version and whether a GPU is available.

2. **Run the Application**
   ```
   python main.py
   ```
   The application should start without errors.

## Troubleshooting

### Common Issues

#### ImportError: No module named 'tensorflow'
- Make sure you've activated the virtual environment
- Try reinstalling: `pip install --upgrade tensorflow`

#### CUDA/GPU not detected
- Verify NVIDIA drivers are installed: `nvidia-smi`
- Check CUDA installation: `nvcc --version`
- Ensure TensorFlow version is compatible with your CUDA version

#### PyQt5 installation issues
- On Linux: `sudo apt-get install python3-pyqt5`
- On macOS: `brew install pyqt5`

#### Memory errors during training
- Reduce batch size in the application settings
- Close other memory-intensive applications

For more issues, please check the [GitHub Issues](https://github.com/yourusername/microscopy-classifier/issues) page or create a new issue.