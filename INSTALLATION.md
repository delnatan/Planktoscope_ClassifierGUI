# Detailed Installation Guide

This guide provides comprehensive instructions for setting up the Planktoscope Object Classifier application on different operating systems.

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- 2GB free disk space

### Recommended Requirements
- Python 3.10
- 8GB RAM
- For GPU acceleration: 
  - macOS: Apple Silicon (M1/M2)
  - Windows/Linux: NVIDIA GPU with at least 4GB VRAM, CUDA 11.2 or higher
- 5GB free disk space

## Installation Methods

We provide two installation methods:
1. **Anaconda/Miniconda** (Recommended): Better environment management and dependency handling
2. **Python Virtual Environment**: For users who prefer standard Python tools

## Installation Steps

### Pre-requisites

#### Installing Anaconda/Miniconda (Recommended)

You have two options for installing Anaconda/Miniconda:

##### Option 1: Use our Install Script

The easiest way to install Miniconda is to use our install script which will automatically detect your system and install the appropriate version:

```bash
# Download the repository
git clone https://github.com/yourusername/planktoscope-classifier.git
cd planktoscope-classifier

# Make the installation script executable
chmod +x install.sh

# Run the installation script and follow the prompts
./install.sh
```

The script will detect if Miniconda is missing and offer to install it for you with the correct version for your platform.

##### Option 2: Manual Installation

If you prefer to install Miniconda manually:

###### Windows
1. Download Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Run the installer and follow the on-screen instructions
3. Verify installation by opening Anaconda Prompt and typing:
   ```
   conda --version
   ```

###### macOS
1. Download Miniconda for macOS from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Open Terminal and navigate to the download location
3. Make the installer executable:
   ```
   chmod +x Miniconda3-latest-MacOSX-arm64.sh  # For Apple Silicon
   # OR
   chmod +x Miniconda3-latest-MacOSX-x86_64.sh  # For Intel Macs
   ```
4. Run the installer:
   ```
   ./Miniconda3-latest-MacOSX-arm64.sh  # For Apple Silicon
   # OR
   ./Miniconda3-latest-MacOSX-x86_64.sh  # For Intel Macs
   ```
5. Follow the prompts and allow the installer to initialize Miniconda
6. Close and reopen Terminal, then verify installation:
   ```
   conda --version
   ```

###### Linux
1. Download Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Open Terminal and navigate to the download location
3. Make the installer executable:
   ```
   chmod +x Miniconda3-latest-Linux-x86_64.sh
   ```
4. Run the installer:
   ```
   ./Miniconda3-latest-Linux-x86_64.sh
   ```
5. Follow the prompts and allow the installer to initialize Miniconda
6. Close and reopen Terminal, then verify installation:
   ```
   conda --version
   ```

#### Installing Git (Required for both methods)

##### Windows
- Download from [git-scm.com](https://git-scm.com/download/win)
- Follow default installation options

##### macOS
```
# If Homebrew is installed
brew install git

# If Homebrew is not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git
```

##### Linux (Ubuntu/Debian)
```
sudo apt update
sudo apt install git
```

### Method 1: Installation with Anaconda/Miniconda (Recommended)

#### Windows/Linux/macOS

1. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/planktoscope-classifier.git
   cd planktoscope-classifier
   ```

2. **Create a Conda Environment**
   ```
   conda create -n planktoscope python=3.10
   conda activate planktoscope
   ```

3. **Install Dependencies**

   ##### For macOS (Apple Silicon or Intel)
   ```bash
   # Install PyQt using conda
   conda install -c conda-forge pyqt=5.15.7

   # For Apple Silicon, install TensorFlow dependencies
   conda install -c apple tensorflow-deps

   # Install NumPy with specific version constraints (important for macOS)
   pip install "numpy>=1.24.0,<1.25.0" --force-reinstall --no-cache-dir

   # Install remaining packages
   pip install scipy==1.10.1 tensorflow-macos==2.10.0 tensorflow-metal==0.6.0 tensorflow-hub==0.12.0
   pip install scikit-learn==1.2.2 scikit-image==0.20.0 pillow==9.5.0 matplotlib==3.7.1 pandas==2.0.1
   pip install opencv-python
   ```

   ##### For Windows/Linux
   ```bash
   # Install PyQt using conda
   conda install -c conda-forge pyqt=5.15.7

   # Install NumPy and other scientific packages
   conda install -c conda-forge numpy=1.24.0 scipy=1.10.1 scikit-learn=1.2.2 scikit-image=0.20.0
   conda install -c conda-forge pillow=9.5.0 matplotlib=3.7.1 pandas=2.0.1 opencv

   # Install TensorFlow
   pip install tensorflow==2.10.0 tensorflow-hub==0.12.0
   ```

4. **For GPU Support on Windows/Linux (optional)**
   ```bash
   # Install CUDA toolkit appropriate for TensorFlow 2.10.0
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

   # Set environment variables for TensorFlow to find CUDA
   # On Linux:
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

   # On Windows (in Command Prompt):
   set LD_LIBRARY_PATH=%LD_LIBRARY_PATH%;%CONDA_PREFIX%\lib\
   ```

### Method 2: Installation with Python Virtual Environment

#### Windows

1. **Install Python**
   - Download Python 3.10 from [python.org](https://www.python.org/downloads/windows/)
   - During installation, check "Add Python to PATH"
   - Verify installation by opening Command Prompt and typing:
     ```
     python --version
     ```

2. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/planktoscope-classifier.git
   cd planktoscope-classifier
   ```

3. **Create a Virtual Environment**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```
   pip install numpy==1.24.0 scipy==1.10.1 scikit-learn==1.2.2 scikit-image==0.20.0
   pip install pillow==9.5.0 matplotlib==3.7.1 pandas==2.0.1 opencv-python
   pip install tensorflow==2.10.0 tensorflow-hub==0.12.0
   pip install PyQt5==5.15.7
   ```

5. **For GPU Support** (optional)
   - Install NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)
   - Install CUDA Toolkit 11.2 from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Install cuDNN 8.1.0 from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

#### macOS

1. **Install Python**
   ```
   # If Homebrew is installed
   brew install python@3.10
   
   # If Homebrew is not installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install python@3.10
   ```

2. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/planktoscope-classifier.git
   cd planktoscope-classifier
   ```

3. **Create a Virtual Environment**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   # Install specific NumPy version (important for macOS)
   pip install "numpy>=1.24.0,<1.25.0" --force-reinstall --no-cache-dir
   
   # Install TensorFlow for macOS
   pip install tensorflow-macos==2.10.0 tensorflow-metal==0.6.0 tensorflow-hub==0.12.0
   
   # Install remaining packages
   pip install scipy==1.10.1 scikit-learn==1.2.2 scikit-image==0.20.0
   pip install pillow==9.5.0 matplotlib==3.7.1 pandas==2.0.1 opencv-python
   pip install PyQt5==5.15.7
   ```

#### Linux (Ubuntu/Debian)

1. **Install Python and Dependencies**
   ```
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev
   ```

2. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/planktoscope-classifier.git
   cd planktoscope-classifier
   ```

3. **Create a Virtual Environment**
   ```
   python3.10 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```
   pip install numpy==1.24.0 scipy==1.10.1 scikit-learn==1.2.2 scikit-image==0.20.0
   pip install pillow==9.5.0 matplotlib==3.7.1 pandas==2.0.1 opencv-python
   pip install tensorflow==2.10.0 tensorflow-hub==0.12.0
   pip install PyQt5==5.15.7
   ```

5. **For GPU Support** (optional)
   - Follow the [TensorFlow GPU support guide](https://www.tensorflow.org/install/gpu) for Linux
   - Install the appropriate CUDA Toolkit (11.2) and cuDNN (8.1.0)

## Quick Installation Script (Recommended for All Platforms)

We provide an automated installation script that handles all dependencies correctly. This script works on macOS, Linux, and Windows (via Git Bash) and will even install Miniconda if you don't have it already:

```bash
# Download the repository
git clone https://github.com/yourusername/planktoscope-classifier.git
cd planktoscope-classifier

# Make the installation script executable
chmod +x install.sh

# Run the installation script
./install.sh
```

What this script does:
1. Checks if Anaconda/Miniconda is installed
2. If not found, offers to download and install the appropriate Miniconda version for your platform
3. Creates a conda environment with Python 3.10
4. Auto-detects your operating system (macOS, Linux, Windows)
5. Installs all required dependencies based on your platform
6. Configures GPU support when available (Apple Silicon or NVIDIA GPU)

This automated approach ensures all dependencies are installed in the correct order and with compatible versions, avoiding common installation issues.

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

#### macOS NumPy/OpenCV compatibility issues
- If you encounter import errors with OpenCV or NumPy, try:
  ```bash
  pip install "numpy>=1.24.0,<1.25.0" --force-reinstall --no-cache-dir
  pip uninstall -y opencv-python
  pip install opencv-python
  ```

#### ImportError: No module named 'tensorflow'
- Make sure you've activated the virtual environment
- For macOS, verify you installed tensorflow-macos instead of tensorflow
- Try reinstalling: `pip install --upgrade tensorflow` or `pip install --upgrade tensorflow-macos`

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

For more issues, please check the [GitHub Issues](https://github.com/yourusername/planktoscope-classifier/issues) page or create a new issue.
