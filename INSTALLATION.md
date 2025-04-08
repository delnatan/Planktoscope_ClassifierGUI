# Detailed Installation Guide

This guide provides comprehensive instructions for setting up the Planktoscope Classifier application on different operating systems.

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

### Universal Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/babo989/Planktoscope_ClassifierGUI.git
   cd Planktoscope_ClassifierGUI
   ```

2. **Run the Installer** (Linux/macOS)
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

Or follow the manual steps below for detailed installation by OS.

---

### Windows

1. **Install Python**
   - Download Python from [python.org](https://www.python.org/downloads/windows/)
   - During installation, check "Add Python to PATH"
   - Verify installation:
     ```bash
     python --version
     ```

2. **Install Git** (if not already installed)
   - Download from [git-scm.com](https://git-scm.com/download/win)

3. **Clone the Repository**
   ```bash
   git clone https://github.com/babo989/Planktoscope_ClassifierGUI.git
   cd Planktoscope_ClassifierGUI
   ```

4. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Optional: Enable GPU Support**
   - Install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn)
   - Make sure your TensorFlow version matches your CUDA version

### macOS

1. **Install Homebrew** (if not already installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python & Git**
   ```bash
   brew install python git
   ```

3. **Clone the Repository**
   ```bash
   git clone https://github.com/babo989/Planktoscope_ClassifierGUI.git
   cd Planktoscope_ClassifierGUI
   ```

4. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Linux (Ubuntu/Debian)

1. **Update Package Manager**
   ```bash
   sudo apt update && sudo apt upgrade
   ```

2. **Install Python, pip, venv, and Git**
   ```bash
   sudo apt install python3 python3-pip python3-venv git
   ```

3. **Clone the Repository**
   ```bash
   git clone https://github.com/babo989/Planktoscope_ClassifierGUI.git
   cd Planktoscope_ClassifierGUI
   ```

4. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Optional: Enable GPU Support**
   - Follow the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu)

## Verifying Installation

1. **Check TensorFlow**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU'))"
   ```

2. **Run the Application**
   ```bash
   python main.py
   ```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'tensorflow'
- Activate your virtual environment
- Run:
  ```bash
  pip install --upgrade tensorflow
  ```

#### CUDA/GPU not detected
- Verify NVIDIA drivers:
  ```bash
  nvidia-smi
  ```
- Check CUDA version:
  ```bash
  nvcc --version
  ```
- Make sure your TensorFlow version matches your CUDA

#### PyQt5 not found
- On Linux:
  ```bash
  sudo apt-get install python3-pyqt5
  ```
- On macOS:
  ```bash
  brew install pyqt5
  ```

#### Memory Errors
- Reduce batch size
- Restart app and close background processes

For more help, visit the [GitHub Issues](https://github.com/babo989/Planktoscope_ClassifierGUI/issues) page.
