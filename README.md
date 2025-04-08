# Planktoscope Object Classifier

A GUI application for processing and classifying microscopy images from Planktoscope with support for both raw images and segmented objects.

## Screenshots
<div align="center">
  <img src="docs/screenshots/Main_Window.png" alt="Main Interface" width="80%">
  <p><em>The main application interface showing raw image processing mode</em></p>
  <br>
  <img src="docs/images/Object_Classification.png" alt="Object Classification" width="80%">
  <p><em>Classification of segmented objects with size filtering</em></p>
  <br>
  <img src="docs/images/Segmented_Mode.png" alt="Thumbnail Gallery" width="80%">
  <p><em>Thumbnail gallery of detected and classified objects</em></p>
</div>

## Features

- **Dual-mode interface**: Process both raw Planktoscope images and segmented objects
- **Interactive segmentation**: Detect objects in raw Planktoscope images using the segmenter logic
- **Classification support**: Classify detected objects using trained models
- **Visualization tools**: View segmentation masks and classification results
- **Batch processing**: Efficiently process and classify multiple images
- **Export functionality**: Export segmented objects with metadata for further analysis

## Installation

### Requirements

- macOS with Apple Silicon (M1/M2) or Intel processor
- Miniconda or Anaconda (recommended)
- CUDA-capable GPU recommended but not required for faster processing

### Quick Installation (Recommended for macOS)

For the most reliable setup on macOS, we provide an installation script that handles all dependencies correctly:

```bash
# Download the repository
git clone https://github.com/yourusername/planktoscope-classifier.git
cd planktoscope-classifier

# Make the installation script executable
chmod +x install.sh

# Run the installation script
./install.sh
```

The script will create a conda environment with all necessary dependencies in the correct order to avoid compatibility issues.

### Manual Installation

If you prefer to install manually, follow these steps:

1. Create a new conda environment with Python 3.10:
   ```bash
   conda create -n planktoscope python=3.10
   conda activate planktoscope
   ```

2. Install PyQt using conda:
   ```bash
   conda install -c conda-forge pyqt=5.15.7
   ```

3. Install TensorFlow system dependencies:
   ```bash
   conda install -c apple tensorflow-deps
   ```

4. Install NumPy with specific version constraints (important for macOS):
   ```bash
   pip install "numpy>=1.24.0,<1.25.0" --force-reinstall --no-cache-dir
   ```

5. Install the remaining Python packages:
   ```bash
   pip install scipy==1.10.1 tensorflow-macos==2.10.0 tensorflow-metal==0.6.0 tensorflow-hub==0.12.0
   pip install scikit-learn==1.2.2 scikit-image==0.20.0 pillow==9.5.0 matplotlib==3.7.1 pandas==2.0.1
   pip install opencv-python
   ```

### Setup for NVIDIA GPU Users (Optional, for Linux/Windows)

If you have an NVIDIA GPU and want to enable GPU acceleration:

```bash
# Activate the environment first
conda activate planktoscope

# Install CUDA toolkit appropriate for your TensorFlow version
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Set environment variables for TensorFlow to find CUDA
# On Linux/macOS:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# On Windows (in Command Prompt):
set LD_LIBRARY_PATH=%LD_LIBRARY_PATH%;%CONDA_PREFIX%\lib\
```

## Usage

### Running the Application

```bash
# Make sure the environment is activated
conda activate planktoscope

# Run the application
python main.py
```
### Segmented Image Mode Workflow

In order to use the classification function, you will need to train a model using the Segmented Mode workflow first. See above

1. **Set Classification Folder**: Set the directory in which your class folders will be created
2. **Add Classes**: Add the number of classes that you wish you use for classification
3. **Load Segmented Objects Folder**: Use this to load the folder containing segmented images
4. **Train Model**: Click "Train Model" to use the class folders and train a custom model using the set epoch and batch sizes

### Raw Image Mode Workflow

In order to use the classification function in Raw Mode, you will need to train a model using the Segmented Mode workflow first. See above

1. **Load Raw Images**: Click "Load Raw Images" and select a folder containing the Planktoscope images
2. **Configure Parameters**: Adjust segmentation parameters in the left panel
3. **Segment Images**: Click "Segment Current Image" or "Segment All Images"
4. **Load Classification Model**: Use the menu to load a trained model (optional)
5. **Classify Objects**: Click "Classify Segmented Objects" to apply the model
6. **Export Results**: Click "Export Objects" to save the segmented and classified objects

### Size Filtering

Use the "Min Object Size" slider to filter objects based on size:
- Small values: Include more objects, potentially with more noise
- Large values: Focus on larger, more significant objects

## Development

### Managing Dependencies

If you add new dependencies to the project, update the requirements.txt file:

```bash
# After installing new packages with pip
pip freeze > requirements.txt
# Edit the file to remove unnecessary packages if needed
```

### Environment Maintenance

```bash
# Update all packages
conda update --all

# Add a new package
conda install -c conda-forge new-package
```

## Troubleshooting

### Common Issues

1. **macOS NumPy/OpenCV compatibility issues**:
   - If you encounter import errors with OpenCV or NumPy, try:
   ```bash
   pip install "numpy>=1.24.0,<1.25.0" --force-reinstall --no-cache-dir
   pip uninstall -y opencv-python
   pip install opencv-python
   ```

2. **Memory errors during segmentation or classification**:
   - Process fewer images at once
   - Reduce image resolution if possible
   - Close other memory-intensive applications

3. **Missing dependencies**:
   - Ensure you've activated the correct environment: `conda activate planktoscope`
   - Verify installations: `conda list` to check installed packages

4. **TensorFlow issues on macOS**:
   - Make sure you have both tensorflow-macos and tensorflow-metal installed
   - Check that you're using compatible versions (2.10.0 recommended)

## Advanced Configuration

The application uses image processing techniques from OpenCV and scikit-image for segmentation. The parameters can be tuned for specific types of Planktoscope images:

- **Min Object Diameter**: The minimum diameter of objects to detect (in μm)
- **Pixel Size**: The physical size of each pixel (in μm/px)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
