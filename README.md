# Planktoscope_ClassifierGUI
A GUI application for training image classification models on Planktoscope images with data augmentation and class imbalance handling.



## Features

- **Dual-mode interface**: Process both raw microscopy images and segmented objects
- **Interactive classification**: View, select, and classify images in an intuitive interface
- **Model training**: Train custom classification models with optimized parameters for small datasets
- **Class balancing**: Automatically handles class imbalance using varable class weights
- **Data augmentation**: Implements gentle augmentation for improving model generalization
- **Batch processing**: Efficiently classifies multiple images in batches
- **TensorRT export**: Optimize models for NVIDIA GPU inference on Jetson edge compute devices (optional)

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU recommended but not required
- TensorFlow 2.x compatible system

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/microscopy-classifier.git
   cd microscopy-classifier
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

```bash
python main.py
```

### Workflow

1. **Set a classification folder**: This is where your labeled images and trained model will be stored. You will add classes, which will create folders as subdirectories
2. **Add classes**: Create classification categories using the "Add New Class" button
3. **Load segmented objects**: Load a folder containing segmented microscopy objects
4. **Label images**: Select images and assign them to classes for training
5. **Train model**: Configure epochs and batch size, then click "Train Model"
6. **Classify new images**: Load new segmented objects and use "Classify Displayed Images"

## Training Guide

### Understanding Training Parameters

- **Epochs**: The number of complete passes through the training dataset
  - Start with 10-15 epochs for small datasets
  - Too few epochs: underfitting (poor accuracy)
  - Too many epochs: overfitting (model memorizes training data)

- **Batch Size**: Number of images processed before model weights are updated
  - Smaller batch size (8-16): Better for smaller datasets
  - Larger batch size (32-64): Faster training, may require more memory

### Interpreting Training Output

When training a model, you'll see output like this:

```
23/23 - 14s - loss: 0.2082 - accuracy: 0.9150 - val_loss: 0.1089 - val_accuracy: 0.9535 - lr: 1.0000e-04 - 14s/epoch - 594ms/step
```

This output contains information about your model's performance:

- **23/23**: Current batch / total batches in the epoch
- **loss: 0.2082**: Training loss (lower is better)
- **accuracy: 0.9150**: Training accuracy (91.5% correct predictions)
- **val_loss: 0.1089**: Validation loss (lower is better)
- **val_accuracy: 0.9535**: Validation accuracy (95.35% correct predictions)
- **lr: 1.0000e-04**: Current learning rate
- **14s/epoch**: Time taken for the complete epoch
- **594ms/step**: Average time per training step

### What to Look For

- **Good training signs**:
  - Both losses (loss and val_loss) decrease over time
  - Both accuracies increase over time
  - Validation accuracy at least 80% by the final epoch
  - Small gap between training and validation metrics

- **Warning signs**:
  - Validation loss increases while training loss continues to decrease (overfitting)
  - Accuracy plateaus at a low value (underfitting or insufficient data)
  - Accuracy stays at the same value across all epochs (possible issues with data or learning rate, this will normally appear as near random accuracy)

## Advanced Configuration

The application is optimized for smaller datasets (hundreds of images) with potential class imbalance. For larger datasets, you may want to modify:

- Increase model capacity (add more layers)
- Unfreeze top layers of the feature extractor for fine-tuning
- Increase learning rate (0.001 instead of 0.0001)
- Use stronger data augmentation

These changes can be made in the `create_model` function within `segmented_mode.py`.

## Troubleshooting

### Common Issues

1. **"Training failed" error**:
   - Ensure at least two classes with images
   - Check for corrupt or invalid image files
   - Verify sufficient memory for the chosen batch size

2. **Low accuracy after training**:
   - Try increasing the number of epochs
   - Add more training examples
   - Ensure proper class balance or adjust class weights
   - Inspect images for mislabeling

3. **GPU memory errors**:
   - Reduce batch size
   - Use a smaller input image size

## License

This project is licensed under the MIT License - see the LICENSE file for details.

