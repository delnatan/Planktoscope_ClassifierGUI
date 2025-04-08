# Comprehensive Training Guide

This guide provides detailed information about the machine learning concepts used in this application and how to interpret training outputs to achieve the best classification results for your microscopy images.

## Understanding Image Classification

Image classification is the task of assigning an input image a label from a set of predefined categories. The Microscopy Classifier implements this using a deep learning approach with a convolutional neural network (CNN) based on EfficientNet.

## Training Process Overview

1. **Data Preparation**: Images are organized into class folders and split into training, validation, and test sets
2. **Data Augmentation**: Small random transformations are applied to increase dataset diversity
3. **Feature Extraction**: A pre-trained EfficientNet extracts meaningful features from images
4. **Classification**: A custom classification head learns to identify your specific classes
5. **Evaluation**: The model is evaluated on a separate validation set to measure performance

## Key Training Parameters

### Epochs

An **epoch** is one complete pass through the entire training dataset.

- **Recommended setting**: 10-20 epochs for small datasets (< 500 images)
- **Too few epochs**: The model won't learn enough (underfitting)
- **Too many epochs**: The model may memorize rather than generalize (overfitting)

**How to choose**: Start with 15 epochs. If validation accuracy is still improving at the end, try increasing to 20-30 epochs. If validation accuracy peaks early then declines, use fewer epochs or enable early stopping.

### Batch Size

A **batch** is a group of training examples processed together before updating model weights.

- **Recommended setting**: 16-32 for most datasets
- **Small batch size (8-16)**: Better for very small datasets, uses less memory, may improve model generalization
- **Large batch size (32-64)**: Faster training, more stable gradients, requires more memory

**How to choose**: If you have <100 images per class, use 8-16. For larger datasets, start with 32. If you encounter memory errors, reduce the batch size.

### Learning Rate

The **learning rate** controls how much the model weights are adjusted during training.

- In this application, a conservative learning rate of 0.0001 is used by default
- The app includes learning rate reduction when performance plateaus

## Understanding Training Output

When training, you'll see output like this:

```
23/23 - 14s - loss: 0.2082 - accuracy: 0.9150 - val_loss: 0.1089 - val_accuracy: 0.9535 - lr: 1.0000e-04 - 14s/epoch - 594ms/step
```

Let's break down each component:

| Component | Example Value | Description |
|-----------|---------------|-------------|
| Batch progress | 23/23 | Current batch / Total batches in the epoch |
| Epoch time | 14s | Total time for this epoch |
| loss | 0.2082 | Training loss (lower is better) |
| accuracy | 0.9150 | Training accuracy (91.5% correct) |
| val_loss | 0.1089 | Validation loss (lower is better) |
| val_accuracy | 0.9535 | Validation accuracy (95.35% correct) |
| lr | 1.0000e-04 | Current learning rate |
| Timing | 14s/epoch - 594ms/step | Performance metrics |

### What Makes a Good Training Run?

#### Ideal Pattern
1. **Both losses decrease** over time
2. **Both accuracies increase** over time
3. **Final validation accuracy** is high (>80% for most use cases)
4. **Training and validation metrics** are reasonably close

#### Signs of Problems

1. **Overfitting**:
   - Training accuracy keeps improving
   - Validation accuracy stalls or decreases
   - Gap between training and validation grows
   - **Solution**: More data augmentation, fewer epochs, more dropout

2. **Underfitting**:
   - Both accuracies are low
   - Loss decreases very slowly
   - **Solution**: More epochs, larger model, higher learning rate

3. **Class Imbalance Issues**:
   - Accuracy seems high but model predicts only majority class
   - **Solution**: Add more examples of minority classes or use class weighting (already implemented)

## Example of a Good Training Run

```
Epoch 1/15
23/23 - 14s - loss: 2.1593 - accuracy: 0.3500 - val_loss: 1.5823 - val_accuracy: 0.4651 - lr: 1.0000e-04
Epoch 5/15
23/23 - 14s - loss: 0.7215 - accuracy: 0.7850 - val_loss: 0.5324 - val_accuracy: 0.8372 - lr: 1.0000e-04
Epoch 10/15
23/23 - 14s - loss: 0.3256 - accuracy: 0.8750 - val_loss: 0.2217 - val_accuracy: 0.9302 - lr: 1.0000e-04
Epoch 15/15
23/23 - 14s - loss: 0.2082 - accuracy: 0.9150 - val_loss: 0.1089 - val_accuracy: 0.9535 - lr: 1.0000e-04
```

Notice how:
- Both losses steadily decrease
- Both accuracies steadily increase
- Final validation accuracy is very high (95.35%)
- Validation metrics are close to training metrics

## Advanced Training Techniques

### Class Weighting

Class weighting helps the model learn from imbalanced datasets by giving more importance to underrepresented classes. The application automatically calculates appropriate weights.

Example output:
```
Class distribution: {0: 36, 1: 91, 2: 160}
Class weights: {0: 2.63, 1: 1.04, 2: 0.59}
```

This means class 0 (with only 36 examples) gets 2.63x more importance than it would without weighting. This can be changed when large quality datasets are used

### Data Augmentation

The application applies gentle data augmentation:
- Horizontal flips
- Small rotations (±5%)
- Minor zoom variations (±5%)
- Slight contrast adjustments (±5%)

For larger datasets, you might want to increase these values to 10-20%.

### Transfer Learning

This application uses transfer learning with a pre-trained EfficientNet B0 model:
- The base model was trained on millions of images
- We use it as a feature extractor (frozen)
- Only the classification head is trained from scratch

For larger datasets, you can modify the code to unfreeze some layers of the base model for fine-tuning.

## Tips for Optimal Results

1. **Balanced classes**: Aim for at least 30-50 images per class, with similar counts across classes
2. **Consistent imaging**: Use similar magnification, lighting, and processing
3. **Representative samples**: Include variations that represent real-world conditions
4. **Proper validation**: Use the test accuracy as the true performance metric
5. **Iterative improvement**: Add more data for classes with poor performance

## Handling Different Dataset Sizes

### Very Small Datasets (<100 total images)
- Increase data augmentation
- Lower learning rate (0.00005)
- Smaller batch size (8)
- More dropout (0.3-0.4)

### Medium Datasets (100-1000 images)
- Default settings work well
- Consider 10-15 epochs

### Large Datasets (>1000 images)
- Less dropout (0.1-0.2)
- Higher learning rate (0.001)
- Larger batch size (64)
- Unfreeze top layers of base model

## Next Steps After Training

After achieving good validation accuracy:
1. **Test on new data**: See how the model performs on completely unseen images. It is nice to separate out some images into a separate folder beforehand
2. **Error analysis**: Examine misclassified images to identify patterns
3. **Model deployment**: Use the trained model for batch classification of new images
4. **Iterative improvement**: Add more training data based on error cases