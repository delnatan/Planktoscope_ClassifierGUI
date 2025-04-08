"""
Inference Utility for Planktoscope Classifier
-----------------------------------------------------------
Utilities to load a trained model and run inference on batches of Planktoscope images or individual files.
This script can be used as a standalone command-line tool for batch classification
without requiring the GUI interface.

Usage examples:
  # Classify a single image
  python inference.py --model model.h5 --image input.jpg
  
  # Classify all images in a folder
  python inference.py --model model.h5 --folder images/ --output results.csv

Author: [Adam Larson]
Date: [4.1.2025]
Version: 1.0
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
import tensorflow_hub as hub
from keras.utils import get_custom_objects

# Required for loading models with TensorFlow Hub layers
get_custom_objects()['KerasLayer'] = hub.KerasLayer


def load_tf_model(model_path, class_file=None):
    """
    Load a trained TensorFlow model and its associated class names.
    
    Args:
        model_path (str): Path to the saved model (.h5 or SavedModel)
        class_file (str, optional): Path to text file with the nice class names
        
    Returns:
        tuple: (model, class_names)
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    # Load the  class names if provided
    class_names = []
    if class_file and os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = f.read().splitlines()
        print(f"Loaded {len(class_names)} classes: {', '.join(class_names)}")
    else:
        # Try to find the class file based on model path
        default_class_file = os.path.splitext(model_path)[0] + "_classes.txt"
        if os.path.exists(default_class_file):
            with open(default_class_file, 'r') as f:
                class_names = f.read().splitlines()
            print(f"Loaded {len(class_names)} classes: {', '.join(class_names)}")
        else:
            print("No class file found. Classes will be indexed numerically.")
    
    return model, class_names


def preprocess_image(path, img_size=(224, 224)):
    """
    Preprocess an Planktoscope image for classification.
    
    Args:
        path (str): Path to the image file
        img_size (tuple): Target size (width, height)
        
    Returns:
        ndarray: Preprocessed image
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0
    return img


def classify_single_image(model, image_path, class_names=None, img_size=(224, 224)):
    """
    Classify a single Planktoscope image.
    
    Args:
        model: Loaded TensorFlow model
        image_path (str): Path to the image file
        class_names (list, optional): List of class names
        img_size (tuple): Target size (width, height)
        
    Returns:
        tuple: (class_name, class_index, confidence)
    """
    try:
        img = preprocess_image(image_path, img_size)
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)
        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx]) * 100
        
        if class_names and class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"Class{class_idx}"
            
        return class_name, class_idx, confidence
    except Exception as e:
        print(f"Error classifying {image_path}: {str(e)}")
        return "Error", -1, 0.0


def classify_folder(model, folder_path, class_names=None, batch_size=8, output_csv="inference_results.csv", img_size=(224, 224)):
    """
    Classify all the Planktoscope images in a folder and optionally save results to CSV.
    
    Args:
        model: Loaded TensorFlow model
        folder_path (str): Folder containing images
        class_names (list, optional): List of class names
        batch_size (int): Number of images to process in a batch
        output_csv (str): Path to save CSV results
        img_size (tuple): Target size (width, height)
        
    Returns:
        list: Classification results [(filename, class, confidence), ...]
    """
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                  if f.lower().endswith(exts)]
    image_paths.sort()
    
    if not image_paths:
        print(f"No valid images found in {folder_path}")
        return []
    
    print(f"Found {len(image_paths)} images to classify")
    
    # Process in batches for efficiency
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Preprocess batch
        for path in batch_paths:
            try:
                img = preprocess_image(path, img_size)
                batch_images.append(img)
            except Exception as e:
                print(f"Error preprocessing {path}: {str(e)}")
                continue
        
        # Make predictions on batch
        if batch_images:
            batch_tensor = tf.stack(batch_images)
            batch_preds = model.predict(batch_tensor)
            
            # Process the batch results
            for j, path in enumerate(batch_paths):
                if j >= len(batch_preds):  # Skip if there was a preprocessing error
                    continue
                    
                pred = batch_preds[j]
                class_idx = np.argmax(pred)
                confidence = float(np.max(pred)) * 100
                
                if class_names and class_idx < len(class_names):
                    class_name = class_names[class_idx]
                else:
                    class_name = f"Class{class_idx}"
                    
                filename = os.path.basename(path)
                results.append((filename, class_name, confidence))
                
        # Print the progress of your thing
        print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images", end="\r")
    
    print("\nClassification complete")
    
    # Compute the class distribution
    class_counts = {}
    for _, cls, _ in results:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count} images ({count/len(results)*100:.1f}%)")
    
    # Write a nice CSV
    if output_csv:
        with open(output_csv, 'w') as fw:
            fw.write("filename,predicted_class,confidence\n")
            for (fn, cls, conf) in results:
                fw.write(f"{fn},{cls},{conf:.3f}\n")
        print(f"\nResults saved to {output_csv}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained microscopy classifier model")
    parser.add_argument("--model", required=True, help="Path to the trained model (.h5 file)")
    parser.add_argument("--class-file", help="Path to text file with class names (optional)")
    
    # Input options (remember these are mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to a single image for classification")
    input_group.add_argument("--folder", help="Path to a folder of images for batch classification")
    
    # Additional cool options
    parser.add_argument("--output", default="inference_results.csv", help="Output CSV file for batch classification")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--img-size", default="224,224", help="Image dimensions (width,height)")
    
    args = parser.parse_args()
    
    # Parse image size
    try:
        width, height = map(int, args.img_size.split(","))
        img_size = (width, height)
    except ValueError:
        print("Error: --img-size should be in format 'width,height'")
        return 1
    
    # Load the model and class names
    model, class_names = load_tf_model(args.model, args.class_file)
    
    # Run inference
    if args.image:
        # Single Planktoscope image classification
        class_name, class_idx, confidence = classify_single_image(
            model, args.image, class_names, img_size
        )
        print(f"\nClassification result for {os.path.basename(args.image)}:")
        print(f"  Class: {class_name}")
        print(f"  Confidence: {confidence:.2f}%")
    
    elif args.folder:
        # Batch classification
        classify_folder(
            model, 
            args.folder, 
            class_names, 
            args.batch_size, 
            args.output,
            img_size
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())