"""
trainer.py

Provides a function to train a TensorFlow classification model on a folder-based dataset:
    dataset_root/
        ClassA/
            image1.jpg
            image2.jpg
            ...
        ClassB/
            image3.jpg
        ...
This function can be called either from the GUI or from a separate script/notebook (e.g. Google Colab).
"""

import os
import tensorflow as tf

def train_model(data_dir, 
                img_size=(224,224),
                batch_size=16,
                epochs=5,
                model_out="my_model.h5"):
    """
    Train a simple TF model (e.g., MobileNetV2 transfer learning) on images in data_dir. (Classification Folder)
    data_dir should have subfolders per class that are added in the GUI. 
    """
    # 1. Load dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    # 2. Prefetch for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # 3. Define a model (transfer-learning with MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_size+(3,),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False  # freeze base
    global_pool = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(len(class_names), activation='softmax')

    inputs = tf.keras.Input(shape=img_size+(3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_pool(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Train
    history = model.fit(train_ds, 
                        validation_data=val_ds,
                        epochs=epochs)
    # 5. Optionally unfreeze base_model for fine-tuning if needed:
    # base_model.trainable = True
    # model.compile(...)
    # model.fit(...)

    # 6. Save the trained model
    model.save(model_out)
    print(f"Model saved to {model_out}")

    return model, history
