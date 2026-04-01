"""
Data pipeline for Alzheimer's MRI classification.
Expects dataset folder structure:
  dataset/
    NonDemented/
    VeryMildDemented/
    MildDemented/
    ModerateDemented/
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
BATCH_SIZE = 64
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


def get_generators(dataset_dir: str):
    """Return train and validation generators from dataset directory."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen
