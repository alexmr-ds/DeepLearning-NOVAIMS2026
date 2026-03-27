"""
Utility functions for the EDA.
"""

import hashlib

import imagehash
import tensorflow as tf
from PIL import Image


def load_image_datasets(dataset_path, image_size=(224, 224), batch_size=32):
    """
    Load train, validation, and test datasets from directory structure.
    """

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path / "train",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path / "validation",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path / "test",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataset, validation_dataset, test_dataset


def get_md5_hash(image_path):
    """
    Generate an MD5 hash for an image file.
    Useful for detecting exact duplicate images.
    """
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_perceptual_hash(image_path):
    """
    Generate perceptual hash (pHash) for an image.

    pHash captures visual similarity between images.
    Images with small hash distance are visually similar.
    """
    try:
        img = Image.open(image_path)
        return imagehash.phash(img)
    except Exception:
        return None
