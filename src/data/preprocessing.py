"""Image preprocessing, augmentation, and CLAHE pipeline."""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

logger = logging.getLogger(__name__)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE to each channel of an RGB image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    channels = cv2.split(image)
    enhanced = [clahe.apply(ch) if ch.dtype == np.uint8 else ch for ch in channels]
    return cv2.merge(enhanced)


def preprocess_image(image_path: str, image_size: tuple, apply_clahe_flag: bool = True) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, tuple(image_size))
    if apply_clahe_flag:
        img = apply_clahe(img)
    return img.astype(np.float32) / 255.0


def preprocess_mask(mask_path: str, image_size: tuple) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, tuple(image_size))
    mask = mask.astype(np.float32) / 255.0
    return np.expand_dims(mask, axis=-1)


def get_classifier_generators(train_df, val_df, test_df, config: dict):
    """Return Keras ImageDataGenerators for the classifier."""
    bs = config["data"]["batch_size"]
    img_size = tuple(config["data"]["image_size"])

    train_aug = ImageDataGenerator(
        rescale=1.0/255, rotation_range=10, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
        horizontal_flip=True, fill_mode="nearest"
    )
    val_aug = ImageDataGenerator(rescale=1.0/255)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    for df in (train_df, val_df, test_df):
        df["has_mask"] = df["has_mask"].astype(str)

    train_gen = train_aug.flow_from_dataframe(train_df, x_col="image_path", y_col="has_mask",
        target_size=img_size, batch_size=bs, class_mode="categorical", shuffle=True)
    val_gen = val_aug.flow_from_dataframe(val_df, x_col="image_path", y_col="has_mask",
        target_size=img_size, batch_size=bs, class_mode="categorical", shuffle=False)
    test_gen = val_aug.flow_from_dataframe(test_df, x_col="image_path", y_col="has_mask",
        target_size=img_size, batch_size=bs, class_mode="categorical", shuffle=False)
    return train_gen, val_gen, test_gen
