"""Data loading, splitting, and validation utilities."""
import os
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataframe(raw_data_dir: str) -> pd.DataFrame:
    """Scan raw_data_dir and build image/mask path DataFrame."""
    records = []
    for patient_dir in sorted(Path(raw_data_dir).rglob("*")):
        if patient_dir.is_dir():
            images = sorted(patient_dir.glob("*.tif"))
            for img_path in images:
                if "_mask" not in img_path.name:
                    mask_path = img_path.parent / (img_path.stem + "_mask" + img_path.suffix)
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        has_mask = int(mask.max() > 0)
                        records.append({
                            "image_path": str(img_path),
                            "mask_path": str(mask_path),
                            "has_mask": has_mask
                        })
    df = pd.DataFrame(records)
    logger.info(f"Built DataFrame with {len(df)} samples ({df['has_mask'].sum()} positive)")
    return df


def split_data(df: pd.DataFrame, config: dict) -> tuple:
    """Stratified train/val/test split."""
    seed = config["data"]["seed"]
    val_ratio = config["data"]["val_split"]
    test_ratio = config["data"]["test_split"]
    train_val, test = train_test_split(df, test_size=test_ratio, stratify=df["has_mask"], random_state=seed)
    val_adj = val_ratio / (1 - test_ratio)
    train, val = train_test_split(train_val, test_size=val_adj, stratify=train_val["has_mask"], random_state=seed)
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def validate_dataset(df: pd.DataFrame) -> bool:
    """Basic data validation checks."""
    assert not df.empty, "Dataset is empty"
    assert df["has_mask"].nunique() == 2, "Dataset must have both positive and negative samples"
    missing = df[~df["image_path"].apply(os.path.exists)]
    if len(missing) > 0:
        logger.warning(f"{len(missing)} image files not found on disk")
    pos_ratio = df["has_mask"].mean()
    assert 0.05 < pos_ratio < 0.95, f"Class imbalance too extreme: {pos_ratio:.2f} positive rate"
    logger.info(f"Dataset validation passed. Positive ratio: {pos_ratio:.2f}")
    return True
