"""Unit tests for data loading and preprocessing."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.preprocessing import apply_clahe, preprocess_mask


def test_apply_clahe_shape():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = apply_clahe(img)
    assert result.shape == img.shape


def test_preprocess_mask_output():
    with patch("cv2.imread") as mock_read, patch("cv2.resize") as mock_resize:
        mock_read.return_value = np.zeros((512, 512), dtype=np.uint8)
        mock_resize.return_value = np.zeros((256, 256), dtype=np.uint8)
        mask = preprocess_mask("dummy.tif", (256, 256))
    assert mask.shape == (256, 256, 1)
    assert mask.dtype == np.float32


def test_dataset_validation_passes():
    from src.data.dataset import validate_dataset
    df = pd.DataFrame({"image_path": ["a.tif"] * 100, "mask_path": ["b.tif"] * 100,
                        "has_mask": [1] * 50 + [0] * 50})
    with patch("os.path.exists", return_value=True):
        assert validate_dataset(df) is True


def test_dataset_validation_fails_imbalance():
    from src.data.dataset import validate_dataset
    df = pd.DataFrame({"image_path": ["a.tif"] * 100, "mask_path": ["b.tif"] * 100,
                        "has_mask": [1] * 99 + [0] * 1})
    with patch("os.path.exists", return_value=True):
        with pytest.raises(AssertionError):
            validate_dataset(df)
