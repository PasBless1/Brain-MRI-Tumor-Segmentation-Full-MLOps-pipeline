"""Unit tests for model architectures."""
import pytest
import numpy as np


def get_mock_config():
    return {
        "data": {"image_size": [64, 64], "batch_size": 2},
        "classifier": {"backbone": "ResNet50", "dense_units": [64, 64], "dropout_rate": 0.3,
                        "learning_rate": 0.0001, "num_classes": 2, "freeze_base": True, "weights": None},
        "segmentation": {"learning_rate": 0.01, "epsilon": 0.1}
    }


def test_classifier_output_shape():
    from src.models.classifier import build_classifier
    config = get_mock_config()
    model = build_classifier(config)
    dummy = np.random.rand(2, 64, 64, 3).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (2, 2)


def test_resunet_output_shape():
    from src.models.resunet import build_resunet
    config = get_mock_config()
    model = build_resunet(config)
    dummy = np.random.rand(2, 64, 64, 3).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (2, 64, 64, 1)


def test_tversky_score_range():
    import tensorflow as tf
    from src.losses import tversky_score
    y_true = tf.constant(np.random.rand(4, 64, 64, 1).astype(np.float32))
    y_pred = tf.constant(np.random.rand(4, 64, 64, 1).astype(np.float32))
    score = tversky_score(y_true, y_pred).numpy()
    assert 0.0 <= score <= 1.0
