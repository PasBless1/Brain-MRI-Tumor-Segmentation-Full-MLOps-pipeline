"""Integration tests for the FastAPI serving endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import io
from PIL import Image


@pytest.fixture
def client():
    with patch("src.serving.app.mlflow.keras.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.9]])
        mock_load.return_value = mock_model
        from src.serving.app import app
        return TestClient(app)


def get_dummy_image_bytes():
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_endpoint_returns_result(client):
    img_bytes = get_dummy_image_bytes()
    response = client.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})
    assert response.status_code == 200
    assert "has_tumor" in response.json()
