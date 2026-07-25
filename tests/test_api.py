import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test that the root endpoint is alive."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Housing Regression API is running 🚀"}


def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "model_path" in data

    assert data["status"] in ["healthy", "unhealthy"]


def test_predict_empty_request():
    response = client.post("/predict", json=[])

    assert response.status_code == 200
    assert response.json() == {"error": "No data provided"}


def test_predict_success():
    sample = pd.read_csv("data/raw/holdout.csv").head(1).to_dict(orient="records")

    response = client.post("/predict", json=sample)

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 1
