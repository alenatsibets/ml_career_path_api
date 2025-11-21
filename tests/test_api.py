# tests/test_api.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data


def test_predict_endpoint():
    # 48 zeros as dummy feature vector
    payload = {"features": [0.0] * 48}

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "predicted_major" in data
    assert "top_5_predictions" in data
    assert isinstance(data["top_5_predictions"], list)
