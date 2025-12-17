import numpy as np
import pytest

from src import api


def test_health_endpoint_reports_status(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)


def test_predict_returns_fallback_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(api, "_model", None)
    monkeypatch.setattr(api, "_model_path", None)

    resp = client.post("/predict", json={"text": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "unknown"
    assert data["proba_spam"] == 0.0
    assert data["model_path"] is None


def test_predict_uses_model_probability(client, monkeypatch, tmp_path):
    class DummyModel:
        def predict_proba(self, _):
            return np.array([[0.1, 0.9]])

    fake_model_path = tmp_path / "model.joblib"
    fake_model_path.write_text("stub")

    monkeypatch.setattr(api, "_model", DummyModel())
    monkeypatch.setattr(api, "_model_path", fake_model_path)

    resp = client.post("/predict", json={"text": "Call now!"})
    assert resp.status_code == 200

    data = resp.json()
    assert data["label"] == "spam"
    assert data["proba_spam"] == pytest.approx(0.9)
    assert data["model_path"] == str(fake_model_path)


def test_metrics_endpoint_exposes_custom_metrics(client, monkeypatch, tmp_path):
    class DummyModel:
        def predict_proba(self, _):
            return np.array([[0.3, 0.7]])

    fake_model_path = tmp_path / "model.joblib"
    fake_model_path.write_text("stub")

    monkeypatch.setattr(api, "_model", DummyModel())
    monkeypatch.setattr(api, "_model_path", fake_model_path)

    client.get("/health")
    client.post("/predict", json={"text": "metrics please"})

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "request_count" in body
    assert "request_latency_seconds_bucket" in body
    assert "prediction_proba_spam_bucket" in body
