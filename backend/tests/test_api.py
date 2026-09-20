from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.main import create_app


def test_health_and_models(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("VALIDATE_MODELS_ON_STARTUP", "false")
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")
    client = TestClient(create_app())
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert body["fast_model"]
    assert body["primary_model"]

    models = client.get("/api/v1/models")
    routing = models.json()["routing"]
    assert routing["cv_extraction"] == body["fast_model"]
    assert routing["question_generation"] == body["primary_model"]


def test_profile_gap_endpoint(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/analyze/profile-gap",
        json={
            "candidate": {
                "competencies": [{"normalized_label": "Python"}]
            },
            "job": {
                "required_skills": [{"normalized_label": "Python", "priority": "required"}],
                "preferred_skills": [{"normalized_label": "Go", "priority": "preferred"}],
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["gaps"]) == 2


def test_select_with_test_embeddings(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/questions/select",
        json={
            "use_test_embeddings": True,
            "k": 2,
            "questions": [
                {"question": "Explain your Python API design", "competency": "python", "priority": 0.9},
                {"question": "Explain your Python API testing", "competency": "python", "priority": 0.8},
                {"question": "Walk through a Kubernetes rollout", "competency": "kubernetes", "priority": 0.7},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["selected"]) == 2
    assert data["mmr"]["lambda"] is not None
