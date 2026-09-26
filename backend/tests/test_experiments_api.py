from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.main import create_app


def test_list_and_get_experiments(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")
    client = TestClient(create_app())
    listed = client.get("/api/v1/experiments")
    assert listed.status_code == 200
    experiments = listed.json()["experiments"]
    assert isinstance(experiments, list)
    if experiments:
        exp_id = experiments[0]["experiment_id"]
        detail = client.get(f"/api/v1/experiments/{exp_id}")
        assert detail.status_code == 200
        body = detail.json()
        assert "summary" in body
        assert "charts" in body
        if body["pairs"]:
            pair = body["pairs"][0]
            assert pair.get("resume_text")
            assert pair.get("job_description_text")


def test_upload_pair_json(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/experiments/uploads/json",
        json={
            "resume_text": "Junior engineer with Python and FastAPI experience building APIs.",
            "job_description_text": "Software engineer role requiring Python, REST APIs, and PostgreSQL.",
            "seniority": "junior",
            "label": "manual-test",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["upload_id"].startswith("upload_")
    listed = client.get("/api/v1/experiments/uploads/list")
    assert listed.status_code == 200
    assert any(u["upload_id"] == data["upload_id"] for u in listed.json()["uploads"])
