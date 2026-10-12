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
        assert "stage" in experiments[0]
        exp_id = experiments[0]["experiment_id"]
        detail = client.get(f"/api/v1/experiments/{exp_id}")
        assert detail.status_code == 200
        body = detail.json()
        assert "summary" in body
        assert "charts" in body
        if body["pairs"]:
            pair = body["pairs"][0]
            # Source texts may be unavailable for metrics-only artifacts
            assert "resume_text" in pair
            assert "job_description_text" in pair


def test_metrics_dashboard_and_human_eval_endpoints(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")
    client = TestClient(create_app())
    dash = client.get("/api/v1/experiments/metrics/dashboard")
    assert dash.status_code == 200
    body = dash.json()
    assert "freeze" in body
    assert "baseline" in body
    assert "scale" in body
    assert "figures" in body

    audit = client.get("/api/v1/experiments/human-eval/assignment-audit")
    assert audit.status_code == 200
    assert "rows" in audit.json()

    ratings = client.get("/api/v1/experiments/human-eval/question-ratings")
    assert ratings.status_code == 200
    assert ratings.json()["schema_version"] == 1

    saved = client.put(
        "/api/v1/experiments/human-eval/question-ratings",
        json={"status": "in_progress", "rater_id": "test", "ratings": []},
    )
    assert saved.status_code == 200
    assert saved.json()["data"]["status"] == "in_progress"


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
