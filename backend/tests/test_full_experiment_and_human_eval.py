"""150-pair role gates, blinded study, rater API, analysis refusal, judge parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.main import create_app
from app.services.full_experiment.batch import call_cache_key, pending_jobs, run_resumable_batch
from app.services.full_experiment.gates import evaluate_paid_gates
from app.services.full_experiment.requirements import summarize_stored_gaps
from app.services.full_experiment.roles import split_roles
from app.services.gap_sensitivity.io_utils import project_root
from app.services.human_eval.analyze import (
    HumanEvalAnalysisError,
    analyze_ratings,
    krippendorff_alpha_ordinal,
)
from app.services.human_eval.judge import (
    JudgeOutput,
    JudgeParseError,
    judge_item,
    parse_judge_output,
    position_bias,
)
from app.services.human_eval.store import JsonlRatingStore
from app.services.human_eval.study import build_blinded_study
from app.services.llm.provider import LLMResult
from app.services.llm.tasks import LLMTask


def test_holdout_is_not_in_development():
    roles = split_roles(project_root())
    assert len(roles["primary_150"]) == 150
    assert len(roles["holdout_30"]) == 30
    assert len(roles["development_120"]) == 120
    assert set(roles["development_120"]).isdisjoint(roles["holdout_30"])
    assert set(roles["primary_30"]).issubset(set(roles["development_120"]))
    assert set(roles["confirmatory_30"]) == set(roles["holdout_30"])


def test_paid_gates_fail_closed():
    gates = evaluate_paid_gates(project_root())
    assert gates["allowed"] is False
    text = " ".join(gates["failures"])
    assert "DIAGNOSE" in text
    assert "hashes do not verify" in text
    assert gates["projected_cny"] == pytest.approx(87.20587362)
    assert gates["projected_cny"] <= gates["scale_budget_cny"]


def test_stored_gap_requirement_summary():
    summary = summarize_stored_gaps(project_root())
    pooled = summary["pooled_existing_60"]
    assert pooled["n_pairs"] == 60
    assert pooled["min"] == 2
    assert pooled["median"] == 17
    assert pooled["max"] == 43
    assert pooled["n_below_10"] == 8


def test_resumable_batch_skips_cache_and_refuses_unconfirmed_spend():
    jobs = [
        {
            "pair_id": "p1",
            "prompt_version": "1.1.0",
            "model": "gpt-5.6-terra",
            "config": "T",
        }
    ]

    class Schema(BaseModel):
        ok: bool

    class Router:
        def __init__(self):
            self.calls = 0

        async def complete_structured(self, **kwargs):
            self.calls += 1
            usage = LLMResult(
                content={},
                provider="fake",
                model="gpt-5.6-terra",
                estimated_cost=0.01,
            )
            return Schema(ok=True), usage

    router = Router()
    cache: dict = {}
    import asyncio

    planned = asyncio.run(
        run_resumable_batch(
            router,
            jobs,
            cache,
            confirm_spend=False,
            max_cost_usd=1,
            task=LLMTask.SIMPLE_TRANSFORMATION,
            schema=Schema,
            system_prompt="s",
            user_prompt_for=lambda job: job["pair_id"],
        )
    )
    assert planned["calls"] == 0
    assert router.calls == 0
    assert planned["n_pending"] == 1

    first = asyncio.run(
        run_resumable_batch(
            router,
            jobs,
            cache,
            confirm_spend=True,
            max_cost_usd=1,
            task=LLMTask.SIMPLE_TRANSFORMATION,
            schema=Schema,
            system_prompt="s",
            user_prompt_for=lambda job: job["pair_id"],
        )
    )
    assert first["calls"] == 1
    second = asyncio.run(
        run_resumable_batch(
            router,
            jobs,
            cache,
            confirm_spend=True,
            max_cost_usd=1,
            task=LLMTask.SIMPLE_TRANSFORMATION,
            schema=Schema,
            system_prompt="s",
            user_prompt_for=lambda job: job["pair_id"],
        )
    )
    assert second["calls"] == 0
    assert router.calls == 1
    assert pending_jobs(jobs, set(cache)) == []
    assert call_cache_key(**jobs[0]) in cache


def _pairs(n: int = 30) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "pair_id": f"pair-{i}",
                "job_text": f"Job {i} needs accounting close.",
                "candidate_text": f"Candidate {i} prepared statements.",
                "T": [f"T question {i} a", f"T question {i} b"],
                "B1": [f"B question {i} a", f"B question {i} b"],
            }
        )
    return rows


def test_study_balance_determinism_and_orders():
    first = build_blinded_study(_pairs(), ["R1", "R2", "R3"], seed=20261019)
    second = build_blinded_study(_pairs(), ["R1", "R2", "R3"], seed=20261019)
    assert first["manifest"] == second["manifest"]
    assert first["condition_key"] == second["condition_key"]
    assert first["manifest"]["n_t_on_x"] == 15
    orders = [tuple(row["item_order"]) for row in first["blinded"]["assignments"]]
    assert len(set(orders)) == 3
    for row in first["blinded"]["assignments"]:
        assert sorted(row["item_order"]) == sorted(
            item["item_id"] for item in first["blinded"]["items"]
        )
    blob = json.dumps(first["blinded"])
    assert "condition" not in blob
    assert "pair-" not in blob


def test_human_eval_api_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMAN_EVAL_ADMIN_TOKEN", "test-admin")
    app = create_app()
    app.state.human_eval_store = JsonlRatingStore(tmp_path)
    client = TestClient(app)
    built = build_blinded_study(_pairs(4), ["R1", "R2"], seed=3)
    loaded = client.post(
        "/api/v1/human-eval/admin/study",
        headers={"X-Human-Eval-Admin": "test-admin"},
        json={"mode": "main", "blinded": built["blinded"]},
    )
    assert loaded.status_code == 200
    issued = client.post(
        "/api/v1/human-eval/admin/tokens",
        headers={"X-Human-Eval-Admin": "test-admin"},
        json={"mode": "main", "rater_id": "R1"},
    )
    assert issued.status_code == 200
    token = issued.json()["token"]
    session = client.get("/api/v1/human-eval/session", params={"token": token, "mode": "main"})
    assert session.status_code == 200
    body = session.json()
    assert "condition" not in json.dumps(body)
    assert body["resume_item_id"] == body["items"][0]["item"]["item_id"]
    item_id = body["resume_item_id"]
    rating = {
        "item_id": item_id,
        "relevance_x": 4,
        "coverage_x": 3,
        "non_redundancy_x": 5,
        "relevance_y": 2,
        "coverage_y": 2,
        "non_redundancy_y": 4,
        "preference": "X",
        "comment": "clearer",
        "elapsed_ms": 1000,
    }
    saved = client.post("/api/v1/human-eval/ratings", params={"token": token, "mode": "main"}, json=rating)
    assert saved.status_code == 200
    assert saved.json()["complete"] is True
    rating["relevance_x"] = 5
    again = client.post("/api/v1/human-eval/ratings", params={"token": token, "mode": "main"}, json=rating)
    assert again.status_code == 200
    exported = client.get(
        "/api/v1/human-eval/admin/export",
        headers={"X-Human-Eval-Admin": "test-admin"},
        params={"mode": "main"},
    )
    rows = [row for row in exported.json()["ratings"] if row["item_id"] == item_id]
    assert len(rows) == 1
    assert rows[0]["relevance_x"] == 5
    bad = client.post(
        "/api/v1/human-eval/ratings",
        params={"token": token, "mode": "main"},
        json={**rating, "relevance_x": 6},
    )
    assert bad.status_code == 422
    missing = client.post(
        "/api/v1/human-eval/ratings",
        params={"token": token, "mode": "main"},
        json={**rating, "item_id": "nope"},
    )
    assert missing.status_code == 404
    assert client.get("/api/v1/human-eval/session", params={"token": "nope", "mode": "main"}).status_code == 401
    demo = client.post(
        "/api/v1/human-eval/admin/study",
        headers={"X-Human-Eval-Admin": "test-admin"},
        json={"mode": "demo", "blinded": built["blinded"]},
    )
    assert demo.status_code == 200
    assert (
        client.get(
            "/api/v1/human-eval/admin/export",
            headers={"X-Human-Eval-Admin": "test-admin"},
            params={"mode": "demo"},
        ).status_code
        == 422
    )
    progress = client.get("/api/v1/human-eval/progress", params={"token": token, "mode": "main"})
    assert progress.json()["n_complete"] == 1
    assert progress.json()["resume_item_id"] != item_id


def test_analysis_refuses_incomplete_and_non_main():
    ratings = [
        {
            "rater_id": "R1",
            "item_id": "item-001",
            "mode": "main",
            "relevance_x": 5,
            "coverage_x": 5,
            "non_redundancy_x": 5,
            "relevance_y": 3,
            "coverage_y": 3,
            "non_redundancy_y": 3,
            "preference": "X",
        }
    ]
    with pytest.raises(HumanEvalAnalysisError, match="mode"):
        analyze_ratings(ratings, {"item-001": {"X": "T", "Y": "B1"}}, mode="pilot", n_items=1)
    with pytest.raises(HumanEvalAnalysisError, match="completion"):
        analyze_ratings(ratings, {"item-001": {"X": "T", "Y": "B1"}}, mode="main", n_items=2)


def test_alpha_perfect_agreement():
    matrix = [[5, 4, 3], [5, 4, 3], [5, 4, 3]]
    assert krippendorff_alpha_ordinal(matrix) == pytest.approx(1.0)


def test_judge_parse_orders_and_dry_run():
    parsed = parse_judge_output(
        {
            "relevance_x": 4,
            "coverage_x": 4,
            "non_redundancy_x": 5,
            "relevance_y": 2,
            "coverage_y": 2,
            "non_redundancy_y": 3,
            "preference": "X",
        }
    )
    swapped = parsed.model_copy(update={"preference": "Y"})
    assert position_bias(parsed, swapped) is True
    with pytest.raises(JudgeParseError):
        parse_judge_output("not-json")

    class Router:
        def __init__(self):
            self.calls = 0

        async def complete_structured(self, **kwargs):
            self.calls += 1
            return parsed, LLMResult(content={}, provider="fake", model="gpt-5.6-luna", estimated_cost=0.0)

    import asyncio

    router = Router()
    item = {
        "item_id": "item-001",
        "job_summary": "job",
        "candidate_summary": "cv",
        "set_x": ["q1"],
        "set_y": ["q2"],
    }
    dry = asyncio.run(
        judge_item(
            router,
            item,
            confirm_spend=False,
            approved=True,
            cache={},
            max_cost_usd=1,
            system_prompt="s",
        )
    )
    assert dry["calls"] == 0
    assert router.calls == 0
    with pytest.raises(RuntimeError, match="not approved"):
        asyncio.run(
            judge_item(
                router,
                item,
                confirm_spend=True,
                approved=False,
                cache={},
                max_cost_usd=1,
                system_prompt="s",
            )
        )
