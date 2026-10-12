"""Tests for frozen evaluation metrics (no network, hashing embeddings)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.main import create_app
from app.services.embeddings.service import HashingEmbeddingService
from app.services.eval_metrics.assignment import calibrate_tau_p, score_with_assignment
from app.services.eval_metrics.constants import FINAL_N
from app.services.eval_metrics.decision import (
    compare_metric,
    decision_rule,
    favourable_differences,
    win_tie_loss,
)
from app.services.eval_metrics.diagnostics import oracle_greedy_selection
from app.services.eval_metrics.freeze import (
    FreezeError,
    file_hashes,
    sha256_file,
    verify_freeze,
    write_freeze,
    write_spec,
)
from app.services.eval_metrics.metrics_core import m1_redundancy, m2_m3_from_assignment
from app.services.eval_metrics.spend import MetricsSpendGuard, SpendAborted
from app.services.eval_metrics.vendi import vendi_score_from_embeddings
from app.services.fit_label_control.stats import holm_adjust
from app.services.gap_sensitivity.io_utils import project_root, write_json


@pytest.fixture
def embeddings():
    return HashingEmbeddingService(dim=32)


def test_vendi_identical_vs_orthogonal():
    ident = np.ones((10, 8), dtype=np.float64)
    ortho = np.eye(10, dtype=np.float64)
    assert abs(vendi_score_from_embeddings(ident) - 1.0) < 1e-6
    assert abs(vendi_score_from_embeddings(ortho) - 10.0) < 1e-6


def test_m1_near_duplicate_rates():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(10, 32))
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    z = m1_redundancy(base, near_dup_threshold=0.99)
    assert z["M1.S1"] == 0.0
    dup = base.copy()
    dup[1] = dup[0]
    one = m1_redundancy(dup, near_dup_threshold=0.99)
    assert one["M1.S1"] == pytest.approx(1.0 / 45.0)


def test_m2_m3_hand_built():
    linked = [True, True, False] + [False] * 7
    idxs = [0, 1, 0] + [None] * 7
    reqs = [
        {"job_competency": "A", "match_label": "missing", "weight": 0.9, "priority": "required"},
        {"job_competency": "B", "match_label": "strong_match", "weight": 0.1, "priority": "preferred"},
    ]
    out = m2_m3_from_assignment(
        linked=linked, r_star_indices=idxs, requirements=reqs, n_questions=FINAL_N
    )
    assert out["M2.P"] == pytest.approx(0.1)  # 1 of 10 linked to gap
    assert out["M3.P"] == pytest.approx(1.0)  # 2 distinct / min(10,2)=2


def test_tau_excludes_own_pair_and_deterministic(embeddings):
    rng = np.random.default_rng(1)
    other = rng.normal(size=(100, 32))
    other = other / np.linalg.norm(other, axis=1, keepdims=True)
    req = rng.normal(size=(5, 32))
    req = req / np.linalg.norm(req, axis=1, keepdims=True)
    t1 = calibrate_tau_p(req_embeddings=req, other_question_embeddings=other)
    t2 = calibrate_tau_p(req_embeddings=req, other_question_embeddings=other)
    assert t1 == t2
    # Own-pair questions not in `other` by construction


def test_favourable_sign_m1_reversed():
    t = np.asarray([0.2, 0.3])
    b = np.asarray([0.5, 0.6])
    # T lower redundancy → favourable positive
    d = favourable_differences(t, b, "M1.P")
    assert np.all(d > 0)


def test_win_tie_loss_and_decision_rule():
    # Synthetic SCALE
    outcomes_b1 = {"M1.P": "win", "M2.P": "win", "M3.P": "tie", "M4.P": "tie"}
    outcomes_b2 = {"M1.P": "tie", "M2.P": "win", "M3.P": "tie", "M4.P": "tie"}
    outcomes_b0 = {"M1.P": "win", "M2.P": "win", "M3.P": "win", "M4.P": "tie"}
    d = decision_rule(
        outcomes_vs_b1=outcomes_b1,
        outcomes_vs_b2=outcomes_b2,
        outcomes_vs_b0=outcomes_b0,
        projected_cost_cny=50.0,
        scale_budget_cny=100.0,
    )
    assert d["verdict"] == "SCALE"

    # DIAGNOSE on a loss
    outcomes_b1_lose = dict(outcomes_b1)
    outcomes_b1_lose["M4.P"] = "lose"
    d2 = decision_rule(
        outcomes_vs_b1=outcomes_b1_lose,
        outcomes_vs_b2=outcomes_b2,
        outcomes_vs_b0=outcomes_b0,
        projected_cost_cny=50.0,
        scale_budget_cny=100.0,
    )
    assert d2["verdict"] == "DIAGNOSE"

    # Budget blocked
    d3 = decision_rule(
        outcomes_vs_b1=outcomes_b1,
        outcomes_vs_b2=outcomes_b2,
        outcomes_vs_b0=outcomes_b0,
        projected_cost_cny=150.0,
        scale_budget_cny=100.0,
    )
    assert d3["verdict"] == "SCALE_BLOCKED_BY_BUDGET"

    assert win_tie_loss({"ci_low": 0.1, "ci_high": 0.2}) == "win"
    assert win_tie_loss({"ci_low": -0.2, "ci_high": -0.1}) == "lose"
    assert win_tie_loss({"ci_low": -0.1, "ci_high": 0.1}) == "tie"


def test_holm_and_compare_metric():
    adj = holm_adjust([0.01, 0.04, None, 0.03])
    assert adj[2] is None
    assert adj[0] <= adj[1]
    rng = np.random.default_rng(0)
    t = rng.normal(0.6, 0.05, size=30)
    b = t - 0.1
    row = compare_metric(t, b, "M3.P")
    assert row["outcome"] == "win"
    assert row["n_pairs"] == 30


def test_oracle_upper_bounds_t(embeddings):
    texts = [f"question about skill {i} python java" for i in range(20)]
    reqs = [
        {"job_competency": "python", "match_label": "missing", "weight": 0.9, "priority": "required"},
        {"job_competency": "java", "match_label": "weak_match", "weight": 0.8, "priority": "required"},
        {"job_competency": "unrelated", "match_label": "strong_match", "weight": 0.1, "priority": "preferred"},
    ]
    pool_emb = np.asarray(embeddings.embed_batch(texts), dtype=np.float64)
    req_emb = np.asarray(
        embeddings.embed_batch([r["job_competency"] for r in reqs]), dtype=np.float64
    )
    tau = 0.0  # link all
    idxs = oracle_greedy_selection(
        pool_texts=texts,
        pool_emb=pool_emb,
        req_emb=req_emb,
        requirements=reqs,
        tau=tau,
        metric="M3.P",
        k=10,
    )
    oracle = score_with_assignment(
        question_embeddings=pool_emb[idxs],
        req_embeddings=req_emb,
        requirements=reqs,
        tau=tau,
    )
    # Random 10
    rand = score_with_assignment(
        question_embeddings=pool_emb[:10],
        req_embeddings=req_emb,
        requirements=reqs,
        tau=tau,
    )
    assert oracle["M3.P"] >= rand["M3.P"] - 1e-9


def test_spend_guard():
    g = MetricsSpendGuard(max_cost=0.01, confirm_spend=False)
    with pytest.raises(SpendAborted):
        g.require_confirm()
    g2 = MetricsSpendGuard(max_cost=0.01, confirm_spend=True, prior_per_pair=0.02)
    with pytest.raises(SpendAborted):
        g2.check_next()


def test_freeze_hash_detects_modification():
    root = project_root()
    metrics_dir = root / "experiments" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = metrics_dir / "metrics_freeze.json"
    backup = freeze_path.read_text(encoding="utf-8") if freeze_path.is_file() else None
    try:
        write_spec()
        tau = {"p1": 0.4, "p2": 0.5}
        from app.services.eval_metrics.assignment import hash_tau_map

        freeze_path = write_freeze(
            tau_map=tau,
            baseline_result_dirs=["exp_test"],
            calibrated_thresholds_hash=hash_tau_map(tau),
        )
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        rel = next(iter(freeze["file_hashes"]))
        freeze["file_hashes"][rel] = "0" * 64
        freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        with pytest.raises(FreezeError):
            verify_freeze(allow_post_hoc=False)
        tagged = verify_freeze(allow_post_hoc=True)
        assert tagged["post_hoc"] is True
        assert "POST-HOC" in tagged["post_hoc_tag"]
    finally:
        if backup is None:
            if freeze_path.is_file():
                freeze_path.unlink()
        else:
            freeze_path.write_text(backup, encoding="utf-8")


def test_experiments_api_still_works(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")
    client = TestClient(create_app())
    listed = client.get("/api/v1/experiments")
    assert listed.status_code == 200
    assert isinstance(listed.json()["experiments"], list)


def test_missing_api_key_skips_extract(monkeypatch):
    from app.services.eval_metrics.extract_missing import maybe_extract_missing
    from app.config.settings import Settings

    get_settings.cache_clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(openai_api_key="")
    info = maybe_extract_missing(
        settings=settings,
        missing_pair_ids=["x"],
        confirm_spend=True,
        max_cost=2.0,
        dry_run=False,
    )
    assert info["ran"] is False
    assert info["skipped_reason"] == "missing_api_key"
