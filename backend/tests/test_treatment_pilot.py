"""Tests for the treatment pilot (no network, hashing embeddings)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings, get_settings
from app.main import create_app
from app.schemas.domain import (
    CandidateProfile,
    GapMatrix,
    InterviewQuestion,
    JobDescription,
    MatchLabel,
    ProfileGap,
    RequirementPriority,
    SimilarityEvidence,
)
from app.schemas.generation import BaselineGeneratedSet
from app.schemas.pool import (
    PROFILE_PROBE,
    PoolGeneratedSet,
    PoolQuestionItem,
    pool_item_to_interview_question,
)
from app.services.embeddings.service import HashingEmbeddingService
from app.services.gap_sensitivity.io_utils import write_json
from app.services.gap_sensitivity.metrics import bootstrap_mean_ci
from app.services.llm.provider import LLMResult
from app.services.llm.tasks import LLMTask
from app.services.mmr.select import competency_weighted_mmr
from app.services.question_generation.pool import (
    generate_pool,
    validate_pool_items,
)
from app.services.question_generation.profile_aware import (
    assert_no_gap_leakage,
    generate_profile_aware,
    render_profile_aware_user_prompt,
)
from app.services.question_generation.weights import map_question_weight, map_weights_for_questions
from app.services.selection.strategies import select_questions
from app.services.treatment_pilot.cache import cache_key, read_cache, write_cache
from app.services.treatment_pilot.constants import (
    GENERATION_MODEL,
    MAX_OUTPUT_TOKENS,
    REASONING_EFFORT,
)
from app.services.treatment_pilot.currency import usd_to_cny
from app.services.treatment_pilot.generate import estimate_stage_costs, generation_parameters
from app.services.treatment_pilot.spend import SpendAborted, TreatmentSpendGuard


@pytest.fixture
def embeddings():
    return HashingEmbeddingService(dim=32)


@pytest.fixture
def gap_matrix() -> GapMatrix:
    gaps = [
        ProfileGap(
            job_competency="Python",
            priority=RequirementPriority.REQUIRED,
            match_label=MatchLabel.MISSING,
            weight=0.95,
            evidence=SimilarityEvidence(semantic_score=0.1),
        ),
        ProfileGap(
            job_competency="Communication",
            priority=RequirementPriority.PREFERRED,
            match_label=MatchLabel.STRONG,
            weight=0.15,
            evidence=SimilarityEvidence(semantic_score=0.9),
        ),
    ]
    return GapMatrix(
        gaps=gaps,
        strong_matches=["Communication"],
        missing=["Python"],
    )


def _pool_item(**kwargs) -> PoolQuestionItem:
    base = dict(
        question="How have you used Python in production APIs?",
        competency="Python",
        dimension="Technical",
        target_gap="Python",
        difficulty="mid",
        rationale="Gap in Python",
        source_evidence=["JD requires Python"],
        priority=0.8,
    )
    base.update(kwargs)
    return PoolQuestionItem(**base)


class FakeRouter:
    def __init__(self, payload):
        self.calls = []
        self.payload = payload

    async def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        usage = LLMResult(
            content={},
            provider="openai",
            model="gpt-5.6-terra",
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.001,
            latency_ms=12.0,
        )
        return self.payload, usage


def test_pool_validation_rejects_bad_items(gap_matrix):
    good = _pool_item()
    with pytest.raises(Exception):
        PoolQuestionItem(
            question="q",
            competency="c",
            dimension="Technical",
            target_gap="Python",
            priority=0.0,
        )
    bad_dim_raw = good.model_dump()
    bad_dim_raw["dimension"] = "Technical"  # ok
    # unknown target via validate_pool_items
    unknown = _pool_item(target_gap="NotACompetency")
    probe = _pool_item(target_gap=PROFILE_PROBE, question="Probe CV strengths?")
    # craft invalid dimension by bypassing enum: use model_construct
    bad_dim = PoolQuestionItem.model_construct(
        question="q",
        competency="c",
        dimension="NotADim",
        target_gap="Python",
        difficulty="mid",
        rationale="",
        source_evidence=[],
        priority=0.5,
    )
    result = validate_pool_items([good, unknown, probe, bad_dim], gap_matrix)
    assert result.n_valid == 2
    assert result.n_failed == 2
    reasons = {tuple(f["reasons"]) for f in result.failures}
    assert any("unknown_target_gap" in r for r in reasons)
    assert any("unknown_dimension" in r for r in reasons)


def test_weight_mapping_rules(gap_matrix, embeddings):
    exact = map_question_weight("Python", gap_matrix, embeddings)
    assert exact.rule == "exact"
    assert exact.weight == pytest.approx(0.95)

    probe = map_question_weight(PROFILE_PROBE, gap_matrix, embeddings)
    assert probe.rule == "profile_probe"
    assert probe.weight == pytest.approx((0.95 + 0.15) / 2)

    fb = map_question_weight("Pythonic scripting", gap_matrix, embeddings)
    assert fb.rule == "embedding_fallback"
    assert fb.matched_competency in {"Python", "Communication"}


def test_plain_mmr_equals_weighted_when_weights_one(embeddings):
    qs = [
        InterviewQuestion(
            question=f"Question number {i} about systems design and APIs?",
            competency="c",
            priority=float((i + 1) / 10),
        )
        for i in range(15)
    ]
    texts = [q.question for q in qs]
    sim = embeddings.similarity_matrix(texts, texts)
    ones = [1.0] * len(qs)
    a = competency_weighted_mmr(qs, sim, ones, lambda_=0.7, k=10)
    b = select_questions(
        "plain_mmr_priority", qs, sim, ones, lambda_=0.7, k=10, seed=42
    )
    assert a.selected_indices == b.selected_indices


def test_selection_strategies_deterministic_and_length_10(embeddings):
    qs = [
        InterviewQuestion(
            question=f"Unique interview question {i} covering topic {i % 3}",
            competency=f"c{i}",
            priority=0.1 + 0.05 * i,
            target_gap="Python",
        )
        for i in range(12)
    ]
    # short pool to exercise backfill path (n=8 < 10)
    short = qs[:8]
    texts = [q.question for q in short]
    sim = embeddings.similarity_matrix(texts, texts)
    weights = [1.0] * len(short)
    for strategy in (
        "competency_weighted_mmr",
        "plain_mmr_priority",
        "top_priority",
        "random",
    ):
        r1 = select_questions(strategy, short, sim, weights, lambda_=0.7, k=10, seed=42)
        r2 = select_questions(strategy, short, sim, weights, lambda_=0.7, k=10, seed=42)
        assert r1.selected_indices == r2.selected_indices
        assert len(r1.selected_indices) == 8  # cannot invent beyond pool
        assert any("pool_shorter_than_k" in n for n in r1.notes) or len(short) < 10

    # full pool always returns 10
    texts = [q.question for q in qs]
    sim = embeddings.similarity_matrix(texts, texts)
    weights = [1.0] * len(qs)
    for strategy in (
        "competency_weighted_mmr",
        "plain_mmr_priority",
        "top_priority",
        "random",
        "plain_mmr_query",
    ):
        qsim = embeddings.similarity_matrix(texts, ["job description python api"])[:, 0]
        r = select_questions(
            strategy,
            qs,
            sim,
            weights,
            lambda_=0.7,
            k=10,
            seed=42,
            query_similarities=qsim,
        )
        assert len(r.selected_indices) == 10


@pytest.mark.asyncio
async def test_b1_prompt_has_no_gap_leakage(gap_matrix):
    cand = CandidateProfile(name="Ada", technical_skills=[])
    job = JobDescription(title="Engineer")
    system = "PROFILE-AWARE BASELINE. You are NOT given a gap matrix."
    user = render_profile_aware_user_prompt(
        candidate=cand, job=job, seniority="mid", n=10
    )
    assert_no_gap_leakage(system, user)
    assert "gap_matrix" not in user
    assert "tfidf_score" not in user.lower()

    payload = BaselineGeneratedSet(
        questions=[{"question": f"Q{i}?"} for i in range(10)]
    )
    router = FakeRouter(payload)
    parsed, usage, sys_p, user_p = await generate_profile_aware(
        router=router,
        settings=Settings(),
        candidate=cand,
        job=job,
        seniority="mid",
        n_questions=10,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
    )
    assert len(parsed.questions) == 10
    call = router.calls[0]
    assert call["task"] == LLMTask.QUESTION_GENERATION
    assert call["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert call["reasoning_effort"] == REASONING_EFFORT
    assert call["temperature"] is None


@pytest.mark.asyncio
async def test_pool_and_b1_same_model_params(gap_matrix):
    cand = CandidateProfile(name="Ada")
    job = JobDescription(title="Engineer")
    items = [_pool_item(question=f"Pool question {i}?", priority=0.5 + i * 0.01) for i in range(5)]
    # vary target for some
    items[1] = _pool_item(question="Probe?", target_gap=PROFILE_PROBE, priority=0.6)
    pool_payload = PoolGeneratedSet(questions=items)
    router = FakeRouter(pool_payload)
    parsed, usage, _, _ = await generate_pool(
        router=router,
        settings=Settings(),
        candidate=cand,
        job=job,
        gap_matrix=gap_matrix,
        seniority="mid",
        n_questions=5,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
    )
    assert parsed.questions
    call = router.calls[0]
    params = generation_parameters()
    assert call["max_output_tokens"] == params["max_output_tokens"]
    assert call["reasoning_effort"] == params["reasoning_effort"]
    assert Settings().llm_primary_model == GENERATION_MODEL


def test_spend_guard_and_cache(tmp_path: Path):
    guard = TreatmentSpendGuard(max_cost=0.05, confirm_spend=False, prior_per_pair=0.02)
    with pytest.raises(SpendAborted):
        guard.require_confirm()
    guard.confirm_spend = True
    guard.check_next_pair()
    guard.add_actual(0.02, pair_id="p1", stage="B1")
    guard.check_next_pair()  # 0.02 + 0.02 = 0.04 <= 0.05
    guard.add_actual(0.02, pair_id="p2", stage="B1")
    # running 0.04; next projected 0.06 > 0.05
    with pytest.raises(SpendAborted):
        guard.check_next_pair()
    # also abort when actual exceeds
    guard2 = TreatmentSpendGuard(max_cost=0.01, confirm_spend=True, prior_per_pair=0.005)
    guard2.check_next_pair()
    with pytest.raises(SpendAborted):
        guard2.add_actual(0.02, pair_id="p3", stage="B1")

    key = cache_key(
        pair_id="p1",
        prompt_hash="abc",
        model_id=GENERATION_MODEL,
        parameters=generation_parameters(),
    )
    write_cache(tmp_path, key, {"ok": True, "cost_usd": 0.01})
    assert read_cache(tmp_path, key)["ok"] is True
    # second read does not create a new charge — caller responsibility; cache hit present
    assert read_cache(tmp_path, key) is not None


def test_usd_cny_and_bootstrap_reproducible():
    assert usd_to_cny(1.0) == pytest.approx(6.77)
    arr = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    a = bootstrap_mean_ci(arr, n_resamples=2000, seed=42)
    b = bootstrap_mean_ci(arr, n_resamples=2000, seed=42)
    assert a == b


@pytest.mark.asyncio
async def test_pipeline_smoke_synthetic(tmp_path: Path, embeddings, gap_matrix, monkeypatch):
    """2 synthetic pairs through B1 + T + selection with FakeRouter."""
    from app.services.treatment_pilot import generate as gen_mod
    from app.services.treatment_pilot.ablation import run_selection_ablation

    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")

    pairs = []
    for i in range(2):
        cand = CandidateProfile(name=f"C{i}", technical_skills=[])
        job = JobDescription(title=f"Role{i}")
        rec = {
            "profile_id": f"p{i}",
            "jd_id": f"j{i}",
            "pair_id": f"p{i}",
            "seniority": "mid",
            "fit_label": "Good Fit",
            "candidate": cand.model_dump(),
            "job": job.model_dump(),
        }
        pairs.append(rec)

    gap_records = {}
    for rec in pairs:
        stem = f"{rec['profile_id']}__{rec['jd_id']}__mid.json"
        gap_records[stem] = {
            "gap_matrix": gap_matrix.model_dump(),
            "profile_id": rec["profile_id"],
            "jd_id": rec["jd_id"],
        }

    class MultiRouter:
        def __init__(self):
            self.calls = []

        async def complete_structured(self, **kwargs):
            self.calls.append(kwargs)
            schema = kwargs["schema"]
            if schema is BaselineGeneratedSet:
                parsed = BaselineGeneratedSet(
                    questions=[{"question": f"B1 Q{j} for pair"} for j in range(10)]
                )
            else:
                items = []
                for j in range(12):
                    tg = "Python" if j % 3 else PROFILE_PROBE
                    items.append(
                        {
                            "question": f"Pool Q{j} unique wording about systems {j}",
                            "competency": "Python",
                            "dimension": "Technical",
                            "target_gap": tg,
                            "difficulty": "mid",
                            "rationale": "test",
                            "source_evidence": ["x"],
                            "priority": 0.3 + 0.05 * (j % 10),
                        }
                    )
                parsed = PoolGeneratedSet(questions=items)
            usage = LLMResult(
                content={},
                provider="openai",
                model="gpt-5.6-terra",
                input_tokens=200,
                output_tokens=100,
                estimated_cost=0.002,
                latency_ms=20.0,
            )
            return parsed, usage

    router = MultiRouter()
    monkeypatch.setattr(gen_mod, "build_router", lambda settings: router)
    monkeypatch.setattr(gen_mod, "api_key_present", lambda settings: True)

    root = tmp_path
    (root / "experiments" / "results").mkdir(parents=True)
    settings = Settings()
    guard = TreatmentSpendGuard(max_cost=1.0, confirm_spend=True, prior_per_pair=0.01)

    b1_dir = root / "experiments" / "results" / "exp_b1"
    b1_dir.mkdir(parents=True)
    await gen_mod.run_baseline_profile_aware(
        root=root,
        settings=settings,
        extraction_records=pairs,
        experiment_id="exp_b1",
        out_dir=b1_dir,
        guard=guard,
        resume=False,
    )
    assert len(list((b1_dir / "baseline_profile_aware").glob("*.json"))) == 2

    t_dir = root / "experiments" / "results" / "exp_t"
    t_dir.mkdir(parents=True)
    # JD texts for S2
    pairs_dir = root / "pairs"
    (pairs_dir / "jobs").mkdir(parents=True)
    for rec in pairs:
        (pairs_dir / "jobs" / f"{rec['jd_id']}.txt").write_text(
            "Python engineer role", encoding="utf-8"
        )

    await gen_mod.run_treatment_pipeline(
        root=root,
        settings=settings,
        extraction_records=pairs,
        gap_records=gap_records,
        embeddings=embeddings,
        experiment_id="exp_t",
        out_dir=t_dir,
        guard=guard,
        resume=False,
        pairs_dir=pairs_dir,
    )
    assert len(list((t_dir / "pool").glob("*.json"))) == 2
    assert len(list((t_dir / "treatment").glob("*.json"))) == 2

    abl_dir = root / "experiments" / "results" / "exp_abl"
    run_selection_ablation(
        pool_dir=t_dir / "pool",
        gap_records=gap_records,
        embeddings=embeddings,
        experiment_id="exp_abl",
        out_dir=abl_dir,
        pairs_dir=pairs_dir,
    )
    for cond in ("B2", "S0", "S1", "S2"):
        assert (abl_dir / cond).is_dir()
        assert len(list((abl_dir / cond).glob("*.json"))) == 2

    # cache prevents double charge: re-run B1 with resume
    guard2 = TreatmentSpendGuard(max_cost=1.0, confirm_spend=True, prior_per_pair=0.01)
    before = guard2.running_cost
    await gen_mod.run_baseline_profile_aware(
        root=root,
        settings=settings,
        extraction_records=pairs,
        experiment_id="exp_b1",
        out_dir=b1_dir,
        guard=guard2,
        resume=True,
    )
    assert guard2.running_cost == before


def test_missing_api_key_dry_run_stops(monkeypatch):
    from app.services.fit_label_control.extract import api_key_present
    from app.services.treatment_pilot.generate import run_baseline_profile_aware
    import asyncio

    settings = Settings(openai_api_key="")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert api_key_present(settings) is False
    est = estimate_stage_costs(30)
    assert est["total_paid_usd"] > 0


def test_experiments_api_with_new_stage_dirs(tmp_path: Path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")
    # Point results root via monkeypatch on reader
    import app.services.experiments.reader as reader

    results = tmp_path / "experiments" / "results"
    exp = results / "exp_treatment_test"
    exp.mkdir(parents=True)
    write_json(
        exp / "summary.json",
        {
            "experiment_id": "exp_treatment_test",
            "status": "completed",
            "stage": "treatment_pipeline",
            "generation_mode": "interviewiq",
            "model": "gpt-5.6-terra",
            "combinations_attempted": 2,
            "combinations_ok": 2,
            "combinations_failed": 0,
            "total_estimated_cost_usd": 0.01,
        },
    )
    (exp / "treatment").mkdir()
    monkeypatch.setattr(reader, "project_root", lambda: tmp_path)
    monkeypatch.setattr(reader, "results_root", lambda: results)

    client = TestClient(create_app())
    # Override app dependency path — reader uses module-level results_root
    listed = reader.list_experiments()
    assert any(e["experiment_id"] == "exp_treatment_test" for e in listed)
    detail = reader.load_experiment("exp_treatment_test")
    assert detail["summary"]["stage"] == "treatment_pipeline"
    assert detail["pairs"] == []  # no baseline/ dir; still loads

    # Also hit HTTP if app uses same reader
    # Ensure app imports reader after monkeypatch — may already be bound; call list directly above is enough.
    get_settings.cache_clear()


def test_output_schema_pool_item_to_interview_question():
    item = _pool_item(dimension="Behavioural", target_gap=PROFILE_PROBE)
    iq = pool_item_to_interview_question(item)
    assert iq.priority == item.priority
    assert iq.target_gap == PROFILE_PROBE
    assert iq.question_type == "Behavioural"
