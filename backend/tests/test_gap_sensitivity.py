"""Tests for gap-sensitivity evidence builders, controls, and metrics.

Uses EMBEDDING_MODEL=hashing-test via HashingEmbeddingService — no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config.settings import Settings
from app.schemas.domain import (
    CandidateProfile,
    ExperienceEntry,
    JobDescription,
    NormalizedCompetency,
    RequirementPriority,
)
from app.services.embeddings.service import HashingEmbeddingService
from app.services.gap_sensitivity.chunking import chunk_cv_text
from app.services.gap_sensitivity.controls import mismatched_partner_indices
from app.services.gap_sensitivity.evidence import (
    CONDITION_A,
    CONDITION_A0,
    CONDITION_B,
    CONDITION_C,
    CONDITION_D,
    CONDITION_E,
    build_evidence,
)
from app.services.gap_sensitivity.metrics import (
    bootstrap_mean_ci,
    wilcoxon_paired,
)
from app.services.gap_sensitivity.scoring import score_requirements


def test_chunker_bullets_and_sentences():
    text = "- Built Python APIs for payments\nWe shipped Kubernetes clusters in prod. Also short."
    chunks = chunk_cv_text(text, min_words=4, max_words=60)
    assert any("Python" in c for c in chunks)
    assert any("Kubernetes" in c for c in chunks)
    # "Also short." has 2 words -> dropped
    assert not any(c.lower() == "also short." for c in chunks)


def test_chunker_empty_and_long_line():
    assert chunk_cv_text("") == []
    assert chunk_cv_text("   \n\n") == []
    words = " ".join(f"word{i}" for i in range(150))
    chunks = chunk_cv_text(words, min_words=4, max_words=60)
    assert len(chunks) >= 2
    assert all(len(c.split()) <= 60 for c in chunks)


def test_evidence_builders_dedup_and_missing_fields():
    profile = CandidateProfile(
        competencies=[
            NormalizedCompetency(normalized_label="Python"),
            NormalizedCompetency(normalized_label="python"),
        ],
        technical_skills=[NormalizedCompetency(normalized_label="FastAPI")],
        experience=[
            ExperienceEntry(
                title="Engineer",
                responsibilities=["Built APIs", "Built APIs"],
                achievements=["Cut latency 20%"],
            )
        ],
    )
    a0 = build_evidence(CONDITION_A0, profile)
    assert a0 == ["Python"]
    b = build_evidence(CONDITION_B, profile)
    assert "Python" in b and "FastAPI" in b
    c = build_evidence(CONDITION_C, profile)
    assert "Engineer" in c
    assert "Cut latency 20%" in c
    assert c.count("Built APIs") == 1

    empty = CandidateProfile()
    assert build_evidence(CONDITION_A0, empty) == [""]
    assert build_evidence(CONDITION_D, empty, None) == [""]
    assert build_evidence(CONDITION_D, empty, "Experienced Java developer with Spring Boot.") != [
        ""
    ]


def test_pipeline_condition_a_includes_skills():
    profile = CandidateProfile(
        competencies=[NormalizedCompetency(normalized_label="Leadership")],
        technical_skills=[NormalizedCompetency(normalized_label="Python")],
    )
    texts = build_evidence(CONDITION_A, profile)
    assert "Python" in texts
    assert "Leadership" in texts


def test_mismatched_control_never_self_and_deterministic():
    a = mismatched_partner_indices(5, seed=42, n_permutations=5)
    b = mismatched_partner_indices(5, seed=42, n_permutations=5)
    assert np.array_equal(a, b)
    assert a.shape == (5, 5)
    assert not np.any(a == np.arange(5)[None, :])
    with pytest.raises(ValueError):
        mismatched_partner_indices(1, seed=0, n_permutations=1)


def test_bootstrap_ci_reproducible():
    values = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    c1 = bootstrap_mean_ci(values, n_resamples=200, seed=42)
    c2 = bootstrap_mean_ci(values, n_resamples=200, seed=42)
    assert c1 == c2
    assert c1["ci_low"] is not None and c1["ci_high"] is not None
    assert c1["ci_low"] <= c1["mean"] <= c1["ci_high"]


def test_wilcoxon_paired_runs():
    x = np.asarray([0.8, 0.7, 0.6, 0.5, 0.4])
    y = np.asarray([0.1, 0.2, 0.15, 0.25, 0.3])
    out = wilcoxon_paired(x, y)
    assert out["n_pairs"] == 5
    assert out["p_value"] is not None
    assert out["rank_biserial"] is not None


def test_end_to_end_two_synthetic_pairs():
    settings = Settings()
    emb = HashingEmbeddingService()
    pairs = [
        (
            CandidateProfile(
                competencies=[NormalizedCompetency(normalized_label="Python")],
                technical_skills=[NormalizedCompetency(normalized_label="Django")],
                experience=[
                    ExperienceEntry(
                        title="Backend engineer",
                        responsibilities=["Implemented REST APIs in Python"],
                    )
                ],
            ),
            JobDescription(
                required_skills=[
                    NormalizedCompetency(
                        normalized_label="Python",
                        priority=RequirementPriority.REQUIRED,
                    ),
                    NormalizedCompetency(
                        normalized_label="Kubernetes",
                        priority=RequirementPriority.REQUIRED,
                    ),
                ]
            ),
            "Jane Engineer. Built Django apps. Python expert for APIs.",
        ),
        (
            CandidateProfile(
                competencies=[NormalizedCompetency(normalized_label="Java")],
                technical_skills=[NormalizedCompetency(normalized_label="Spring")],
            ),
            JobDescription(
                required_skills=[
                    NormalizedCompetency(
                        normalized_label="Java",
                        priority=RequirementPriority.REQUIRED,
                    )
                ]
            ),
            "John Developer with Java and Spring Boot microservices experience.",
        ),
    ]
    for profile, job, cv in pairs:
        for cond in (
            CONDITION_A,
            CONDITION_A0,
            CONDITION_B,
            CONDITION_C,
            CONDITION_D,
            CONDITION_E,
        ):
            evidence = build_evidence(cond, profile, cv)
            result = score_requirements(job, evidence, settings, emb)
            assert len(result.scores) == len(job.required_skills)
            assert result.evidence_size >= 0
            for score in result.scores:
                assert 0.0 <= score.combined <= 1.0 + 1e-6 or score.combined >= 0.0
