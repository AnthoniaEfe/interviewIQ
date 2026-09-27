import numpy as np

from app.config.settings import Settings
from app.schemas.domain import (
    CandidateProfile,
    InterviewQuestion,
    JobDescription,
    MatchLabel,
    NormalizedCompetency,
    RequirementPriority,
)
from app.services.embeddings.service import HashingEmbeddingService
from app.services.gap_analysis.matrix import build_gap_matrix, classify_match
from app.services.mmr.select import competency_weighted_mmr, redundant_pairs


def test_classify_match_thresholds():
    assert classify_match(tfidf=1.0, bm25=10.0, semantic=1.0, bm25_max=10.0) == MatchLabel.STRONG
    assert classify_match(tfidf=0.0, bm25=0.0, semantic=0.0, bm25_max=1.0) == MatchLabel.MISSING


def test_gap_matrix_uses_typed_skills_when_competencies_present():
    profile = CandidateProfile(
        competencies=[NormalizedCompetency(normalized_label="Leadership")],
        technical_skills=[NormalizedCompetency(normalized_label="Python")],
    )
    job = JobDescription(
        required_skills=[
            NormalizedCompetency(normalized_label="Python", priority=RequirementPriority.REQUIRED),
        ]
    )
    matrix = build_gap_matrix(profile, job, Settings(), HashingEmbeddingService())
    assert matrix.gaps[0].job_competency == "Python"
    assert matrix.gaps[0].match_label in {MatchLabel.STRONG, MatchLabel.PARTIAL}


def test_gap_matrix_detects_missing_and_match():
    profile = CandidateProfile(
        competencies=[
            NormalizedCompetency(normalized_label="Python"),
            NormalizedCompetency(normalized_label="FastAPI"),
        ]
    )
    job = JobDescription(
        required_skills=[
            NormalizedCompetency(normalized_label="Python", priority=RequirementPriority.REQUIRED),
            NormalizedCompetency(normalized_label="Kubernetes", priority=RequirementPriority.REQUIRED),
        ]
    )
    matrix = build_gap_matrix(profile, job, Settings(), HashingEmbeddingService())
    labels = {g.job_competency: g.match_label for g in matrix.gaps}
    assert labels["Python"] in {MatchLabel.STRONG, MatchLabel.PARTIAL}
    assert labels["Kubernetes"] in {MatchLabel.WEAK, MatchLabel.MISSING}


def test_mmr_prefers_diverse_questions():
    questions = [
        InterviewQuestion(question="python depth one", competency="python", priority=1.0),
        InterviewQuestion(question="python depth two", competency="python", priority=0.95),
        InterviewQuestion(question="kubernetes design", competency="kubernetes", priority=0.8),
    ]
    sim = np.array(
        [
            [1.0, 0.95, 0.1],
            [0.95, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
    )
    selection = competency_weighted_mmr(
        questions,
        sim,
        [1.0, 1.0, 1.2],
        lambda_=0.7,
        k=2,
    )
    selected_competencies = {questions[i].competency for i in selection.selected_indices}
    assert "kubernetes" in selected_competencies
    assert len(selection.mmr_scores) == 2


def test_redundant_pairs_threshold():
    sim = np.array([[1.0, 0.9], [0.9, 1.0]])
    pairs = redundant_pairs(sim, 0.85)
    assert pairs == [(0, 1, 0.9)]
