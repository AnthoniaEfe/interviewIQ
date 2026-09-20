from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.config.settings import Settings
from app.schemas.domain import CandidateProfile, GapMatrix, InterviewQuestion, JobDescription
from app.core.dependencies import get_embeddings
from app.services.embeddings.service import HashingEmbeddingService
from app.services.gap_analysis.matrix import build_gap_matrix
from app.services.mmr.select import competency_weighted_mmr, redundant_pairs

router = APIRouter()


class GapRequest(BaseModel):
    candidate: CandidateProfile
    job: JobDescription


class SelectRequest(BaseModel):
    questions: list[InterviewQuestion]
    competency_weights: list[float] | None = None
    k: int | None = None
    mmr_lambda: float | None = None
    similarity_threshold: float | None = None
    use_test_embeddings: bool = False


@router.post("/analyze/profile-gap", response_model=GapMatrix)
async def profile_gap(payload: GapRequest, request: Request) -> GapMatrix:
    settings: Settings = request.app.state.settings
    embeddings = get_embeddings(request.app) if settings.enable_semantic else None
    return build_gap_matrix(payload.candidate, payload.job, settings, embeddings)


@router.post("/questions/select")
async def select_questions(payload: SelectRequest, request: Request) -> dict:
    settings: Settings = request.app.state.settings
    questions = payload.questions
    if not questions:
        return {"selected": [], "redundant_pairs": [], "mmr": None}

    embeddings = (
        HashingEmbeddingService()
        if payload.use_test_embeddings
        else get_embeddings(request.app)
    )

    texts = [q.question for q in questions]
    sim = embeddings.similarity_matrix(texts, texts)
    threshold = payload.similarity_threshold or settings.question_similarity_threshold
    pairs = redundant_pairs(sim, threshold)
    weights = payload.competency_weights or [q.priority or 1.0 for q in questions]
    k = payload.k or settings.final_question_count
    lambda_ = payload.mmr_lambda if payload.mmr_lambda is not None else settings.mmr_lambda
    selection = competency_weighted_mmr(
        questions,
        sim,
        weights,
        lambda_=lambda_,
        k=k,
    )
    selected = [questions[i] for i in selection.selected_indices]
    return {
        "selected": [q.model_dump() for q in selected],
        "redundant_pairs": [
            {"i": i, "j": j, "similarity": score} for i, j, score in pairs
        ],
        "mmr": {
            "indices": selection.selected_indices,
            "relevance_scores": selection.relevance_scores,
            "redundancy_scores": selection.redundancy_scores,
            "mmr_scores": selection.mmr_scores,
            "competency_weights": selection.competency_weights,
            "lambda": lambda_,
        },
    }
