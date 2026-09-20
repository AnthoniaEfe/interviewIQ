from __future__ import annotations

from fastapi import APIRouter, Request

from app.config.settings import Settings
from app.services.llm.tasks import LLMTask
from app.services.llm.usage import usage_store

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "fast_model": settings.llm_fast_model,
        "primary_model": settings.llm_primary_model,
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
    }


@router.get("/models")
async def models(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    llm_router = request.app.state.llm_router
    return {
        "provider": settings.llm_provider,
        "fast_model": settings.llm_fast_model,
        "primary_model": settings.llm_primary_model,
        "fallback_model": settings.llm_fallback_model or None,
        "routing": {
            "cv_extraction": llm_router.model_for(LLMTask.CV_EXTRACTION),
            "jd_extraction": llm_router.model_for(LLMTask.JD_EXTRACTION),
            "profile_analysis": llm_router.model_for(LLMTask.PROFILE_ANALYSIS),
            "gap_analysis": llm_router.model_for(LLMTask.GAP_ANALYSIS),
            "question_generation": llm_router.model_for(LLMTask.QUESTION_GENERATION),
            "question_validation": llm_router.model_for(LLMTask.QUESTION_VALIDATION),
        },
        "note": "Configured IDs are used as-is. Unavailable IDs fail closed; they are never substituted.",
    }


@router.get("/usage")
async def usage() -> dict:
    return usage_store.summary()
