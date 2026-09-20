from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.config.settings import get_settings
from app.core.exceptions import InterviewIQError, ModelUnavailableError
from app.core.logging import configure_logging
from app.services.embeddings.service import HashingEmbeddingService
from app.services.llm.factory import build_router
from app.services.llm.openai_provider import verify_configured_models


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    llm_router = build_router(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.validate_models_on_startup and settings.openai_api_key:
            await verify_configured_models(app.state.llm_router.provider, settings)
        yield

    app = FastAPI(
        title="InterviewIQ",
        description="Research-grade interview question generation with profile-gap analysis and competency-weighted MMR.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.llm_router = llm_router
    if settings.embedding_model == "hashing-test":
        app.state.embeddings = HashingEmbeddingService()
    else:
        app.state.embeddings = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable(_: Request, exc: ModelUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": exc.public_message})

    @app.exception_handler(InterviewIQError)
    async def app_error(_: Request, exc: InterviewIQError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.public_message})

    return app


app = create_app()
