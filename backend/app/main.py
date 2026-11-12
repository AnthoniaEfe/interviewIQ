from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.config.settings import get_settings
from app.core.exceptions import InterviewIQError, ModelUnavailableError
from app.core.logging import configure_logging
from app.services.llm.factory import build_router
from app.services.llm.openai_provider import verify_configured_models


def _cors_origins(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    llm_router = build_router(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Do not load Sentence-Transformers / CUDA here. Embeddings are built
        # lazily when a route needs them.
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
    # Never eagerly load torch / MiniLM at import or startup.
    app.state.embeddings = None
    origins = _cors_origins(settings.cors_allow_origins)
    regex = (getattr(settings, "cors_allow_origin_regex", None) or "").strip()
    if origins or regex:
        cors_kwargs: dict = {
            "allow_origins": origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
        if regex:
            cors_kwargs["allow_origin_regex"] = regex
        app.add_middleware(CORSMiddleware, **cors_kwargs)
    app.include_router(api_router)

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable(_: Request, exc: ModelUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": exc.public_message})

    @app.exception_handler(InterviewIQError)
    async def app_error(_: Request, exc: InterviewIQError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.public_message})

    return app


app = create_app()
