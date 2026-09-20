from fastapi import FastAPI

from app.services.embeddings.service import HashingEmbeddingService, build_embedding_service


def get_embeddings(app: FastAPI):
    existing = getattr(app.state, "embeddings", None)
    if existing is not None:
        return existing
    settings = app.state.settings
    if settings.embedding_model == "hashing-test":
        app.state.embeddings = HashingEmbeddingService()
    else:
        app.state.embeddings = build_embedding_service(
            settings.embedding_model,
            settings.embedding_device,
        )
    return app.state.embeddings
