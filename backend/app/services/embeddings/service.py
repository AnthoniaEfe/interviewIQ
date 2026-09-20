from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingService(ABC):
    model_name: str

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def similarity(self, a: str, b: str) -> float:
        va = self.embed_text(a)
        vb = self.embed_text(b)
        return float(_cosine(va, vb))

    def similarity_matrix(self, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        if not texts_a or not texts_b:
            return np.zeros((len(texts_a), len(texts_b)))
        ea = self.embed_batch(texts_a)
        eb = self.embed_batch(texts_b)
        ea_norm = ea / np.clip(np.linalg.norm(ea, axis=1, keepdims=True), 1e-12, None)
        eb_norm = eb / np.clip(np.linalg.norm(eb, axis=1, keepdims=True), 1e-12, None)
        return ea_norm @ eb_norm.T


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class HashingEmbeddingService(EmbeddingService):
    """Deterministic CPU embedding used in tests. Not for production research runs."""

    def __init__(self, dim: int = 64, model_name: str = "hashing-test") -> None:
        self.dim = dim
        self.model_name = model_name

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                vectors[i, hash(token) % self.dim] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm:
                vectors[i] /= norm
        return vectors


class SentenceTransformerEmbeddingService(EmbeddingService):
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=self.device)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_embedding_service(
    model_name: str,
    device: str = "auto",
    *,
    use_hashing_fallback: bool = False,
) -> EmbeddingService:
    if use_hashing_fallback:
        return HashingEmbeddingService(model_name=model_name)
    return SentenceTransformerEmbeddingService(model_name=model_name, device=device)
