"""Embedding pair features (document-level and chunk-pooled)."""

from __future__ import annotations

import numpy as np

from app.services.embeddings.service import EmbeddingService, HashingEmbeddingService
from app.services.gap_sensitivity.chunking import chunk_cv_text


def pair_features(emb: EmbeddingService, resumes: list[str], jobs: list[str]) -> np.ndarray:
    rv = emb.embed_batch(resumes)
    jv = emb.embed_batch(jobs)
    return np.concatenate([rv, jv, np.abs(rv - jv), rv * jv], axis=1)


def _pool_doc(emb: EmbeddingService, text: str) -> np.ndarray:
    chunks = chunk_cv_text(text, min_words=4, max_words=80) or [text or " "]
    mat = emb.embed_batch(chunks)
    mean = mat.mean(axis=0)
    mx = mat.max(axis=0)
    return np.concatenate([mean, mx], axis=0)


def chunk_pair_features(
    emb: EmbeddingService, resumes: list[str], jobs: list[str]
) -> np.ndarray:
    rows = []
    for r, j in zip(resumes, jobs, strict=True):
        rv = _pool_doc(emb, r)
        jv = _pool_doc(emb, j)
        rows.append(np.concatenate([rv, jv, np.abs(rv - jv), rv * jv]))
    return np.vstack(rows)


def truncation_rate(
    texts: list[str],
    embeddings: EmbeddingService,
    *,
    max_seq: int = 256,
) -> dict:
    n = len(texts)
    n_long = 0
    tok = getattr(getattr(embeddings, "_model", None), "tokenizer", None)
    for t in texts:
        if tok is not None:
            try:
                ids = tok.encode(t, add_special_tokens=True, truncation=False)
                if len(ids) > max_seq:
                    n_long += 1
                continue
            except Exception:
                pass
        # Word-piece proxy when tokenizer is unavailable (hashing tests).
        if len(str(t).split()) > max_seq:
            n_long += 1
    return {
        "n": n,
        "n_exceeding_max_seq": n_long,
        "rate": (n_long / n) if n else None,
        "max_seq": max_seq,
        "used_tokenizer": tok is not None,
    }


def is_hashing(emb: EmbeddingService) -> bool:
    return isinstance(emb, HashingEmbeddingService)
