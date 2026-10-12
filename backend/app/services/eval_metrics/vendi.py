"""Vendi score (parameter-free diversity) with a cosine kernel."""

from __future__ import annotations

import numpy as np


def vendi_score_from_embeddings(E: np.ndarray) -> float | None:
    """Vendi score for unit-norm (or to-be-normalised) embedding rows.

    K = E E^T; eigenvalues of K/n; VS = exp(-sum l_i ln l_i).
    Range is 1..n for n items.
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2 or E.shape[0] == 0:
        return None
    n = E.shape[0]
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    E = E / norms
    K = E @ E.T
    # Numerical symmetry
    K = 0.5 * (K + K.T)
    evals = np.linalg.eigvalsh(K / n)
    evals = np.clip(evals, 0.0, None)
    s = float(evals.sum())
    if s <= 0:
        return 1.0
    evals = evals / s
    # Entropy; 0 * ln 0 := 0
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(evals > 0, np.log(evals), 0.0)
    entropy = -float(np.sum(evals * logs))
    return float(np.exp(entropy))


def vendi_score_from_texts(texts: list[str], embed_batch) -> float | None:
    if not texts:
        return None
    E = np.asarray(embed_batch(texts), dtype=np.float64)
    return vendi_score_from_embeddings(E)
