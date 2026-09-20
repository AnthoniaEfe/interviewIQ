from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_TOKEN_RE = re.compile(r"[a-z0-9+#]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfMatcher:
    """Deterministic TF-IDF cosine similarity between job and candidate texts."""

    def __init__(self, min_similarity: float = 0.0) -> None:
        self.min_similarity = min_similarity
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=True,
            norm="l2",
        )

    def similarity(self, candidate_text: str, job_text: str) -> tuple[float, list[str]]:
        if not candidate_text.strip() or not job_text.strip():
            return 0.0, []
        matrix = self.vectorizer.fit_transform([job_text, candidate_text])
        score = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
        feature_names = self.vectorizer.get_feature_names_out()
        job_vec = matrix[0].toarray()[0]
        cand_vec = matrix[1].toarray()[0]
        matched = [
            term
            for term, j, c in zip(feature_names, job_vec, cand_vec, strict=True)
            if j > 0 and c > 0
        ]
        if score < self.min_similarity:
            return score, matched
        return score, matched

    def pairwise_matrix(self, job_texts: list[str], candidate_texts: list[str]) -> list[list[float]]:
        if not job_texts or not candidate_texts:
            return []
        corpus = job_texts + candidate_texts
        matrix = self.vectorizer.fit_transform(corpus)
        n_jobs = len(job_texts)
        job_mat = matrix[:n_jobs]
        cand_mat = matrix[n_jobs:]
        sim = cosine_similarity(job_mat, cand_mat)
        return sim.tolist()
