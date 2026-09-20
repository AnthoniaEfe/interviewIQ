from __future__ import annotations

from rank_bm25 import BM25Okapi

from app.services.retrieval.tfidf import tokenize


class BM25Matcher:
    """BM25 lexical retrieval of candidate competencies given a job requirement."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, min_score: float = 0.0) -> None:
        self.k1 = k1
        self.b = b
        self.min_score = min_score

    def scores(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        tokenized = [tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        raw = bm25.get_scores(tokenize(query))
        return [float(score) for score in raw]

    def best_match(self, query: str, documents: list[str]) -> tuple[int | None, float]:
        ranked = self.scores(query, documents)
        if not ranked:
            return None, 0.0
        best_index = max(range(len(ranked)), key=lambda i: ranked[i])
        best_score = ranked[best_index]
        if best_score < self.min_score:
            return best_index, best_score
        return best_index, best_score
