from app.services.retrieval.bm25 import BM25Matcher
from app.services.retrieval.tfidf import TfidfMatcher


def test_tfidf_identical_texts_high_similarity():
    matcher = TfidfMatcher()
    score, terms = matcher.similarity("python fastapi pytest", "python fastapi pytest")
    assert score > 0.99
    assert "python" in terms


def test_tfidf_disjoint_texts_low_similarity():
    matcher = TfidfMatcher()
    score, terms = matcher.similarity("python fastapi", "kubernetes terraform")
    assert score < 0.2
    assert terms == []


def test_tfidf_empty_returns_zero():
    matcher = TfidfMatcher()
    score, terms = matcher.similarity("", "python")
    assert score == 0.0
    assert terms == []


def test_bm25_ranks_relevant_document_first():
    matcher = BM25Matcher()
    docs = ["excel accounting payroll", "python fastapi rest api", "nursing patient care"]
    idx, score = matcher.best_match("python rest api", docs)
    assert idx == 1
    assert score > 0
