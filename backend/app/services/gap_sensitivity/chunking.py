"""Raw CV text chunkers for evidence-set condition D."""

from __future__ import annotations

import re

_BULLET_RE = re.compile(r"^\s*(?:[-*•●▪◦]|\d+[.)])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WHITESPACE_RE = re.compile(r"\s+")


def _word_count(text: str) -> int:
    return len([w for w in _WHITESPACE_RE.split(text.strip()) if w])


def _split_long(text: str, max_words: int) -> list[str]:
    words = [w for w in _WHITESPACE_RE.split(text.strip()) if w]
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


def chunk_cv_text(
    text: str,
    *,
    min_words: int = 4,
    max_words: int = 60,
) -> list[str]:
    """Split raw CV text into sentence/bullet chunks.

    Splitter rules (documented for the report):
    1. Normalise newlines; treat each non-empty line as a candidate unit.
    2. Lines that look like bullets (``-``, ``*``, ``•``, numbered) are kept
       as whole units (after stripping the bullet marker).
    3. Non-bullet lines are further split on sentence boundaries
       (``.!?`` followed by whitespace) or residual newlines.
    4. Chunks shorter than ``min_words`` are dropped.
    5. Chunks longer than ``max_words`` are split into contiguous
       non-overlapping windows of ``max_words``.
    6. Case-insensitive deduplication; original casing of the first seen form
       is kept.
    """
    if not text or not text.strip():
        return []

    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    units: list[str] = []
    for line in normalised.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _BULLET_RE.match(stripped):
            units.append(_BULLET_RE.sub("", stripped).strip())
            continue
        pieces = _SENTENCE_SPLIT_RE.split(stripped)
        units.extend(p.strip() for p in pieces if p and p.strip())

    seen: set[str] = set()
    out: list[str] = []
    for unit in units:
        cleaned = _WHITESPACE_RE.sub(" ", unit).strip()
        if not cleaned:
            continue
        for piece in _split_long(cleaned, max_words):
            if _word_count(piece) < min_words:
                continue
            key = piece.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(piece)
    return out
