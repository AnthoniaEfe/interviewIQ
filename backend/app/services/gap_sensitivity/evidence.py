"""Candidate evidence-set builders for gap sensitivity conditions."""

from __future__ import annotations

from collections.abc import Callable

from app.schemas.domain import CandidateProfile
from app.services.gap_analysis.matrix import _candidate_texts
from app.services.gap_sensitivity.chunking import chunk_cv_text

# Condition ids used throughout tables / figures.
CONDITION_A = "A_competencies"
CONDITION_A0 = "A0_competencies_only"
CONDITION_B = "B_comp_skills"
CONDITION_C = "C_comp_skills_exp"
CONDITION_D = "D_raw_cv_chunks"
CONDITION_E = "E_union"

# Main ladder reported in thesis tables. A is production evidence (regression).
# A0 is the sparse competencies-only counterfactual motivating the brief.
MAIN_CONDITIONS = (
    CONDITION_A,
    CONDITION_A0,
    CONDITION_B,
    CONDITION_C,
    CONDITION_D,
    CONDITION_E,
)

EvidenceBuilder = Callable[[CandidateProfile, str | None], list[str]]


def _dedupe(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        if text is None:
            continue
        cleaned = str(text).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def evidence_pipeline_current(profile: CandidateProfile, _cv_text: str | None = None) -> list[str]:
    """Production ``_candidate_texts`` (skills + competencies + experience, …)."""
    return list(_candidate_texts(profile))


def evidence_competencies_only(profile: CandidateProfile, _cv_text: str | None = None) -> list[str]:
    labels = [c.normalized_label for c in profile.competencies]
    return _dedupe(labels) or [""]


def evidence_comp_skills(profile: CandidateProfile, _cv_text: str | None = None) -> list[str]:
    labels = [c.normalized_label for c in profile.competencies]
    labels += [s.normalized_label for s in profile.technical_skills]
    return _dedupe(labels) or [""]


def evidence_comp_skills_exp(profile: CandidateProfile, _cv_text: str | None = None) -> list[str]:
    """B plus experience titles, responsibilities, and achievements.

    ``ExperienceEntry`` schema fields used: title, responsibilities, achievements.
    Technologies are intentionally omitted here so C follows the brief's
    "role/title + responsibility/achievement bullets" definition; production
    Condition A already includes technologies via ``_candidate_texts``.
    """
    texts = evidence_comp_skills(profile)
    if texts == [""]:
        texts = []
    for job in profile.experience:
        if job.title:
            texts.append(job.title)
        texts.extend(job.responsibilities or [])
        texts.extend(job.achievements or [])
    return _dedupe(texts) or [""]


def evidence_raw_cv_chunks(profile: CandidateProfile, cv_text: str | None = None) -> list[str]:
    del profile  # unused; condition D ignores extraction
    if not cv_text:
        return [""]
    chunks = chunk_cv_text(cv_text)
    return chunks or [""]


def evidence_union(profile: CandidateProfile, cv_text: str | None = None) -> list[str]:
    texts = evidence_comp_skills_exp(profile)
    if texts == [""]:
        texts = []
    texts.extend(evidence_raw_cv_chunks(profile, cv_text))
    return _dedupe([t for t in texts if t]) or [""]


BUILDERS: dict[str, EvidenceBuilder] = {
    CONDITION_A: evidence_pipeline_current,
    CONDITION_A0: evidence_competencies_only,
    CONDITION_B: evidence_comp_skills,
    CONDITION_C: evidence_comp_skills_exp,
    CONDITION_D: evidence_raw_cv_chunks,
    CONDITION_E: evidence_union,
}


CONDITION_DESCRIPTIONS: dict[str, str] = {
    CONDITION_A: (
        "Production pipeline evidence via ``_candidate_texts``: technical/soft/"
        "domain skills, certifications, competencies, experience technologies "
        "and responsibilities (deduplicated). This is the stored-result baseline."
    ),
    CONDITION_A0: "Extracted competency ``normalized_label`` strings only.",
    CONDITION_B: "Competencies + extracted technical skills (case-insensitive dedupe).",
    CONDITION_C: (
        "B + experience title, responsibility bullets, and achievement bullets."
    ),
    CONDITION_D: "Raw CV sentence/bullet chunks (no extraction).",
    CONDITION_E: "Union of C and D.",
}


def build_evidence(
    condition: str,
    profile: CandidateProfile,
    cv_text: str | None = None,
) -> list[str]:
    if condition not in BUILDERS:
        raise KeyError(f"Unknown evidence condition: {condition}")
    return BUILDERS[condition](profile, cv_text)
