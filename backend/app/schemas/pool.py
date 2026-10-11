"""Schemas for InterviewIQ pool generation (treatment). Do not edit domain.py."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.domain import CompetencyType, InterviewQuestion

POOL_DIMENSIONS = (
    "Technical",
    "Behavioural",
    "Situational",
    "Cultural/Motivational",
    "Role-Specific",
)

PoolDimension = Literal[
    "Technical",
    "Behavioural",
    "Situational",
    "Cultural/Motivational",
    "Role-Specific",
]

DIMENSION_TO_COMPETENCY_TYPE: dict[str, CompetencyType] = {
    "Technical": CompetencyType.TECHNICAL,
    "Behavioural": CompetencyType.SOFT,
    "Situational": CompetencyType.OTHER,
    "Cultural/Motivational": CompetencyType.SOFT,
    "Role-Specific": CompetencyType.DOMAIN,
}

PROFILE_PROBE = "profile_probe"


class PoolQuestionItem(BaseModel):
    question: str
    competency: str
    dimension: PoolDimension
    target_gap: str
    difficulty: Literal["junior", "mid", "senior", "easy", "medium", "hard"] = "medium"
    rationale: str = ""
    source_evidence: list[str] = Field(default_factory=list)
    priority: float

    @field_validator("priority")
    @classmethod
    def priority_in_open_unit_interval(cls, value: float) -> float:
        if value is None or not (0.0 < float(value) <= 1.0):
            raise ValueError("priority must be in (0, 1]")
        return float(value)

    @field_validator("question", "competency", "target_gap")
    @classmethod
    def non_empty_str(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("field must be non-empty")
        return str(value).strip()


class PoolGeneratedSet(BaseModel):
    questions: list[PoolQuestionItem] = Field(min_length=1)


def pool_item_to_interview_question(item: PoolQuestionItem) -> InterviewQuestion:
    return InterviewQuestion(
        question=item.question,
        competency=item.competency,
        competency_type=DIMENSION_TO_COMPETENCY_TYPE[item.dimension],
        target_gap=item.target_gap,
        question_type=item.dimension,
        difficulty=item.difficulty,
        rationale=item.rationale or None,
        source_evidence=list(item.source_evidence),
        priority=float(item.priority),
    )
