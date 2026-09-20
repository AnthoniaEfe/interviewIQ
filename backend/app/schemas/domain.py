from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CompetencyOrigin(str, Enum):
    CANDIDATE = "candidate"
    JOB = "job"


class CompetencyType(str, Enum):
    TECHNICAL = "technical"
    SOFT = "soft"
    DOMAIN = "domain"
    EDUCATION = "education"
    CERTIFICATION = "certification"
    LEADERSHIP = "leadership"
    OTHER = "other"


class RequirementPriority(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    UNSPECIFIED = "unspecified"


class MatchLabel(str, Enum):
    STRONG = "strong_match"
    PARTIAL = "partial_match"
    WEAK = "weak_match"
    MISSING = "missing"


class GenerationMode(str, Enum):
    BASELINE = "baseline"
    INTERVIEWIQ = "interviewiq"


class SourceSpan(BaseModel):
    source_text: str
    source_section: str | None = None
    origin: CompetencyOrigin


class NormalizedCompetency(BaseModel):
    normalized_label: str
    original_mentions: list[str] = Field(default_factory=list)
    competency_type: CompetencyType = CompetencyType.TECHNICAL
    priority: RequirementPriority = RequirementPriority.UNSPECIFIED
    confidence: float = 1.0
    source: SourceSpan | None = None


class EducationEntry(BaseModel):
    degree: str | None = None
    institution: str | None = None
    field: str | None = None
    year: str | None = None
    source_text: str | None = None


class ExperienceEntry(BaseModel):
    title: str | None = None
    employer: str | None = None
    duration: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    source_text: str | None = None


class ProjectEntry(BaseModel):
    name: str | None = None
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    source_text: str | None = None


class CandidateProfile(BaseModel):
    name: str | None = None
    professional_summary: str | None = None
    seniority_indicators: list[str] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    technical_skills: list[NormalizedCompetency] = Field(default_factory=list)
    soft_skills: list[NormalizedCompetency] = Field(default_factory=list)
    certifications: list[NormalizedCompetency] = Field(default_factory=list)
    domain_experience: list[NormalizedCompetency] = Field(default_factory=list)
    competencies: list[NormalizedCompetency] = Field(default_factory=list)


class JobDescription(BaseModel):
    title: str | None = None
    seniority: str | None = None
    years_of_experience: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    required_skills: list[NormalizedCompetency] = Field(default_factory=list)
    preferred_skills: list[NormalizedCompetency] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    certifications: list[NormalizedCompetency] = Field(default_factory=list)
    domain_knowledge: list[NormalizedCompetency] = Field(default_factory=list)
    soft_skills: list[NormalizedCompetency] = Field(default_factory=list)
    leadership_requirements: list[str] = Field(default_factory=list)
    system_design_requirements: list[str] = Field(default_factory=list)
    competencies: list[NormalizedCompetency] = Field(default_factory=list)


class SimilarityEvidence(BaseModel):
    tfidf_score: float | None = None
    tfidf_matched_terms: list[str] = Field(default_factory=list)
    bm25_score: float | None = None
    semantic_score: float | None = None
    matched_candidate_competency: str | None = None


class ProfileGap(BaseModel):
    job_competency: str
    priority: RequirementPriority
    match_label: MatchLabel
    weight: float
    evidence: SimilarityEvidence
    candidate_evidence: list[str] = Field(default_factory=list)


class GapMatrix(BaseModel):
    gaps: list[ProfileGap]
    strong_matches: list[str] = Field(default_factory=list)
    partial_matches: list[str] = Field(default_factory=list)
    weak_matches: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class InterviewQuestion(BaseModel):
    question: str
    competency: str
    competency_type: CompetencyType = CompetencyType.TECHNICAL
    target_gap: str | None = None
    question_type: str = "technical_depth"
    difficulty: Literal["junior", "mid", "senior", "easy", "medium", "hard"] = "medium"
    rationale: str | None = None
    source_evidence: list[str] = Field(default_factory=list)
    priority: float = 0.0


class QuestionSet(BaseModel):
    questions: list[InterviewQuestion]


class LLMUsage(BaseModel):
    provider: str
    model: str
    task_type: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    estimated_cost: float = 0.0
    latency_ms: float = 0.0
    retry_count: int = 0
    fallback_used: bool = False
    fallback_reason: str | None = None


class GenerationMetadata(BaseModel):
    experiment_id: str | None = None
    generation_mode: GenerationMode
    provider: str
    model: str
    prompt_version: str
    embedding_model: str
    mmr_lambda: float
    question_similarity_threshold: float
    question_pool_size: int
    final_question_count: int
    usage: list[LLMUsage] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str | None = None
