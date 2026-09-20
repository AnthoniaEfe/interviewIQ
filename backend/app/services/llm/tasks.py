from enum import Enum


class LLMTask(str, Enum):
    CV_EXTRACTION = "cv_extraction"
    JD_EXTRACTION = "jd_extraction"
    NORMALIZATION = "normalization"
    CLASSIFICATION = "classification"
    PROFILE_ANALYSIS = "profile_analysis"
    GAP_ANALYSIS = "gap_analysis"
    QUESTION_GENERATION = "question_generation"
    QUESTION_REFINEMENT = "question_refinement"
    QUESTION_VALIDATION = "question_validation"
    SIMPLE_TRANSFORMATION = "simple_transformation"


FAST_TASKS = {
    LLMTask.CV_EXTRACTION,
    LLMTask.JD_EXTRACTION,
    LLMTask.NORMALIZATION,
    LLMTask.CLASSIFICATION,
    LLMTask.QUESTION_VALIDATION,
    LLMTask.SIMPLE_TRANSFORMATION,
}

PRIMARY_TASKS = {
    LLMTask.PROFILE_ANALYSIS,
    LLMTask.GAP_ANALYSIS,
    LLMTask.QUESTION_GENERATION,
    LLMTask.QUESTION_REFINEMENT,
}
