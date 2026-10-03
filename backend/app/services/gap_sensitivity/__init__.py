"""Evidence-set sensitivity analysis for profile-gap matrix scores.

Local-only (no LLM). Candidate evidence varies; job requirements, weights,
thresholds, and the embedding model stay fixed.
"""

from app.services.gap_sensitivity.analyze import run_gap_sensitivity

__all__ = ["run_gap_sensitivity"]
