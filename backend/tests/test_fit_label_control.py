"""Tests for fit-label control (no network, hashing embeddings)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.settings import Settings
from app.schemas.domain import (
    CandidateProfile,
    ExperienceEntry,
    JobDescription,
    NormalizedCompetency,
    RequirementPriority,
)
from app.services.embeddings.service import HashingEmbeddingService
from app.services.fit_label_control.extract import dry_run_payload
from app.services.fit_label_control.mismatch import mismatched_partner_indices_by_label
from app.services.fit_label_control.sampling import load_used_pair_ids, sample_no_fit_pairs
from app.services.fit_label_control.score import score_records
from app.services.fit_label_control.spend import SpendAborted, SpendGuard
from app.services.fit_label_control.stats import (
    bootstrap_auc_ci,
    jonckheere_terpstra,
    jonckheere_terpstra_statistic,
)
from app.services.fit_label_control.tables import apply_decision_rule, bundle_condition
from app.services.gap_sensitivity.io_utils import project_root


def test_sample_no_fit_deterministic_and_disjoint():
    used = load_used_pair_ids(project_root())
    df = pd.DataFrame(
        {
            "resume_text": [f"resume {i}" for i in range(80)],
            "job_description_text": [f"job {i}" for i in range(80)],
            "label": ["No Fit"] * 50 + ["Good Fit"] * 15 + ["Potential Fit"] * 15,
            "label_id": [0] * 50 + [2] * 15 + [1] * 15,
            "pair_id": [f"fake_{i:05d}" for i in range(80)],
        }
    )
    a = sample_no_fit_pairs(df, n=10, seed=42, exclude_ids=used)
    b = sample_no_fit_pairs(df, n=10, seed=42, exclude_ids=used)
    assert list(a["pair_id"]) == list(b["pair_id"])
    assert set(a["label"]) == {"No Fit"}
    assert set(a["pair_id"]).isdisjoint(used)
    c = sample_no_fit_pairs(
        df, n=10, seed=2026, exclude_ids=used | set(a["pair_id"].astype(str))
    )
    assert set(c["pair_id"]).isdisjoint(set(a["pair_id"]))
    assert set(c["pair_id"]).isdisjoint(used)


def test_spend_guard_refuses_without_confirm_and_aborts_over_max():
    g = SpendGuard(max_cost=1.0, confirm_spend=False)
    with pytest.raises(SpendAborted, match="confirm-spend"):
        g.require_confirm()
    g2 = SpendGuard(max_cost=0.01, confirm_spend=True, prior_per_pair=0.05)
    with pytest.raises(SpendAborted, match="max-cost"):
        g2.check_next_pair()
    g3 = SpendGuard(max_cost=1.0, confirm_spend=True)
    g3.add_actual(0.4, pair_id="a")
    g3.add_actual(0.4, pair_id="b")
    with pytest.raises(SpendAborted, match="exceeded"):
        g3.add_actual(0.4, pair_id="c")
    dry = dry_run_payload(
        [{"pair_id": "x"}, {"pair_id": "y"}],
        cache={"x": "unused"},
        prior_per_pair=0.01,
    )
    assert dry["n_cached"] == 1
    assert dry["n_uncached"] == 1
    assert dry["estimated_cost_usd"] == pytest.approx(0.01)


def test_jonckheere_terpstra_toy_increasing():
    groups = [
        np.asarray([0.0, 0.1, 0.2]),
        np.asarray([0.5, 0.6]),
        np.asarray([0.9, 1.0, 1.1]),
    ]
    stat = jonckheere_terpstra_statistic(groups)
    assert stat == pytest.approx(21.0)
    out = jonckheere_terpstra(groups, n_permutations=500, seed=42, alternative="greater")
    assert out["p_value"] is not None
    assert out["p_value"] < 0.05
    reversed_groups = list(reversed(groups))
    out_rev = jonckheere_terpstra(
        reversed_groups, n_permutations=500, seed=42, alternative="greater"
    )
    assert out_rev["p_value"] > out["p_value"]


def test_bootstrap_auc_ci_reproducible():
    y = np.asarray([0, 0, 0, 1, 1, 1])
    s = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    a = bootstrap_auc_ci(y, s, n_resamples=200, seed=42)
    b = bootstrap_auc_ci(y, s, n_resamples=200, seed=42)
    assert a == b
    assert a["auc"] is not None and a["auc"] > 0.9
    assert a["ci_low"] <= a["auc"] <= a["ci_high"]


def test_same_label_mismatched_never_self():
    labels = ["No Fit"] * 4 + ["Good Fit"] * 3 + ["Potential Fit"] * 3
    partners = mismatched_partner_indices_by_label(labels, seed=42, n_permutations=5)
    n = len(labels)
    assert partners.shape == (5, n)
    assert not np.any(partners == np.arange(n)[None, :])
    for p in range(5):
        for i in range(n):
            assert labels[int(partners[p, i])] == labels[i]
    again = mismatched_partner_indices_by_label(labels, seed=42, n_permutations=5)
    assert np.array_equal(partners, again)


def _synthetic_pair(label: str, match: bool) -> tuple[dict, str, str]:
    skill = "Python" if match else "COBOL"
    profile = CandidateProfile(
        competencies=[NormalizedCompetency(normalized_label=skill)],
        technical_skills=[NormalizedCompetency(normalized_label=skill)],
        experience=[
            ExperienceEntry(
                title="Engineer",
                responsibilities=[f"Used {skill} daily"],
                technologies=[skill],
            )
        ],
    )
    job = JobDescription(
        required_skills=[
            NormalizedCompetency(
                normalized_label="Python",
                priority=RequirementPriority.REQUIRED,
            )
        ]
    )
    rec = {
        "pair_id": f"{label}_{skill}_{match}",
        "profile_id": f"{label}_{skill}_{match}",
        "jd_id": f"{label}_{skill}_{match}",
        "fit_label": label,
        "seniority": "mid",
        "candidate": profile.model_dump(),
        "job": job.model_dump(),
    }
    cv = f"Candidate skilled in {skill}."
    jd = "We require Python."
    return rec, cv, jd


def test_end_to_end_synthetic_pairs(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-test")
    settings = Settings()
    emb = HashingEmbeddingService()
    records, cvs, jds = [], [], []
    for i in range(2):
        r, cv, jd = _synthetic_pair("Good Fit", True)
        r["pair_id"] = f"g{i}"
        records.append(r)
        cvs.append(cv)
        jds.append(jd)
        r, cv, jd = _synthetic_pair("Potential Fit", True)
        r["pair_id"] = f"p{i}"
        records.append(r)
        cvs.append(cv)
        jds.append(jd)
        r, cv, jd = _synthetic_pair("No Fit", False)
        r["pair_id"] = f"n{i}"
        records.append(r)
        cvs.append(cv)
        jds.append(jd)
    scored = score_records(records, cvs, jds, settings, emb)
    assert len(scored["A"]["rows"]) == 6
    bundle = bundle_condition(scored["A"]["rows"])
    assert bundle["n_pairs"] == 6
    assert len(bundle["descriptives"]) == 3
    decision = apply_decision_rule(
        bundle["gradient_gap_rate"],
        bundle["gradient_gap_rate"],
        bundle["discrimination"]["gap_rate"],
        bundle["discrimination"]["gap_rate"],
    )
    assert "outcome" in decision
    partners = scored["mismatch_partners"]
    labels = [r["fit_label"] for r in records]
    for i in range(partners.shape[1]):
        assert labels[int(partners[0, i])] == labels[i]
        assert int(partners[0, i]) != i
