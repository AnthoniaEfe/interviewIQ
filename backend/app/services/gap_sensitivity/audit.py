"""Human audit sheet and false-gap examples (no LLM)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from app.schemas.domain import CandidateProfile, MatchLabel
from app.services.gap_sensitivity.evidence import CONDITION_A, CONDITION_D
from app.services.gap_sensitivity.io_utils import request_id
from app.services.gap_sensitivity.scoring import PairScoreResult

AUDIT_SEED = 42
CV_PREVIEW_CHARS = 1500


def _stratified_pair_indices(
    records: list[dict],
    *,
    n: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        lab = str(rec.get("fit_label") or "unknown")
        by_label.setdefault(lab, []).append(i)
    # Aim for balanced Good Fit / Potential Fit when possible.
    labels = sorted(by_label.keys())
    per = max(1, n // max(len(labels), 1))
    chosen: list[int] = []
    for lab in labels:
        pool = list(by_label[lab])
        rng.shuffle(pool)
        chosen.extend(pool[:per])
    if len(chosen) < n:
        remaining = [i for i in range(len(records)) if i not in set(chosen)]
        rng.shuffle(remaining)
        chosen.extend(remaining[: n - len(chosen)])
    return sorted(chosen[:n])


def build_audit_rows(
    *,
    split: str,
    records: list[dict],
    cv_texts: list[str | None],
    true_results: dict[str, list[PairScoreResult]],
    n_pairs: int = 10,
    seed: int = AUDIT_SEED,
) -> list[dict[str, Any]]:
    idxs = _stratified_pair_indices(records, n=n_pairs, seed=seed)
    rows: list[dict[str, Any]] = []
    for i in idxs:
        rec = records[i]
        cand = CandidateProfile.model_validate(rec["candidate"])
        cv = cv_texts[i] or ""
        truncated = len(cv) > CV_PREVIEW_CHARS
        preview = cv[:CV_PREVIEW_CHARS]
        a_scores = true_results[CONDITION_A][i].scores
        d_scores = true_results[CONDITION_D][i].scores
        recovered = []
        for a_s, d_s in zip(a_scores, d_scores, strict=True):
            if a_s.match_label == MatchLabel.MISSING and d_s.match_label in {
                MatchLabel.STRONG,
                MatchLabel.PARTIAL,
            }:
                recovered.append(
                    {
                        "requirement": a_s.job_competency,
                        "priority": a_s.priority.value,
                        "a_label": a_s.match_label.value,
                        "a_combined": a_s.combined,
                        "d_label": d_s.match_label.value,
                        "d_combined": d_s.combined,
                        "d_semantic": d_s.semantic,
                        "matched_cv_chunk": d_s.matched_candidate,
                    }
                )
        recovered = recovered[:5]
        exp_entries = []
        for job in cand.experience:
            exp_entries.append(
                {
                    "title": job.title,
                    "employer": job.employer,
                    "responsibilities": list(job.responsibilities or []),
                    "achievements": list(job.achievements or []),
                    "technologies": list(job.technologies or []),
                }
            )
        rows.append(
            {
                "split": split,
                "pair_id": rec.get("pair_id") or request_id(rec),
                "profile_id": rec.get("profile_id"),
                "fit_label": rec.get("fit_label"),
                "cv_text_preview": preview,
                "cv_truncated": truncated,
                "extracted_competencies": [
                    c.normalized_label for c in cand.competencies
                ],
                "extracted_technical_skills": [
                    s.normalized_label for s in cand.technical_skills
                ],
                "extracted_experience": exp_entries,
                "recovered_requirements_A_missing_to_D": recovered,
                "missed_skills_in_cv": "",
                "extraction_recall_estimate": "",
                "notes": "",
            }
        )
    return rows


def write_audit_sheet(path_csv: Path, path_md: Path, rows: list[dict[str, Any]]) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for row in rows:
        recovered = row["recovered_requirements_A_missing_to_D"]
        flat_rows.append(
            {
                "split": row["split"],
                "pair_id": row["pair_id"],
                "fit_label": row["fit_label"],
                "cv_truncated": row["cv_truncated"],
                "cv_text_preview": row["cv_text_preview"],
                "extracted_competencies": " | ".join(row["extracted_competencies"]),
                "extracted_technical_skills": " | ".join(
                    row["extracted_technical_skills"]
                ),
                "experience_titles": " | ".join(
                    (e.get("title") or "") for e in row["extracted_experience"]
                ),
                "n_recovered_examples": len(recovered),
                "recovered_requirements": " || ".join(
                    f"{r['requirement']} (A={r['a_combined']:.3f}->{r['d_label']} "
                    f"{r['d_combined']:.3f}; chunk={r['matched_cv_chunk']!r})"
                    for r in recovered
                ),
                "missed_skills_in_cv": "",
                "extraction_recall_estimate": "",
                "notes": "",
            }
        )
    with path_csv.open("w", encoding="utf-8", newline="") as fh:
        if flat_rows:
            writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    lines = [
        "# Gap sensitivity audit sheet",
        "",
        "Annotator columns (`missed_skills_in_cv`, `extraction_recall_estimate`, "
        "`notes`) are intentionally blank.",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['split']} / {row['pair_id']} [{row['fit_label']}]")
        lines.append("")
        lines.append(f"- CV truncated: {row['cv_truncated']}")
        lines.append(f"- Competencies: {row['extracted_competencies']}")
        lines.append(f"- Technical skills: {row['extracted_technical_skills']}")
        lines.append("")
        lines.append("### CV preview")
        lines.append("")
        lines.append("```")
        lines.append(row["cv_text_preview"])
        lines.append("```")
        lines.append("")
        lines.append("### Experience (extracted)")
        lines.append("")
        for exp in row["extracted_experience"]:
            lines.append(f"- **{exp.get('title') or '(no title)'}** @ {exp.get('employer')}")
            for bullet in (exp.get("responsibilities") or [])[:8]:
                lines.append(f"  - {bullet}")
        lines.append("")
        lines.append("### Recovered requirements (A=missing → D=strong/partial)")
        lines.append("")
        if not row["recovered_requirements_A_missing_to_D"]:
            lines.append("_None in this pair._")
        for r in row["recovered_requirements_A_missing_to_D"]:
            lines.append(
                f"- `{r['requirement']}`: A combined={r['a_combined']:.4f} → "
                f"D {r['d_label']} combined={r['d_combined']:.4f} "
                f"(semantic={r['d_semantic']:.4f}); "
                f"matched chunk: {r['matched_cv_chunk']!r}"
            )
        lines.append("")
        lines.append("| missed_skills_in_cv | extraction_recall_estimate | notes |")
        lines.append("| --- | --- | --- |")
        lines.append("|  |  |  |")
        lines.append("")
    path_md.write_text("\n".join(lines), encoding="utf-8")


def build_false_gap_examples(
    *,
    split_results: list[tuple[str, list[dict], dict[str, list[PairScoreResult]]]],
    top_n: int = 25,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for split, records, true_results in split_results:
        for i, rec in enumerate(records):
            a_scores = true_results[CONDITION_A][i].scores
            d_scores = true_results[CONDITION_D][i].scores
            for a_s, d_s in zip(a_scores, d_scores, strict=True):
                delta = d_s.combined - a_s.combined
                examples.append(
                    {
                        "split": split,
                        "pair_id": rec.get("pair_id") or request_id(rec),
                        "fit_label": rec.get("fit_label"),
                        "requirement": a_s.job_competency,
                        "priority": a_s.priority.value,
                        "a_label": a_s.match_label.value,
                        "d_label": d_s.match_label.value,
                        "a_combined": a_s.combined,
                        "d_combined": d_s.combined,
                        "delta_combined": delta,
                        "a_semantic": a_s.semantic,
                        "d_semantic": d_s.semantic,
                        "matched_cv_chunk": d_s.matched_candidate,
                        "a_matched": a_s.matched_candidate,
                    }
                )
    examples.sort(key=lambda r: r["delta_combined"], reverse=True)
    return examples[:top_n]


def write_false_gap_examples(path: Path, examples: list[dict[str, Any]]) -> None:
    lines = [
        "# False-gap examples (largest A→D combined-score increases)",
        "",
        "Top requirements by `combined(D) - combined(A)` across both splits.",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(
            f"## {i}. {ex['split']} / {ex['pair_id']} — `{ex['requirement']}`"
        )
        lines.append("")
        lines.append(
            f"- Fit label: {ex['fit_label']}; priority: {ex['priority']}"
        )
        lines.append(
            f"- A: {ex['a_label']} combined={ex['a_combined']:.4f} "
            f"semantic={ex['a_semantic']:.4f}; matched={ex['a_matched']!r}"
        )
        lines.append(
            f"- D: {ex['d_label']} combined={ex['d_combined']:.4f} "
            f"semantic={ex['d_semantic']:.4f}; "
            f"matched CV chunk={ex['matched_cv_chunk']!r}"
        )
        lines.append(f"- Δ combined: {ex['delta_combined']:.4f}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
