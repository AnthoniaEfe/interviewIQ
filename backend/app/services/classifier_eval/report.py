"""Append computed classifier-reframing results (does not edit the frozen protocol)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt_ci(d: dict | None, key: str | None = None) -> str:
    if d is None:
        return "null"
    if key:
        inner = d.get(key)
        if not isinstance(inner, dict):
            v = d.get(key)
            return "null" if v is None else f"{v:.3f}"
        d = inner
    v, lo, hi = d.get("value"), d.get("ci_low"), d.get("ci_high")
    if v is None:
        return "null"
    if lo is None or hi is None:
        return f"{v:.3f} [CI null]"
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def append_results(report_path: Path, payload: dict[str, Any]) -> None:
    marker = "\n## Computed results (written after the frozen protocol)\n"
    existing = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    if marker.strip() in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"

    lines = [
        marker,
        "",
        f"Experiment id: `{payload.get('experiment_id')}`.",
        f"Git commit: `{payload.get('git_commit')}`.",
        f"Embedding: `{payload.get('embedding_model')}` on `{payload.get('embedding_device')}`.",
        "",
        "### Regression check (V0 vs stored test metrics)",
        "",
        f"```json\n{ _json(payload.get('regression_v0')) }\n```",
        "",
        "### Leakage audit",
        "",
        f"```json\n{ _json(payload.get('leakage')) }\n```",
        "",
        "### Truncation",
        "",
        f"```json\n{ _json(payload.get('truncation')) }\n```",
        "",
        "### Headline metrics (original test split)",
        "",
        "| variant | accuracy | balanced acc | macro-F1 | QWK | binary AUC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    results = payload.get("results_test") or {}
    for name, res in results.items():
        if name.endswith("_as_binary") or name == "V2":
            b = res if name == "V2" else res
            if name == "V2":
                b = res.get("binary") or {}
            lines.append(
                f"| {name} | {_n(b.get('accuracy'))} | {_n(b.get('balanced_accuracy'))} | "
                f"{_n(b.get('macro_f1'))} | n/a | {_n(b.get('auc'))} |"
            )
            continue
        ci = res.get("ci") or {}
        acc = ci.get("accuracy") or {"value": res.get("accuracy")}
        ba = ci.get("balanced_accuracy") or {"value": res.get("balanced_accuracy")}
        f1 = ci.get("macro_f1") or {"value": res.get("macro_f1")}
        qwk = ci.get("qwk") or {"value": res.get("qwk")}
        b = res.get("binary") or {}
        bauc = (res.get("ci") or {}).get("binary_auc") or {"value": b.get("auc")}
        lines.append(
            f"| {name} | {_fmt_ci(acc)} | {_fmt_ci(ba)} | {_fmt_ci(f1)} | {_fmt_ci(qwk)} | {_fmt_ci(bauc)} |"
        )
    base = payload.get("baselines_test") or {}
    maj = base.get("majority") or {}
    chance = base.get("chance_uniform") or {}
    strat = base.get("stratified_random") or {}
    lines += [
        "",
        "### Computed baselines (test labels)",
        "",
        f"- Majority class={maj.get('majority_class')}: accuracy={_n(maj.get('accuracy'))}, "
        f"balanced accuracy={_n(maj.get('balanced_accuracy'))}, macro-F1={_n(maj.get('macro_f1'))}.",
        f"- Uniform chance (MC): accuracy={_n(chance.get('accuracy'))}, "
        f"balanced accuracy={_n(chance.get('balanced_accuracy'))}, macro-F1={_n(chance.get('macro_f1'))}.",
        f"- Stratified random (MC): accuracy={_n(strat.get('accuracy'))}, "
        f"balanced accuracy={_n(strat.get('balanced_accuracy'))}, macro-F1={_n(strat.get('macro_f1'))}.",
        "",
        "### Improved-over-V0 rule (literal)",
        "",
        "A variant is improved only if selected on group CV **and** the paired-bootstrap 95% CI of "
        "macro-F1 gain vs V0 on test excludes 0 (lower bound > 0).",
        "",
        "| variant | selected_on_validation | gain | CI | excludes 0 (lower>0) | improved_over_V0 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, g in (payload.get("gains_vs_V0") or {}).items():
        lines.append(
            f"| {name} | {g.get('selected_on_validation')} | {_n(g.get('gain'))} | "
            f"[{_n(g.get('ci_low'))}, {_n(g.get('ci_high'))}] | {g.get('improves')} | "
            f"**{g.get('improved_over_V0')}** |"
        )
    lines += [
        "",
        "### Gap-lite features",
        "",
        "Gap-lite is **not identical to the production LLM-extracted matrix**. "
        "If these features predict fit labels, that is **indirect, independent support for the "
        "gap-matrix idea only**, not validation of the production matrix.",
        "",
        "Feature names:",
        "",
        ", ".join(f"`{n}`" for n in (payload.get("gap_lite_feature_names") or [])),
        "",
        "### Importances (V4_LR mean |coef|)",
        "",
        f"```json\n{_json(payload.get('gap_lite_importances'))}\n```",
        "",
        "### Hyperparameters chosen on train group-CV",
        "",
        f"```json\n{_json(payload.get('hyperparameters'))}\n```",
        "",
        "### Deviations",
        "",
    ]
    for d in payload.get("deviations") or []:
        lines.append(f"- {d}")
    lines += [
        "",
        "### What this means for the thesis",
        "",
        _thesis_paragraph(payload),
        "",
        "### Figures",
        "",
        "- `docs/figures/fig30_classifier_headline.png`",
        "- `docs/figures/fig31_per_class_f1.png`",
        "- `docs/figures/fig32_confusion.png`",
        "- `docs/figures/fig33_binary_roc.png`",
        "- `docs/figures/fig34_gap_lite_importance.png`",
        "",
        "### Limitations",
        "",
        "- Single public dataset; label noise is unknown.",
        "- Group CV reduces but does not eliminate all forms of leakage (paraphrase, templates).",
        "- Gap-lite uses JD sentence/bullet chunks, not Luna-extracted requirements.",
        "",
    ]
    report_path.write_text(existing.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


def _n(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _json(obj) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)


def _thesis_paragraph(payload: dict[str, Any]) -> str:
    gains = payload.get("gains_vs_V0") or {}
    any_imp = any(g.get("improved_over_V0") for g in gains.values())
    if any_imp:
        names = [k for k, g in gains.items() if g.get("improved_over_V0")]
        return (
            f"Under the pre-registered rule, the following variant(s) improved over V0: "
            f"{', '.join(names)}. Claims remain limited to this dataset and the frozen protocol."
        )
    return (
        "No variant met the pre-registered “improved over V0” rule. The classifier should be "
        "presented as **exploratory**, reporting macro-F1, balanced accuracy, and quadratic-weighted "
        "kappa against the **computed** majority baseline rather than accuracy alone."
    )
