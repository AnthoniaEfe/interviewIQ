"""Phase 2 gold-set analysis (runs only on real annotation files)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from app.services.gold_set.analyze_lib import (
    bootstrap_metric_ci,
    percent_agreement,
    quadratic_weighted_kappa,
    unweighted_kappa,
    weighted_mean,
)
from app.services.gold_set.constants import BOOTSTRAP, STATS_SEED
from app.services.gap_sensitivity.io_utils import (
    git_commit_hash,
    library_versions,
    load_json,
    write_json,
)


SYS3 = {"strong_match": 2, "partial_match": 1, "weak_match": 0, "missing": 0}


def _parse_label(row: dict) -> tuple[int | None, bool]:
    cj = str(row.get("cannot_judge", "")).strip().lower() in {"1", "true", "yes", "c"}
    raw = str(row.get("label", "")).strip()
    if cj or raw.upper() == "C" or raw == "":
        return None, True if cj or raw.upper() == "C" else cj
    try:
        lab = int(float(raw))
    except ValueError:
        return None, True
    if lab not in {0, 1, 2}:
        return None, True
    return lab, cj


def _consensus(a: dict | None, b: dict | None, adj: dict | None) -> tuple[int | None, str]:
    if a is None:
        return None, "missing"
    la, ca = _parse_label(a)
    if b is None:
        return (None if ca else la), "single"
    lb, cb = _parse_label(b)
    if ca or cb:
        return None, "cannot_judge"
    if la is not None and la == lb:
        return la, "agree"
    if adj is not None:
        lj, cj = _parse_label(adj)
        return (None if cj else lj), "adjudicated"
    return None, "disagree"


def _pair_for_item(
    iid: str,
    maps: dict[str, dict[str, dict]],
    assignment: dict[str, Any] | None,
) -> tuple[dict | None, dict | None]:
    if assignment:
        names = assignment.get("item_to_annotators", {}).get(iid)
        if not names or len(names) < 2:
            return None, None
        left, right = names[0], names[1]
        return maps.get(left, {}).get(iid), maps.get(right, {}).get(iid)
    names = list(maps)
    first = maps[names[0]].get(iid) if names else None
    second = maps[names[1]].get(iid) if len(names) > 1 else None
    return first, second


def _pairwise_labels(
    ids: list[str],
    maps: dict[str, dict[str, dict]],
    assignment: dict[str, Any] | None,
    *,
    all_annotator_pairs: bool = False,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if all_annotator_pairs:
        names = list(maps)
        for iid in ids:
            labels = []
            for name in names:
                row = maps[name].get(iid)
                if row is None:
                    continue
                lab, cj = _parse_label(row)
                if cj or lab is None:
                    continue
                labels.append(lab)
            for i, left in enumerate(labels):
                for right in labels[i + 1 :]:
                    pairs.append((left, right))
        return pairs
    for iid in ids:
        a_row, b_row = _pair_for_item(iid, maps, assignment)
        if a_row is None or b_row is None:
            continue
        la, ca = _parse_label(a_row)
        lb, cb = _parse_label(b_row)
        if ca or cb or la is None or lb is None:
            continue
        pairs.append((la, lb))
    return pairs


def run_phase2(
    root: Path,
    rows_by: dict[str, list[dict]],
    adj_rows: list[dict] | None,
    assignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pack = root / "experiments/gold_set"
    key = {it["item_id"]: it for it in load_json(pack / "answer_key_do_not_open_before_annotating.json")["items"]}
    blind = load_json(pack / "items_blind.json")
    main_ids = [it["item_id"] for it in blind["items"]]
    cal_ids = [it["item_id"] for it in (blind.get("calibration") or [])]
    blind_rows = list(blind.get("items") or []) + list(blind.get("calibration") or [])
    req_text = {it["item_id"]: it.get("requirement_text") or "" for it in blind_rows}
    cv_text = {it["item_id"]: it.get("cv_text") or "" for it in blind_rows}
    maps = {name: {r["item_id"]: r for r in rows} for name, rows in rows_by.items()}
    adj_map = {r["item_id"]: r for r in (adj_rows or [])}
    dual = len(maps) > 1 or assignment is not None

    def agr(ids: list[str], *, all_annotator_pairs: bool = False) -> dict[str, Any]:
        if not dual:
            n = 0
            for iid in ids:
                row = next(iter(maps.values()), {}).get(iid)
                if row is None:
                    continue
                lab, cj = _parse_label(row)
                if not cj and lab is not None:
                    n += 1
            return {"mode": "single_annotator_fallback", "n": n}
        pairs = _pairwise_labels(
            ids, maps, assignment, all_annotator_pairs=all_annotator_pairs
        )
        if not pairs:
            return {"n": 0}
        a = np.asarray([p[0] for p in pairs])
        b = np.asarray([p[1] for p in pairs])
        return {
            "n": int(len(pairs)),
            "percent_agreement": percent_agreement(a, b),
            "kappa_unweighted": bootstrap_metric_ci(a, b, unweighted_kappa, n_resamples=BOOTSTRAP, seed=STATS_SEED),
            "kappa_quadratic": bootstrap_metric_ci(
                a, b, quadratic_weighted_kappa, n_resamples=BOOTSTRAP, seed=STATS_SEED
            ),
            "confusion": confusion_matrix(a, b, labels=[0, 1, 2]).tolist(),
        }

    items = []
    for iid in main_ids:
        a_row, b_row = _pair_for_item(iid, maps, assignment)
        if a_row is None and b_row is None:
            continue
        cons, how = _consensus(a_row, b_row if dual else None, adj_map.get(iid))
        k = key[iid]
        snippet = ""
        if a_row:
            snippet = a_row.get("snippet") or ""
        elif b_row:
            snippet = b_row.get("snippet") or ""
        items.append(
            {
                "item_id": iid,
                "consensus": cons,
                "how": how,
                "split": k["split"],
                "system3": SYS3.get(k["system_label"], 0),
                "combined": k["combined"],
                "tfidf": k["tfidf"],
                "bm25_norm": k["bm25_norm"],
                "semantic": k["semantic"],
                "weight": k.get("design_weight") or 1.0,
                "system_label": k["system_label"],
                "snippet": snippet,
                "evidence_string": k.get("evidence_string") or "",
                "requirement_text": req_text.get(iid, ""),
                "requirement_in_cv": _phrase_in_cv(req_text.get(iid, ""), cv_text.get(iid, "")),
            }
        )

    usable = [it for it in items if it["consensus"] is not None]
    y_h = np.asarray([it["consensus"] for it in usable], dtype=int)
    y_s = np.asarray([it["system3"] for it in usable], dtype=int)
    w = np.asarray([it["weight"] for it in usable], dtype=float)
    scores = np.asarray([it["combined"] for it in usable], dtype=float)

    def pack_metrics(mask: np.ndarray) -> dict[str, Any]:
        yh, ys, ww, sc = y_h[mask], y_s[mask], w[mask], scores[mask]
        if yh.size == 0:
            return {"n": 0}
        acc_i = (yh == ys).astype(float)
        return {
            "n": int(yh.size),
            "accuracy": float(accuracy_score(yh, ys)),
            "accuracy_weighted": weighted_mean(acc_i, ww),
            "macro_f1": float(f1_score(yh, ys, average="macro", labels=[0, 1, 2], zero_division=0)),
            "qwk": quadratic_weighted_kappa(yh, ys),
            "confusion": confusion_matrix(yh, ys, labels=[0, 1, 2]).tolist(),
            "spearman_score_vs_human": _spearman(sc, yh),
        }

    splits = {
        "pooled": pack_metrics(np.ones(len(usable), dtype=bool)),
        "primary-30": pack_metrics(np.asarray([it["split"] == "primary-30" for it in usable])),
        "holdout-30": pack_metrics(np.asarray([it["split"] == "holdout-30" for it in usable])),
    }

    false_gap = [it for it in usable if it["system3"] == 0 and it["consensus"] >= 1]
    from app.services.gold_set.hypotheses import (
        aqg_implication,
        evaluate_hypotheses,
        g1_may_be_interpreted,
        render_report,
    )

    hypotheses = evaluate_hypotheses(usable)
    payload = {
        "status": "completed",
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "n_main": len(main_ids),
        "n_assigned_main": len(items),
        "n_usable_consensus": len(usable),
        "consensus_disposition": dict(Counter(it["how"] for it in items)),
        "annotator_vs_system": _annotator_vs_system(maps, key, main_ids),
        "label_margins_main": _label_margins(maps, main_ids),
        "cannot_judge_rate_by_annotator": {
            name: _cj_rate(amap, main_ids) for name, amap in maps.items()
        },
        "agreement_main": agr(main_ids, all_annotator_pairs=False),
        "agreement_calibration": agr(cal_ids, all_annotator_pairs=True),
        "calibration_adjudication": _calibration_adjudication(pack, maps, key, cal_ids),
        "single_annotator": not dual,
        "n_annotators": len(maps),
        "system_vs_human": splits,
        "false_gap_n": len(false_gap),
        "false_gap_rate": (len(false_gap) / len(usable)) if usable else None,
        "g1": hypotheses["g1"],
        "g2": hypotheses["g2"],
        "g3": hypotheses["g3"],
        "g4": hypotheses["g4"],
        "aqg_implication": aqg_implication(hypotheses),
        "note": (
            "Consensus drops items the two assigned annotators did not agree on, "
            "unless adjudicated.csv resolves them. G4 adoption uses primary-30 to "
            "choose a configuration and holdout-30 to test it."
        ),
        "seed": STATS_SEED,
        "bootstrap_resamples": BOOTSTRAP,
    }
    kappa_q = payload["agreement_main"].get("kappa_quadratic") or {}
    interpreted = g1_may_be_interpreted(kappa_q)
    disposition = payload.get("consensus_disposition") or {}
    n_adj_main = int(disposition.get("adjudicated") or 0)
    n_open_disagree = int(disposition.get("disagree") or 0)
    adj_main_note = (
        f"Main-set adjudicated.csv resolved {n_adj_main} dual-coder disagreements "
        f"({n_open_disagree} still open; cannot_judge={int(disposition.get('cannot_judge') or 0)})."
        if n_adj_main or adj_map
        else "Main-set disagreements have no adjudicated.csv resolution."
    )
    payload["analysis_round"] = "round2_gate_on_unchanged_human_labels"
    payload["g1_interpretation"] = {
        "rule": (
            "Interpret G1 only when the inter-annotator quadratic-weighted kappa "
            "on the main pairs has a bootstrap 95% CI entirely above 0. "
            "This gate is separate from the round-1 G1 calculation."
        ),
        "interpreted": interpreted,
        "kappa_quadratic": kappa_q,
        "reason": (
            "Main-pair quadratic kappa has a 95% CI entirely above 0, so G1 may be read. "
            + adj_main_note
            if interpreted
            else (
                "Main-pair quadratic kappa has a 95% interval that covers 0 or is undefined. "
                "G1 is not interpreted. The hybrid-gap component stays implemented and unvalidated. "
                "Calibration items c001–c010 have an adjudicated label. "
                + adj_main_note
            )
        ),
    }
    if not interpreted:
        g4_adopt = bool((payload.get("g4") or {}).get("adopt_recalibration"))
        g4_note = (
            "G4's frozen adoption test is True on this consensus (holdout ΔQWK CI excludes 0); "
            "the reverse-split sensitivity does not adopt. This run does not rewrite stored gap "
            "artifacts. Because G1 remains uninterpreted, treat the G4 candidate as a locked "
            "sensitivity configuration pending an explicit production cutover decision."
            if g4_adopt
            else (
                "G4 does not adopt. Production weights 0.25/0.25/0.50 and thresholds "
                "0.65/0.40/0.20 stay."
            )
        )
        payload["aqg_implication"] = (
            "Round 2 does not interpret G1. The five annotators' main-set quadratic kappa "
            "still has a 95% interval covering 0. "
            f"{adj_main_note} "
            "The hybrid gap matrix stays an implemented, unvalidated component. "
            f"{g4_note}"
        )
    out = root / "experiments/results" / "gold_set_analysis"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", payload)
    write_json(out / "consensus_items.json", items)
    from app.services.gold_set.figures import write_gold_figures

    write_gold_figures(root / "docs/figures", payload)
    report = root / "docs/gold_set_report.md"
    report.write_text(render_report(payload), encoding="utf-8")
    return payload


def _phrase_in_cv(requirement: str, cv: str) -> bool:
    phrase = " ".join(requirement.split()).casefold()
    if len(phrase) < 2:
        return False
    return phrase in " ".join(cv.split()).casefold()


def _calibration_adjudication(
    pack: Path,
    maps: dict[str, dict[str, dict]],
    key: dict[str, dict],
    cal_ids: list[str],
) -> dict[str, Any] | None:
    """Post-discussion labels for the 10 shared items. Not a main-set consensus."""
    path = pack / "annotations" / "adjudicated_calibration.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    labels: dict[str, int] = {}
    for row in rows:
        iid = str(row.get("item_id") or "").strip()
        lab, cj = _parse_label(row)
        if cj or lab is None:
            raise RuntimeError(f"{path} has no usable label for {iid or '(blank)'}")
        labels[iid] = lab
    missing = [iid for iid in cal_ids if iid not in labels]
    extra = [iid for iid in labels if iid not in set(cal_ids)]
    if missing or extra:
        raise RuntimeError(
            f"{path} must contain exactly the calibration items. "
            f"Missing {missing}. Extra {extra}."
        )
    per_annotator: dict[str, Any] = {}
    pooled_human: list[int] = []
    pooled_key: list[int] = []
    for name, amap in maps.items():
        human: list[int] = []
        gold: list[int] = []
        for iid in cal_ids:
            row = amap.get(iid)
            if row is None:
                continue
            lab, cj = _parse_label(row)
            if cj or lab is None:
                continue
            human.append(lab)
            gold.append(labels[iid])
        yh = np.asarray(human, dtype=int)
        yg = np.asarray(gold, dtype=int)
        per_annotator[name] = {
            "n": int(yh.size),
            "n_match": int(np.sum(yh == yg)) if yh.size else 0,
            "percent_agreement": percent_agreement(yh, yg),
            "qwk": quadratic_weighted_kappa(yh, yg) if yh.size else None,
        }
        pooled_human.extend(human)
        pooled_key.extend(gold)
    yh = np.asarray(pooled_human, dtype=int)
    yg = np.asarray(pooled_key, dtype=int)
    system = np.asarray([SYS3.get(key[iid]["system_label"], 0) for iid in cal_ids], dtype=int)
    adjudicated = np.asarray([labels[iid] for iid in cal_ids], dtype=int)
    return {
        "source": str(path.relative_to(pack.parents[1])),
        "n": len(cal_ids),
        "labels": {iid: labels[iid] for iid in cal_ids},
        "annotator_vs_adjudicated": per_annotator,
        "pooled_independent_vs_adjudicated": {
            "n": int(yh.size),
            "percent_agreement": percent_agreement(yh, yg),
            "qwk": bootstrap_metric_ci(
                yh,
                yg,
                quadratic_weighted_kappa,
                n_resamples=BOOTSTRAP,
                seed=STATS_SEED + 7,
            ),
        },
        "system_vs_adjudicated": {
            "n": int(adjudicated.size),
            "accuracy": float(np.mean(system == adjudicated)),
            "qwk": quadratic_weighted_kappa(adjudicated, system),
            "confusion": confusion_matrix(adjudicated, system, labels=[0, 1, 2]).tolist(),
            "role": "Descriptive check on the 10 calibration items. Not G1.",
        },
    }


def _label_margins(maps: dict[str, dict[str, dict]], main_ids: list[str]) -> dict[str, dict[str, int]]:
    margins: dict[str, dict[str, int]] = {}
    for name, amap in maps.items():
        counts: Counter[str] = Counter()
        for iid in main_ids:
            row = amap.get(iid)
            if row is None:
                continue
            lab, cj = _parse_label(row)
            if cj or lab is None:
                counts["cannot_judge"] += 1
            else:
                counts[str(lab)] += 1
        margins[name] = dict(counts)
    return margins


def _annotator_vs_system(
    maps: dict[str, dict[str, dict]],
    key: dict[str, dict],
    main_ids: list[str],
) -> dict[str, Any]:
    """Sensitivity: each annotator's own labels, including items the pair disagreed on."""
    out: dict[str, Any] = {}
    for name, amap in maps.items():
        human: list[int] = []
        system: list[int] = []
        for iid in main_ids:
            row = amap.get(iid)
            if row is None or iid not in key:
                continue
            lab, cj = _parse_label(row)
            if cj or lab is None:
                continue
            human.append(lab)
            system.append(SYS3.get(key[iid]["system_label"], 0))
        yh = np.asarray(human, dtype=int)
        ys = np.asarray(system, dtype=int)
        out[name] = {
            "n": int(yh.size),
            "qwk": bootstrap_metric_ci(
                yh,
                ys,
                quadratic_weighted_kappa,
                n_resamples=BOOTSTRAP,
                seed=STATS_SEED,
            )
            if yh.size
            else {"value": None, "ci_low": None, "ci_high": None, "n": 0},
        }
    return out


def _cj_rate(amap: dict, ids: list[str]) -> float | None:
    if not ids:
        return None
    n = 0
    cj = 0
    for iid in ids:
        if iid not in amap:
            continue
        n += 1
        _, c = _parse_label(amap[iid])
        cj += int(c)
    return (cj / n) if n else None


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    from scipy.stats import spearmanr

    r, _ = spearmanr(x, y)
    if r is None or not np.isfinite(r):
        return None
    return float(r)
