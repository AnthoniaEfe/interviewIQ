"""Phase 2 gold-set analysis (runs only on real annotation files)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
                "requirement_text": None,
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
    payload = {
        "status": "completed",
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "n_main": len(main_ids),
        "n_usable_consensus": len(usable),
        "cannot_judge_rate_by_annotator": {
            name: _cj_rate(amap, main_ids) for name, amap in maps.items()
        },
        "agreement_main": agr(main_ids, all_annotator_pairs=False),
        "agreement_calibration": agr(cal_ids, all_annotator_pairs=True),
        "single_annotator": not dual,
        "n_annotators": len(maps),
        "system_vs_human": splits,
        "false_gap_n": len(false_gap),
        "false_gap_rate": (len(false_gap) / len(usable)) if usable else None,
        "note": "Signal AUC, recalibration, and figures are produced by the same stage when n is sufficient.",
        "seed": STATS_SEED,
        "bootstrap_resamples": BOOTSTRAP,
    }
    out = root / "experiments/results" / "gold_set_analysis"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", payload)
    report = root / "docs/gold_set_report.md"
    report.write_text(
        "# Gold-set report\n\n"
        "Frozen hypotheses: see `docs/gold_set_protocol.md` (G1–G4).\n\n"
        f"Usable consensus items: {len(usable)} / {len(main_ids)}.\n\n"
        f"False-gap count (system gap & human ≥ 1): {len(false_gap)} "
        f"(rate {payload['false_gap_rate']}).\n\n"
        "G4 adoption was not evaluated in this compact write-path if recalibration tables "
        "are absent from summary.json.\n",
        encoding="utf-8",
    )
    return payload


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
