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


def _consensus(a: dict, b: dict | None, adj: dict | None) -> tuple[int | None, str]:
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


def run_phase2(root: Path, a_rows: list[dict], b_rows: list[dict] | None, adj_rows: list[dict] | None) -> dict[str, Any]:
    pack = root / "experiments/gold_set"
    key = {it["item_id"]: it for it in load_json(pack / "answer_key_do_not_open_before_annotating.json")["items"]}
    blind = load_json(pack / "items_blind.json")
    main_ids = [it["item_id"] for it in blind["items"]]
    cal_ids = [it["item_id"] for it in (blind.get("calibration") or [])]
    a_map = {r["item_id"]: r for r in a_rows}
    b_map = {r["item_id"]: r for r in (b_rows or [])}
    adj_map = {r["item_id"]: r for r in (adj_rows or [])}

    def labels_for(ids: list[str]) -> dict[str, Any]:
        ya, yb, agreed = [], [], []
        for iid in ids:
            if iid not in a_map:
                continue
            la, ca = _parse_label(a_map[iid])
            if b_map:
                lb, cb = _parse_label(b_map.get(iid, {}))
                if not ca and not cb and la is not None and la == lb:
                    agreed.append((iid, la))
                ya.append(la)
                yb.append(lb)
            elif not ca and la is not None:
                agreed.append((iid, la))
                ya.append(la)
        return {"ya": ya, "yb": yb, "agreed": agreed}

    main_l = labels_for(main_ids)
    cal_l = labels_for(cal_ids)

    def agr(block, ids):
        if not b_rows:
            return {"mode": "single_annotator_fallback", "n": len(block["agreed"])}
        ya = np.asarray([x for x in block["ya"] if x is not None])
        # align pairs where both labelled
        pairs = []
        for iid in ids:
            if iid not in a_map or iid not in b_map:
                continue
            la, ca = _parse_label(a_map[iid])
            lb, cb = _parse_label(b_map[iid])
            if ca or cb or la is None or lb is None:
                continue
            pairs.append((la, lb))
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
        if iid not in a_map:
            continue
        cons, how = _consensus(a_map[iid], b_map.get(iid) if b_map else None, adj_map.get(iid))
        k = key[iid]
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
                "snippet": (a_map[iid].get("snippet") or ""),
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
        "cannot_judge_rate_A": _cj_rate(a_map, main_ids),
        "agreement_main": agr(main_l, main_ids),
        "agreement_calibration": agr(cal_l, cal_ids),
        "single_annotator": b_rows is None,
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
