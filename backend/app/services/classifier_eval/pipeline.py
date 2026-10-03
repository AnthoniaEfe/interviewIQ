"""Classifier reframing evaluation pipeline (local embeddings only)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config.settings import Settings
from app.services.classifier_eval.embed_features import (
    chunk_pair_features,
    is_hashing,
    pair_features,
    truncation_rate,
)
from app.services.classifier_eval.gap_lite import FEATURE_NAMES, gap_lite_matrix
from app.services.classifier_eval.groups import leakage_report
from app.services.classifier_eval.figures import (
    figure_30_headline,
    figure_31_per_class_f1,
    figure_32_confusion,
    figure_33_binary_roc,
    figure_34_importances,
)
from app.services.classifier_eval.metrics import (
    binary_metrics,
    bootstrap_metric_ci,
    chance_baseline,
    holm_adjust,
    majority_baseline,
    mcnemar_accuracy,
    paired_bootstrap_gain,
    quadratic_weighted_kappa,
    stratified_random_baseline,
    three_class_metrics,
)
from app.services.classifier_eval.report import append_results
from app.services.datasets.resume_fit import load_split
from app.services.embeddings.service import build_embedding_service
from app.services.gap_sensitivity.io_utils import (
    git_commit_hash,
    library_versions,
    project_root,
    utc_experiment_id,
    write_json,
)

SEED = 42
BOOT = 2000
TOL = 0.002
EXPECTED = {
    "accuracy": 0.46389994314951677,
    "macro_f1": 0.4239011172161828,
    "weighted_f1": 0.46155480465348275,
}


def _lr(balanced: bool = True, C: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced" if balanced else None,
                    C=C,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=SEED,
        max_depth=3,
        learning_rate=0.1,
        max_iter=80,
    )


def _group_cv_macro_f1(est, X, y, groups, n_splits: int = 5) -> float:
    n_groups = len(np.unique(groups))
    splits = max(2, min(n_splits, n_groups, len(y) // 3 or 1))
    if splits < 2 or n_groups < 2:
        m = clone(est)
        m.fit(X, y)
        return float(f1_score(y, m.predict(X), average="macro", zero_division=0))
    gkf = GroupKFold(n_splits=splits)
    scores = []
    for tr, va in gkf.split(X, y, groups):
        m = clone(est)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[va])
        scores.append(f1_score(y[va], pred, average="macro", zero_division=0))
    return float(np.mean(scores))


def _tune_lr_C(X, y, groups, balanced: bool = True) -> tuple[Pipeline, float, float]:
    best_c, best_s = 1.0, -1.0
    for C in (0.25, 1.0, 4.0):
        s = _group_cv_macro_f1(_lr(balanced=balanced, C=C), X, y, groups)
        if s > best_s:
            best_s, best_c = s, C
    est = _lr(balanced=balanced, C=best_c)
    est.fit(X, y)
    return est, best_c, best_s


def _binarise(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y) >= 1).astype(int)


def evaluate_predictions(
    y: np.ndarray,
    yhat: np.ndarray,
    proba: np.ndarray | None,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    m = three_class_metrics(y, yhat, proba=proba)
    yb = _binarise(y)
    yhb = _binarise(yhat)
    scores = None
    if proba is not None:
        if proba.shape[1] == 2:
            scores = proba[:, 1]
        elif proba.shape[1] >= 3:
            scores = 1.0 - proba[:, 0]
    b = binary_metrics(yb, yhb, scores)

    def acc(a, p):
        return float(np.mean(a == p))

    def mf1(a, p):
        return float(f1_score(a, p, average="macro", zero_division=0))

    def ba(a, p):
        return float(balanced_accuracy_score(a, p))

    def qwk(a, p):
        v = quadratic_weighted_kappa(a, p)
        return float(v) if v is not None else 0.0

    ci = {
        "accuracy": bootstrap_metric_ci(y, yhat, acc, n_resamples=BOOT, seed=seed),
        "macro_f1": bootstrap_metric_ci(y, yhat, mf1, n_resamples=BOOT, seed=seed),
        "balanced_accuracy": bootstrap_metric_ci(y, yhat, ba, n_resamples=BOOT, seed=seed),
        "qwk": bootstrap_metric_ci(y, yhat, qwk, n_resamples=BOOT, seed=seed),
    }
    if scores is not None:

        def auc_fn(a, p, extra=None):
            from sklearn.metrics import roc_auc_score

            ya = _binarise(a)
            if extra is None:
                return None
            if len(np.unique(ya)) < 2:
                return None
            return float(roc_auc_score(ya, extra))

        ci["binary_auc"] = bootstrap_metric_ci(
            y, yhat, auc_fn, n_resamples=BOOT, seed=seed, extra=scores
        )
    return {**m, "binary": b, "ci": ci}


def run_classifier_eval(
    *,
    root: Path | None = None,
    settings: Settings | None = None,
    embeddings=None,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    smoke: bool = False,
    skip_gap_lite: bool = False,
) -> dict[str, Any]:
    root = root or project_root()
    settings = settings or Settings()
    if embeddings is None:
        embeddings = build_embedding_service(
            "hashing-test" if smoke else settings.embedding_model,
            "cpu" if smoke else settings.embedding_device,
            use_hashing_fallback=smoke or settings.embedding_model == "hashing-test",
        )
    cache_dir = root / "experiments/datasets/resume_job_fit/cache"
    if train_df is None:
        train_df = load_split("train", cache_dir=cache_dir)
        test_df = load_split("test", cache_dir=cache_dir)
    assert test_df is not None

    leak = leakage_report(train_df, test_df)
    tr, te = leak["train"], leak["test"]
    y_tr = tr["label_id"].to_numpy()
    y_te = te["label_id"].to_numpy()
    groups = tr["group_id"].to_numpy()

    resumes_tr = tr["resume_text"].astype(str).tolist()
    jobs_tr = tr["job_description_text"].astype(str).tolist()
    resumes_te = te["resume_text"].astype(str).tolist()
    jobs_te = te["job_description_text"].astype(str).tolist()

    feat_dir = root / "experiments/datasets/resume_job_fit/models_reframing"
    feat_dir.mkdir(parents=True, exist_ok=True)
    orig_feat = root / "experiments/datasets/resume_job_fit/cache/features"

    def cached(name: str, builder):
        path = feat_dir / name
        if path.exists() and not smoke:
            print(f"[classifier_eval] load {path}")
            return np.load(path)
        print(f"[classifier_eval] compute {name}")
        arr = builder()
        if not smoke:
            np.save(path, arr)
        return arr

    minilm_tr = orig_feat / "train_features_all-MiniLM-L6-v2.npy"
    minilm_te = orig_feat / "test_features_all-MiniLM-L6-v2.npy"
    if (
        not smoke
        and not is_hashing(embeddings)
        and minilm_tr.is_file()
        and minilm_te.is_file()
    ):
        print("[classifier_eval] load original MiniLM pair-feature cache for V0/V1/V2")
        X0_tr = np.load(minilm_tr)
        X0_te = np.load(minilm_te)
        if X0_tr.shape[0] != len(y_tr) or X0_te.shape[0] != len(y_te):
            print("[classifier_eval] original cache row count mismatch; recomputing")
            X0_tr = cached("X0_train.npy", lambda: pair_features(embeddings, resumes_tr, jobs_tr))
            X0_te = cached("X0_test.npy", lambda: pair_features(embeddings, resumes_te, jobs_te))
    else:
        X0_tr = cached("X0_train.npy", lambda: pair_features(embeddings, resumes_tr, jobs_tr))
        X0_te = cached("X0_test.npy", lambda: pair_features(embeddings, resumes_te, jobs_te))

    trunc_r = truncation_rate(resumes_tr + resumes_te, embeddings)
    trunc_j = truncation_rate(jobs_tr + jobs_te, embeddings)

    models: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    probas: dict[str, np.ndarray | None] = {}
    cv_scores: dict[str, Any] = {}
    hypers: dict[str, Any] = {}

    # V0: reproduce existing (C=1, balanced)
    v0 = _lr(balanced=True, C=1.0)
    v0.fit(X0_tr, y_tr)
    models["V0"] = v0
    preds["V0"] = v0.predict(X0_te)
    probas["V0"] = v0.predict_proba(X0_te)
    hypers["V0"] = {"C": 1.0, "class_weight": "balanced", "tuned": False}

    v0_metrics = three_class_metrics(y_te, preds["V0"], proba=probas["V0"])
    regression = {
        "accuracy": v0_metrics["accuracy"],
        "macro_f1": v0_metrics["macro_f1"],
        "weighted_f1": v0_metrics["weighted_f1"],
        "per_class": v0_metrics["per_class"],
        "passed": (
            abs(v0_metrics["accuracy"] - EXPECTED["accuracy"]) <= TOL
            and abs(v0_metrics["macro_f1"] - EXPECTED["macro_f1"]) <= TOL
            and abs(v0_metrics["weighted_f1"] - EXPECTED["weighted_f1"]) <= TOL
        ),
        "expected": EXPECTED,
        "tolerance": TOL,
    }
    if smoke:
        regression["passed"] = True
        regression["note"] = "smoke: expected metrics not applicable"
    elif not regression["passed"]:
        print("[classifier_eval] V0 regression FAILED — stopping variants as specified.", regression)

    yb_tr, yb_te = _binarise(y_tr), _binarise(y_te)
    Xg_tr = Xg_te = None
    if regression["passed"]:
        X3_tr = cached("X3_train.npy", lambda: chunk_pair_features(embeddings, resumes_tr, jobs_tr))
        X3_te = cached("X3_test.npy", lambda: chunk_pair_features(embeddings, resumes_te, jobs_te))
        v1, c1, s1 = _tune_lr_C(X0_tr, y_tr, groups, balanced=True)
        models["V1"] = v1
        preds["V1"] = v1.predict(X0_te)
        probas["V1"] = v1.predict_proba(X0_te)
        hypers["V1"] = {"C": c1, "class_weight": "balanced", "group_cv_macro_f1": s1}
        cv_scores["V1"] = s1

        v2, c2, s2 = _tune_lr_C(X0_tr, yb_tr, groups, balanced=True)
        models["V2"] = v2
        preds["V2"] = v2.predict(X0_te)
        probas["V2"] = v2.predict_proba(X0_te)
        hypers["V2"] = {"C": c2, "task": "binary_no_fit_vs_fitpos", "group_cv_macro_f1": s2}

        v3, c3, s3 = _tune_lr_C(X3_tr, y_tr, groups, balanced=True)
        models["V3"] = v3
        preds["V3"] = v3.predict(X3_te)
        probas["V3"] = v3.predict_proba(X3_te)
        hypers["V3"] = {"C": c3, "group_cv_macro_f1": s3, "pooling": "mean+max chunks"}
        cv_scores["V3"] = s3

        X5_tr = X5_te = None
        if not skip_gap_lite:
            Xg_tr = cached(
                "Xg_train.npy",
                lambda: gap_lite_matrix(resumes_tr, jobs_tr, settings, embeddings),
            )
            Xg_te = cached(
                "Xg_test.npy",
                lambda: gap_lite_matrix(resumes_te, jobs_te, settings, embeddings),
            )
            v4lr, c4, s4 = _tune_lr_C(Xg_tr, y_tr, groups, balanced=True)
            models["V4_LR"] = v4lr
            preds["V4_LR"] = v4lr.predict(Xg_te)
            probas["V4_LR"] = v4lr.predict_proba(Xg_te)
            hypers["V4_LR"] = {"C": c4, "group_cv_macro_f1": s4}
            cv_scores["V4_LR"] = s4
            hgb = _hgb()
            cv_scores["V4_HGB"] = _group_cv_macro_f1(hgb, Xg_tr, y_tr, groups)
            hgb.fit(Xg_tr, y_tr)
            models["V4_HGB"] = hgb
            preds["V4_HGB"] = hgb.predict(Xg_te)
            probas["V4_HGB"] = hgb.predict_proba(Xg_te)
            hypers["V4_HGB"] = {"max_depth": 3, "learning_rate": 0.1, "max_iter": 80}

            best_emb = "V3" if cv_scores.get("V3", -1) >= cv_scores.get("V1", -1) else "V1"
            Xe_tr = X3_tr if best_emb == "V3" else X0_tr
            Xe_te = X3_te if best_emb == "V3" else X0_te
            X5_tr = np.concatenate([Xe_tr, Xg_tr], axis=1)
            X5_te = np.concatenate([Xe_te, Xg_te], axis=1)
            v5lr, c5, s5 = _tune_lr_C(X5_tr, y_tr, groups, balanced=True)
            models["V5_LR"] = v5lr
            preds["V5_LR"] = v5lr.predict(X5_te)
            probas["V5_LR"] = v5lr.predict_proba(X5_te)
            hypers["V5_LR"] = {"C": c5, "embed": best_emb, "group_cv_macro_f1": s5}
            cv_scores["V5_LR"] = s5
            hgb5 = _hgb()
            cv_scores["V5_HGB"] = _group_cv_macro_f1(hgb5, X5_tr, y_tr, groups)
            hgb5.fit(X5_tr, y_tr)
            models["V5_HGB"] = hgb5
            preds["V5_HGB"] = hgb5.predict(X5_te)
            probas["V5_HGB"] = hgb5.predict_proba(X5_te)

        cand = {k: cv_scores[k] for k in ("V1", "V3", "V5_LR") if k in cv_scores}
        best_name = max(cand, key=cand.get)
        Xo_tr, Xo_te = {
            "V1": (X0_tr, X0_te),
            "V3": (X3_tr, X3_te),
            "V5_LR": (X5_tr, X5_te),
        }[best_name]
        y_ge1 = (y_tr >= 1).astype(int)
        y_ge2 = (y_tr >= 2).astype(int)
        o1, _, _ = _tune_lr_C(Xo_tr, y_ge1, groups, balanced=True)
        o2, _, _ = _tune_lr_C(Xo_tr, y_ge2, groups, balanced=True)
        p1 = o1.predict(Xo_te)
        p2 = o2.predict(Xo_te)
        y6 = np.zeros(len(y_te), dtype=int)
        y6[p1 == 1] = 1
        y6[p2 == 1] = 2
        preds["V6"] = y6
        probas["V6"] = None
        hypers["V6"] = {"feature_set": best_name, "stacked": ["P(y>=Potential)", "P(y>=Good)"]}
        models["V6_ge1"] = o1
        models["V6_ge2"] = o2
        cv_scores["V6"] = cand[best_name]

    # Metrics on original test
    results = {}
    for name, yhat in preds.items():
        if name == "V2":
            results[name] = {
                "binary": binary_metrics(yb_te, yhat, probas[name][:, 1] if probas[name] is not None else None),
                "note": "binary model; 3-class metrics not applicable",
            }
            continue
        results[name] = evaluate_predictions(y_te, yhat, probas.get(name))

    # Collapse V0/V1 to binary
    for src in ("V0", "V1"):
        if src in preds:
            results[f"{src}_as_binary"] = binary_metrics(
                yb_te, _binarise(preds[src]), 1.0 - probas[src][:, 0]
            )

    disjoint_mask = leak["disjoint_mask"]
    disjoint_results = {}
    if disjoint_mask.sum() > 0:
        yd = y_te[disjoint_mask]
        for name, yhat in preds.items():
            if name == "V2":
                continue
            disjoint_results[name] = three_class_metrics(yd, yhat[disjoint_mask])

    baselines = {
        "majority": majority_baseline(y_te),
        "chance_uniform": chance_baseline(y_te, seed=SEED),
        "stratified_random": stratified_random_baseline(y_te, seed=SEED),
    }

    def mf1(a, p):
        return float(f1_score(a, p, average="macro", zero_division=0))

    three_class_vars = [k for k in preds if k != "V2"]
    gains = {}
    mcnemar = {}
    pvals = []
    for name in three_class_vars:
        if name == "V0":
            continue
        gains[name] = paired_bootstrap_gain(
            y_te, preds["V0"], preds[name], mf1, n_resamples=BOOT, seed=SEED
        )
        mc = mcnemar_accuracy(y_te, preds["V0"], preds[name])
        mcnemar[name] = mc
        pvals.append(mc["p_value"])
    holm = holm_adjust(pvals)
    for name, hp in zip([n for n in three_class_vars if n != "V0"], holm, strict=True):
        mcnemar[name]["holm_p"] = hp
        selected = name in cv_scores  # tuned on train CV
        gains[name]["selected_on_validation"] = selected
        gains[name]["improved_over_V0"] = bool(
            selected and gains[name]["improves"]
        )

    # Permutation importance on best 3-class variant (macro-F1 on test — only for
    # reporting gap-lite coeffs; selection already frozen). Use train CV winner.
    importances = None
    if Xg_tr is not None and "V4_LR" in models:
        coef = models["V4_LR"].named_steps["model"].coef_
        importances = {
            "model": "V4_LR",
            "note": "Mean |coefficient| across classes after scaling (gap-lite-only LR).",
            "features": [
                {"name": n, "mean_abs_coef": float(np.mean(np.abs(coef[:, i])))}
                for i, n in enumerate(FEATURE_NAMES)
            ],
        }

    experiment_id = utc_experiment_id()
    out_dir = root / "experiments/results" / experiment_id
    if smoke:
        out_dir = root / "experiments/results" / f"{experiment_id}_smoke"
    combined = root / "experiments/results/classifier_reframing_combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not smoke:
        for name, mdl in models.items():
            joblib.dump(mdl, feat_dir / f"{name}.joblib")

    payload = {
        "experiment_id": experiment_id,
        "stage": "classifier_eval",
        "status": "completed" if regression["passed"] else "regression_failed",
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "embedding_model": getattr(embeddings, "model_name", None),
        "embedding_device": getattr(embeddings, "device", None),
        "seed": SEED,
        "regression_v0": regression,
        "leakage": {k: v for k, v in leak.items() if k not in {"train", "test", "disjoint_mask"}},
        "truncation": {"resumes": trunc_r, "jobs": trunc_j},
        "hyperparameters": hypers,
        "group_cv_macro_f1": cv_scores,
        "results_test": results,
        "results_disjoint_test": disjoint_results,
        "baselines_test": baselines,
        "gains_vs_V0": gains,
        "mcnemar_vs_V0": mcnemar,
        "gap_lite_importances": importances,
        "gap_lite_feature_names": FEATURE_NAMES,
        "deviations": [
            "V0 already uses class_weight=balanced in train_fit_classifier.py; "
            "V1 is the same feature family with C chosen by group CV ({0.25, 1, 4}).",
            "Gap-lite scores each pair with production per-pair TF-IDF/BM25/semantic "
            "(not a train-global vectorizer) via score_requirements on JD chunks vs CV chunks.",
            "Holdout gold-set raw texts are read from experiment_pairs_150 (holdout30 profiles dir is empty).",
            "Smoke runs do not write classifier_reframing_combined/.",
        ],
        "smoke": smoke,
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "summary.json", payload)
    if not smoke:
        combined.mkdir(parents=True, exist_ok=True)
        write_json(combined / "summary.json", payload)
        _write_tables(combined, payload)
        fig_dir = root / "docs/figures"
        _write_figures(fig_dir, payload, y_te, preds, probas, yb_te)
        append_results(root / "docs/classifier_reframing_report.md", payload)
    return payload


def _write_tables(combined: Path, payload: dict[str, Any]) -> None:
    rows = []
    for name, res in (payload.get("results_test") or {}).items():
        if name == "V2" or name.endswith("_as_binary"):
            b = res.get("binary") if name == "V2" else res
            rows.append(
                {
                    "variant": name,
                    "accuracy": b.get("accuracy"),
                    "balanced_accuracy": b.get("balanced_accuracy"),
                    "macro_f1": b.get("macro_f1"),
                    "qwk": None,
                    "binary_auc": b.get("auc"),
                }
            )
            continue
        ci = res.get("ci") or {}
        rows.append(
            {
                "variant": name,
                "accuracy": (ci.get("accuracy") or {}).get("value", res.get("accuracy")),
                "accuracy_ci_low": (ci.get("accuracy") or {}).get("ci_low"),
                "accuracy_ci_high": (ci.get("accuracy") or {}).get("ci_high"),
                "balanced_accuracy": (ci.get("balanced_accuracy") or {}).get(
                    "value", res.get("balanced_accuracy")
                ),
                "balanced_accuracy_ci_low": (ci.get("balanced_accuracy") or {}).get("ci_low"),
                "balanced_accuracy_ci_high": (ci.get("balanced_accuracy") or {}).get("ci_high"),
                "macro_f1": (ci.get("macro_f1") or {}).get("value", res.get("macro_f1")),
                "macro_f1_ci_low": (ci.get("macro_f1") or {}).get("ci_low"),
                "macro_f1_ci_high": (ci.get("macro_f1") or {}).get("ci_high"),
                "qwk": (ci.get("qwk") or {}).get("value", res.get("qwk")),
                "qwk_ci_low": (ci.get("qwk") or {}).get("ci_low"),
                "qwk_ci_high": (ci.get("qwk") or {}).get("ci_high"),
                "binary_auc": (ci.get("binary_auc") or {}).get("value", (res.get("binary") or {}).get("auc")),
            }
        )
    pd.DataFrame(rows).to_csv(combined / "headline_metrics.csv", index=False)
    pd.DataFrame(payload.get("gains_vs_V0") or {}).T.to_csv(combined / "gains_vs_V0.csv")
    pd.DataFrame([payload.get("leakage") or {}]).to_csv(combined / "leakage_audit.csv", index=False)


def _write_figures(
    fig_dir: Path,
    payload: dict[str, Any],
    y_te: np.ndarray,
    preds: dict[str, np.ndarray],
    probas: dict[str, np.ndarray | None],
    yb_te: np.ndarray,
) -> None:
    rows = []
    for name, res in (payload.get("results_test") or {}).items():
        if name == "V2" or str(name).endswith("_as_binary"):
            continue
        ci = res.get("ci") or {}
        rows.append(
            {
                "variant": name,
                "macro_f1": (ci.get("macro_f1") or {}).get("value"),
                "macro_f1_ci_low": (ci.get("macro_f1") or {}).get("ci_low"),
                "macro_f1_ci_high": (ci.get("macro_f1") or {}).get("ci_high"),
                "balanced_accuracy": (ci.get("balanced_accuracy") or {}).get("value"),
                "balanced_accuracy_ci_low": (ci.get("balanced_accuracy") or {}).get("ci_low"),
                "balanced_accuracy_ci_high": (ci.get("balanced_accuracy") or {}).get("ci_high"),
                "accuracy": (ci.get("accuracy") or {}).get("value"),
                "accuracy_ci_low": (ci.get("accuracy") or {}).get("ci_low"),
                "accuracy_ci_high": (ci.get("accuracy") or {}).get("ci_high"),
            }
        )
    maj = (payload.get("baselines_test") or {}).get("majority") or {}
    figure_30_headline(fig_dir / "fig30_classifier_headline.png", rows, maj.get("accuracy"))
    per_class = {}
    for name, res in (payload.get("results_test") or {}).items():
        if name in preds and name != "V2" and "per_class" in res:
            per_class[name] = {k: v.get("f1") for k, v in res["per_class"].items()}
    if per_class:
        figure_31_per_class_f1(fig_dir / "fig31_per_class_f1.png", per_class)
    three = [k for k in preds if k != "V2"]
    best = "V0"
    best_f1 = -1.0
    for name in three:
        f1 = (payload.get("results_test") or {}).get(name, {}).get("macro_f1")
        if f1 is not None and f1 > best_f1:
            best, best_f1 = name, f1
    if "V0" in preds and best in preds:
        from sklearn.metrics import confusion_matrix

        cm0 = confusion_matrix(y_te, preds["V0"], labels=[0, 1, 2])
        cmb = confusion_matrix(y_te, preds[best], labels=[0, 1, 2])
        figure_32_confusion(fig_dir / "fig32_confusion.png", cm0, cmb, "V0", best)
    curves = []
    if "V0" in probas and probas["V0"] is not None:
        curves.append(("V0 collapsed", yb_te, 1.0 - probas["V0"][:, 0]))
    if "V2" in probas and probas["V2"] is not None:
        curves.append(("V2", yb_te, probas["V2"][:, 1]))
    if curves:
        figure_33_binary_roc(fig_dir / "fig33_binary_roc.png", curves)
    imp = (payload.get("gap_lite_importances") or {}).get("features")
    if imp:
        figure_34_importances(fig_dir / "fig34_gap_lite_importance.png", imp)
