"""Test de integración: benchmark diminuto de extremo a extremo."""
import numpy as np
import pandas as pd

from qmkl_bench import report
from qmkl_bench.data import load_dataset
from qmkl_bench.evaluate import run_task


def _tiny(cfg):
    cfg["data"]["max_samples"] = 60
    cfg["cv"].update(n_seeds=2, n_splits=3, inner_splits=2)
    cfg["preprocess"]["n_components"] = [4]
    cfg["models"]["gradient_boosting"].update(params={"n_estimators": 10}, grid={})
    cfg["models"]["random_forest"].update(params={"n_estimators": 10}, grid={})
    cfg["models"]["svm"]["C_grid"] = [1.0]
    cfg["models"]["qsvm"].update(reps=[1], entanglement=["linear"])
    cfg["models"]["qmkl"]["n_random_weights"] = 3
    cfg["bootstrap"]["n_resamples"] = 100
    return cfg


def test_end_to_end_reproducible_and_cached(cfg):
    cfg = _tiny(cfg)
    X, y = load_dataset("synthetic", cfg)
    rec1, oof1, diag1, st1 = run_task(X, y, 4, cfg["seed"], cfg)
    rec2, oof2, _, st2 = run_task(X, y, 4, cfg["seed"], cfg)
    assert st1["misses"] == 3 and st2["misses"] == 0 and st2["hits"] == 3
    a, b = pd.DataFrame(rec1), pd.DataFrame(rec2)
    pd.testing.assert_frame_equal(a.drop(columns="fit_seconds"), b.drop(columns="fit_seconds"))
    assert oof1 == oof2

    models = set(a["model"])
    assert {"gb", "rf", "svm_rbf", "svm_linear", "qsvm_r1_linear", "qmkl_cv", "qmkl_cka"} == models
    # Mismos splits para todos los modelos: cada uno predice cada muestra exactamente una vez.
    o = pd.DataFrame(oof1, columns=["model", "n_components", "seed", "fold", "idx", "score", "pred"])
    for _, g in o.groupby("model"):
        assert sorted(g["idx"]) == list(range(len(y)))
    assert o.groupby("idx")["fold"].nunique().eq(1).all()

    rec3, oof3, diag3, _ = run_task(X, y, 4, cfg["seed"] + 1, cfg)
    folds = pd.DataFrame(rec1 + rec3)
    o = pd.DataFrame(oof1 + oof3, columns=o.columns)
    summary, dists = report.summarize(folds, o, y, cfg)
    for m in ("auc_roc", "auc_pr", "recall"):
        assert summary[f"{m}_mean"].between(0, 1).all() and summary[f"{m}_std"].notna().all()
        assert (summary[f"{m}_ci_lo"] <= summary[f"{m}_ci_hi"]).all()
    sig = report.significance(folds, summary, dists, cfg)
    assert len(sig) == 6 and sig["p_ttest"].between(0, 1).all()
    meta = {"n_samples": len(y), "n_positive": int(y.sum()), "n_features": X.shape[1], "wall_seconds": 1.0}
    text = report.build_report("synthetic", meta, summary, sig, folds, pd.DataFrame(diag1 + diag3), cfg)
    assert "Recall" in text and "qmkl_cv" in text
