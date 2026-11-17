"""Bucle de evaluación: todos los modelos sobre exactamente los mismos splits."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from .config import seeds
from .kernels import KernelCache, centered_alignment, classical_kernels
from .preprocess import build_preprocessor

FAMILY = {"gb": "classical", "rf": "classical", "svm_rbf": "classical", "svm_linear": "classical"}


def family_of(model: str) -> str:
    if model.endswith("_raw"):
        return "reference"
    if model.startswith("qsvm"):
        return "quantum"
    if model.startswith("qmkl"):
        return "hybrid"
    return FAMILY[model]


def _inner_cv(cfg: dict, seed: int) -> StratifiedKFold:
    return StratifiedKFold(cfg["cv"]["inner_splits"], shuffle=True, random_state=seed)


def _svc(C: float, cfg: dict) -> SVC:
    return SVC(kernel="precomputed", C=C, class_weight=cfg.get("class_weight"))


def select_svm(candidates: dict[str, np.ndarray], y: np.ndarray, cfg: dict, seed: int):
    """Elige (kernel, C) por AUC-ROC media en CV interna sobre matrices de Gram del train."""
    folds = list(_inner_cv(cfg, seed).split(np.zeros(len(y)), y))
    best = (-np.inf, None, None)
    for name, K in candidates.items():
        for C in cfg["models"]["svm"]["C_grid"]:
            aucs = []
            for tr, va in folds:
                clf = _svc(C, cfg).fit(K[np.ix_(tr, tr)], y[tr])
                aucs.append(roc_auc_score(y[va], clf.decision_function(K[np.ix_(va, tr)])))
            score = float(np.mean(aucs))
            if score > best[0] + 1e-12:
                best = (score, name, C)
    return best[1], best[2], best[0]


def qmkl_weight_candidates(bank: dict[str, np.ndarray], y: np.ndarray, n_random: int, seed: int):
    """Candidatos en el símplex: vértices, uniforme, proporcional al alineamiento y Dirichlet."""
    m = len(bank)
    align = np.clip([centered_alignment(K, y) for K in bank.values()], 0.0, None)
    cands = [np.eye(m)[i] for i in range(m)] + [np.full(m, 1 / m)]
    if align.sum() > 0:
        cands.append(align / align.sum())
    rng = np.random.default_rng(seed)
    cands += list(rng.dirichlet(np.ones(m), size=n_random))
    return cands, align


def _combine(bank: dict[str, np.ndarray], w: np.ndarray) -> np.ndarray:
    return sum(wi * K for wi, K in zip(w, bank.values()))


def _fit_tree(kind: str, X_tr, y_tr, X_te, cfg: dict, seed: int):
    spec = cfg["models"]["gradient_boosting" if kind == "gb" else "random_forest"]
    cls = GradientBoostingClassifier if kind == "gb" else RandomForestClassifier
    weights = (
        compute_sample_weight(cfg["class_weight"], y_tr) if cfg.get("class_weight") else None
    )
    search = GridSearchCV(
        cls(random_state=seed, **spec.get("params", {})),
        spec.get("grid") or {},
        scoring="roc_auc",
        cv=_inner_cv(cfg, seed),
    ).fit(X_tr, y_tr, sample_weight=weights)
    return search.predict_proba(X_te)[:, 1], search.predict(X_te), search.best_params_


def _record(model, n_components, seed, fold, y_te, score, pred, seconds, selected):
    return {
        "model": model,
        "family": family_of(model),
        "n_components": n_components,
        "seed": seed,
        "fold": fold,
        "auc_roc": roc_auc_score(y_te, score),
        "auc_pr": average_precision_score(y_te, score),
        "recall": recall_score(y_te, pred),
        "fit_seconds": seconds,
        "selected": json.dumps(selected, sort_keys=True),
    }


def run_task(X: np.ndarray, y: np.ndarray, n_components: int | None, seed: int, cfg: dict):
    """Evalúa todos los modelos para una (n_components, semilla). n_components=None -> vista cruda."""
    outer = StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    cache = KernelCache(Path(cfg["cache"]["dir"]))
    mcfg = cfg["models"]
    records, oof, diagnostics = [], [], []

    def emit(model, fold, te, score, pred, seconds, selected):
        nc = n_components if n_components is not None else X.shape[1]
        records.append(_record(model, nc, seed, fold, y[te], score, pred, seconds, selected))
        for i, s, p in zip(te, score, pred):
            oof.append((model, nc, seed, fold, int(i), float(s), int(p)))

    for fold, (tr, te) in enumerate(outer.split(X, y)):
        y_tr = y[tr]
        if n_components is None:
            for kind in ("gb", "rf"):
                t0 = time.perf_counter()
                score, pred, sel = _fit_tree(kind, X[tr], y_tr, X[te], cfg, seed)
                emit(f"{kind}_raw", fold, te, score, pred, time.perf_counter() - t0, sel)
            continue

        prep = build_preprocessor(n_components, tuple(cfg["preprocess"]["feature_range"]), seed)
        prep.fit(X[tr])
        Z = prep.transform(X)

        for kind in ("gb", "rf"):
            t0 = time.perf_counter()
            score, pred, sel = _fit_tree(kind, Z[tr], y_tr, Z[te], cfg, seed)
            emit(kind, fold, te, score, pred, time.perf_counter() - t0, sel)

        classical = classical_kernels(Z, tr, mcfg["svm"]["rbf_gamma_factors"])
        quantum = {}
        for reps in mcfg["qsvm"]["reps"]:
            for ent in mcfg["qsvm"]["entanglement"]:
                before, t0 = cache.misses, time.perf_counter()
                K = cache.get(Z, reps, ent)
                name = f"zz_r{reps}_{ent}"
                quantum[name] = K
                off = K[np.ix_(tr, tr)][~np.eye(len(tr), dtype=bool)]
                diagnostics.append(
                    {
                        "kernel": name,
                        "n_components": n_components,
                        "seed": seed,
                        "fold": fold,
                        "offdiag_mean": float(off.mean()),
                        "offdiag_std": float(off.std()),
                        "alignment": centered_alignment(K[np.ix_(tr, tr)], y_tr),
                        "kernel_seconds": time.perf_counter() - t0,
                        "cache_hit": cache.misses == before,
                    }
                )
        bank = {**quantum, **classical}
        train = {k: K[np.ix_(tr, tr)] for k, K in bank.items()}

        def fit_eval(model, candidates_train, full_of):
            t0 = time.perf_counter()
            name, C, inner = select_svm(candidates_train, y_tr, cfg, seed)
            K = full_of(name)
            clf = _svc(C, cfg).fit(K[np.ix_(tr, tr)], y_tr)
            K_te = K[np.ix_(te, tr)]
            sel = {"kernel": name, "C": C, "inner_auc": round(inner, 4)}
            emit(model, fold, te, clf.decision_function(K_te), clf.predict(K_te),
                 time.perf_counter() - t0, sel)
            return sel

        rbf = {k: v for k, v in train.items() if k.startswith("rbf")}
        fit_eval("svm_rbf", rbf, bank.__getitem__)
        fit_eval("svm_linear", {"linear": train["linear"]}, bank.__getitem__)
        for name in quantum:
            fit_eval(f"qsvm_{name[3:]}", {name: train[name]}, bank.__getitem__)

        # QMKL-CKA: selección del kernel mejor alineado, como en arXiv:2506.14920.
        t0 = time.perf_counter()
        cands, align = qmkl_weight_candidates(train, y_tr, mcfg["qmkl"]["n_random_weights"], seed)
        best = list(train)[int(np.argmax(align))]
        fit_eval("qmkl_cka", {best: train[best]}, bank.__getitem__)

        # QMKL-CV: pesos en el símplex elegidos por CV interna.
        combos = {f"w{i}": _combine(train, w) for i, w in enumerate(cands)}
        weights_of = {f"w{i}": w for i, w in enumerate(cands)}
        sel = fit_eval("qmkl_cv", combos, lambda n: _combine(bank, weights_of[n]))
        w = weights_of[sel["kernel"]]
        records[-1]["selected"] = json.dumps(
            {"C": sel["C"], "inner_auc": sel["inner_auc"],
             "weights": {k: round(float(v), 4) for k, v in zip(bank, w)}},
            sort_keys=True,
        )
        records[-1]["fit_seconds"] = time.perf_counter() - t0

    stats = {"hits": cache.hits, "misses": cache.misses, "kernel_seconds": cache.compute_seconds}
    return records, oof, diagnostics, stats


def run_benchmark(X: np.ndarray, y: np.ndarray, cfg: dict):
    """Lanza todas las combinaciones (n_components x semilla) en paralelo."""
    tasks = [(nc, s) for nc in cfg["preprocess"]["n_components"] for s in seeds(cfg)]
    if cfg["models"].get("raw_feature_reference"):
        tasks += [(None, s) for s in seeds(cfg)]
    out = Parallel(n_jobs=cfg["runtime"]["n_jobs"])(
        delayed(run_task)(X, y, nc, s, cfg) for nc, s in tasks
    )
    records = [r for o in out for r in o[0]]
    oof = [r for o in out for r in o[1]]
    diagnostics = [r for o in out for r in o[2]]
    cache_stats = {k: sum(o[3][k] for o in out) for k in ("hits", "misses", "kernel_seconds")}
    return records, oof, diagnostics, cache_stats
