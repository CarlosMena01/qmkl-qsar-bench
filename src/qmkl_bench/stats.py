"""Métricas vectorizadas, bootstrap estratificado y tests de significancia."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

METRICS = ("auc_roc", "auc_pr", "recall")


def auc_roc(y: np.ndarray, score: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AUC-ROC ponderada (Mann-Whitney, empates = 1/2) para cada fila de pesos w (B, n)."""
    pos, neg = y == 1, y == 0
    wins = (score[pos, None] > score[None, neg]) + 0.5 * (score[pos, None] == score[None, neg])
    wp, wn = w[:, pos], w[:, neg]
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((wp @ wins) * wn).sum(axis=1) / (wp.sum(axis=1) * wn.sum(axis=1))


def auc_pr(y: np.ndarray, score: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Average precision ponderada, con el tratamiento de empates de scikit-learn."""
    order = np.argsort(-score, kind="stable")
    s, ys, ws = score[order], y[order], w[:, order]
    tps = np.cumsum(ws * ys, axis=1)
    total = np.cumsum(ws, axis=1)
    block_end = np.append(s[:-1] != s[1:], True)
    tps_e, total_e = tps[:, block_end], total[:, block_end]
    gained = np.diff(tps_e, axis=1, prepend=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(total_e > 0, tps_e / total_e, 0.0)
        return (gained * precision).sum(axis=1) / tps[:, -1]


def recall(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Recall ponderado de la clase positiva."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w * (y * pred)).sum(axis=1) / (w * y).sum(axis=1)


def compute_metric(name, y, score, pred, w=None) -> np.ndarray:
    w = np.ones((1, len(y))) if w is None else w
    if name == "auc_roc":
        return auc_roc(y, score, w)
    if name == "auc_pr":
        return auc_pr(y, score, w)
    if name == "recall":
        return recall(y, pred, w)
    raise KeyError(name)


def stratified_bootstrap_weights(y: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Multiplicidades (B, N) de un bootstrap con reemplazo dentro de cada clase."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n_resamples, len(y)))
    for c in (0, 1):
        members = np.flatnonzero(y == c)
        w[:, members] = rng.multinomial(len(members), np.full(len(members), 1 / len(members)), n_resamples)
    return w


def bootstrap_distribution(name, y, scores, preds, folds, w) -> np.ndarray:
    """Distribución bootstrap del mismo estadístico que se reporta como estimación puntual.

    scores, preds, folds: (n_seeds, N) con la predicción out-of-fold y el fold de test de
    cada muestra. La métrica se calcula dentro de cada fold (los scores de folds distintos
    vienen de modelos distintos y no son comparables), se promedia entre folds y después
    entre semillas.
    """
    per_seed = []
    for s, p, f in zip(scores, preds, folds):
        per_fold = [
            compute_metric(name, y[f == k], s[f == k], p[f == k], w[:, f == k]) for k in np.unique(f)
        ]
        per_seed.append(np.nanmean(per_fold, axis=0))
    return np.mean(per_seed, axis=0)


def percentile_ci(dist: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(dist, [alpha, 1 - alpha])
    return float(lo), float(hi)


def bootstrap_pvalue(diff: np.ndarray) -> float:
    """p bilateral de H0: diferencia = 0 a partir de la distribución bootstrap pareada."""
    b = len(diff)
    tail = min((diff <= 0).sum(), (diff >= 0).sum())
    return float(min(1.0, 2 * (tail + 1) / (b + 1)))


def corrected_resampled_ttest(diffs: np.ndarray, test_train_ratio: float) -> tuple[float, float]:
    """t-test remuestreado corregido (Nadeau & Bengio, 2003) para CV repetida.

    Las diferencias por fold no son independientes porque los conjuntos de train se
    solapan; la corrección infla la varianza con n_test / n_train.
    """
    diffs = np.asarray(diffs, dtype=float)
    j = len(diffs)
    var = diffs.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = diffs.mean() / np.sqrt((1 / j + test_train_ratio) * var)
    return float(t), float(2 * sps.t.sf(abs(t), df=j - 1))


def holm(pvalues: list[float]) -> list[float]:
    """Ajuste de Holm-Bonferroni."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
