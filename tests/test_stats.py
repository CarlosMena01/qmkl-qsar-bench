import numpy as np
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

from qmkl_bench import stats


def _data(seed=0, n=80):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    score = np.round(y * 0.8 + rng.normal(size=n), 1)  # redondeo => empates
    return y, score, (score > 0.4).astype(int)


def test_metrics_match_sklearn_unweighted():
    y, score, pred = _data()
    assert np.isclose(stats.compute_metric("auc_roc", y, score, pred)[0], roc_auc_score(y, score))
    assert np.isclose(stats.compute_metric("auc_pr", y, score, pred)[0], average_precision_score(y, score))
    assert np.isclose(stats.compute_metric("recall", y, score, pred)[0], recall_score(y, pred))


def test_metrics_match_sklearn_under_bootstrap_weights():
    y, score, pred = _data(1)
    w = stats.stratified_bootstrap_weights(y, 25, seed=3)
    assert (w.sum(axis=1) == len(y)).all() and (w[:, y == 1].sum(axis=1) == y.sum()).all()
    for i, wi in enumerate(w):
        keep = wi > 0
        kw = dict(sample_weight=wi[keep])
        assert np.isclose(stats.auc_roc(y, score, w)[i], roc_auc_score(y[keep], score[keep], **kw))
        assert np.isclose(stats.auc_pr(y, score, w)[i], average_precision_score(y[keep], score[keep], **kw))
        assert np.isclose(stats.recall(y, pred, w)[i], recall_score(y[keep], pred[keep], **kw))


def test_bootstrap_ci_contains_point_estimate():
    y, score, pred = _data(2, n=200)
    folds = np.arange(200) % 5
    w = stats.stratified_bootstrap_weights(y, 500, seed=0)
    dist = stats.bootstrap_distribution("auc_roc", y, score[None], pred[None], folds[None], w)
    point = np.mean([roc_auc_score(y[folds == k], score[folds == k]) for k in range(5)])
    lo, hi = stats.percentile_ci(dist, 0.95)
    assert lo < point < hi


def test_corrected_ttest_is_more_conservative_than_naive():
    from scipy import stats as sps

    d = np.random.default_rng(0).normal(0.02, 0.03, 25)
    _, p = stats.corrected_resampled_ttest(d, 0.25)
    assert p > sps.ttest_1samp(d, 0).pvalue
    assert stats.corrected_resampled_ttest(np.zeros(25), 0.25) == (0.0, 1.0)


def test_holm():
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
