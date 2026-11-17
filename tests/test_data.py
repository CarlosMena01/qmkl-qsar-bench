import numpy as np
import pytest

from qmkl_bench.data import load_dataset


def test_synthetic_shape_and_determinism(cfg):
    X1, y1 = load_dataset("synthetic", cfg)
    X2, y2 = load_dataset("synthetic", cfg)
    assert X1.shape == (500, 20) and set(np.unique(y1)) == {0, 1}
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_max_samples_is_stratified(cfg):
    _, y_full = load_dataset("synthetic", cfg)
    cfg["data"]["max_samples"] = 100
    X, y = load_dataset("synthetic", cfg)
    assert len(y) == 100 and X.shape[0] == 100
    assert abs(y.mean() - y_full.mean()) < 0.02


def test_class_sep_controls_difficulty(cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def acc(sep):
        cfg["data"]["datasets"]["synthetic"]["class_sep"] = sep
        X, y = load_dataset("synthetic", cfg)
        return cross_val_score(LogisticRegression(max_iter=500), X, y, cv=3).mean()

    assert acc(3.0) > acc(0.2) + 0.1


def test_unknown_dataset(cfg):
    with pytest.raises(KeyError):
        load_dataset("nope", cfg)
