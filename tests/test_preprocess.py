import numpy as np
import pytest

from qmkl_bench.preprocess import build_preprocessor


@pytest.fixture()
def X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(120, 15)) * rng.uniform(0.1, 50, size=15)


@pytest.mark.parametrize("n_components", [4, 6, 8])
def test_output_dims_and_range(X, n_components):
    Z = build_preprocessor(n_components).fit_transform(X)
    assert Z.shape == (120, n_components)
    np.testing.assert_allclose(Z.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.max(axis=0), 1.0, atol=1e-12)


def test_unseen_data_is_clipped_to_range(X):
    prep = build_preprocessor(4).fit(X[:80])
    Z = prep.transform(X[80:] * 10)
    assert Z.min() >= 0.0 and Z.max() <= 1.0


def test_fit_uses_train_only(X):
    """Cambiar las filas de test no altera la transformación del train (sin fuga)."""
    a = build_preprocessor(4).fit(X[:80]).transform(X[:80])
    X2 = X.copy()
    X2[80:] += 100
    b = build_preprocessor(4).fit(X2[:80]).transform(X2[:80])
    np.testing.assert_array_equal(a, b)


def test_custom_feature_range(X):
    Z = build_preprocessor(4, (0.0, np.pi)).fit_transform(X)
    np.testing.assert_allclose(Z.max(axis=0), np.pi)


def test_deterministic(X):
    np.testing.assert_array_equal(
        build_preprocessor(6, seed=1).fit_transform(X), build_preprocessor(6, seed=1).fit_transform(X)
    )
