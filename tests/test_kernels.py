import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qmkl_bench import kernels
from qmkl_bench.evaluate import _combine, qmkl_weight_candidates
from qmkl_bench.kernels import KernelCache, classical_kernels, quantum_kernel

CONFIGS = [(n, r, e) for n in (4, 6, 8) for r in (1, 2) for e in ("linear", "full")]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    return rng.random((40, 8)), rng.integers(0, 2, 40)


def assert_valid_gram(K, tol=1e-8):
    assert K.shape[0] == K.shape[1]
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -tol


@pytest.mark.parametrize("n,reps,ent", CONFIGS)
def test_quantum_kernel_symmetric_psd(data, n, reps, ent):
    K = quantum_kernel(data[0][:, :n], reps, ent)
    assert_valid_gram(K)
    np.testing.assert_allclose(np.diag(K), 1.0)
    assert K.min() >= 0.0 and K.max() <= 1.0 + 1e-12


@pytest.mark.parametrize("reps,ent", [(1, "linear"), (2, "full")])
def test_quantum_kernel_matches_reference_fidelity(data, reps, ent):
    """El Gram calculado con Aer coincide con la fidelidad de estados calculada aparte."""
    Z = data[0][:6, :4]
    fmap = kernels._zz(4, reps=reps, entanglement=ent)
    states = [Statevector(fmap.assign_parameters(z)) for z in Z]
    ref = np.array([[abs(a.inner(b)) ** 2 for b in states] for a in states])
    np.testing.assert_allclose(quantum_kernel(Z, reps, ent), ref, atol=1e-10)


def test_entanglement_and_reps_change_kernel(data):
    Z = data[0][:10, :4]
    assert not np.allclose(quantum_kernel(Z, 1, "linear"), quantum_kernel(Z, 1, "full"))
    assert not np.allclose(quantum_kernel(Z, 1, "linear"), quantum_kernel(Z, 2, "linear"))


def test_classical_kernels_symmetric_psd(data):
    for K in classical_kernels(data[0], np.arange(30), [0.1, 1.0, 10.0]).values():
        assert_valid_gram(K)


def test_qmkl_combinations_symmetric_psd(data):
    Z, y = data
    bank = {**classical_kernels(Z, np.arange(40), [1.0]), "zz": quantum_kernel(Z[:, :4], 1, "linear")}
    cands, _ = qmkl_weight_candidates(bank, y, n_random=10, seed=0)
    assert len(cands) >= 10 + len(bank) + 1
    for w in cands:
        assert w.min() >= 0 and np.isclose(w.sum(), 1.0)
        assert_valid_gram(_combine(bank, w))


def test_cache_roundtrip(data, tmp_path, monkeypatch):
    Z = data[0][:12, :4]
    cache = KernelCache(tmp_path)
    K1 = cache.get(Z, 1, "linear")
    assert (cache.hits, cache.misses) == (0, 1)
    monkeypatch.setattr(kernels, "quantum_kernel", lambda *a: pytest.fail("debía salir de la caché"))
    np.testing.assert_array_equal(cache.get(Z, 1, "linear"), K1)
    assert cache.hits == 1
    # Otros datos u otro circuito son entradas distintas.
    assert KernelCache.key(Z, {"reps": 1}) != KernelCache.key(Z + 1e-9, {"reps": 1})
    assert KernelCache.key(Z, {"reps": 1}) != KernelCache.key(Z, {"reps": 2})
