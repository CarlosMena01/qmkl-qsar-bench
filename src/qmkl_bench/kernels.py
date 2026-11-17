"""Kernels cuánticos (ZZFeatureMap en Qiskit Aer) y clásicos, con caché en disco."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

try:  # qiskit >= 2.1
    from qiskit.circuit.library import zz_feature_map as _zz
except ImportError:  # pragma: no cover - qiskit 1.x
    from qiskit.circuit.library import ZZFeatureMap

    def _zz(n, reps, entanglement):
        return ZZFeatureMap(n, reps=reps, entanglement=entanglement).decompose()


def zz_statevectors(Z: np.ndarray, reps: int, entanglement: str) -> np.ndarray:
    """|phi(z)> para cada fila de Z, simulado con Aer (método statevector)."""
    Z = np.asarray(Z, dtype=float)
    fmap = _zz(Z.shape[1], reps=reps, entanglement=entanglement)
    circuit = fmap.copy()
    circuit.save_statevector()
    sim = AerSimulator(method="statevector", max_parallel_threads=1)
    compiled = transpile(circuit, sim, optimization_level=0)
    binds = {p: Z[:, i].tolist() for i, p in enumerate(fmap.parameters)}
    result = sim.run(compiled, parameter_binds=[binds]).result()
    return np.array([np.asarray(result.get_statevector(i)) for i in range(len(Z))])


def quantum_kernel(Z: np.ndarray, reps: int, entanglement: str) -> np.ndarray:
    """Kernel de fidelidad K_ij = |<phi(z_i)|phi(z_j)>|^2.

    En simulación exacta basta un circuito por punto (N simulaciones) y el Gram sale de
    productos internos; en hardware serían O(N^2) circuitos de overlap.
    """
    sv = zz_statevectors(Z, reps, entanglement)
    K = np.abs(sv.conj() @ sv.T) ** 2
    K = (K + K.T) / 2.0
    np.fill_diagonal(K, 1.0)
    return K


@dataclass
class KernelCache:
    """Caché de matrices de Gram indexada por el contenido de los datos y el spec del kernel."""

    directory: Path
    hits: int = 0
    misses: int = 0
    compute_seconds: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(Z: np.ndarray, spec: dict) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(spec, sort_keys=True).encode())
        h.update(str(Z.shape).encode())
        h.update(np.ascontiguousarray(Z, dtype=np.float64).tobytes())
        return h.hexdigest()

    def get(self, Z: np.ndarray, reps: int, entanglement: str) -> np.ndarray:
        spec = {"kind": "zz_fidelity", "reps": reps, "entanglement": entanglement}
        path = self.directory / f"{self.key(Z, spec)}.npy"
        if path.exists():
            self.hits += 1
            return np.load(path)
        start = time.perf_counter()
        K = quantum_kernel(Z, reps, entanglement)
        self.compute_seconds += time.perf_counter() - start
        self.misses += 1
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, K)
        os.replace(tmp, path)
        return K


def rbf_gamma(Z_train: np.ndarray, factor: float) -> float:
    """gamma = factor / (d * var), la heurística 'scale' de scikit-learn multiplicada por un factor."""
    return factor / (Z_train.shape[1] * Z_train.var())


def classical_kernels(Z: np.ndarray, train_idx: np.ndarray, gamma_factors: list[float]) -> dict:
    """RBF (varios anchos, fijados con el train) y lineal normalizado por la traza del train."""
    out = {}
    for factor in gamma_factors:
        out[f"rbf_g{factor:g}"] = rbf_kernel(Z, gamma=rbf_gamma(Z[train_idx], factor))
    K_lin = linear_kernel(Z)
    scale = K_lin[train_idx, train_idx].mean()
    out["linear"] = K_lin / scale if scale > 0 else K_lin
    return out


def centered_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Centered kernel alignment entre K y el kernel ideal yy^T (y en {-1, +1})."""
    n = len(y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    s = np.where(y > 0, 1.0, -1.0)
    Yc = H @ np.outer(s, s) @ H
    denom = np.linalg.norm(Kc) * np.linalg.norm(Yc)
    return float((Kc * Yc).sum() / denom) if denom > 0 else 0.0
