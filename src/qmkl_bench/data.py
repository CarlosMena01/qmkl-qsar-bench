"""Carga de datos: ``load_dataset(name)`` -> (X, y), independiente del origen."""
from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

Backend = Callable[[dict, dict], tuple[np.ndarray, np.ndarray]]
BACKENDS: dict[str, Backend] = {}


def register(name: str) -> Callable[[Backend], Backend]:
    def deco(fn: Backend) -> Backend:
        BACKENDS[name] = fn
        return fn

    return deco


@register("synthetic")
def _synthetic(spec: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    params = {k: v for k, v in spec.items() if k != "backend"}
    X, y = make_classification(random_state=cfg["seed"], **params)
    return X, y


def _download(url: str, dest: Path, sha256: str | None) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "qmkl-bench"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.rename(dest)
    if sha256:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"checksum inesperado para {dest}: {digest} (esperado {sha256})")
    return dest


@register("arff_url")
def _arff_url(spec: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    name = spec.get("filename") or hashlib.sha256(spec["url"].encode()).hexdigest()[:16] + ".arff"
    path = _download(spec["url"], Path(cfg["data"]["dir"]) / name, spec.get("sha256"))
    records, _ = arff.loadarff(io.StringIO(path.read_text(encoding="utf-8")))
    frame = pd.DataFrame(records)
    target = frame.pop(spec["target"])
    labels = target.map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
    y = (labels == str(spec["positive_label"])).to_numpy().astype(int)
    X = frame.to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("el dataset contiene valores faltantes; no se imputan implícitamente")
    return X, y


def load_dataset(name: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (X, y) con y en {0, 1}, submuestreado de forma estratificada a ``max_samples``."""
    try:
        spec = cfg["data"]["datasets"][name]
    except KeyError as exc:
        raise KeyError(f"dataset '{name}' no definido en la configuración") from exc
    X, y = BACKENDS[spec["backend"]](spec, cfg)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("se espera un problema binario con ambas clases presentes")
    max_samples = cfg["data"].get("max_samples")
    if max_samples and len(y) > max_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, stratify=y, random_state=cfg["seed"]
        )
    return X, y
