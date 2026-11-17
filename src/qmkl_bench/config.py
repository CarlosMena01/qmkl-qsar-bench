"""Carga de la configuración YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG, **overrides: Any) -> dict:
    """Lee el YAML y aplica overrides con claves punteadas (``data.max_samples=200``)."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return cfg


def seeds(cfg: dict) -> list[int]:
    return [cfg["seed"] + i for i in range(cfg["cv"]["n_seeds"])]
