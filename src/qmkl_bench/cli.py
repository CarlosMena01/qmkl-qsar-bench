"""CLI: ``bench data``, ``bench train``, ``bench report`` y ``bench all``."""
from __future__ import annotations

import json
import platform
import re
import time
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from . import report as rpt
from .config import DEFAULT_CONFIG, load_config
from .data import load_dataset

app = typer.Typer(add_completion=False, help="Benchmark de kernels cuánticos vs. clásicos.")

ConfigOpt = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Ruta al YAML de configuración.")
DatasetOpt = typer.Option(None, "--dataset", "-d", help="Dataset (por defecto, data.name del YAML).")
MaxOpt = typer.Option(None, "--max-samples", help="Sobrescribe data.max_samples.")
README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"


def _setup(config: Path, dataset: Optional[str], max_samples: Optional[int]):
    cfg = load_config(config, **{"data.name": dataset, "data.max_samples": max_samples})
    return cfg, cfg["data"]["name"], Path(cfg["output"]["dir"]) / cfg["data"]["name"]


@app.command()
def data(config: Path = ConfigOpt, dataset: Optional[str] = DatasetOpt, max_samples: Optional[int] = MaxOpt):
    """Descarga/genera el dataset y muestra un resumen."""
    cfg, name, _ = _setup(config, dataset, max_samples)
    X, y = load_dataset(name, cfg)
    typer.echo(f"{name}: X={X.shape}, positivos={int(y.sum())} ({y.mean():.1%})")


@app.command()
def train(config: Path = ConfigOpt, dataset: Optional[str] = DatasetOpt, max_samples: Optional[int] = MaxOpt):
    """Ejecuta el benchmark completo y registra las corridas en MLflow."""
    from .evaluate import run_benchmark
    from .tracking import log_benchmark

    cfg, name, out = _setup(config, dataset, max_samples)
    X, y = load_dataset(name, cfg)
    typer.echo(f"{name}: X={X.shape}, positivos={int(y.sum())}; entrenando...")
    start = time.perf_counter()
    records, oof, diagnostics, cache_stats = run_benchmark(X, y, cfg)
    wall = time.perf_counter() - start

    out.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(records).sort_values(["n_components", "model", "seed", "fold"])
    diag = pd.DataFrame(diagnostics)
    oof_df = pd.DataFrame(oof, columns=["model", "n_components", "seed", "fold", "idx", "score", "pred"])
    folds.to_csv(out / "fold_metrics.csv", index=False)
    diag.to_csv(out / "kernel_diagnostics.csv", index=False)
    oof_df.to_csv(out / "oof_predictions.csv.gz", index=False)
    np.save(out / "y.npy", y)
    meta = {
        "dataset": name, "n_samples": int(len(y)), "n_features": int(X.shape[1]),
        "n_positive": int(y.sum()), "wall_seconds": wall, "kernel_cache": cache_stats,
        "versions": {p: version(p) for p in ("qiskit", "qiskit-aer", "scikit-learn", "numpy", "mlflow")},
        "python": platform.python_version(), "config": cfg,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    run_id = log_benchmark(cfg, name, X.shape, folds, diag, cache_stats, wall,
                           [str(out / "fold_metrics.csv"), str(out / "kernel_diagnostics.csv")])
    typer.echo(f"Hecho en {wall:.0f} s (kernels: {cache_stats['misses']} calculados, "
               f"{cache_stats['hits']} desde caché). MLflow run {run_id}. Resultados en {out}/")


@app.command()
def report(
    config: Path = ConfigOpt, dataset: Optional[str] = DatasetOpt,
    readme: Optional[Path] = typer.Option(None, "--readme", help="Inserta la tabla entre los marcadores del README."),
):
    """Genera el reporte (tablas, IC bootstrap, significancia) a partir de la última corrida."""
    cfg, name, out = _setup(config, dataset, None)
    if not (out / "fold_metrics.csv").exists():
        raise typer.BadParameter(f"no hay resultados en {out}/; ejecuta `bench train` primero")
    meta = json.loads((out / "run_meta.json").read_text())
    cfg = meta["config"]  # el reporte usa la configuración con la que se entrenó
    folds = pd.read_csv(out / "fold_metrics.csv")
    diag = pd.read_csv(out / "kernel_diagnostics.csv")
    oof = pd.read_csv(out / "oof_predictions.csv.gz")
    y = np.load(out / "y.npy")
    summary, dists = rpt.summarize(folds, oof, y, cfg)
    sig = rpt.significance(folds, summary, dists, cfg)
    summary.to_csv(out / "summary.csv", index=False)
    sig.to_csv(out / "significance.csv", index=False)
    text = rpt.build_report(name, meta, summary, sig, folds, diag, cfg)
    (out / "report.md").write_text(text, encoding="utf-8")
    typer.echo(text)
    if readme:
        body = re.sub(r"^# .*\n", "", text, count=1).replace("\n## ", "\n### ")
        block = f"{README_START}\n{body.strip()}\n{README_END}"
        content = readme.read_text(encoding="utf-8")
        pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
        if not pattern.search(content):
            raise typer.BadParameter(f"{readme} no contiene los marcadores {README_START} / {README_END}")
        readme.write_text(pattern.sub(lambda _: block, content), encoding="utf-8")


@app.command(name="all")
def run_all(config: Path = ConfigOpt, dataset: Optional[str] = DatasetOpt, max_samples: Optional[int] = MaxOpt):
    """data + train + report en un solo comando."""
    data(config, dataset, max_samples)
    train(config, dataset, max_samples)
    report(config, dataset, None)


if __name__ == "__main__":
    app()
