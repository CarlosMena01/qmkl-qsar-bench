"""Registro de corridas en MLflow: una corrida padre y una anidada por (modelo, n_components, semilla)."""
from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402

from .stats import METRICS  # noqa: E402


def log_benchmark(
    cfg: dict, dataset: str, shape: tuple, folds: pd.DataFrame, diag: pd.DataFrame,
    cache_stats: dict, wall_seconds: float, artifacts: list[str],
) -> str:
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"bench-{dataset}") as parent:
        mlflow.log_params(
            {
                "dataset": dataset,
                "n_samples": shape[0],
                "n_features": shape[1],
                "master_seed": cfg["seed"],
                "n_seeds": cfg["cv"]["n_seeds"],
                "n_splits": cfg["cv"]["n_splits"],
                "inner_splits": cfg["cv"]["inner_splits"],
                "class_weight": cfg.get("class_weight"),
                "n_components": cfg["preprocess"]["n_components"],
            }
        )
        mlflow.log_dict(cfg, "config.yaml")
        mlflow.log_metrics(
            {
                "wall_seconds": wall_seconds,
                "kernel_cache_hits": cache_stats["hits"],
                "kernel_cache_misses": cache_stats["misses"],
                "kernel_compute_seconds": cache_stats["kernel_seconds"],
            }
        )
        for path in artifacts:
            mlflow.log_artifact(path)
        kernel_time = (
            diag.groupby(["n_components", "seed"])["kernel_seconds"].sum() if len(diag) else {}
        )
        for (model, nc, seed), g in folds.groupby(["model", "n_components", "seed"]):
            with mlflow.start_run(run_name=f"{model}-d{nc}-s{seed}", nested=True):
                mlflow.log_params(
                    {"model": model, "family": g["family"].iloc[0], "n_components": nc,
                     "seed": seed, "dataset": dataset}
                )
                for _, row in g.iterrows():
                    mlflow.log_param(f"selected_fold{row['fold']}", row["selected"][:500])
                    mlflow.log_metrics(
                        {f"fold_{m}": row[m] for m in METRICS} | {"fold_fit_seconds": row["fit_seconds"]},
                        step=int(row["fold"]),
                    )
                metrics = {m: g[m].mean() for m in METRICS}
                metrics |= {f"{m}_fold_std": g[m].std(ddof=1) for m in METRICS}
                metrics["fit_seconds"] = g["fit_seconds"].sum()
                if g["family"].iloc[0] in ("quantum", "hybrid"):
                    metrics["kernel_seconds"] = float(kernel_time.get((nc, seed), 0.0))
                mlflow.log_metrics(metrics)
        return parent.info.run_id
