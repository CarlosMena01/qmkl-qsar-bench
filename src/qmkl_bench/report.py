"""Agregación de resultados: media ± desviación, IC bootstrap y tests de significancia."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats

LABELS = {"auc_roc": "AUC-ROC", "auc_pr": "AUC-PR", "recall": "Recall"}
QUANTUM_SIDE = ("quantum", "hybrid")


def _oof_arrays(oof: pd.DataFrame, n: int) -> dict:
    """(model, n_components) -> (scores, preds) con forma (n_seeds, N)."""
    out = {}
    for key, g in oof.groupby(["model", "n_components"]):
        seeds = sorted(g["seed"].unique())
        scores = np.empty((len(seeds), n))
        preds = np.empty((len(seeds), n), dtype=int)
        fold = np.empty((len(seeds), n), dtype=int)
        for i, s in enumerate(seeds):
            gs = g[g["seed"] == s]
            scores[i, gs["idx"].to_numpy()] = gs["score"].to_numpy()
            preds[i, gs["idx"].to_numpy()] = gs["pred"].to_numpy()
            fold[i, gs["idx"].to_numpy()] = gs["fold"].to_numpy()
        out[key] = (scores, preds, fold)
    return out


def summarize(folds: pd.DataFrame, oof: pd.DataFrame, y: np.ndarray, cfg: dict):
    """Tabla resumen y distribuciones bootstrap por (modelo, n_components, métrica)."""
    boot_cfg = cfg["bootstrap"]
    w = stats.stratified_bootstrap_weights(y, boot_cfg["n_resamples"], cfg["seed"])
    arrays = _oof_arrays(oof, len(y))
    per_seed = folds.groupby(["model", "family", "n_components", "seed"], as_index=False).agg(
        {**{m: "mean" for m in stats.METRICS}, "fit_seconds": "sum"}
    )
    rows, dists = [], {}
    for (model, family, nc), g in per_seed.groupby(["model", "family", "n_components"]):
        row = {"model": model, "family": family, "n_components": nc, "n_seeds": len(g),
               "fit_seconds_mean": g["fit_seconds"].mean(), "fit_seconds_std": g["fit_seconds"].std(ddof=1)}
        scores, preds, fold = arrays[(model, nc)]
        for m in stats.METRICS:
            dist = stats.bootstrap_distribution(m, y, scores, preds, fold, w)
            dists[(model, nc, m)] = dist
            lo, hi = stats.percentile_ci(dist, boot_cfg["confidence"])
            row |= {f"{m}_mean": g[m].mean(), f"{m}_std": g[m].std(ddof=1),
                    f"{m}_ci_lo": lo, f"{m}_ci_hi": hi}
        rows.append(row)
    return pd.DataFrame(rows), dists


def significance(folds: pd.DataFrame, summary: pd.DataFrame, dists: dict, cfg: dict) -> pd.DataFrame:
    """Mejor QSVM y mejor QMKL vs. mejor clásico (elegidos por AUC-ROC media) para cada n_components."""
    ratio = 1.0 / (cfg["cv"]["n_splits"] - 1)
    rows = []
    for nc in cfg["preprocess"]["n_components"]:
        sub = summary[summary["n_components"] == nc]
        c = sub[sub["family"] == "classical"].sort_values("auc_roc_mean").iloc[-1]["model"]
        f = folds[folds["n_components"] == nc].set_index(["seed", "fold"])
        fc = f[f["model"] == c]
        for side in QUANTUM_SIDE:
            q = sub[sub["family"] == side].sort_values("auc_roc_mean").iloc[-1]["model"]
            fq = f[f["model"] == q]
            for m in stats.METRICS:
                d = (fq[m] - fc[m].reindex(fq.index)).to_numpy()
                t, p_t = stats.corrected_resampled_ttest(d, ratio)
                boot = dists[(q, nc, m)] - dists[(c, nc, m)]
                lo, hi = stats.percentile_ci(boot, cfg["bootstrap"]["confidence"])
                rows.append({"n_components": nc, "metric": m, "side": side, "challenger": q,
                             "best_classical": c, "diff_mean": d.mean(), "diff_ci_lo": lo,
                             "diff_ci_hi": hi, "t_corrected": t, "p_ttest": p_t,
                             "p_bootstrap": stats.bootstrap_pvalue(boot)})
    sig = pd.DataFrame(rows)
    sig["p_ttest_holm"] = np.nan
    for m in stats.METRICS:
        mask = sig["metric"] == m
        sig.loc[mask, "p_ttest_holm"] = stats.holm(sig.loc[mask, "p_ttest"].tolist())
    return sig


def _cell(row, m: str) -> str:
    return (f"{row[f'{m}_mean']:.3f} ± {row[f'{m}_std']:.3f} "
            f"[{row[f'{m}_ci_lo']:.3f}, {row[f'{m}_ci_hi']:.3f}]")


def results_table(summary: pd.DataFrame, cfg: dict) -> str:
    order = {"classical": 0, "quantum": 1, "hybrid": 2, "reference": 3}
    lines = []
    ref = summary[summary["family"] == "reference"]
    for nc in cfg["preprocess"]["n_components"]:
        sub = pd.concat([summary[(summary["n_components"] == nc) & (summary["family"] != "reference")], ref])
        sub = sub.assign(_o=sub["family"].map(order)).sort_values(["_o", "auc_roc_mean"], ascending=[True, False])
        best = sub[sub["family"] != "reference"]["auc_roc_mean"].max()
        lines += [f"**{nc} componentes PCA / qubits**", "",
                  "| Modelo | Familia | AUC-ROC | AUC-PR | Recall | Tiempo de ajuste (s) |",
                  "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            name = f"`{r['model']}`"
            if r["family"] != "reference" and r["auc_roc_mean"] == best:
                name = f"**{name}**"
            lines.append(f"| {name} | {r['family']} | {_cell(r, 'auc_roc')} | {_cell(r, 'auc_pr')} | "
                         f"{_cell(r, 'recall')} | {r['fit_seconds_mean']:.1f} ± {r['fit_seconds_std']:.1f} |")
        lines.append("")
    return "\n".join(lines)


def significance_table(sig: pd.DataFrame) -> str:
    lines = ["| Qubits | Métrica | Retador | Mejor clásico | Δ (retador − clásico) | IC bootstrap de Δ | t corregido | p (t-test) | p Holm | p (bootstrap) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in sig.sort_values(["side", "n_components"], kind="stable", ascending=[False, True]).iterrows():
        lines.append(
            f"| {r['n_components']} | {LABELS[r['metric']]} | `{r['challenger']}` ({r['side']}) | `{r['best_classical']}` | "
            f"{r['diff_mean']:+.3f} | [{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}] | {r['t_corrected']:.2f} | "
            f"{r['p_ttest']:.3f} | {r['p_ttest_holm']:.3f} | {r['p_bootstrap']:.3f} |")
    return "\n".join(lines)


def diagnostics_table(diag: pd.DataFrame) -> str:
    g = diag.groupby(["n_components", "kernel"]).agg(
        off_mean=("offdiag_mean", "mean"), off_sd=("offdiag_mean", "std"),
        al_mean=("alignment", "mean"), al_sd=("alignment", "std")).reset_index()
    lines = ["| Qubits | Kernel | Media fuera de la diagonal | Alineamiento centrado con y |", "|---|---|---|---|"]
    for _, r in g.iterrows():
        lines.append(f"| {r['n_components']} | `{r['kernel']}` | {r['off_mean']:.3f} ± {r['off_sd']:.3f} | "
                     f"{r['al_mean']:.3f} ± {r['al_sd']:.3f} |")
    return "\n".join(lines)


def qmkl_weights_table(folds: pd.DataFrame) -> str:
    import json

    q = folds[folds["model"] == "qmkl_cv"]
    w = pd.DataFrame([json.loads(s)["weights"] for s in q["selected"]], index=q.index)
    w["n_components"] = q["n_components"]
    cols = [c for c in w.columns if c != "n_components"]
    lines = ["| Qubits | " + " | ".join(f"`{c}`" for c in cols) + " | Peso cuántico total |",
             "|---|" + "---|" * (len(cols) + 1)]
    for nc, g in w.groupby("n_components"):
        cells = [f"{g[c].mean():.2f} ± {g[c].std(ddof=1):.2f}" for c in cols]
        qsum = g[[c for c in cols if c.startswith("zz")]].sum(axis=1)
        lines.append(f"| {nc} | " + " | ".join(cells) + f" | {qsum.mean():.2f} ± {qsum.std(ddof=1):.2f} |")
    return "\n".join(lines)


def build_report(dataset: str, meta: dict, summary, sig, folds, diag, cfg) -> str:
    conf = int(cfg["bootstrap"]["confidence"] * 100)
    return "\n".join([
        f"# Resultados: `{dataset}`", "",
        f"N = {meta['n_samples']} filas ({meta['n_positive']} positivas), {meta['n_features']} variables originales. "
        f"{cfg['cv']['n_seeds']} semillas × CV estratificada de {cfg['cv']['n_splits']} folds. "
        f"Tiempo total de la corrida: {meta['wall_seconds']:.0f} s.", "",
        f"Cada celda es media ± desviación estándar entre semillas (media de los folds de cada semilla) y, entre "
        f"corchetes, el IC bootstrap estratificado al {conf}% ({cfg['bootstrap']['n_resamples']} remuestreos) "
        "de ese mismo estadístico sobre las predicciones out-of-fold. `gb_raw` / `rf_raw` usan los descriptores "
        "originales sin PCA y son solo una referencia: no entran en el test de significancia.", "",
        "## Tabla de resultados", "", results_table(summary, cfg),
        "## Significancia: mejor cuántico y mejor híbrido vs. mejor clásico", "",
        "Dos comparaciones por número de qubits: el mejor QSVM puro (`quantum`) y el mejor QMKL (`hybrid`) contra "
        "el mejor modelo clásico, todos elegidos a posteriori por AUC-ROC media, lo que favorece al lado con más "
        "candidatos. Las diferencias son pareadas sobre los mismos folds. La prueba principal es el t-test "
        "remuestreado corregido de Nadeau y Bengio (2003) sobre las diferencias por fold, que incluye la "
        "variabilidad del entrenamiento; *p Holm* lo ajusta por todas las comparaciones de la misma métrica. El "
        "bootstrap pareado solo remuestrea las muestras de test con los modelos ya entrenados, por lo que es más "
        "optimista.", "",
        significance_table(sig), "",
        "## Pesos aprendidos por QMKL-CV", "",
        "Media ± desviación de los pesos elegidos en los folds externos.", "", qmkl_weights_table(folds), "",
        "## Diagnóstico de los kernels cuánticos", "",
        "Una media fuera de la diagonal cercana a 0 indica concentración exponencial del kernel (matriz de Gram "
        "próxima a la identidad).", "", diagnostics_table(diag), "",
    ])
