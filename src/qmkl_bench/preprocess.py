"""Preprocesamiento: MinMax -> PCA -> MinMax a [0, 1].

El segundo escalado es necesario porque las componentes de PCA no viven en [0, 1],
que es el rango que el feature map recibe como ángulos. Todo se ajusta solo con el
train de cada fold; el test se transforma y se recorta al rango.
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def build_preprocessor(
    n_components: int, feature_range: tuple[float, float] = (0.0, 1.0), seed: int = 0
) -> Pipeline:
    lo, hi = feature_range
    return Pipeline(
        [
            ("scale_in", MinMaxScaler()),
            ("pca", PCA(n_components=n_components, svd_solver="full", random_state=seed)),
            ("scale_out", MinMaxScaler(feature_range=(lo, hi), clip=True)),
        ]
    )
