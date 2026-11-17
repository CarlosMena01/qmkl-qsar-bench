# Resultados: `qsar_biodeg`

N = 500 filas (169 positivas), 41 variables originales. 5 semillas × CV estratificada de 5 folds. Tiempo total de la corrida: 202 s.

Cada celda es media ± desviación estándar entre semillas (media de los folds de cada semilla) y, entre corchetes, el IC bootstrap estratificado al 95% (2000 remuestreos) de ese mismo estadístico sobre las predicciones out-of-fold. `gb_raw` / `rf_raw` usan los descriptores originales sin PCA y son solo una referencia: no entran en el test de significancia.

## Tabla de resultados

**4 componentes PCA / qubits**

| Modelo | Familia | AUC-ROC | AUC-PR | Recall | Tiempo de ajuste (s) |
|---|---|---|---|---|---|
| **`svm_rbf`** | classical | 0.862 ± 0.002 [0.828, 0.893] | 0.760 ± 0.015 [0.714, 0.817] | 0.798 ± 0.018 [0.739, 0.850] | 0.9 ± 0.0 |
| `gb` | classical | 0.846 ± 0.010 [0.811, 0.878] | 0.736 ± 0.026 [0.694, 0.792] | 0.762 ± 0.018 [0.701, 0.821] | 10.1 ± 0.1 |
| `rf` | classical | 0.844 ± 0.005 [0.808, 0.877] | 0.740 ± 0.020 [0.696, 0.797] | 0.747 ± 0.024 [0.684, 0.803] | 13.6 ± 0.2 |
| `svm_linear` | classical | 0.843 ± 0.005 [0.808, 0.877] | 0.712 ± 0.012 [0.669, 0.778] | 0.813 ± 0.044 [0.760, 0.861] | 0.3 ± 0.0 |
| `qsvm_r1_linear` | quantum | 0.854 ± 0.008 [0.820, 0.886] | 0.739 ± 0.026 [0.695, 0.799] | 0.791 ± 0.017 [0.737, 0.840] | 0.3 ± 0.0 |
| `qsvm_r2_linear` | quantum | 0.849 ± 0.009 [0.814, 0.880] | 0.733 ± 0.025 [0.692, 0.792] | 0.792 ± 0.027 [0.734, 0.843] | 0.4 ± 0.0 |
| `qsvm_r1_full` | quantum | 0.846 ± 0.005 [0.809, 0.879] | 0.730 ± 0.017 [0.683, 0.788] | 0.780 ± 0.020 [0.723, 0.830] | 0.4 ± 0.0 |
| `qsvm_r2_full` | quantum | 0.829 ± 0.012 [0.791, 0.863] | 0.705 ± 0.030 [0.663, 0.766] | 0.768 ± 0.028 [0.713, 0.818] | 0.4 ± 0.0 |
| **`qmkl_cv`** | hybrid | 0.862 ± 0.002 [0.828, 0.893] | 0.760 ± 0.015 [0.714, 0.817] | 0.798 ± 0.018 [0.739, 0.850] | 13.7 ± 0.1 |
| `qmkl_cka` | hybrid | 0.859 ± 0.005 [0.825, 0.891] | 0.756 ± 0.013 [0.711, 0.815] | 0.797 ± 0.018 [0.738, 0.850] | 0.3 ± 0.0 |
| `rf_raw` | reference | 0.923 ± 0.001 [0.896, 0.946] | 0.882 ± 0.006 [0.847, 0.915] | 0.791 ± 0.024 [0.733, 0.843] | 14.6 ± 0.4 |
| `gb_raw` | reference | 0.919 ± 0.005 [0.891, 0.943] | 0.875 ± 0.009 [0.839, 0.910] | 0.799 ± 0.014 [0.745, 0.849] | 17.9 ± 0.4 |

**6 componentes PCA / qubits**

| Modelo | Familia | AUC-ROC | AUC-PR | Recall | Tiempo de ajuste (s) |
|---|---|---|---|---|---|
| **`svm_rbf`** | classical | 0.879 ± 0.006 [0.848, 0.906] | 0.790 ± 0.012 [0.748, 0.838] | 0.819 ± 0.005 [0.765, 0.865] | 0.9 ± 0.0 |
| `svm_linear` | classical | 0.870 ± 0.008 [0.838, 0.897] | 0.763 ± 0.018 [0.719, 0.820] | 0.821 ± 0.029 [0.770, 0.868] | 0.3 ± 0.0 |
| `rf` | classical | 0.866 ± 0.006 [0.832, 0.895] | 0.774 ± 0.019 [0.728, 0.826] | 0.763 ± 0.022 [0.704, 0.818] | 13.7 ± 0.1 |
| `gb` | classical | 0.864 ± 0.007 [0.831, 0.892] | 0.767 ± 0.023 [0.725, 0.817] | 0.776 ± 0.015 [0.721, 0.828] | 11.2 ± 0.2 |
| `qsvm_r1_linear` | quantum | 0.876 ± 0.004 [0.843, 0.903] | 0.775 ± 0.012 [0.732, 0.828] | 0.818 ± 0.008 [0.764, 0.865] | 0.3 ± 0.0 |
| `qsvm_r2_linear` | quantum | 0.868 ± 0.008 [0.834, 0.897] | 0.775 ± 0.022 [0.734, 0.826] | 0.802 ± 0.017 [0.746, 0.851] | 0.3 ± 0.0 |
| `qsvm_r1_full` | quantum | 0.865 ± 0.003 [0.829, 0.895] | 0.776 ± 0.014 [0.733, 0.826] | 0.775 ± 0.029 [0.718, 0.826] | 0.3 ± 0.0 |
| `qsvm_r2_full` | quantum | 0.842 ± 0.008 [0.803, 0.874] | 0.748 ± 0.021 [0.705, 0.803] | 0.700 ± 0.027 [0.638, 0.753] | 0.3 ± 0.0 |
| `qmkl_cka` | hybrid | 0.879 ± 0.008 [0.848, 0.905] | 0.787 ± 0.017 [0.745, 0.838] | 0.813 ± 0.023 [0.758, 0.861] | 0.3 ± 0.0 |
| `qmkl_cv` | hybrid | 0.877 ± 0.005 [0.845, 0.904] | 0.786 ± 0.014 [0.743, 0.834] | 0.821 ± 0.017 [0.768, 0.867] | 13.6 ± 0.1 |
| `rf_raw` | reference | 0.923 ± 0.001 [0.896, 0.946] | 0.882 ± 0.006 [0.847, 0.915] | 0.791 ± 0.024 [0.733, 0.843] | 14.6 ± 0.4 |
| `gb_raw` | reference | 0.919 ± 0.005 [0.891, 0.943] | 0.875 ± 0.009 [0.839, 0.910] | 0.799 ± 0.014 [0.745, 0.849] | 17.9 ± 0.4 |

**8 componentes PCA / qubits**

| Modelo | Familia | AUC-ROC | AUC-PR | Recall | Tiempo de ajuste (s) |
|---|---|---|---|---|---|
| `svm_rbf` | classical | 0.917 ± 0.005 [0.892, 0.940] | 0.866 ± 0.008 [0.830, 0.904] | 0.841 ± 0.019 [0.788, 0.887] | 0.9 ± 0.0 |
| `rf` | classical | 0.911 ± 0.005 [0.884, 0.935] | 0.866 ± 0.013 [0.832, 0.900] | 0.773 ± 0.016 [0.716, 0.825] | 13.9 ± 0.1 |
| `gb` | classical | 0.907 ± 0.008 [0.880, 0.931] | 0.853 ± 0.020 [0.818, 0.890] | 0.792 ± 0.035 [0.737, 0.840] | 12.4 ± 0.1 |
| `svm_linear` | classical | 0.896 ± 0.006 [0.868, 0.923] | 0.790 ± 0.008 [0.748, 0.851] | 0.852 ± 0.022 [0.803, 0.897] | 0.3 ± 0.0 |
| `qsvm_r1_linear` | quantum | 0.907 ± 0.004 [0.877, 0.932] | 0.855 ± 0.011 [0.818, 0.896] | 0.817 ± 0.007 [0.761, 0.865] | 0.3 ± 0.0 |
| `qsvm_r2_linear` | quantum | 0.899 ± 0.002 [0.868, 0.925] | 0.843 ± 0.010 [0.804, 0.887] | 0.804 ± 0.024 [0.749, 0.851] | 0.3 ± 0.0 |
| `qsvm_r1_full` | quantum | 0.896 ± 0.005 [0.864, 0.924] | 0.839 ± 0.015 [0.798, 0.883] | 0.778 ± 0.021 [0.717, 0.828] | 0.3 ± 0.0 |
| `qsvm_r2_full` | quantum | 0.869 ± 0.010 [0.834, 0.898] | 0.799 ± 0.014 [0.759, 0.846] | 0.556 ± 0.111 [0.500, 0.610] | 0.3 ± 0.0 |
| **`qmkl_cka`** | hybrid | 0.919 ± 0.006 [0.892, 0.941] | 0.870 ± 0.007 [0.833, 0.907] | 0.825 ± 0.008 [0.768, 0.872] | 0.3 ± 0.0 |
| `qmkl_cv` | hybrid | 0.916 ± 0.006 [0.890, 0.939] | 0.863 ± 0.010 [0.827, 0.902] | 0.838 ± 0.028 [0.784, 0.885] | 13.5 ± 0.1 |
| `rf_raw` | reference | 0.923 ± 0.001 [0.896, 0.946] | 0.882 ± 0.006 [0.847, 0.915] | 0.791 ± 0.024 [0.733, 0.843] | 14.6 ± 0.4 |
| `gb_raw` | reference | 0.919 ± 0.005 [0.891, 0.943] | 0.875 ± 0.009 [0.839, 0.910] | 0.799 ± 0.014 [0.745, 0.849] | 17.9 ± 0.4 |

## Significancia: mejor cuántico y mejor híbrido vs. mejor clásico

Dos comparaciones por número de qubits: el mejor QSVM puro (`quantum`) y el mejor QMKL (`hybrid`) contra el mejor modelo clásico, todos elegidos a posteriori por AUC-ROC media, lo que favorece al lado con más candidatos. Las diferencias son pareadas sobre los mismos folds. La prueba principal es el t-test remuestreado corregido de Nadeau y Bengio (2003) sobre las diferencias por fold, que incluye la variabilidad del entrenamiento; *p Holm* lo ajusta por todas las comparaciones de la misma métrica. El bootstrap pareado solo remuestrea las muestras de test con los modelos ya entrenados, por lo que es más optimista.

| Qubits | Métrica | Retador | Mejor clásico | Δ (retador − clásico) | IC bootstrap de Δ | t corregido | p (t-test) | p Holm | p (bootstrap) |
|---|---|---|---|---|---|---|---|---|---|
| 4 | AUC-ROC | `qsvm_r1_linear` (quantum) | `svm_rbf` | -0.008 | [-0.018, +0.003] | -0.79 | 0.437 | 1.000 | 0.144 |
| 4 | AUC-PR | `qsvm_r1_linear` (quantum) | `svm_rbf` | -0.021 | [-0.041, +0.001] | -0.62 | 0.539 | 1.000 | 0.065 |
| 4 | Recall | `qsvm_r1_linear` (quantum) | `svm_rbf` | -0.007 | [-0.030, +0.015] | -0.25 | 0.805 | 1.000 | 0.590 |
| 6 | AUC-ROC | `qsvm_r1_linear` (quantum) | `svm_rbf` | -0.003 | [-0.014, +0.007] | -0.29 | 0.774 | 1.000 | 0.491 |
| 6 | AUC-PR | `qsvm_r1_linear` (quantum) | `svm_rbf` | -0.015 | [-0.034, +0.008] | -0.62 | 0.538 | 1.000 | 0.229 |
| 6 | Recall | `qsvm_r1_linear` (quantum) | `svm_rbf` | -0.001 | [-0.023, +0.021] | -0.03 | 0.973 | 1.000 | 0.922 |
| 8 | AUC-ROC | `qsvm_r1_linear` (quantum) | `svm_rbf` | -0.011 | [-0.023, +0.001] | -1.25 | 0.223 | 1.000 | 0.080 |
| 8 | AUC-PR | `qsvm_r1_linear` (quantum) | `svm_rbf` | -0.011 | [-0.030, +0.011] | -0.60 | 0.555 | 1.000 | 0.325 |
| 8 | Recall | `qsvm_r1_linear` (quantum) | `svm_rbf` | -0.025 | [-0.051, -0.001] | -0.89 | 0.382 | 1.000 | 0.045 |
| 4 | AUC-ROC | `qmkl_cv` (hybrid) | `svm_rbf` | +0.000 | [+0.000, +0.000] | 0.00 | 1.000 | 1.000 | 1.000 |
| 4 | AUC-PR | `qmkl_cv` (hybrid) | `svm_rbf` | +0.000 | [+0.000, +0.000] | 0.00 | 1.000 | 1.000 | 1.000 |
| 4 | Recall | `qmkl_cv` (hybrid) | `svm_rbf` | +0.000 | [+0.000, +0.000] | 0.00 | 1.000 | 1.000 | 1.000 |
| 6 | AUC-ROC | `qmkl_cka` (hybrid) | `svm_rbf` | -0.000 | [-0.006, +0.006] | -0.05 | 0.960 | 1.000 | 0.910 |
| 6 | AUC-PR | `qmkl_cka` (hybrid) | `svm_rbf` | -0.003 | [-0.014, +0.009] | -0.15 | 0.879 | 1.000 | 0.676 |
| 6 | Recall | `qmkl_cka` (hybrid) | `svm_rbf` | -0.006 | [-0.015, +0.002] | -0.32 | 0.749 | 1.000 | 0.187 |
| 8 | AUC-ROC | `qmkl_cka` (hybrid) | `svm_rbf` | +0.001 | [-0.003, +0.006] | 0.45 | 0.660 | 1.000 | 0.541 |
| 8 | AUC-PR | `qmkl_cka` (hybrid) | `svm_rbf` | +0.003 | [-0.005, +0.012] | 0.43 | 0.670 | 1.000 | 0.380 |
| 8 | Recall | `qmkl_cka` (hybrid) | `svm_rbf` | -0.017 | [-0.030, -0.004] | -0.96 | 0.347 | 1.000 | 0.009 |

## Pesos aprendidos por QMKL-CV

Media ± desviación de los pesos elegidos en los folds externos.

| Qubits | `linear` | `rbf_g0.1` | `rbf_g1` | `rbf_g10` | `zz_r1_full` | `zz_r1_linear` | `zz_r2_full` | `zz_r2_linear` | Peso cuántico total |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.00 ± 0.00 | 0.68 ± 0.48 | 0.32 ± 0.48 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 6 | 0.01 ± 0.04 | 0.62 ± 0.48 | 0.30 ± 0.45 | 0.00 ± 0.01 | 0.03 ± 0.09 | 0.01 ± 0.05 | 0.02 ± 0.06 | 0.01 ± 0.04 | 0.07 ± 0.19 |
| 8 | 0.04 ± 0.08 | 0.47 ± 0.48 | 0.38 ± 0.45 | 0.02 ± 0.05 | 0.03 ± 0.07 | 0.02 ± 0.06 | 0.02 ± 0.05 | 0.01 ± 0.04 | 0.09 ± 0.18 |

## Diagnóstico de los kernels cuánticos

Una media fuera de la diagonal cercana a 0 indica concentración exponencial del kernel (matriz de Gram próxima a la identidad).

| Qubits | Kernel | Media fuera de la diagonal | Alineamiento centrado con y |
|---|---|---|---|
| 4 | `zz_r1_full` | 0.275 ± 0.017 | 0.137 ± 0.023 |
| 4 | `zz_r1_linear` | 0.319 ± 0.031 | 0.127 ± 0.023 |
| 4 | `zz_r2_full` | 0.178 ± 0.016 | 0.104 ± 0.021 |
| 4 | `zz_r2_linear` | 0.251 ± 0.066 | 0.104 ± 0.022 |
| 6 | `zz_r1_full` | 0.147 ± 0.010 | 0.126 ± 0.014 |
| 6 | `zz_r1_linear` | 0.200 ± 0.026 | 0.122 ± 0.019 |
| 6 | `zz_r2_full` | 0.062 ± 0.005 | 0.088 ± 0.009 |
| 6 | `zz_r2_linear` | 0.139 ± 0.043 | 0.104 ± 0.020 |
| 8 | `zz_r1_full` | 0.098 ± 0.008 | 0.147 ± 0.013 |
| 8 | `zz_r1_linear` | 0.150 ± 0.026 | 0.138 ± 0.018 |
| 8 | `zz_r2_full` | 0.025 ± 0.003 | 0.092 ± 0.007 |
| 8 | `zz_r2_linear` | 0.088 ± 0.030 | 0.119 ± 0.018 |
