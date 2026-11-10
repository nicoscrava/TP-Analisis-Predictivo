# AP_2026Q2 — Competencia Kaggle (Examen 3, Análisis Predictivo ITBA)

Clasificación binaria: predecir si una publicación de MercadoLibre Argentina (2013–2015) es `new` o `used` (`condition`). Métrica: accuracy.

## Modelo final

Promedio 50/50 de las probabilidades de dos modelos que codifican las variables categóricas de forma distinta. Los dos usan como probabilidad de texto el **promedio de una regresión logística y de BETO** (BERT en español, `dccuchile/bert-base-spanish-wwm-cased`, con *fine-tuning* sobre título + garantía):

| Modelo | Codificación de vendedor y categoría | Accuracy CV (5 folds) |
|---|---|---|
| CatBoost (parámetros por defecto) | categóricas crudas, *ordered target statistics* de CatBoost | 0,9431 |
| HistGradientBoosting | `TargetEncoder` sobre vendedor, categoría y sus combinaciones | 0,9425 |
| **Promedio 50/50 (modelo final)** | — | **0,9435** |

**Leaderboard público: 0,94438** (`submission/submission_beto_entrega.csv`).

Además de la probabilidad de texto, los dos modelos usan TF-IDF del título y features numéricas de la publicación: cantidades, envío y pago, fotos, fechas y la **duración de la publicación** (`stop_time` − `start_time`; las que duran 60 días + unos segundos son casi todas usadas).

**Submission de respaldo (la segunda elegida):** el mismo modelo **sin BETO** (P7c + duración). CV 0,9420 (0,9422 con otra semilla de CV) y público 0,94247 (`submission/submission_p7c_duracion.csv`). No necesita GPU y entrena en ~2 min.

El código compartido está en `features.py` y `texto_beto.py`. El análisis exploratorio, en `eda.ipynb`. El detalle del desarrollo (todos los experimentos, resultados y decisiones), en `trabajo_realizado/BITACORA.md`.

## Cómo reproducir

1. Descargar `train_data.jsonlines` y `test_data.jsonlines` desde Kaggle y ponerlos en `data/`.
2. Crear el entorno:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **`train.ipynb` (~100 min):** hace el *fine-tuning* de 5 BETO (uno por fold, para la probabilidad *out-of-fold* del train) y entrena los dos modelos con todo el train.
   - Guarda `modelos_beto/fold_1..5/` (pesos de BETO) y `modelo_final.joblib`.
   - El tiempo está medido en una Mac M4 con GPU (MPS). Conviene correrlo con la compu enchufada y sin apps pesadas: a batería y con poca memoria tarda muchísimo más.
   - La primera vez descarga BETO de Hugging Face (~440 MB).
   - Si `modelos_beto/fold_k/` ya tiene sus pesos y su `oof.npy`, saltea el *fine-tuning* de ese fold.
4. **`predict.ipynb` (~7 min):** calcula la probabilidad de BETO del test (promedio de los 5), aplica los dos modelos y genera `submission/submission.csv`.

### Archivos grandes

- Los dos modelos entrenados **están en el repo**: `modelo_final.joblib` (~40 MB, con BETO) y `modelo_final_p7c_duracion.joblib` (~37 MB, sin BETO).
- Los pesos de BETO, `modelos_beto/` (~2,2 GB), no entran en git por tamaño. Están en el Release, en 5 zips, uno por fold, cada uno con su `oof.npy`:

  **Release: https://github.com/nicoscrava/TP-Analisis-Predictivo/releases/tag/v1.0**

  Hay que descomprimirlos en la raíz del repo, así: `modelos_beto/fold_1/` … `modelos_beto/fold_5/`.
- Los datos (`data/`) se bajan de Kaggle.

### Reproducibilidad

- **Con los pesos guardados,** `predict.ipynb` genera una submission **idéntica byte a byte** a `submission_beto_entrega.csv`. Está verificado dos veces en la máquina donde se entrenó.
- **Si se vuelve a correr `train.ipynb`,** cambia ~0,5% de las predicciones: en la verificación, 140 de 30.000. Todas las semillas están fijas en 42, pero el entrenamiento de BETO en MPS no es determinista bit a bit, y diferencias mínimas en sus probabilidades mueven algunos cortes de los árboles.
- Correr `predict.ipynb` en CPU o en otra GPU podría cambiar alguna predicción cercana a 0,5. Esto no está verificado.

### Reproducir cada submission

`train.ipynb` y `predict.ipynb` tienen al principio una única constante, `CON_BETO`, que elige cuál de las dos submissions elegidas se reproduce. Las dos escriben `submission/submission.csv`.

| | `CON_BETO = True` (entrega) | `CON_BETO = False` (respaldo) |
|---|---|---|
| Submission que reproduce | `submission_beto_entrega.csv` (público 0,94438) | `submission_p7c_duracion.csv` (público 0,94247) |
| Modelo | `con_beto=True` → `modelo_final.joblib` | `con_beto=False` → `modelo_final_p7c_duracion.joblib` |
| Necesita `modelos_beto/` | sí (Release) | no |
| `train.ipynb` | 1,7 min con los pesos del Release; ~100 min desde cero | 1,7 min |
| `predict.ipynb` | 7,3 min (BETO sobre el test, en MPS) | 0,2 min |
| Resultado | **idéntica byte a byte** | **idéntica byte a byte** |

Verificado el 2026-11-10 en dos copias limpias del repo, sin los `.joblib` y con los tiempos medidos en una Mac M4 a batería. Cada copia corrió `train.ipynb` y después `predict.ipynb`, y su `submission.csv` se comparó con la submission enviada. Para el respaldo no hace falta GPU ni los pesos de BETO.

### Replicar el modelo final

**(a) Con los pesos de BETO del Release** (`modelos_beto/fold_1..5/`, cada uno con su `oof.npy`), `train.ipynb` reproduce exactamente el modelo final y la submission:
- Saltea el *fine-tuning* de los 5 BETO y tarda **~2 min**: medido 1,7 min, más ~7 min de `predict.ipynb`.
- La submission que genera es **idéntica byte a byte** a `submission_beto_entrega.csv`.
- El `modelo_final.joblib` nuevo da **predicciones idénticas bit a bit** a las del original.
  - Los dos archivos difieren en unos pocos bytes, todos metadatos: el identificador interno y la hora de entrenamiento que CatBoost guarda en el modelo, y un id de memoria que el TF-IDF guarda en su lista de *stop words*.
- Verificado el 2026-11-10 en una copia limpia del repo, sin `modelo_final.joblib`.

**(b) Desde cero** (sin `modelos_beto/`), `train.ipynb` reentrena los 5 BETO (**~100 min** en MPS).
- MPS no es determinista bit a bit, así que cambia **~0,5% de las predicciones**: 140 de 30.000 en la verificación.
- Por eso la submission no queda idéntica.

### Dependencias

`requirements.txt` incluye librerías que el modelo final no usa y que quedaron de los experimentos y de la interpretación:
- `sentence-transformers`: embeddings;
- `tensorflow` y `pytorch-tabnet`: redes neuronales;
- `optuna`: tuning;
- `lightgbm`: comparación;
- `shap`: interpretación;
- `seaborn`.

La entrega necesita `pandas`, `numpy`, `scikit-learn`, `catboost`, `torch`, `transformers`, `joblib`, `ipykernel` y `matplotlib` (este último, solo para los gráficos de `eda.ipynb`).

## Estructura

```
eda.ipynb             análisis exploratorio (justifica las decisiones de features.py)
train.ipynb           entrena los 5 BETO y los dos modelos; guarda modelos_beto/ y modelo_final.joblib
                      (con CON_BETO = False: modelo_final_p7c_duracion.joblib, sin BETO)
predict.ipynb         genera submission/submission.csv
features.py           features determinísticas, encoders, modelo de texto y los dos pipelines
texto_beto.py         fine-tuning y predicción de BETO
modelos_beto/         pesos de los 5 BETO (no está en git: van en el Release)
modelo_final.joblib   CatBoost + HistGB entrenados, con BETO (entrega)
modelo_final_p7c_duracion.joblib   CatBoost + HistGB entrenados, sin BETO (respaldo)
data/                 train_data.jsonlines y test_data.jsonlines de Kaggle (no está en git)
submission/           en git solo las dos elegidas: submission_beto_entrega.csv y submission_p7c_duracion.csv
requirements.txt
trabajo_realizado/    exploración y experimentos (no forman parte de la entrega)
```

Dentro de `trabajo_realizado/`:
- `BITACORA.md`: cronología completa del trabajo (metodología, experimentos, resultados y decisiones).
- Notebooks `01`–`34`:
  - `01`–`07`: EDA original y primeras features;
  - `08`–`12`: análisis de errores, modelo de texto, vendedor, columnas anidadas y tuning;
  - `13_comparacion_modelos`: comparación de los modelos del material con el modelo final (tabla de la presentación);
  - `18_barrido_campos`: encontró la duración;
  - `21_shap`: interpretación con SHAP (gráficos en `figuras/`);
  - `26`–`34`: experimentos con BETO y otros que no pasaron.
- `laboratorio.ipynb`: pruebas de features y del modelo final.
- `anteriores/`: versiones viejas de `train`/`predict`.
