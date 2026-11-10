# Bitácora del TP3 — Análisis Predictivo (ITBA)

**Para quien lee esto sin haber estado:** documento autocontenido con **todo** lo hecho en el TP, en orden, con los números medidos, las decisiones y su porqué. El código está en las notebooks y en `features.py`; acá está la historia y el razonamiento. Lo confirmado (medido) y lo interpretado se distinguen explícitamente.

Última actualización: 2026-11-10 (Fases M a X, Paso 8 y Paso 9: `13_comparacion_modelos` corrida, repo reorganizado, `eda.ipynb` nueva y **submissions finales elegidas**). **Estado: la ENTREGA ACTUAL es P7c + duración + BETO (Paso 8): promedio 50/50 de CatBoost + HistGB con combinaciones, los dos con `duracion_dias` y con `prob_texto_new` = (logística + BETO)/2 (variante (a) de la Fase T). CV 0,9435 (solo semilla 42; la semilla 7 no se corre), público 0,94438.** Está en `features.py` (`con_beto=True`), `texto_beto.py`, `train.ipynb` y `predict.ipynb` (verificados). El modelo anterior (P7c + duración: CV 0,9420, semilla 7 0,9422, público 0,94247) quedó en `modelo_final_p7c_duracion.joblib`. **Experimentos de BETO y formato terminados:** Fases V (BETO con contexto), W (artefactos de formato) y X (BETO con preentrenamiento adaptado) no pasan. La semilla 7 de la Fase T no se corre (decisión mía, 2026-11-04). **Semana 3 empezada:** `13_comparacion_modelos` corrida el 2026-11-09 (tabla en el Paso 9). Plan: sección 9.

---

## 0. Resumen en 30 segundos

- **Problema:** clasificar publicaciones de MercadoLibre Argentina (2013–2015) como `new` o `used`. Competencia Kaggle **AP_2026Q2**, métrica **accuracy**.
- **Estado de los experimentos:** se cerraron el 2026-10-13 (Fases M–P: nada pasó), pero los reabrí para probar BETO: el fine-tuning (Fase Q) no pasó como feature extra, y **promediado con la logística en una sola probabilidad de texto (Fase T) sí pasó** (+0,0015, 5/5, semilla 42). Esa es la entrega actual (Paso 8). Después: Fases R, S, U, V (BETO con contexto) y W (artefactos de formato), y X (BETO con preentrenamiento adaptado), donde nada pasó.
- **De dónde a dónde:** baseline de regresión logística con 5 variables = 0,822 → **modelo final = 0,9420 en CV** (5 folds; 0,9422 con otra semilla).
- **Leaderboard público:** la Base anterior (0,9339 en CV) dio **0,93476**; `submission_catboost.csv` (CatBoost P6b, 0,9365 en CV) dio 0,93695 (primero en su momento); el modelo P7c + duración (`submission_p7c_duracion.csv`, 0,9420 en CV) dio 0,94247; **la entrega actual con BETO (`submission_beto_entrega.csv`, 0,9435 en CV) dio 0,94438**. La CV predijo bien el leaderboard en todas las submissions.
- **Entrega actual = el modelo de abajo + BETO:** la probabilidad de texto pasa a ser el promedio de la logística y de un BETO (transformer en español) ajustado para `new`/`used` sobre título + garantía (Fase T, Paso 8).
- **Modelo final (P7c + duración):** promedio 50/50 de dos modelos que codifican las categóricas distinto:
  - `CatBoostClassifier` (parámetros por defecto) con las categóricas **crudas**: 0,9414;
  - `HistGradientBoostingClassifier` con `TargetEncoder` sobre vendedor, categoría y sus **combinaciones**: 0,9412.

  Los dos usan la probabilidad de un **modelo de texto** (regresión logística sobre título + garantía) y features numéricas de la publicación, incluidas fotos, fechas y la **duración de la publicación** (`stop_time` − `start_time`). Promedio: **0,9420**.
- **Lo que más sumó:** el vendedor (+0,037), el título (+0,028 con TF-IDF y +0,006 con el modelo de texto), las features "de catálogo" (+0,028), codificar las categóricas con CatBoost (+0,0026), promediarlo con el HistGB con combinaciones (+0,0011) y la duración de la publicación (+0,0044).
- **Lo que no sumó:** tuning de hiperparámetros (HistGB y CatBoost), promediar semillas (HistGB y CatBoost), más categóricas crudas, `text_features` de CatBoost, precio/conteo del vendedor sobre CatBoost, mover el umbral, títulos repetidos, publicación "padre", perfil del vendedor, flags de método de pago, embeddings del título, URL de la miniatura, redes neuronales (MLP, TabNet), tuning de CatBoost con Optuna, kNN de títulos, stacking con contexto, pseudo-labeling, modelo especialista, modelo en cascada, el resto de las fechas al segundo (Fase M), LightGBM (Fase N), cantidades y precios relativos (Fase O), precio compartido entre vendedores, limpieza de etiquetas, propagación por vendedor y conteos sobre train + test (Fase P), y los ensambles con modelos más flojos (voting simple/ponderado, Hill Climber, Stacking de la `16_ensambles`).

---

## 1. El problema y las reglas

### Datos
- `data/train_data.jsonlines`: 70.000 filas, 45 columnas, con `condition` (≈ 53,7% `new`). `data/test_data.jsonlines`: 30.000 filas, sin `condition`.
- Un JSON por línea; varias columnas anidadas (dicts como `shipping` y `seller_address`; listas como `pictures`, `attributes`, `tags`, `variations`).
- El test parece una muestra aleatoria del mismo universo: mismo rango de fechas, ~72% de vendedores compartidos con train. Por eso una **StratifiedKFold aleatoria** es una validación razonable, y el leaderboard lo confirmó (CV 0,9339 → público 0,93476).

### Submission (formato confirmado por Kaggle)
CSV con header y 30.000 filas, columnas `ID` y `condition`:
- `ID` = número de fila **empezando en 1**, en el orden de `test_data.jsonlines`. Con ID 0-based o con el `id` "MLA…", Kaggle da error.
- `condition` = `"new"` / `"used"` (clase, no probabilidad).

### Reglas del examen
- La nota depende mucho del leaderboard **privado** (30% del test); el público es el 70%. Máximo 5 submissions por día; al final se eligen 2.
- Las submissions las subo a mano.
- Entrega: presentación de máx. 5 min (baseline → selección de modelos → modelo final → limitaciones y mejoras) + `train.ipynb` (entrena y guarda el modelo final) + `predict.ipynb` (genera exactamente la submission), con **solo** el código necesario.
- Defensa individual: tengo que poder explicar todo el código.
- Todo se basa en el material de la materia (`material/`, copia local del Drive). Si se usa algo que no está, hay que decirlo y explicarlo desde cero.

---

## 2. Metodología (cómo se decidió todo)

1. **Validación cruzada**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, siempre la misma. Métrica principal: accuracy; secundaria: roc_auc. Se elige **por CV, nunca por el leaderboard público**.
2. **Todo lo que aprende de los datos va adentro de un `Pipeline`** (capeo, encoders, TF-IDF, modelo de texto), así en cada fold se ajusta solo con el train de ese fold. Lo **determinístico** (aplanar columnas, contar fotos, parsear fechas) va afuera, porque da lo mismo calcularlo antes o después.
3. **Modelo fijo para elegir features**: `HistGradientBoostingClassifier(learning_rate=0.1, max_depth=None, max_iter=200, random_state=42)`. Así la única diferencia entre dos pruebas son las features.
4. **Bloques acumulativos, una cosa por vez.**
5. **Criterio para aceptar una mejora:**
   - Al principio: ganancia media claramente mayor que el desvío entre folds.
   - Desde el Paso 3 del plan: **comparación pareada fold por fold**. Como dos configuraciones se miden sobre los mismos 5 folds, se resta fold por fold. **Mejora real** = gana en 5 de 5 folds, o en 4 de 5 con la media de la diferencia mayor que su desvío. Intuición: si no hubiera diferencia real, ganar en un fold sería como tirar una moneda, y ganar en los 5 por azar tiene probabilidad (1/2)^5 ≈ 3%.
   - *Por qué el cambio:* el desvío entre folds incluye que hay folds más difíciles **para cualquier modelo**, y esa variación no tiene que ver con la diferencia entre dos configuraciones. La vara pareada es la justa.
6. **Sesgo de selección**: cuantas más pruebas se hacen sobre los mismos folds, más probable que alguna gane "de suerte". Por eso: (a) desconfiar de lo que pasa "justo"; (b) el ganador final se re-midió con **otra semilla de CV** (`random_state=7`); (c) en el tuning, no aceptar mejoras chicas.
7. `random_state=42` en todo.
8. **Cuánto ruido tiene el leaderboard** (para no sobre-reaccionar, discutido el 2026-10-13). El público se calcula sobre ~21.000 filas y el privado sobre ~9.000. El desvío por azar de una accuracy de 0,94 es √(0,94·0,06/n): **±0,0016 en el público y ±0,0025 en el privado**. Además, cambiar la semilla de CatBoost cambia el 0,73% de las predicciones (Fase E). Una distancia de 0,002-0,003 con otro participante puede darse vuelta en el privado; una de ≥ 0,005 indica algo real (como fue la duración). Elegir submissions por el público sobreajusta a esas 21.000 filas.

---

## 3. Herramientas propias (`features.py`)

Todo lo que comparten las notebooks y la entrega vive en `features.py`.

| Nombre | Tipo | Qué hace | ¿Usa el target? |
|---|---|---|---|
| `aplanar_anidadas(df)` | función determinística | Abre las columnas anidadas en columnas planas, conteos y flags (`n_fotos`, `n_atributos`, `tiene_garantia`, `free_shipping`, …) | No |
| `agregar_features_contenido(df)` | función determinística | Fotos (resolución de la primera, % cuadradas, % con proporción de cámara 4:3, cantidad de tamaños distintos), fechas (`hora_inicio`, `gap_actualizacion`) y `warranty_texto` (garantía como texto; nula → "") | No |
| `agregar_categoricas_crudas(df)` | función determinística | Copia las 6 categóricas como texto (`cat_seller_id`, …) para CatBoost | No |
| `agregar_logs(df)` | función determinística | `log_price`, `log_available_quantity` | No |
| `CapeoPorMoneda` | transformer | Winsoriza `price`/`base_price` al percentil 99,5 **por moneda**, aprendido en el fit | No |
| `VendedorEncoder` | transformer | Por `seller_id`: cantidad de publicaciones, precio medio y `seller_pct_new` (% de `new` con *leave-one-out*) | Sí (con *leave-one-out*) |
| `PrediccionOOF` | transformer | Envuelve un clasificador y devuelve su probabilidad de `new` como feature (ver abajo) | Sí (con *cross-fitting*) |
| `armar_modelo_texto(C=2)` | función | Regresión logística sobre TF-IDF del título (palabras 1–2 + caracteres 3–5) y de la garantía (palabras 1–2) | — |
| `armar_pipeline_final()` | función | El pipeline final completo (sección 6) | — |
| Constantes | — | `COLUMNAS_CONTENIDO`, `COLUMNAS_NUMERICAS`, `COLUMNAS_CATEGORICAS`, `COLUMNAS_CAT_CRUDAS`, `INDICES_CAT` | — |

### `PrediccionOOF`: la pieza clave del modelo de texto
- **Problema:** el TF-IDF grande del título tiene decenas de miles de columnas. El HistGB/CatBoost no las puede manejar, pero una regresión logística sí.
- **Solución:** entrenar la regresión logística sobre el texto y pasarle al modelo principal **solo su probabilidad de `new`**, como una columna (`prob_texto_new`).
- **Riesgo de leakage:** si la probabilidad de las filas de train se calcula con un modelo entrenado sobre esas mismas filas, sale "demasiado buena" (memorizó los títulos). El modelo principal aprendería a confiar ciegamente en ella, y en test, donde es más floja, se equivocaría.
- **Cómo lo evita (*cross-fitting*, igual que `TargetEncoder` de sklearn):**
  - En `fit_transform` (train): CV interna de 5. Cada fila recibe la probabilidad de una regresión logística que **no la vio**.
  - Además entrena una regresión logística con todo el train, que es la que usa `transform` sobre validación/test.
  - Así la feature tiene en train la misma calidad que en test. Es la idea del **Stacking**.

### Target encoding: los tres resguardos que se usaron
El material muestra el target encoder "naive" (promedio del target por categoría, incluyendo la propia fila), que filtra la respuesta. Se usaron tres versiones con resguardo:

| Versión | Dónde | Cómo evita el leakage | Debilidad |
|---|---|---|---|
| *Leave-one-out* | `VendedorEncoder` (`seller_pct_new`) | Cada fila usa el promedio de **las otras** filas de su grupo | "Inversión" en grupos chicos: vendedor con 1 `new` y 1 `used` → la `new` ve 0% y la `used` 100% |
| *Cross-fitting* + *smoothing* | `sklearn.preprocessing.TargetEncoder` (`category_id`) | El valor de cada fila se calcula con los otros 4/5 del train; las categorías chicas se acercan al promedio global | Cada fila se codifica con solo 4/5 de los datos |
| ***Ordered target statistics*** | CatBoost (modelo final) | Ordena las filas al azar; cada una se codifica solo con las **anteriores** de su categoría + un *prior*. Usa varias permutaciones | — (es la que mejor funcionó) |

---

## 4. Cronología

### Fase A — Primeras notebooks y comparación de modelos (01–07, `train.ipynb`)

| Notebook | Qué se hizo | Resultado |
|---|---|---|
| `01_eda` | EDA. Aplanado de anidadas (`aplanar_anidadas`). Nulos → flags de presencia. `price` con errores de carga → capeo al percentil 99,5 por moneda. `log(price)`. | Decisiones de preprocesamiento |
| `train.ipynb` (baseline) | Regresión logística con `listing_type_id`, `buying_mode`, log(price), log cantidades, `automatic_relist` | **0,822** acc / 0,903 auc |
| `02_features` | Grupos de features que ya construía `aplanar_anidadas`: B (cantidades), C (contenido), D (envío/pago), E (vendedor/precio), F (tags) | 0,825 → **0,853**. F no aporta |
| `03_titulo` | TF-IDF del título (500 palabras). *TF-IDF no está en el material* | → **0,881** (+0,028) |
| `04_vendedor` | `VendedorEncoder`: conteo y precio medio (sin target) + `seller_pct_new` (LOO) | → 0,893 → **0,918** (+0,037 en total) |
| `05_submit` | Utilidad para subir CSV con Kaggle CLI (la corro a mano) | — |
| `06_eda_nuevas` / `07_nuevas_features` | `category_id` con `TargetEncoder`; fechas (L) y redondez del precio (M) | `category_id` +0,0049 → **0,9233**; L y M al límite del ruido |

**`train.ipynb`** (estado al terminar la Fase A): `GridSearchCV` (5 folds, `refit="roc_auc"`) sobre 11 modelos del material + 3 ensambles, con las features hasta `category_id`:

| Modelo | Accuracy | roc_auc | Público Kaggle |
|---|---|---|---|
| Stacking (GB + RF + kNN + Árbol) | 0,9238 | 0,9773 | 0,91995 |
| Gradient Boosting (HistGB) | 0,9237 | 0,9783 | 0,91928 |
| Voting (soft) | 0,9191 | 0,9757 | 0,91466 |
| Voting (hard) | 0,9179 | — | |
| Random Forest | 0,9179 | 0,9725 | |
| Regresión logística | 0,9130 | 0,9714 | |
| SVM lineal | 0,9130 | 0,9714 | |
| Extra Trees | 0,9130 | 0,9712 | |
| LDA | 0,9118 | 0,9704 | |
| AdaBoost | 0,9110 | 0,9693 | |
| Árbol de decisión | 0,9080 | 0,9670 | |
| kNN | 0,8976 | 0,9605 | |
| QDA | 0,7206 | 0,9487 | |
| Naive Bayes (Gaussiano) | 0,6430 | 0,9455 | |

(Las submissions previas de HistGB y AdaBoost con features viejas dieron 0,82280 y 0,81019.)

### Fase B — Plan de experimentos de 7 pasos (08–13)

Objetivo: pasar de ≈ 0,924 a 0,93–0,94. Modelo fijo HistGB, "Base" = mejor configuración vigente.

**Paso 1 — Análisis de errores (`08_errores`).** Predicciones *out-of-fold* con `cross_val_predict` (material: `S8_Ensambles.ipynb`). Base: 0,9233.
- Los **vendedores con 1 sola publicación** son el 28,5% de las filas pero el **54% de los errores** (accuracy 0,854). Para ellos el vendedor no aporta: el modelo depende solo del contenido de la publicación.
- Tipo de publicación, precio y categoría casi no discriminan errores.
- El 82% de los errores son "dudosos" (confianza < 0,9). Solo el 18% (1,4% de las filas) son errores "seguros", probable **ruido de etiqueta**: títulos "Nuevo!!!" o "Sin Uso" etiquetados `used`. (Corrección: "Departamento a estrenar", estampillas "Mint" y libros etiquetados `new` **no** son ruido, la etiqueta es razonable y el error es del modelo; el único caso discutible de ese grupo es un "Sillón antiguo" etiquetado `new`.)
- El texto de `warranty` separa mucho ("fábrica" → 98% `new`) y solo se usaba como flag.
- **Recomendación:** atacar primero el contenido (texto, fotos), después el vendedor.

**Paso 2 — Modelo de texto (`09_texto`).** Nace `PrediccionOOF`.
- T1 (regresión logística sola sobre el título, C = 2) = 0,8227.
- **T2a (Base + `prob_texto_new`) = 0,9291 (+0,0058), se suma.**
- T2b (sin TF-IDF 500) = 0,9277, equivalente.
- T3 (sumar `warranty` al texto) = +0,0011 y T4 (largo, mayúsculas, dígitos del título) = +0,0003: no pasan en ese momento.

**Paso 4 — Columnas anidadas (`11_anidadas`).** *Feature engineering* (material: `AP_Categóricas.ipynb`).
- `attributes` es redundante con `category_id`; `variations` ya está en `n_variaciones`.
- **Fotos**: la señal nueva más clara. Cuadradas ≈ 88% `new` (catálogo); 4:3 de celular ≈ 36% `new`.
- N (4 features de fotos) = +0,0018, marginal.

**Paso 3 — Vendedor v2 (`10_vendedor_v2`).** Primera vez con comparación pareada.
- V1 (`seller_pct_new` con `TargetEncoder` en vez de LOO): 4/5 folds, no pasa por poco.
- V2 (conteo del vendedor en train + test): 4/5 folds, no pasa. Además no serviría en producción.

**Paso 5 — Las marginales juntas (`11b_marginales`).** Seis mejoras chicas que por separado no pasaban (T3, N, L, M, V1, V2):
- Juntas: **+0,0051, 5/5**.
- *Ablation* (sacar de a una): N y L aportan; T3 en el borde; M, V1 y V2 sobran.
- Confirmación: **T2a + N + L + T3 = 0,9339 ± 0,0018**, 5/5 contra la Base. Pasa a ser la nueva Base.

**Paso 6 — Tuning del HistGB (`12_tuning`).** `RandomizedSearchCV`, 30 combinaciones (`learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`), *early stopping*.
- Mejor: 0,9344 contra 0,9339 del fijo, 3/5 folds → **no pasa**.
- **Superficie plana**: el top 10 entre 0,9339 y 0,9344. Lo que mueve el resultado son las features, no los hiperparámetros.
- Hallazgo: el HistGB fijo ya usaba *early stopping* sin pedirlo (`"auto"` con más de 10.000 filas) y frenaba en ~110–120 árboles.
- Se mantiene el modelo fijo.

**Paso 7 (empezado) — `13_comparacion_modelos`.** Escrita y probada: los 11 modelos + 3 ensambles con las features finales y `refit="accuracy"`. Incluye imputación por mediana + indicador de faltante para las fotos (material de datos faltantes), porque los modelos lineales y kNN no aceptan nulos.
- **No se corrió entera**: tarda ~3 h, y las features todavía podían cambiar. Se corre una sola vez, al final.
- Resultado parcial antes de cortarla (preliminar): LDA 0,9227, SVM 0,9225, regresión logística 0,9221, árbol 0,9197, kNN 0,8963, NB 0,822, QDA 0,8031. Todos por debajo del HistGB (0,934).

### Fase C — Laboratorio y última ronda de features

**`laboratorio.ipynb`**: herramienta para probar features rápido.
- `probar(nombre, columnas_numericas_extra=…, transformers_extra=…, sacar=…, modelo=…, parametros_fit=…)`: mide en ~1 minuto, con los folds en paralelo y el modelo de texto en caché en `trabajo_realizado/.cache_laboratorio/`.
- `comparar_pareado(a, b)` aplica el criterio.
- Registro persistente en `trabajo_realizado/.cache_laboratorio/resultados.csv`.
- También se midió ahí la grilla de Gradient Boosting de `train.ipynb` con las features finales: el mejor es exactamente el HistGB fijo (0,9339).

**Paso 0 — Submission de la Base (`14_submission_base`)** → `submission/submission_base_0934.csv`. **Público: 0,93476** (CV 0,9339). Confirma que la CV mide bien.

**Última ronda (todo contra la Base 0,9339, comparación pareada):**

| Prueba | Qué | Accuracy | Folds | Veredicto |
|---|---|---|---|---|
| P1 | Publicación "padre" (`parent_item_id`) | — | — | **Inaplicable**: el 77% tiene padre, pero solo 1 de 53.934 padres está en train (0 en test), y no hay hermanos |
| P2 | Títulos repetidos: `TargetEncoder` del título normalizado + conteo de gemelas | 0,9341 | 2/5 | Ruido. Solo el 1,5% del test tiene un título gemelo en train |
| P3a / P3b | Perfil del vendedor sin target (% garantía, % fotos cuadradas, % envío gratis, n_fotos, % tienda oficial), con y sin la propia fila | 0,9337 / 0,9340 | 2/5 | Ruido |
| V3 | `VendedorEncoder` sin contarse a sí misma en conteo y precio medio | 0,9333 | 2/5 | No mejora (el desajuste teórico no pesa) |
| P4 | Promedio de 5 HistGB con semillas 42–46 | 0,9340 | 3/5 | Ruido. La única fuente de azar es el 10% del *early stopping* |
| P5 | Umbral de decisión elegido con 4 folds y evaluado en el quinto | 0,9339 | 3/5 | El umbral elegido fue 0,495 en las 5 rotaciones: el modelo está bien calibrado. Se mantiene 0,5 |
| P6a | CatBoost (por defecto) en lugar del HistGB, mismas features | 0,9338 | 2/5 | Igual que el HistGB |
| **P6b** | **CatBoost con `seller_id`, `category_id` y las 4 categóricas crudas como `cat_features`, sin `VendedorEncoder` ni `TargetEncoder`** | **0,9365 ± 0,0017** | **5/5, +0,0026** | **Mejora real** |

**Cierre:**
- **Robustez de P6b** con otra semilla de CV (`random_state=7`): Base 0,9332 → P6b 0,9359, **5/5, +0,0028**. Se mantiene.
- Solo pasó una prueba, así que no hubo combinación ni *ablation*.
- `armar_pipeline_final()` en `features.py` reproduce P6b **exacto** (diferencia 0 en los 5 folds).
- **Submission candidata** (`15_submission_final`) → `submission/submission_catboost.csv`. Formato OK; 52,9% `new` predicho; 588 predicciones (2,0%) distintas de la de la Base. (Después se subió: público 0,93695.)

### Fase D — Ronda corta post-congelamiento (contra CatBoost P6b = 0,9365)

**Parte A — Métodos de pago (`laboratorio.ipynb`).**
- EDA de `non_mercado_pago_payment_methods`: tarjeta de crédito (25% de las filas, 71% `new` contra 48% sin), transferencia (51%, +15 puntos), efectivo (67%, +5), acordar con el comprador (8%, −9), giro postal (1,7%, −32). Las demás son < 1%. Train y test tienen las mismas proporciones.
- `shipping` no tiene nada nuevo: `free_methods` ≈ `free_shipping` (coinciden salvo 9 filas); `tags`, `dimensions` y `methods` casi siempre vacíos.
- **PA: 5 flags de método de pago** (dummies, `AP_Categóricas.ipynb`) sobre la configuración final: 0,9368, 3/5 folds, media +0,0002 → **ruido. No se toca nada.** Lectura: es la misma distinción tienda/particular que ya captura `seller_id`.

**Parte B — Ensambles (`16_ensambles.ipynb`).**
- **Probabilidades out-of-fold** de 6 modelos, calculadas una vez y guardadas en `trabajo_realizado/.cache_laboratorio/oof_ensambles.csv`:

  | Modelo | Accuracy | Correlación con CatBoost | % desacuerdo con CatBoost |
  |---|---|---|---|
  | CatBoost P6b | 0,9365 | 1 | — |
  | HistGB (Base) | 0,9339 | 0,991 | 2,0% |
  | Random Forest (`train.ipynb`: 200 árboles, prof. 20) | 0,9264 | 0,974 | 2,8% |
  | LDA (13 parcial: `lsqr`) | 0,9227 | 0,973 | 3,7% |
  | Regresión logística (13 parcial: L1, C = 1) | 0,9221 | 0,977 | 3,7% |
  | kNN (13 parcial: k = 15, distancia) | 0,8963 | 0,917 | 7,6% |

- **Ensambles.** Todo lo que se elige (pesos, meta-modelo) se elige con 4 folds y se evalúa en el quinto, rotando. Comparación pareada contra CatBoost solo:

  | Ensamble | Accuracy | Folds vs CatBoost |
  |---|---|---|
  | (a) Promedio 50/50 CatBoost + HistGB | 0,9359 | **pierde 5/5** |
  | (b) CatBoost ponderado con HistGB / RF / logística / LDA / kNN | 0,9363–0,9366 | ruido (el w elegido para CatBoost queda en 0,8–1) |
  | (c1) Hill Climber literal (slide 75, con `softmax`) | 0,9315 | pierde 5/5 |
  | (c2) Hill Climber con normalización simple | 0,9322 | pierde 5/5 |
  | (d) Stacking (meta-modelo: regresión logística) | 0,9364 | 2/5 |

- **Hallazgo sobre la slide 75:** el pseudocódigo normaliza los pesos con `softmax`, que los aplasta hacia "todos iguales" (con 6 modelos, un peso no puede pasar de 0,168). Tal como está escrito, el Hill Climber es casi un promedio simple. La versión con normalización simple tampoco se aleja mucho: arranca uniforme y, con pasos de 0,01 sobre la accuracy (una métrica "escalonada"), se traba en un óptimo local.
- **Conclusión: ningún ensamble mejora. El modelo final sigue siendo CatBoost P6b solo.** Los otros modelos o son casi iguales a CatBoost (HistGB) o bastante peores. Es el caso de `S8_Ensambles.ipynb`: con un modelo dominante, votar diluye su performance.

### Fase E — Ronda de robustez sobre CatBoost (C1–C5, `laboratorio.ipynb`)

Contexto: con `submission_catboost.csv`, quedé **primero en el leaderboard público**. Objetivo: robustez y alguna mejora chica. Todo contra P6b (0,9365), con comparación pareada.

| Prueba | Qué | Accuracy | Folds vs P6b | Tiempo | Veredicto |
|---|---|---|---|---|---|
| C1 | Promedio de 5 CatBoost (semillas 42–46) | 0,9367 ± 0,0012 | 3/5 | 6,4 min CV + 3 min test | No pasa |
| C2 | Tuning: `RandomizedSearchCV`, 15 combinaciones de `depth` {4,6,8} × `learning_rate` {0,03, 0,06, 0,1} × `l2_leaf_reg` {1,3,10}, con *early stopping* sobre el 10% del train del fold | mejor: 0,9364 (depth 8, lr 0,03, l2 3) | 2/5 | 54 min | No pasa: los valores por defecto ganan |
| C3 | + `currency_id`, `status`, `official_store_id`, `tags` como categóricas crudas | 0,9360 | **0/5** | 2,0 min | Empeora |
| C4a | Sin TF-IDF 500 | 0,9365 ± 0,0011 | 2/5 | 1,0 min | Igual: el TF-IDF es **redundante** con `prob_texto_new` |
| C4b1 / C4b2 | Título como `text_features` de CatBoost (con / sin TF-IDF 500) | 0,9370 / 0,9368 | 3/5 | 4,4 / 3,5 min | No pasa |
| C5 | + `seller_avg_log_price` + conteo del vendedor en train + test | 0,9366 | 3/5 | 1,8 min | No pasa |

- **Azar de CatBoost** (C1): `bootstrap_type="MVS"` con `subsample=0.8` (cada árbol usa el 80% de las filas), `random_strength=1` (ruido en los cortes) y permutaciones de los *ordered target statistics*. Actúa en cada árbol, a diferencia del HistGB (solo el 10% del *early stopping*).
- **Robustez confirmada:**
  - las semillas sueltas dan 0,9364–0,9366 (la 42 no fue suerte);
  - el promedio de semillas baja un poco el desvío entre folds (0,0017 → 0,0012) pero no la media;
  - entre dos semillas cambia el **0,73% de las predicciones del test (~218 filas)**, así que diferencias de ~0,002–0,004 en el leaderboard pueden ser solo azar.
- **Tuning (C2):** superficie plana, como con el HistGB. El *early stopping* cuesta ~0,001, porque entrena con el 90% del train, y ninguna combinación lo recupera. Coherente con la slide 68 ("funciona muy bien… con los parámetros de default"). Detalle técnico: el envoltorio `CatBoostConEarlyStopping` tiene que heredar `ClassifierMixin` antes que `BaseEstimator` para que sklearn 1.9 lo reconozca como clasificador.
- **`text_features` de CatBoost** (C4b, no está en el material): por defecto tokeniza por espacios sin pasar a minúsculas, arma diccionarios de palabras y pares, y calcula BoW + Naive Bayes.
- **Resultado: nada pasó. La configuración final no cambia** (sin cambios en `features.py`, sin `submission_catboost_v2.csv`).
- **Simplificación posible, no aplicada:** sacar el TF-IDF 500 (C4a) da la misma accuracy con un pipeline más simple. Requeriría revalidar y regenerar la submission.

### Fase F — Combinaciones de categóricas (ronda P7, `laboratorio.ipynb`)

`agregar_combos` (código propio, en `features.py`): 19 columnas `c_*`. 13 son combinaciones de categóricas como texto (vendedor × categoría, vendedor × tipo de publicación, categoría × bin de precio, vendedor × categoría × tipo, …) y 6 son categóricas sueltas (moneda, status, tienda oficial, tags, métodos de pago, ids de atributos). No usa target y va después de `aplanar_anidadas`. **Idea:** un vendedor que vende celulares nuevos y ropa usada tiene 50% `new` en total, pero 100% / 0% según la categoría.

| Prueba | CV semilla 42 | folds vs P6b | CV semilla 7 | Tiempo |
|---|---|---|---|---|
| CatBoost P6b (referencia) | 0,9365 | — | 0,9359 | 1,6 min |
| P7a: HistGB + `TargetEncoder`(`seller_id`, `category_id`, 19 `c_*`) + `VendedorEncoder` sin target | 0,9371 | 4/5, no pasa (pero +0,0032 y 5/5 contra la Base HistGB) | 0,9364 | 0,7 min |
| P7b: CatBoost P6b + 19 `c_*` como categóricas crudas | 0,9366 | 2/5, no pasa | — | 3,6 min |
| P7c ponderado (w en 4 folds) | 0,9376 | 5/5 | 0,9369 (4/5) | — |
| **P7c: promedio 50/50 CatBoost P6b + HistGB P7a** | **0,9376** | **5/5, +0,0011** | **0,9373 (5/5, +0,0013)** | — |

- Las combinaciones le sirven mucho al HistGB (le dan las interacciones que CatBoost arma solo) y nada a CatBoost.
- **P7c 50/50 es la primera mejora sobre P6b**, y robusta: 10 de 10 folds entre las dos semillas. La ganancia es chica (+0,001). Correlación entre los modelos: 0,993. La diferencia con la `16_ensambles` es que ahora los dos son igual de buenos.
- **Submission (sin subir):** `submission/submission_catboost_histgb.csv` (`trabajo_realizado/17_submission_p7c.ipynb`): 266 predicciones (0,9%) distintas de `submission_catboost.csv`.
- **Decisión: P7c 50/50 es el modelo final.** Por eso `train.ipynb` y `predict.ipynb` tienen que entrenar y promediar los dos modelos, y el pipeline del HistGB P7a tiene que pasar a `features.py`.

### Fase G — Embeddings y URL de la miniatura (ronda P8–P9, `laboratorio.ipynb`)

Referencia: **P7c 50/50 = 0,9376** (semilla 7: 0,9373). Cada feature se midió primero en el HistGB P7a (0,9371).

**P9 — Embeddings del título** (redes neuronales y embeddings **no** están en `material/`).
- **Qué son:** `paraphrase-multilingual-MiniLM-L12-v2` es una red neuronal pre-entrenada (por otros, con millones de pares de frases en más de 50 idiomas) que convierte cada frase en 384 números, de modo que frases con **significado** parecido den vectores parecidos. No la entrenamos: solo leemos su salida (*transfer learning*). Se instalaron `torch` y `sentence-transformers` (en `requirements.txt`).
- **No es leakage:** la red nunca vio nuestras etiquetas, y el embedding de cada título depende solo de ese título, así que se calcula una vez, afuera del `Pipeline`. Lo que aprende de las etiquetas es la regresión logística de arriba, dentro de `PrediccionOOF`.
- Embeddings de los 100.000 títulos en la GPU del M4 (`device="mps"`): **0,6 minutos**. Guardados en `trabajo_realizado/.cache_laboratorio/emb_minilm_{train,test}.npy`.

| Prueba (sobre HistGB P7a) | Accuracy | Folds vs P7a |
|---|---|---|
| P9a: + `prob_embed_new` (regresión logística sobre embeddings, `PrediccionOOF`) | 0,9367 | 2/5, −0,0004 |
| P9b: texto = TF-IDF + embeddings (reemplaza `prob_texto_new`) | 0,9361 | 1/5, −0,0010 |
| P9c: 384 embeddings como columnas | 0,9358 | **0/5**, −0,0013 |

- **Diagnóstico:** una regresión logística sobre los embeddings **sola** da **0,7247**, contra 0,8227 sobre TF-IDF del título (0,8414 con la garantía).
- **Por qué (ejemplo para la presentación).** Similitud coseno entre títulos ilustrativos:

  | | iPhone nuevo | iPhone usado | Nike nuevas | Nike usadas |
  |---|---|---|---|---|
  | "Iphone 5 16gb Nuevo En Caja Sellado" | 1,00 | **0,82** | 0,25 | 0,13 |
  | "Iphone 5 16gb Usado Con Detalles" | 0,82 | 1,00 | 0,11 | 0,19 |
  | "Zapatillas Nike Air Nuevas En Caja" | 0,25 | 0,11 | 1,00 | **0,85** |
  | "Zapatillas Nike Air Usadas Con Detalles" | 0,13 | 0,19 | 0,85 | 1,00 |

  **El embedding agrupa por producto, no por condición:** "iPhone nuevo" y "iPhone usado" son casi el mismo vector (0,82), mientras que "iPhone nuevo" y "Nike nuevas" quedan lejos (0,25). Justo la palabra que decide la etiqueta ("nuevo"/"usado") casi no mueve el vector. Con títulos reales pasa lo mismo: los vecinos más parecidos de "Pizarron 20x30 + Imagen + Bolsita Artesanal" fueron "Cubre Optica Zanella" (0,67) y "Album Figuritas Bandana…" (0,63), títulos sin relación con la condición. El TF-IDF, en cambio, ve directamente "nuevo", "usado", "sin uso".
- **No probé modelos más grandes** (mpnet, e5): el problema es de fondo (embeddings de paráfrasis), no de tamaño.

**P8 — URL de la miniatura** (las fotos ya no existen: 29 de 30 URLs redirigen a "imagen no disponible"; se usa el texto de la URL).
- Regex `/(\d+)-(MLA\d+)_(\d+)-`: matchea el 99% de las URLs. El sufijo de 6 dígitos es MMYYYY (meses 1–12, años 2011–2015). Los sufijos de 4 dígitos (6,7%) tienen **36% `new`** contra 55%. Por año de la foto: 2012 42%, 2014–2015 57%.
- La flag "foto de otra publicación" quedó **siempre en 0**: el MLA de la URL coincide siempre con el `id`. El ~1% con ~31% `new` son las URLs que no matchean (publicaciones sin fotos), ya cubiertas por `n_fotos = 0`.
- **P8** (largo del sufijo, mes y año de la foto, meses hasta `start_time`, flag) sobre HistGB P7a: **0,9371, 2/5, media 0,0000 → no suma.**

**Cierre de la ronda: nada pasó. Modelo final = P7c 50/50 (CatBoost P6b + HistGB P7a).**

### Fase H — Redes neuronales (ronda N, `laboratorio.ipynb`)

**Material.** Fuente principal: **AP**, `material/redes predictivo/Intro a Redes para datos estructurados.pdf`:
- neurona z = w·x + b, y = f(z) (slide 5); red *fully connected* (6–7); backpropagation (8);
- *cross-entropy* (9); descenso por gradiente, mini-batch y épocas (10–14); *learning rate* (16);
- **Adam** (19–21: *"para la mayoría de nuestros casos, usaremos Adam"*; `SparseAdam` para tensores sparse);
- normalización (23); dropout (24); **TabNet** (26–32);
- **slide 3: en clasificación con datos tabulares, XGBoost, HistGradientBoosting y Random Forest les ganan a las redes**.

Complemento: **SIA**, `material/redes neuronales/presentacion redes neuronales.pdf` y `S10_Redes_Neuronales_MLP.ipynb` (Keras).

**TabNet en una línea:** red para datos tabulares que, en cada paso de decisión, arma una máscara de atención que elige qué features mirar (parecido a un árbol que elige variables, pero con atención "blanda" aprendida por gradiente). Hiperparámetros: `n_d = n_a`, `n_steps`, `gamma` (reuso de features entre pasos, 1–2), `lambda_sparse` (máscaras ralas), Adam, `max_epochs` / `patience` / `batch_size`.

**Instalaciones:**
- `torch` (ya estaba): MPS del M4 funciona.
- `pytorch-tabnet`: instalado para esta ronda.
- `tensorflow` 2.21 + Keras: en CPU, porque `tensorflow-metal` no es compatible y se desinstaló.

Todo quedó en `requirements.txt` (no lo necesita el modelo final).

**Primera versión (SIA, Keras / sklearn):**
- MLP de texto con `MLPClassifier` sobre el TF-IDF (238.008 columnas, ~15 millones de parámetros): **no terminó** en el límite de 2 horas. Se lanzó sin medir antes un ajuste (error).
- MLP de Keras como tercer modelo: 0,9302 solo; con el ensamble no mejora (peso 0–5%).

**Segunda versión (AP, PyTorch).** Antes de cada corrida se midió un ajuste en un fold.
- **N1, MLP de texto en PyTorch.** La capa oculta es un `nn.EmbeddingBag` en modo suma (capa densa sobre las columnas no nulas del TF-IDF), entrenada con `SparseAdam` + Adam, dropout 0,3 y *early stopping*. Tarda 0,4 min por ajuste; MPS y CPU tardan lo mismo, así que se usó CPU para paralelizar los folds.

| Prueba | Accuracy | Comparación | Tiempo |
|---|---|---|---|
| MLP de texto solo | 0,8179 | regresión logística sobre el mismo TF-IDF: 0,8414 | 1,1 min |
| N1a: en lugar de `prob_texto_new` (HistGB P7a) | 0,9348 | **0/5** vs P7a | 4,2 min |
| N1b: como feature extra (HistGB P7a) | 0,9367 | 2/5 vs P7a | 4,4 min |
| N2: TabNet solo (`n_d = n_a = 16`, `n_steps = 4`, `gamma = 1,5`, `lambda_sparse = 1e-4`, Adam 0,02, `patience = 10`) | 0,9254 | 0/5 vs P7c | 9,7 min (MPS; un fold: 2,1 min MPS vs 3,5 CPU) |
| CatBoost + HistGB + TabNet, pesos iguales / elegidos en 4 folds | 0,9366 / 0,9376 | 0/5 / 3/5 (TabNet recibe 0–5%) | — |
| N3: MLP PyTorch solo (128–64, dropout 0,3, Adam) | 0,9313 | 0/5 vs P7c | 0,3 min |
| CatBoost + HistGB + MLP, pesos iguales / elegidos en 4 folds | 0,9375 / 0,9377 | 1/5 / 3/5 (MLP recibe 0–15%) | — |

- Sobre el mismo TF-IDF, el MLP rinde **menos** que la regresión logística: con 15 millones de parámetros y 56.000 filas sobreajusta antes de aprender combinaciones útiles.
- Como terceros modelos del ensamble, las redes quedan 0,6–1,2 puntos por debajo de los árboles y reciben pesos de 0–15%, sin mejorar a P7c.
- **Resultado: nada pasó. El modelo final sigue siendo P7c 50/50.** Coherente con la slide 3 del PDF de AP. Buen argumento para la "selección de modelos" de la presentación.

### Fase I — Ronda final de ideas (I1–I4, `laboratorio.ipynb`)

Contra P7c 50/50 (0,9376); las ideas de features se midieron primero en el HistGB P7a (0,9371).

| Prueba | Qué | Accuracy | Folds | Tiempo |
|---|---|---|---|---|
| I1a | kNN de títulos: % de `new` entre los 10 / 30 títulos más parecidos (coseno sobre TF-IDF) del train, con *cross-fitting* interno. Material: kNN para regresión (`material/Knn/Copia de AP_kNN.ipynb`); el coseno da los mismos vecinos que la distancia euclídea del material porque el TF-IDF tiene largo 1 | 0,9365 | 0/5 vs P7a | 1,9 min |
| I1b | Ídem, solo vecinos de la misma `category_id` | 0,9365 | 1/5 vs P7a | 1,9 min |
| I2 | Stacking con contexto: meta-modelo sobre las 2 probabilidades OOF + log publicaciones del vendedor, garantía, log precio, tipo de publicación y tienda oficial, entrenado en 4 folds y evaluado en el quinto. Regresión logística / HistGB chico | 0,9373 / 0,9377 | 1/5 / 3/5 vs P7c | segundos |
| I3 | Pseudo-labeling: se suman al train las filas del fold y del test con probabilidad > 0,97 o < 0,03, con la etiqueta predicha (~9.000 + ~19.000 por fold; pseudo-etiquetas correctas en el 99,3–99,5%) | 0,9365 | 1/5 vs P7a | 6,5 min |
| I4 | Especialista: HistGB P7a entrenado solo con vendedores de ≤ 1 publicación, promediado con P7c en ese grupo (peso elegido en 4 folds: 0,1–0,2) | 0,9378 | 2/5 vs P7c | 0,7 min |

- **I1:** el "voto" de los títulos parecidos ya lo resume mejor el modelo de texto (`prob_texto_new`).
- **I2:** los dos modelos se equivocan en publicaciones parecidas, así que el contexto no ayuda a decidir en cuál confiar.
- **I3:** las pseudo-etiquetas son casi siempre correctas justamente porque son los casos fáciles, y no enseñan nada nuevo. En vendedores sin historial sube en 3 folds y baja en 2.
- **I4:** el especialista solo es **peor** que P7c en su propio grupo (0,8825 contra 0,8967): entrenar solo con vendedores sin historial le quita los ejemplos de todos los demás.
- **Resultado: nada pasó. Modelo final: P7c 50/50.** No hubo combinación, robustez ni submission nueva.

### Fase J — Última prueba: modelo en cascada (*two-stage*, `laboratorio.ipynb`)

```
publicación → Etapa 1: P7c → P(new)
   ├─ seguro (P < 0,2 ó > 0,8) → queda P7c
   └─ duda (0,2 ≤ P ≤ 0,8)     → Etapa 2: especialista entrenado SOLO con la zona de duda
                                  (a) HistGB sobre features P7a + P(new) de P7c
                                  (b) regresión logística sobre TF-IDF grande + logit de P(new) de P7c
                                  → reemplaza a P7c, o se promedia con él
```

**Por qué no es leakage:**
- la zona se define con las probabilidades OOF de P7c;
- el especialista del fold k se entrena solo con la zona de los otros 4 folds, con preprocesamiento ajustado en esos folds;
- umbrales, especialista y modo se eligen con 4 folds y se evalúan en el quinto.

Sesgo chico conocido: las OOF de P7c de las filas de entrenamiento vienen de modelos que vieron el quinto fold.

**Material:** umbral de decisión (S8 de SIA; ROC en `Repaso de Metricas.pdf`); regresión logística; Gradient Boosting. La cascada no está en el material. Sin oversampling ni SMOTE.

| | Zona 0,2/0,8 | Zona 0,3/0,7 |
|---|---|---|
| Filas | 7.679 (11,0%) | 4.492 (6,4%) |
| % `new` (balanceada, sin SMOTE) | 46,6% | 48,0% |
| Accuracy de P7c | 0,6567 | 0,6013 |
| % de los errores de P7c | 60,3% | 41,0% |
| Especialista (a) HistGB | 0,6317 | 0,5628 |
| Especialista (b) regresión logística + TF-IDF | 0,6197 | 0,5715 |

- **Resultado:** cascada con configuración elegida en 4 folds = **0,9364**, 1/5 contra P7c (−0,0012), 2,8 min. **No pasa.**
- **Dato útil para la presentación:** fuera de la zona de duda, P7c acierta el **97,2%**; el 60% de sus errores está en el 11% de publicaciones donde duda.
- **Por qué no funciona:** los especialistas son **peores** que P7c en su propia zona. Con solo ~6.000 casos difíciles, sin los fáciles, no encuentran un patrón nuevo: en esas filas la información es genuinamente ambigua (incluye el ruido de etiqueta de `08_errores`).
- **Cierre definitivo de experimentos: modelo final P7c 50/50.**

### Fase K — Barrido sistemático de campos (`18_barrido_campos.ipynb`)

Motivación: dos personas del leaderboard llegan a ~0,945 contra nuestro 0,937.

**Qué se hizo:**
- Inventario de los 82 campos y sub-campos del jsonlines (tipo, % nulos, valores distintos, uso actual).
- 28 campos no usados (o usados solo como conteo/flag) medidos uno por uno: señal sola (HistGB con solo esa feature; `TargetEncoder` si es categórica) y aporte marginal sobre el HistGB P7a con comparación pareada.

**Resultado: un solo campo aporta, y mucho: `duracion_dias` = `stop_time` − `start_time`.**
- Sobre el HistGB P7a: +0,0041, 5/5.
- Confirmado en los dos modelos y en el promedio:

  | | Semilla 42 | Semilla 7 |
  |---|---|---|
  | P7c (referencia) | 0,9376 | 0,9373 |
  | CatBoost + duración | 0,9414 | 0,9406 |
  | HistGB P7a + duración | 0,9412 | 0,9412 |
  | **P7c + duración (promedio 50/50)** | **0,9420** (5/5, +0,0044) | **0,9422** (5/5, +0,0049) |

- **La señal está en los segundos:**
  - 66.143 publicaciones duran 60 días exactos (55% `new`);
  - 1.343 duran 60 días + 3–117 segundos (**2,5% `new`**; 60,0001 días: 807 filas, 0,5% `new`);
  - casi ninguna de estas últimas tiene `parent_item_id` (1% contra 80%): son publicaciones creadas por primera vez, casi todas `bronze`/`silver` y `buy_it_now`.

  **Hipótesis (no verificada):** artefacto de cómo MercadoLibre calcula el vencimiento de las publicaciones nuevas (las republicadas copian 60 días justos).
- **Sola casi no predice (0,563)**, porque afecta a ~2% de las filas. Pero son filas difíciles que ahora acierta casi siempre.
- **No es leakage:** el campo está en el test con la misma distribución (5,8% no exactas contra 5,5% en train) y no usa el target. Es un artefacto de la plataforma, no del producto: para un sistema real habría que verificar que se genere igual.
- `06_eda_nuevas` había descartado la duración porque miró días enteros (96,7% "60 días"). **Lección: la señal estaba en los segundos.**
- **Ningún campo tiene señal sola > 0,70.** Los mejores solos (`ciudad` 0,632, `fotos_size_igual_max` 0,632) no aportan sobre P7a.
- El resto (ciudad, `sub_status`, `deal_ids`, envío, tipos de pago, variaciones, valores de atributos, números de `id`/`parent`/descripción, días entre creación e inicio, descuento, tipo de `permalink`, `video_id`) queda en el ruido o empeora.
- **Corrección metodológica:** 4 campos figuraban como "pasa" con diferencia +0,0000 por un artefacto de redondeo (no cambiaban ninguna predicción). Corregido.
- **Submission (sin subir):** `submission/submission_p7c_duracion.csv` (`trabajo_realizado/19_submission_duracion.ipynb`), 322 predicciones (1,1%) distintas de P7c.
- **Adoptado como modelo final:** `features.py` calcula `duracion_dias` en `agregar_features_contenido` y la suma a los dos pipelines en la misma posición en la que se midió (CatBoost: al final de las numéricas, `INDICES_CAT` = 526–531; HistGB: última columna). `train.ipynb` y `predict.ipynb` regenerados: **la submission es idéntica byte a byte a `submission_p7c_duracion.csv`**.

### Fase L — Tuning de CatBoost con Optuna y SHAP (`20_tuning_catboost_optuna.ipynb`, `21_shap.ipynb`)

**Tuning con Optuna (TPE).** TPE no está en el material: aprende de las pruebas anteriores, separándolas en "buenas" y "malas" y proponiendo valores frecuentes entre las buenas.
- Espacio: `depth` 4–8, `learning_rate` 0,02–0,2, `l2_leaf_reg` 1–30, `random_strength` 0,1–10; *early stopping* sobre el 10% del train de cada fold.
- **24 pruebas** (de 35 previstas): la máquina, con 18 GB de RAM, se saturó con 5 CatBoost en paralelo.
- Mejor (prueba 18): `depth` 7, `learning_rate` 0,021, `l2_leaf_reg` 3,0, `random_strength` 0,11 → 0,9411 con *early stopping*, contra 0,9406 del default en las mismas condiciones.
- **Confirmación** reentrenando con el 100% del train del fold (3.300 árboles fijos):

  | | Semilla 42 | Semilla 7 |
  |---|---|---|
  | Tuneado vs default | 0,9417 vs 0,9414 (4/5, +0,0003 < desvío) | 0,9408 vs 0,9406 (3/5) |
  | P7c con tuneado vs P7c | 0,9420 vs 0,9420 (3/5) | 0,9423 vs 0,9422 (3/5) |

  **No pasa. Se mantiene CatBoost por defecto** (como en C2 y la slide 68). La confirmación tardó ~36 h por falta de memoria; con la máquina libre son ~30 min.

**SHAP** (`trabajo_realizado/figuras/`). Cuando se hizo no estaba en el material; **después apareció `material/SHAP_Values.ipynb`** (presentación de tema del Examen 2: SHAP sobre regresión lineal y Random Forest con el dataset de diabetes), que sirve para citarlo. Modelos entrenados con los folds 2–5 y explicados en el fold 1 (CatBoost 0,9415, HistGB 0,9428, promedio 0,9431).
- **CatBoost**, SHAP nativo (`shap_global_catboost.png`): `listing_type_id` > `prob_texto_new` > `seller_id` > cantidades > `category_id` > `tiene_parent_item` > TF-IDF > precio > fotos. `duracion_dias` en el puesto 13: pocas filas, pero con mucho peso. Palabras con más peso: *nuevo, estado, nueva, sin, impecable*.
- **HistGB**, `shap.TreeExplainer` (`shap_global_histgb.png`): `prob_texto_new` > `listing_type_id = free` > TE de vendedor × tipo de publicación > TF-IDF > cantidades > combinaciones del vendedor.
- **Ejemplos** (`shap_ejemplo_acierto.png`, `shap_ejemplo_error.png`, `shap_ejemplo_duracion.png`):
  - acierto: repuesto VW Gol de un vendedor con 405 publicaciones;
  - error: "Bisel Marco Frontal… LG G Pro", etiquetado `used` con P(new) = 0,999 (todo indica comercio de repuestos nuevos: probable etiqueta discutible);
  - duración: libro `used` donde lo que más empuja hacia `used` es `duracion_dias` = 60,00009 (en la imagen guardada la etiqueta muestra "60" por redondeo; el código ya está corregido).
- Nota: inicialmente se creyó que `shap` se trababa con el HistGB; en realidad la máquina estaba saturada.

**CatBoost en el material:** `material/Arboles/Árboles.pdf` (slide 68: "Gradient Boosting preparado para variables categóricas… funciona muy bien con variables categóricas sin transformar y con los parámetros de default"), `material/Arboles/Modelos.ipynb` y el `Catboost.ipynb` del profe (github.com/LCaravaggio/AnalisisPredictivo, 04_Árboles), que usa `CatBoostClassifier()` por defecto. El mecanismo de *ordered target statistics* **no** está explicado en el material.

---

### Fase M — Barrido fino de timestamps (`22_barrido_timestamps.ipynb`, semana 1)

Motivación: la duración funcionó porque la señal estaba en los **segundos**, y las demás fechas se habían mirado en días. Se barrieron `start_time`, `stop_time`, `date_created` y `last_updated` al segundo y al milisegundo.

**Formato (confirmado):** `start_time`/`stop_time` vienen como enteros en milisegundos desde 1970; `date_created`/`last_updated` como texto ISO (`…T19:19:14.000Z`). Se leyeron directo del jsonlines y se pasaron a milisegundos enteros, así que las diferencias son exactas. Componentes en hora argentina (UTC−3).

**EDA (descriptivo):**
- **Diferencias y sobrantes respecto de días redondos (6 pares):** lo único que separa fuerte son los sobrantes de +5 a +11 s en inicio→fin / creada→fin (0-2% `new`), que son las mismas filas de `duracion_dias`. `date_created` = `start_time` (80%) o 1 s después (19%), sin diferencia de `new`.
- **Componentes:** la hora va de 40% a 61% de `new` (ya está como `hora_inicio`); el día del mes de `last_updated` de 44% a 62% (efecto de calendario, ligado a `gap_actualizacion`). Segundos y minutos están dentro de lo esperable por azar.
- **Milisegundos y redondos:** `start_time` y `date_created` nunca tienen milisegundos. **`last_updated` con ms ≠ 000: 4,8% de las filas, 66% `new`**, pero es casi todo "se actualizó días después" (en el tramo de 1 a 30 días: 64% contra 64%). Segundos = 0 aparece en ~1,7%, lo que da el azar.
- **Coincidencias exactas:** diferencias de 5-12 puntos en grupos chicos.
- **Cargas por vendedor** (train + test juntos, sin target): a 1-10 s de otra publicación del vendedor → 76% `new`; más de 100 en la misma hora → 100% `new` (536 filas). Entre los vendedores con más de 50 publicaciones, a 1-10 s → **80% `new` contra 32-44%** del resto del mismo grupo (señal *dentro* del vendedor).
  - **No es leakage:** usa solo `seller_id` y `start_time`, que están en el test, nunca `condition`. Se calcula sobre train + test para que las dos se describan igual. Limitación transductiva, como V2: en producción no existiría la "siguiente".
  - **Bug encontrado y corregido:** al ordenar filas con empates (varias en el mismo segundo), la "anterior" dependía de la posición en el archivo, y "0 s desde la anterior" daba 14,6% en train contra 30% en test. Se redefinió sobre instantes **distintos** (estrictamente anterior/posterior) + `n_mismo_segundo`. Después del arreglo, train y test tienen la misma distribución.

**Medición** (igual que la 18, contra el HistGB P7a con duración = 0,9412, recalculado e idéntico al de la 18): 52 candidatas sueltas + 5 familias como bloque.

| | Aporte sobre P7a + duración |
|---|---|
| Mejores sueltas | `actualizada_hora` +0,0005 (3/5, desvío 0,0009); sobrante actualizada→fin +0,0003 (3/5); `seg_desde_anterior` +0,0002 (2/5) |
| A: diferencias y sobrantes (10) | −0,0007 (1/5) |
| B: componentes (19) | −0,0006 (2/5) |
| C: ms y redondos (10) | +0,0002 (3/5) |
| D: coincidencias (7) | −0,0002 (2/5) |
| E: cargas por vendedor (6) | −0,0000 (3/5) |

- **Nada pasa.** Señal sola: 0,536-0,573. 15 de 57 con media positiva y 39 negativa: es lo esperable al sumar columnas sin información nueva.
- **Lectura:** la información fuerte de las fechas ya estaba entera en `duracion_dias`. Las ráfagas por vendedor no suman sobre P7a. **Hipótesis (no verificada):** las combinaciones vendedor × categoría / tipo / precio ya separan las líneas de producto de cada tienda.
- **Sin cambios:** no hubo confirmación con la semilla 7 (no hubo candidatas), ni submission nueva, ni cambios en `features.py`. Tiempo: 55 min (debería ser ~30; algunas candidatas tardaron 4-5 min por carga de la máquina).

### Fase N — LightGBM (`23_lightgbm.ipynb`, semana 1)

**Material:** LightGBM solo aparece **nombrado** (`Árboles.pdf`, slide 65, el logo, entre XGBoost y Gradient Boosting / Histogram Gradient Boosting). Explicado desde cero en la notebook: Gradient Boosting con histogramas como el HistGB (slide 67), con categóricas nativas de alta cardinalidad (`min_data_per_group`, `cat_smooth`, `max_cat_threshold`).

| Prueba | Accuracy | Contra | Diferencia (folds) |
|---|---|---|---|
| L1: `LGBMClassifier` por defecto con el pipeline del HistGB P7a + duración | 0,9409 | HistGB P7a + duración (0,9412) | −0,0004 (1/5) |
| L2a: entradas del CatBoost P6b + duración, 6 categóricas nativas (`OrdinalEncoder` en el fold), por defecto 100/10 | 0,9324 | ídem | −0,0089 (0/5) |
| L2b / L2c: más suavizado (200/50, 500/100) | 0,9259 / 0,9229 | ídem | 0/5 |
| L2d / L2e: menos suavizado (20/1; 5/1 + `max_cat_threshold` 256) | 0,9320 / 0,9263 | ídem | 0/5 |
| CatBoost + HistGB + L1, pesos iguales | 0,9419 | modelo final (0,9420) | −0,0001 (1/5) |
| Ídem, pesos elegidos con 4 folds | 0,9417 | modelo final | −0,0003 (2/5); L1 recibe peso 0 en 4 de 5 folds |

- **L1 ≈ HistGB** (correlación 0,998): misma familia, no aporta diversidad.
- **Categóricas nativas: mucho peor** que el target encoding con *cross-fitting* (HistGB) y que las *ordered target statistics* (CatBoost, 0,9414 con las mismas entradas). Suavizar en cualquier sentido no lo arregla. **Interpretación (no verificada):** LightGBM ordena los ~28.000 vendedores en cada corte con el target de las mismas filas que entrena, así que memoriza los vendedores chicos; el `TargetEncoder` y CatBoost codifican cada fila sin su propia etiqueta. **Para la presentación: cómo se codifica `seller_id` importa más que qué boosting se use.**
- Tiempo: ~0,3 min por prueba. Sin cambios, sin submission. OOF en `.cache_laboratorio/oof_lightgbm.csv`.

### Fase O — Features derivadas (`24_features_derivadas.ipynb`)

*Feature engineering* (`Copia de AP_Categóricas.ipynb`, sección "Feature Engineering": la superficie por distrito es el análogo del precio por categoría). Medición como la 18 y la 22, contra HistGB P7a + duración (0,9412).

- **D1, cantidades relativas** (determinísticas, afuera del `Pipeline`): `ratio_vendido`, `ratio_disponible`, `vendio_algo`, `diferencia_inicial_disponible`.
  - Chequeo: `initial_quantity` nunca es 0 (train ni test); igual se dejó NaN si fuera 0. `sold_quantity` > `initial_quantity` en el 6% (suposición: `sold` acumula ventas de publicaciones anteriores).
  - EDA: cualquier señal de venta da 74-99% `new`, pero el 83% no vendió nada. `vendio_algo` = 1 (4,5%) es un subconjunto de `sold_quantity` > 0 (96,8% contra 86,1% `new`).
- **D2, precio relativo a la categoría** (`PrecioRelativoGrupo`, adentro del `Pipeline`): log_price − mediana de la categoría, aprendida en el fold; menos de 5 filas → mediana global. Variantes: (a) `category_id`; (b) `category_id` × moneda. Va adentro aunque no use el target porque aprende una mediana: afuera, las filas de validación ayudarían a calcular la mediana de su propia categoría, y el test no.
- **D3, precio relativo al vendedor** (`PrecioRelativoVendedor`): log_price − promedio de las **otras** publicaciones del vendedor. En `fit_transform` excluye la propia fila, y deja NaN si el vendedor no tiene otras; en `transform`, el promedio de todas las del fit, y NaN si es un vendedor no visto. Así un vendedor con 1 publicación (train) y uno nuevo (test) se ven igual (NaN, no 0).

| Candidata | Señal sola | Con P7a | Diferencia (folds) |
|---|---|---|---|
| D1a / D1b / D1c / D1d | 0,596 / 0,537 / 0,537 / 0,537 | 0,9411 / 0,9412 / 0,9409 / 0,9406 | −0,0001 (3/5) / +0,0000 (2/5) / −0,0003 (0/5) / −0,0007 (0/5) |
| D1 juntas | 0,596 | 0,9412 | +0,0000 (2/5) |
| D2a / D2b | 0,600 / 0,601 | 0,9410 / 0,9408 | −0,0003 (2/5) / −0,0005 (1/5) |
| D3 | 0,558 | 0,9412 | −0,0000 (3/5) |

- **Nada pasa**, así que no hubo combinación, *ablation* ni confirmación. ~0,8 min por candidata.
- **Lectura:** D1 ya está en `sold_quantity`. D2, en `log_price` + `category_id` + `c_cat_price`. D3, en el `VendedorEncoder` sin target + `c_seller_price`.
- **Nota:** la "señal sola" de 0,537 (D1b-d) es el % de la clase mayoritaria. Con una sola feature binaria, el árbol predice `new` de los dos lados del corte (96% y 52%), así que no cambia ninguna predicción aunque la feature tenga información.
- Sin cambios en `features.py` ni submission.

### Fase P — Ronda final de ideas de otra IA (`25_ideas_finales.ipynb`) — **cierre de experimentos**

Ideas surgidas al pasarle a otra IA un resumen con el contexto completo del TP. Se midieron contra el HistGB P7a + duración (0,9412), con comparación pareada. Pieza común: predicciones OOF **internas** (CV de 5 dentro del train de cada fold externo, HistGB P7a + duración, 25 ajustes, 4,5 min), guardadas en `oof_interna_p7a_duracion.csv`.

- **R0, precio "de referencia" compartido** (idea propia):
  - Features: vendedores distintos y publicaciones con el mismo (`category_id`, `price` exacto), y vendedores con el mismo `price` en cualquier categoría. Precio **crudo** (el capeo inventaría precios compartidos).
  - EDA: % `new` de 64,1% (precio único en la categoría) a 15,8% (> 20 vendedores). El 33% comparte el par. **No es solo redondez:** entre precios no múltiplos de 100, compartido = 33% contra único = 71%.
  - R0a cuenta sobre train + test (transductiva). R0b cuenta solo con el train del fold (`PrecioCompartido`, adentro del `Pipeline`, sirve en producción); en `transform` cada fila se suma a sí misma, así se describe igual que en `fit_transform`.
  - **Señal sola 0,705 (la más alta del TP), pero no suma:** R0a −0,0003 (2/5); R0b −0,0001 (2/5); contra P7a + M (0,9416): R0a −0,0004, R0b +0,0001 (3/5). El bloque M solo: +0,0003 (2/5).
- **R1, limpieza de etiquetas (*confident learning*, no está en el material):**
  - Sospechosa = P(clase etiquetada) OOF interna < umbral. Se reentrena sin ellas y se evalúa el fold completo sin limpiar.
  - Saca 0,7% / 1,5% / 2,7% (umbrales 0,05 / 0,10 / 0,20). Cambio: −0,0006 / −0,0005 / −0,0021. **El umbral elegido con 4 folds es 0 (no limpiar) en los 5 folds.**
  - Los 20 ejemplos son casos ambiguos o errores del modelo, no etiquetas mal puestas: "Campera … Sin Usar" `new` con P(new) 0,04; libros, revistas y VHS viejos `new` de vendedores grandes; "Yegua Zaina de siete años" `new`.
- **R2, propagación por vendedor:**
  - Feature: promedio de las predicciones (etapa 1: HistGB P7a + duración) de las **otras** publicaciones del mismo vendedor en el mismo lote sin etiquetas, y cuántas son.
  - Lote = fold de validación (en la entrega, el test); en train = fold interno, con predicciones OOF internas (*cross-fitting*).
  - +0,0000 (2/5). Sin historial: +0,0005 (2/5); sin historial con otras en el lote (~317 filas por fold): −0,0011 (3/5). Solo el 53% de las filas tiene vecinas en el lote.
- **R3, conteos sobre train + test** (vendedor, categoría, título normalizado; transductivo): +0,0003 (3/5) / −0,0002 / −0,0001; juntas −0,0000.

**Nada pasa. Se cierran los experimentos** (decisión: si nada pasaba, documentarlo y cerrar). El modelo final no cambia: sin confirmación, sin cambios en `features.py`, sin submission. ~11 min por corrida completa.

### Fase Q — Fine-tuning de BETO sobre el título y la garantía (`26_bert_texto.ipynb`, `bert_texto.py`)

La planteé como "la última prueba grande". Es distinto de P9: en lugar de usar embeddings **congelados** de paráfrasis, se **entrena** el transformer para `new` vs `used`.

- **Material:**
  - AP, `Intro a Redes para datos estructurados.pdf`: red, backpropagation, Adam, dropout y slide 25 "Transformers" (solo el diagrama).
  - SIA, `material/redes neuronales/S12 - Redes Neuronales - NLP.pdf` + `S12_Redes_Neuronales_NLP.ipynb` (**nuevo**, apareció después del 13/10): tokens, bolsa de palabras, embeddings, atención y embeddings contextuales, transformer, pre-entrenamiento como *transfer learning*; usa un DistilBERT ya ajustado por otros. S12 también muestra que la bolsa de palabras + logística es difícil de superar.
  - **No está en el material:** hacer nosotros el fine-tuning (AdamW, *scheduler* lineal con *warmup*), BETO, WordPiece y la cabeza sobre `[CLS]`. La slide 29 de S12 dice que la arquitectura transformer "va más allá de los contenidos de este curso".
- **Configuración** (`trabajo_realizado/bert_texto.py`): `dccuchile/bert-base-spanish-wwm-cased`, entrada `[CLS] título [SEP] garantía [SEP]`, `max_length` 64, 2 épocas, AdamW lr 2e-5 (*weight decay* 0,01), *warmup* lineal el 10% de los pasos y bajada lineal a 0, batch 32, MPS. BETO no trae los pesos del "pooler": se inicializan al azar y se entrenan.
- **Cross-fitting exacto con 15 ajustes en lugar de 30:**
  - 5 "globales" `sin_k` (56.000 filas) predicen el fold k y el test;
  - 10 "de a pares" `sin_j_k` (42.000 filas) predicen j y k;
  - para el fold externo k: validación ← `sin_k`; train del fold j ← `sin_j_k`, que no vio ni la fila ni la validación;
  - en la entrega: train = OOF de los globales; test = promedio de los 5 globales.
- **Tiempos:** 20 min por global y 14 min por par, **~4,4 h en total**.
  - Un ajuste se trabó ~11 h con la máquina a batería y el swap lleno (17,8 de 19 GB). Se cortó y se relanzó con la compu enchufada.
- **Resultados:**

| Prueba | Accuracy | Contra | Diferencia (folds) |
|---|---|---|---|
| Texto solo: BETO | 0,8497 | logística 0,8414 | **+0,0083 (5/5)** |
| Texto solo: promedio logística + BETO | 0,8589 | — | +0,0175 |
| Filtro sesgado (OOF globales en todos los folds): HistGB P7a + BETO extra | 0,9424 | 0,9412 | +0,0012 (4/5) |
| **Exacto: HistGB P7a + BETO extra** | **0,9426** | 0,9412 | **+0,0013 (4/5), desvío 0,0015 → no pasa** |
| Exacto: BETO en lugar de `prob_texto_new` | 0,9415 | 0,9412 | +0,0003 (3/5) |

- **Lectura:**
  - El fine-tuning **sí aprende la tarea**: BETO es el mejor modelo de texto del TP, mientras que con embeddings congelados (P9) se llegaba a 0,7247. Además se equivoca en otras filas que la logística (correlación 0,889).
  - Dentro del modelo completo, la ganancia es chica y despareja (−0,0009 / +0,0016 / +0,0015 / +0,0037 / +0,0006), y no supera el criterio. Las filas que BETO arregla respecto de la logística, en su mayoría, ya las resolvían el vendedor y las demás features (interpretación).
- **Decisión:** no pasa en el HistGB, así que no se mide en el promedio ni con la semilla 7 (eso habría costado 15 ajustes más), y no hay submission. Mirar el promedio después de no pasar habría sido sesgo de selección; se ofreció como opción fuera de protocolo y no se recomendó.

### Fase R — Atipicidad respecto de productos parecidos + umbral por segmento + precio exacto (`27_atipicidad_producto.ipynb`)

Idea propia: medir cuán atípica es cada publicación respecto de **productos parecidos** (no de su categoría ni de su vendedor). Mi EDA previo, versión burda (categoría + 3 primeras palabras, ARS): en vendedores sin historial, 29% `new` en el 20% más barato contra 55% en el 20% más caro, pero con cobertura del 3%.

- **Vecinos:**
  - TF-IDF del título sobre train + test (palabras 1-2 + caracteres 3-5, como el modelo de texto), normalizado, así que el producto escalar es el coseno (S12, slide 15).
  - Los 30 más parecidos dentro de la misma `category_id` y moneda, sin la propia fila ni las del mismo vendedor. Es la idea de kNN (`Presentacion_kNN.pdf`), pero promediando **features** de los vecinos, no su target (eso fue I1).
  - **No es leakage:** no usa `condition`, y validación y test buscan vecinos en el mismo universo. Es transductivo, como R0a y R3.
- **Features** (k = 10 y 30; vecino válido si similitud ≥ 0,3, fijado a priori; NaN si hay menos de 3): precio relativo y percentil, cantidad, fotos, garantía, % del mismo tipo de publicación, similitud media y cantidad de vecinos válidos.
- **Cobertura: 18,4%** (mediana de la similitud con el vecino más parecido: 0,27). EDA donde hay vecinos: precio del quintil más barato al más caro, 38,5% → 52,2% `new` (vendedores con 1 publicación); cantidad distinta de los parecidos, 83-95% `new` contra 39% igual.
- **Resultados contra HistGB P7a + duración (0,9412):** k = 10 −0,0003 (1/5); k = 30 −0,0003 (2/5); cada feature −0,0001 a −0,0006. En vendedores sin historial también baja (0,8923 → 0,890-0,892).
- **Umbral por segmento** (OOF del modelo final, umbral elegido con 4 folds y evaluado en el quinto, grilla 0,30-0,70):
  - sin historial +0,00007 (3/5); `free` +0,00001 (1/5); **los dos, +0,00009 (4/5): pasa con la semilla 42 por ~6 predicciones**;
  - **con la semilla 7: −0,00003 (0/5)**, y los umbrales elegidos cambian (sin historial 0,48-0,49 → 0,50-0,51; `free` 0,54-0,57 → 0,58-0,70);
  - **no se confirma:** el modelo está calibrado también por segmento.
- **`TargetEncoder` de (vendedor, precio exacto) y (vendedor, categoría, precio exacto):** −0,0003 / −0,0001 / −0,0004.
- **Nada pasa.** Sin cambios ni submission. Se guardaron las OOF del modelo final con la semilla 7 (`oof_final_semilla7.csv`, 2,4 min) para confirmaciones futuras.
- Nota: el 27/10 la terminal no tenía permiso para leer `material/` (Google Drive). Se citó el material relevado el 25/10.

### Fase S — Especialista PONDERADO para vendedores sin historial (`28_especialista_ponderado.ipynb`)

Idea propia: corregir I4 (Fase I), que entrenó el especialista **solo** con vendedores de ≤ 1 publicación y quedó peor que el modelo final en su propio grupo.

- **Especialista:** HistGB P7a + duración entrenado con **todo** el train del fold y `sample_weight` = w (2, 3, 5) en las filas de vendedores con 1 sola publicación en el train del fold (~17.300 de 56.000), 1 en el resto. Idea de AdaBoost (`Árboles.pdf`, slide 62); el `sample_weight` como parámetro no está en el material.
- **Ablación por grupos** (solo en el especialista): sin texto, sin vendedor (`seller_id`, 7 `c_seller_*`, `VendedorEncoder`), sin fotos, sin fechas (incluida la duración), sin precio. No está en el material (lo más cercano: Lasso, slide 36 de regularización; *random feature selection*, SIA S8 slide 17).
- **Uso:** en las filas sin historial, (1 − α)·P7c + α·especialista, α ∈ {0; 0,25; 0,5; 0,75; 1}. w, grupo y α elegidos con 4 folds y evaluados en el quinto (91 candidatos; α = 0 gana los empates).
- **No es leakage:** el peso sale del conteo de publicaciones del vendedor en el train del fold (no usa `condition`), y todo lo que aprende sigue adentro del `Pipeline`.
- **Ablación** (accuracy del especialista solo en vendedores sin historial; P7a sin peso: 0,8923, modelo final: 0,8931): sin peso de grupo 0,8908-0,8929; sin vendedor 0,8914-0,8927; sin precio 0,890-0,891; sin fotos 0,887; sin fechas 0,877-0,879; sin texto 0,869-0,870. **Ningún grupo sobra**; el peso tampoco mejora al especialista solo (w = 2: +0,0006; w = 5: −0,0015).
- **Semilla 42:** los 5 folds eligen (w = 2, sin vendedor, α = 0,5): **0,94233, +0,00033, 5/5, desvío 0,00012 → pasa**. Sin historial: 0,8931 → 0,8941 (5/5).
- **Semilla 7 (robustez):** se recalcularon las 18 variantes con esos folds. Elección anidada: **0,9417, −0,00046, 0/5 → no se confirma**. Los elegidos cambian entre folds (w de 2 a 5, α de 0,25 a 0,75, y un fold sin sacar el vendedor). La misma combinación de la semilla 42 (w = 2, sin vendedor, α = 0,5) da −0,00019 (1/5). Sin historial: 0,8932 → 0,8917.
- **Conclusión: no pasa.** El +0,0003 de la semilla 42 era ruido de partición (como el umbral por segmento de la Fase R). Sin cambios ni submission. ~23 min (18 variantes × 2 semillas, ~0,7 min cada una). OOF en `oof_especialista.csv` y `oof_especialista_semilla7.csv`.

### Fase T — Probabilidad de texto combinada: logística + BETO (`29_texto_combinado.ipynb`)

Idea propia: la ablación de la 28 mostró que el texto es lo que más pesa en vendedores sin historial (−0,023 si se saca), así que la palanca es la calidad de la probabilidad de texto. Se usan las probabilidades guardadas de BETO (Fase Q), **sin reentrenarlo**.

- **Compatibilidad del cross-fitting** (fold externo k): logística = `PrediccionOOF` (OOF interna sobre el train del fold; validación con todo el train del fold); BETO = `sin_j_k` para las filas de train del fold j, `sin_k` para la validación. Las dos cumplen la misma regla (ninguna probabilidad de una fila de train sale de un modelo que la vio; ninguna vio la validación), así que combinarlas no crea leakage. **Control:** la logística calculada afuera del `Pipeline` reproduce las OOF de P7a con diferencia máxima 1e-16.
- **Parecido** (OOF de texto solo): correlación 0,889 (0,843 en sin historial); discrepan en el 11,7% de las filas (15,9% en sin historial). Accuracy logística / BETO / promedio: total 0,8414 / 0,8497 / 0,8589; **sin historial 0,7835 / 0,7926 / 0,8036**. Donde discrepan, BETO acierta el 53,6%.
- **Variantes en lugar de `prob_texto_new`** (material: voting *soft* SIA S8 slides 6-7; logit = log-odds `Presentación_Logística.pdf` slide 34; *log-loss* `Árboles.pdf` slides 12-13):

| Variante | HistGB P7a (vs 0,9412) | Sin historial (0,8923) | Promedio final (vs 0,9420) | Sin historial (0,8931) |
|---|---|---|---|---|
| (a) promedio simple (L + B)/2 | 0,9425, +0,0013 (5/5) | 0,8948 | **0,9435, +0,0015 (5/5)** | 0,8962 |
| (b) promedio de logits | 0,9421, +0,0009 (5/5) | 0,8941 | 0,9437, +0,0017 (5/5) | 0,8962 |
| (c) peso elegido con 4 folds (*log-loss*: w_BETO = 0,6 en los 5) | 0,9427, +0,0015 (5/5) | 0,8950 | 0,9435, +0,0015 (4/5) | 0,8958 |
| (d) L, B y B − L por separado | 0,9423, +0,0010 (4/5) | 0,8933 | 0,9434, +0,0014 (4/5) | 0,8952 |

- **CatBoost solo** (con la misma columna, contra el CatBoost actual 0,9414): (a) 0,9431 +0,0017 / (b) 0,9428 +0,0014 / (c) 0,9424 +0,0010 / (d) 0,9430 +0,0016, **las 4 con 5/5**. La mejora no depende de uno solo de los dos modelos.
- **Lectura:** en la Fase Q, BETO como feature **extra** no pasaba (+0,0013, 4/5, desvío 0,0015). **Promediado con la logística**, en una sola probabilidad, pasa en los dos niveles. Promediar dos modelos de texto que se equivocan en filas distintas da una probabilidad mejor (0,8589 contra 0,8497) y el árbol no tiene que aprender a combinarlas. Las 4 variantes dan lo mismo dentro del ruido; elegir la mejor del promedio final sería optimista: se elige **(a)**, la más simple, sin parámetros y 5/5 en los dos niveles (y la mejor en el CatBoost solo).
- **⚠️ La variante (a) se confirmó SOLO con la semilla 42.** La confirmación con la semilla 7 (regla del proyecto) **queda pendiente por tiempo de cómputo**: los 15 ajustes de BETO corresponden a los folds de la semilla 42 y, con otros folds, la probabilidad de BETO de las filas de train saldría de modelos que vieron la validación (leakage). Confirmarla = 15 ajustes nuevos de BETO (~4,4 h, enchufada y sin apps pesadas).
- **Submission de prueba** (`30_submission_texto_combinado.ipynb`, **sin subir**): `submission/submission_texto_combinado.csv`.
  - Columna de texto: train = (OOF de la logística + OOF de los 5 BETO globales)/2; test = (logística con todo el train + promedio de los 5 BETO globales, `test_bert.csv`)/2.
  - **Control:** con la logística sola calculada afuera del `Pipeline`, la submission sale idéntica a `submission_p7c_duracion.csv` (0 diferencias).
  - **Cambian 366 de 30.000 predicciones (1,22%)**: 187 new → used y 179 used → new (% new 52,89% → 52,86%). ~2 min.
  - `predict.ipynb` todavía **no** la reproduce: en `bert/` hay solo probabilidades, no los pesos de BETO (ver propuesta de entrega abajo).
- **Propuesta de entrega (pendiente de decidir, no se tocó `features.py`, `train` ni `predict`):** `train.ipynb` hace el fine-tuning de BETO (5 modelos con los folds de la semilla 42, ~100 min), guarda los pesos y las OOF del train; `predict.ipynb` carga los pesos y solo predice el test. Detalle en el chat del 2026-10-27.
- OOF en `oof_texto_combinado.csv`, tabla en `texto_combinado.csv`. ~9 min.

### Fase U — Ensamble de modelos de texto (`31_texto_ensamble.ipynb`)

Idea propia: si lo que sumó en la Fase T fue combinar modelos de texto que se equivocan distinto, sumar más modelos de texto baratos. **Referencia: variante (a) de la Fase T** (HistGB 0,9425; promedio final 0,9435; sin historial 0,8962). Semilla 42 (la de BETO guardado).

- **Modelos nuevos** (título + garantía, cada uno en `PrediccionOOF` → mismo cross-fitting que la logística, compatible con los folds de afuera; control: la variante (a) recalculada reproduce las OOF de la Fase T con diferencia 1e-16 en HistGB y CatBoost):
  - `MultinomialNB()` sobre conteos de palabras 1-2 y `BernoulliNB()` sobre presencia de palabras 1-2 (`Presentación_NB_LDA_QDA.pdf` slides 11 y 27; parámetros por defecto como en `Copia de AP_NaiveBayes.ipynb`);
  - logística C = 2 solo con caracteres 3-5 / solo con palabras 1-2;
  - `LinearSVC` (`SVM (1).pdf` slide 52) + `CalibratedClassifierCV(method="sigmoid")` (*Platt scaling*, **no está en el material**) sobre el TF-IDF de la logística actual.
- **Cada modelo solo** (accuracy total / sin historial; correlación con la logística / con BETO):

| Modelo | Total | Sin historial | Corr. logística | Corr. BETO |
|---|---|---|---|---|
| logística (actual) | 0,8414 | 0,7835 | 1 | 0,889 |
| BETO | 0,8497 | 0,7926 | 0,889 | 1 |
| NB multinomial | 0,8168 | 0,7397 | 0,878 | 0,809 |
| NB Bernoulli | 0,8055 | 0,7389 | 0,864 | 0,804 |
| logística caracteres | 0,8322 | 0,7733 | 0,974 | 0,882 |
| logística palabras | 0,8370 | 0,7753 | 0,980 | 0,880 |
| SVM lineal calibrado | 0,8351 | 0,7768 | 0,986 | 0,863 |

- **Combinaciones (promedio simple, una sola columna en lugar de `prob_texto_new`)**, contra la variante (a):

| Candidata | HistGB | Promedio final | Sin historial (final) |
|---|---|---|---|
| a (referencia) | 0,9425 | 0,9435 | 0,8962 |
| a + NB multinomial | −0,0014 (0/5) | −0,0013 (0/5) | 0,8938 |
| a + NB Bernoulli | −0,0005 (1/5) | −0,0006 (1/5) | 0,8950 |
| a + logística caracteres | −0,0002 (2/5) | −0,0006 (1/5) | 0,8950 |
| a + logística palabras | −0,0006 (1/5) | −0,0010 (0/5) | 0,8936 |
| a + SVM lineal | −0,0004 (1/5) | −0,0007 (1/5) | 0,8944 |
| todos (7) | −0,0011 (1/5) | −0,0015 (1/5) | 0,8936 |
| subconjunto elegido por *log-loss* (4 folds → quinto) | = a (elige **ningún** modelo nuevo en los 5 folds) | = a | 0,8962 |
| elección anidada entre candidatas (accuracy, 4 folds → quinto) | = a (elige a en los 5 folds) | = a | 0,8962 |

- **Lectura:** **nada pasa; todo empeora.** Ya como texto solo, agregar cualquier modelo sube la *log-loss* (0,3293 → 0,337-0,348) y baja la accuracy (0,8589 → 0,846-0,858). Los modelos nuevos son de la **misma familia** que la logística (bolsa de palabras/caracteres + modelo lineal): la logística de caracteres, la de palabras y el SVM tienen correlación 0,97-0,99 con la logística actual (son casi la misma), y los Naive Bayes se equivocan distinto pero son bastante peores (0,81-0,82). Un promedio con pesos iguales los cuenta igual que a BETO, así que diluyen su aporte. La lección de la Fase T se precisa: **suma un modelo que se equivoca distinto Y es al menos tan bueno** (BETO); otro modelo lineal sobre las mismas palabras no aporta nada nuevo.
- **Parte 4 (segundo transformer, RoBERTa-BNE):** la condición era "si sumar modelos sigue mejorando", y no mejora. No se propone.
- Sin cambios ni submission. ~19 min (modelos de texto 2 min; 16 corridas de HistGB/CatBoost ~17 min). Probabilidades en `textos_31/`, OOF en `oof_texto_ensamble.csv`, tabla en `texto_ensamble.csv`. Durante la primera corrida joblib avisó "A worker stopped" (memoria llena: swap ~13 de 13,3 GB); el control dio igual, así que el resultado no se vio afectado.

### Fase V — BETO con contexto tabular serializado como texto (`32_beto_contexto.ipynb`, `bert_contexto.py`) — **TERMINADA (2026-11-04): no pasa**

Idea propia: en vendedores sin historial el texto es la palanca principal (ablación de la 28), y BETO hoy ve solo "título [SEP] garantía". Acá ve además el contexto de la publicación como texto.

- **Entrada:** `título [SEP] garantía [SEP] categoría: MLA… [SEP] precio: 15 ARS [SEP] tipo: bronze [SEP] compra: buy_it_now [SEP] cantidad: 1 [SEP] vendidas: 0 [SEP] fotos: 5 [SEP] envío gratis: no`. Sin `seller_id` ni nada que use el target: todas las variables están en el test y se calculan fila por fila, así que la entrada no es leakage.
- **Configuración** (`trabajo_realizado/bert_contexto.py`): reusa `bert_texto.py` (misma receta que la Fase Q). Cambian `max_length`, que pasa a 128 (con 64 se truncaba el 57% de las filas; con 128, el 0,6%), y la carpeta de salida, `.cache_laboratorio/bert_contexto/`.
- **Por qué podría ayudar:** los árboles ya tienen esas variables, pero ven el texto resumido en una sola probabilidad. BETO puede cruzar palabras con contexto ("Lenovo i5" a 45.000 con 1 unidad ≠ "remera" a 45.000). **Riesgo:** que aprenda lo mismo que los árboles y se equivoque en las mismas filas (lección de la Fase U).
- **Versión rápida (decisión mía, para no esperar ~8 h):** solo los **5 globales** `sin_k`, sin los 10 de a pares. Eso **sesga** la comparación dentro del HistGB: las filas de train del fold j reciben `sin_j`, que vio el fold k de validación. En la Fase Q, ese sesgo fue chico (+0,0012 sesgada contra +0,0013 exacta). Por eso:
  - se compara contra la variante (a) **sesgada igual** (BETO común con sus 5 globales), que es la comparación de igual a igual, y también contra la (a) exacta (0,9435);
  - **regla de decisión que fijé:** integrar solo si el promedio final mejora **+0,003 o más y gana 5/5**. Si no, se descarta.
  - Candidatas: logística + BETO con contexto; logística + BETO + BETO con contexto.
- **Material:** transformer y atención en SIA `S12 - Redes Neuronales - NLP.pdf` y en AP `Intro a Redes para datos estructurados.pdf`, slide 25. **No está en el material:** el fine-tuning (ver Fase Q) y serializar variables tabulares como texto.
- **Entrenamiento:**
  - `sin_1` se entrenó el 27/10 (42 min).
  - `sin_2` a `sin_5` el 2026-11-04, de 17:24 a 20:00, con `bert_contexto/cola_globales_logs.sh`. Es la misma cola de antes, pero guarda el log completo en `sin_k.log`. Se lanzó con `nohup` + `caffeinate -i -w`, enchufada y con Chrome cerrado. Cada modelo tardó ~39 min (2,0 min cada 200 pasos, igual que `sin_1`).
  - La notebook 32 tardó 6,4 min.
- **Cambios en la 32** (2026-11-04, editando `armar_32.py`):
  - tablas fold por fold con accuracy y *log-loss*;
  - regla contra la entrega medida igual (a sesgada);
  - nueva sección (d): promedio de 3 modelos.
- **(a) BETO con contexto solo** (OOF de los 5 globales, sin sesgo):

| Fold | Acc BETO | Acc BETO + contexto | Log-loss BETO | Log-loss BETO + contexto |
|---|---|---|---|---|
| 1 | 0,8469 | 0,9058 | 0,3525 | 0,2333 |
| 2 | 0,8506 | 0,9083 | 0,3454 | 0,2331 |
| 3 | 0,8491 | 0,9081 | 0,3501 | 0,2329 |
| 4 | 0,8503 | 0,9090 | 0,3432 | 0,2329 |
| 5 | 0,8519 | 0,9086 | 0,3428 | 0,2290 |
| **Media** | **0,8497** | **0,9080** (+0,0582, 5/5) | **0,3468** | **0,2322** (−0,1146) |

  - Sin historial: 0,7926 → **0,8644**; con historial: 0,8751 → 0,9273.
  - Correlación con BETO 0,858; con la logística 0,816; con CatBoost / HistGB de la entrega 0,932 / 0,925. Discrepa con BETO en el 12,0% de las filas.
  - Promediarlo con la logística no mejora el texto (0,9073); con la logística y BETO, baja (0,8942).
- **(b)-(c) Como columna de texto en los árboles** (sesgada: OOF de los globales en las filas de train). Fold por fold del promedio final:

| Fold | Entrega (a exacta) | Entrega (a sesgada) | L + BETO-ctx | Dif vs a sesgada | L + BETO + BETO-ctx | Dif vs a sesgada |
|---|---|---|---|---|---|---|
| 1 | 0,94321 | 0,94350 | 0,94414 | +0,00064 | 0,94414 | +0,00064 |
| 2 | 0,94221 | 0,94264 | 0,94179 | −0,00086 | 0,94307 | +0,00043 |
| 3 | 0,94421 | 0,94336 | 0,94321 | −0,00014 | 0,94329 | −0,00007 |
| 4 | 0,94350 | 0,94307 | 0,94343 | +0,00036 | 0,94429 | +0,00121 |
| 5 | 0,94450 | 0,94329 | 0,94436 | +0,00107 | 0,94493 | +0,00164 |
| **Media** | **0,94353** | **0,94317** | **0,94339** | **+0,00021 (3/5)** | **0,94394** | **+0,00077 (4/5)** |

  - El sesgo medido en la referencia (a sesgada − a exacta) es **−0,00036**: acá la versión sesgada salió **más baja**, no inflada. Es ruido de la partición de las filas de train, del mismo orden que las diferencias que se miden.
  - HistGB solo, contra la a exacta (0,9425): L + BETO-ctx 0,9430 (+0,0005, 4/5); L + BETO + BETO-ctx 0,9428 (+0,0003, 4/5).
  - **Regla de decisión (+0,003 y 5/5 contra la a sesgada): ninguna pasa.** La mejor (L + BETO + BETO-ctx) da +0,00077 con 4/5: es 4 veces menos de lo que pide la regla.
  - Con el criterio de siempre (4/5 con media > desvío), contra la a sesgada daría +0,00077 > 0,00060. Contra la a exacta (la entrega real), da +0,00042 con desvío 0,00069 → no pasa. Como la regla de esta fase era más exigente y la CV es sesgada, **no se toma como candidata**.
- **(d) Promedio de 3 modelos** (CatBoost y HistGB exactos de la entrega + BETO con contexto solo; pesos de a 0,1 elegidos con 4 folds y evaluados en el quinto, rotando):

| Fold | Entrega | 3 modelos | Dif | Pesos (cat, hgb, ctx) |
|---|---|---|---|---|
| 1 | 0,94321 | 0,94307 | −0,00014 | (0,6; 0,2; 0,2) |
| 2 | 0,94221 | 0,94236 | +0,00014 | (0,3; 0,5; 0,2) |
| 3 | 0,94421 | 0,94350 | −0,00071 | (0,3; 0,5; 0,2) |
| 4 | 0,94350 | 0,94379 | +0,00029 | (0,5; 0,4; 0,1) |
| 5 | 0,94450 | 0,94393 | −0,00057 | (0,6; 0,3; 0,1) |
| **Media** | **0,94353** | **0,94333** | **−0,00020 (2/5)** | |

  - Sin historial: 0,8962 → 0,8963. **No pasa.**
  - Por qué esta CV no está sesgada: las tres probabilidades de cada fila vienen de modelos que no la vieron, y sobre BETO con contexto no se entrena nada: solo se eligen 2 pesos con los otros 4 folds.
  - Matiz: las OOF de esos 4 folds salieron de modelos que sí vieron el fold evaluado. Es lo mismo que en la 16 y en la 23, y con 2 parámetros es despreciable.
- **Lectura:**
  - Como modelo de texto, BETO con contexto es por lejos el mejor del TP (0,908; sin historial 0,864).
  - Pero lo que agrega el contexto (precio, tipo, cantidades, fotos, envío) **los árboles ya lo tenían**. Correlación con el CatBoost: 0,93, contra 0,889 de BETO con la logística.
  - Dentro del modelo completo, la ganancia queda en el ruido de la partición (±0,0009 por fold). Es la lección de las Fases U y Q: suma un modelo que se equivoca **distinto**, y el contexto lo hizo equivocarse **parecido** a los árboles.
- **Decisión:** no pasa. Sin cambios en la entrega ni submission.
- **Archivos:** OOF en `.cache_laboratorio/oof_beto_contexto.csv`, tabla en `beto_contexto.csv`, probabilidades en `bert_contexto/sin_k.npz` (incluyen el test).

### Fase W — Búsqueda de artefactos de FORMATO (`33_artefactos.ipynb`, armada con `.cache_laboratorio/armar_33.py`) — **no pasa**

Idea propia (2026-11-04), a partir de la lección de la duración: la señal estaba en **cómo viene escrito** el dato (60 días + unos segundos), no en su valor. Se buscan artefactos parecidos en todos los campos del JSON.

- **Zona de duda:** se usó la OOF de la entrega (variante a exacta) y se tomaron las filas con P(new) entre 0,2 y 0,8 más los errores. Son **8.527 filas (12,2%)**, 50,1% `new`.
  - La entrega acierta el 53,6% en la zona (incluye los 3.953 errores).
  - De los errores, 2.295 están entre 0,2 y 0,8 y el resto son errores "seguros".
- **224 features de formato** (46 constantes descartadas), calculadas con `json.loads` sobre el JSON crudo (conserva entero/decimal y null / `""` / `[]`). ~20 s.
  - **Genéricas**, de cada campo y subcampo (dicts recursivos; en las listas, el primer elemento):
    - estado: clave ausente / null / texto vacío / lista vacía / dict vacío / tipo del valor;
    - largo de los textos, cantidad de claves, largo de las listas;
    - decimales y centavos de los números con coma.
  - **Específicas:**
    - título original: mayúsculas, palabras capitalizadas, espacios dobles, signos, dígitos, caracteres no ASCII;
    - garantía;
    - `permalink` contra el slug del título actual (título editado);
    - precio: terminaciones 9/99/0/00/000, dígitos significativos, entero;
    - segundos y ms de las 4 fechas;
    - fotos:
      - formato del id: el nuevo trae el **mes y año de subida** (`…_052015`), y de ahí sale `foto_meses_antes_max` (meses entre la foto y la creación de la publicación);
      - id de otra publicación, ids repetidos, orden, servidores, `size` contra `max_size`;
    - orden de tags y de métodos de pago;
    - formato de `descriptions`;
    - precio entero o decimal en variaciones;
    - `seller_custom_field`.
- **Ranking dentro de la zona** (cada feature sola: grupo → clase mayoritaria, ajustado con 4 folds y evaluado en el quinto, rotando):
  - 46 features ganan en 5/5 folds, pero las ganancias son chicas: accuracy sola de 0,53 contra 0,495 de la clase mayoritaria.
  - La mejor es `foto_largo_prefijo` (+0,038).
  - Siguen las terminaciones del precio (`termina_00` +0,036, `digitos_significativos` +0,026, con 13 puntos de diferencia de % `new`), el largo del id de la foto (+0,033), `foto_meses_antes_max` (+0,029, 9 puntos) y `titulo_n_digitos` (+0,023).
- **Filtro antes de elegir** (agregado después de mirar el primer ranking). Algunas features no eran artefactos nuevos:
  - `largo|listing_type_id` es el valor disfrazado ("free" = 4 letras);
  - `estado|parent_item_id` = `tiene_parent_item`;
  - el largo de `url` / `secure_url` es el largo del id de la foto + una constante.

  Regla: se descarta una candidata si es casi función (≥ 99,5% de las filas) de una columna que el modelo ya usa o de una ya elegida. Solo se consideran las que ganan 5/5 y no se probaron en las Fases G ni M; se eligen hasta 5. El filtro no mira el target.
- **Elegidas:** `foto_largo_prefijo`, `base_price_termina_00`, `largo|pictures[0].id`, `foto_meses_antes_max`, `base_price_digitos_significativos`. Las cinco aparecen en el test con la misma frecuencia (diferencias < 0,5 puntos).
- **Medición:**
  - control: el HistGB sin extras reproduce la OOF de la entrega (diferencia 1e-16);
  - tiempos: HistGB 0,5 min, CatBoost 1,6 min por candidata (5 folds en paralelo);
  - las features nuevas van al **final** del `ColumnTransformer` (`INDICES_CAT` no cambia), con la columna de texto exacta de la variante a.

| Candidata | HistGB (vs 0,9425) | Promedio final (vs 0,9435) | Folds | Desvío |
|---|---|---|---|---|
| `foto_largo_prefijo` | +0,0001 (3/5) | +0,0003 | 3/5 | 0,0005 |
| `base_price_termina_00` | −0,0000 (3/5) | +0,0001 | 2/5 | 0,0008 |
| `largo|pictures[0].id` | +0,0003 (4/5) | +0,0001 | 1/5 | 0,0004 |
| `foto_meses_antes_max` | −0,0000 (3/5) | +0,0002 | **4/5** | 0,0003 → media < desvío |
| `base_price_digitos_significativos` | +0,0002 (2/5) | +0,0000 | 2/5 | 0,0005 |
| Las 5 juntas | +0,0004 (4/5) | +0,0002 | 2/5 | 0,0007 |

- Diferencias fold por fold del promedio final (folds 1 a 5):
  - `foto_largo_prefijo`: +0,0006 / −0,0001 / +0,0004 / +0,0009 / −0,0004;
  - `foto_meses_antes_max`: +0,0004 / +0,0005 / +0,0001 / −0,0003 / +0,0003;
  - las 5 juntas: +0,0011 / −0,0003 / −0,0004 / +0,0010 / −0,0003.
- **Ejemplos crudos** (en la notebook): fotos con prefijo de 3 dígitos y sufijo `_072013` (fotos subidas en 2013 en publicaciones de 2015); precios 2.100 / 10.000 / 1.000 en el grupo "termina en 00" (46% `new` en la zona contra 52%).
- **Lectura:**
  - Los artefactos existen y son reales: las fotos traen su mes de subida, los precios redondos son más `used` (35% contra 59% en todo el train) y el largo del id de la foto marca la época en que se subió.
  - Pero el modelo **ya los aprovecha** a través de otras columnas: vendedor, precio + categoría, `n_fotos`, fechas. Dentro de la zona de duda separan apenas 3-4 puntos de accuracy, y sumados al modelo quedan en el ruido de la partición (±0,001 por fold).
  - No apareció nada parecido a la duración: un grupo chico y casi puro (2,5% `new`) que el modelo no pudiera ver de otra forma.
- **Decisión:** nada pasa. Sin cambios ni submission.
- **Tropiezos** (corregidos antes de concluir):
  1. En pandas 3, `.astype(str)` deja los NaN como faltantes. Por eso el filtro de duplicados no veía las 541 filas sin fotos (dejaba pasar `url` y `secure_url`, iguales al id), y en el ranking NaN no era un grupo propio. Se corrigió con `a_texto()`.
  2. La OOF de "todas juntas" quedó guardada en la caché de la primera corrida, que tenía otras 5 features. Se borró y se recalculó; ahora el nombre de la columna lleva la lista de features.
- **Archivos:** `.cache_laboratorio/artefactos_ranking.csv` (ranking completo), `oof_artefactos.csv`, `artefactos.csv`. La notebook tarda ~6 min con la caché; desde cero, ~16 min.

### Fase X — BETO con preentrenamiento adaptado al dominio (`34_beto_dapt.ipynb`, `bert_dapt.py`) — **TERMINADA (2026-11-09): no pasa**

- **Qué hace:**
  1. **Paso 1, MLM:** sobre "título [SEP] garantía" de train + test (100.000 textos), sin la etiqueta. Se tapa el 15% de los tokens (80% `[MASK]`, 10% otro token, 10% igual) y BETO aprende a adivinarlos. 1 época, AdamW lr 5e-5, *warmup* del 10%, batch 32, `max_length` 64, semilla 42. Guarda el modelo en `bert_dapt/mlm/`.
  2. **Paso 2:** el mismo fine-tuning de la Fase Q/T (`bert_texto.main()`), con los 5 folds de la semilla 42 (globales `sin_k`), partiendo del modelo del paso 1. Guarda en `.cache_laboratorio/bert_dapt/`; no pisa `bert/`, `bert_contexto/` ni `modelos_beto/`.
  3. **Notebook 34:**
     - (a) BETO adaptado solo contra BETO, fold por fold (accuracy y *log-loss*);
     - (b) columna (logística + BETO adaptado)/2 en HistGB y CatBoost, contra la variante a **sesgada igual** (de la 32) y contra la a exacta.
- **Por qué no es leakage:**
  - el paso 1 no lee `condition` (`textos_sin_etiqueta()` solo carga `title` y `warranty`);
  - leer los textos de la validación y del test es transductivo, como los conteos sobre train + test; en el uso real, los títulos a predecir están disponibles;
  - las etiquetas entran recién en el paso 2, con el cross-fitting de siempre.
- **CV sesgada** (solo globales), como en la Fase V. La versión exacta necesita 10 ajustes de a pares más (~14 min cada uno, ~2,3 h), y `bert_dapt.py --excluir j,k` ya los soporta.
- **Tiempo medido** (60 pasos cada uno, a batería): MLM 0,43 s/paso → **~23 min**; fine-tuning 0,33 s/paso → ~19 min + ~2 de predicción por global (la Fase Q midió 19-20). **Total ≈ 23 + 5 × 21 ≈ 2,1 h**; la notebook 34, ~5 min más.
- **Lanzar** (desde `trabajo_realizado/`, enchufada y sin apps pesadas): `nohup .cache_laboratorio/bert_dapt/cola_dapt.sh` + `caffeinate -i -w PID`. Termina con `bert_dapt/cola.terminado`. Después: `armar_34.py` (ya corrido) y ejecutar `34_beto_dapt.ipynb`.
- **Material:** SIA `S12 - Redes Neuronales - NLP.pdf`, slides 34-35 (pre-entrenamiento como *transfer learning*). **No está en el material:** MLM, el preentrenamiento adaptado al dominio y el fine-tuning.
- **Corrida (2026-11-08, 22:21 → 00:30, enchufada, con Chrome en segundo plano):**
  - MLM, 21,0 min: la pérdida pasó de ~5,2 a ~2,9. Se verificó que los fine-tunings cargan desde `bert_dapt/mlm` y no desde el BETO original;
  - 5 globales de ~21 min cada uno;
  - notebook 34, 2,3 min.
- **(a) BETO adaptado solo, contra BETO** (OOF de los globales, sin sesgo):

| Fold | Acc BETO | Acc adaptado | Log-loss BETO | Log-loss adaptado |
|---|---|---|---|---|
| 1 | 0,8469 | 0,8563 | 0,3525 | 0,3415 |
| 2 | 0,8506 | 0,8554 | 0,3454 | 0,3369 |
| 3 | 0,8491 | 0,8536 | 0,3501 | 0,3437 |
| 4 | 0,8503 | 0,8556 | 0,3432 | 0,3391 |
| 5 | 0,8519 | 0,8576 | 0,3428 | 0,3354 |
| **Media** | **0,8497** | **0,8557** (+0,0060, 5/5) | **0,3468** | **0,3393** |

  - Sin historial: 0,7926 → 0,7995.
  - Correlación con BETO: 0,967.
  - Logística + BETO: 0,8589 → logística + adaptado: 0,8629.
- **(b) Columna (logística + BETO adaptado)/2** (CV sesgada, solo globales). Promedio final, fold por fold:

| Fold | a exacta | a sesgada | Logística + adaptado | Dif vs a sesgada | Dif vs a exacta |
|---|---|---|---|---|---|
| 1 | 0,94321 | 0,94350 | 0,94407 | +0,00057 | +0,00086 |
| 2 | 0,94221 | 0,94264 | 0,94157 | −0,00107 | −0,00064 |
| 3 | 0,94421 | 0,94336 | 0,94329 | −0,00007 | −0,00093 |
| 4 | 0,94350 | 0,94307 | 0,94450 | +0,00143 | +0,00100 |
| 5 | 0,94450 | 0,94329 | 0,94414 | +0,00086 | −0,00036 |
| **Media** | **0,94353** | **0,94317** | **0,94351** | **+0,0003 (3/5)** | **−0,0000 (2/5)** |

  - HistGB: contra la a sesgada +0,0005 (4/5, desvío 0,0006); contra la a exacta +0,0004 (4/5, desvío 0,0007).
  - Sin historial (final): a exacta 0,8962, a sesgada 0,8952, adaptado 0,8958.
- **Lectura:**
  - El preentrenamiento **sí** mejora a BETO como modelo de texto: +0,006 en los 5 folds y menos *log-loss*.
  - Pero es casi el mismo modelo (correlación 0,967 con BETO): se equivoca en las mismas filas, apenas menos. En el modelo completo, lo que gana queda en el ruido de la partición (±0,001 por fold).
  - Es la lección de las Fases Q, U y V: al modelo final le suma un texto que se equivoca **distinto**, no uno un poco mejor en las mismas filas.
- **Decisión:** no pasa. Sin cambios en la entrega ni submission. Por eso no se corren los 10 ajustes de a pares de la versión exacta.
- **Archivos:** `bert_dapt/mlm/` (modelo adaptado, ~0,4 GB), `bert_dapt/sin_k.npz` (incluyen el test), `oof_beto_dapt.csv`.

### Paso 7 — Entrega (estado al 2026-10-13)

- **`features.py`:**
  - `armar_pipeline_final()` (CatBoost) y `armar_pipeline_histgb_combos()` (HistGB P7a) incluyen `duracion_dias`, en la misma posición en la que se midió;
  - `agregar_features_contenido` calcula `duracion_dias`.
- **`train.ipynb`:** prepara el train con las 4 funciones determinísticas, entrena los dos modelos con todo el train y los guarda en `modelo_final.joblib` (`{"catboost": ..., "histgb": ...}`, ~2 min). Sin CV, por decisión mía.
- **`predict.ipynb`:** carga `modelo_final.joblib`, aplica las mismas 4 funciones al test, promedia 50/50 y genera `submission/submission.csv` con asserts de formato. **Verificado: idéntica byte a byte a `submission/submission_p7c_duracion.csv`**, que se subió a Kaggle: **público 0,94247**.
- Versiones viejas de `train`/`predict` (comparación de 14 modelos) en `trabajo_realizado/anteriores/`.
- **`13_comparacion_modelos.ipynb`:** lista (11 modelos + 3 ensambles + CatBoost + HistGB con combinaciones + promedio), **sin correr**: la corro una sola vez en la semana 3.
  - Celda `MODELOS` con `True`/`False` por modelo. Por defecto, uno por familia: regresión logística, árbol, Random Forest, Gradient Boosting y el modelo final. La comparación completa de los 14 ya está en la Fase A. Voting y Stacking combinan solo los modelos corridos.
  - Se sumó `duracion_dias` al preprocesador de los modelos de sklearn (faltaba: la 13 era anterior a la Fase K).
  - Probada con una copia de grillas mínimas (regresión logística + árbol + Voting + Stacking): anda de punta a punta.
  - **Por qué no se reduce la 13 borrando modelos:** Preferí elegir al correrla. La tabla completa de 14 modelos ya existe (Fase A), y en la presentación alcanza con un modelo por familia.
  - **Por qué el modelo final no entra en el Voting/Stacking de la 13**: esos ensambles comparten un mismo `Pipeline` de sklearn, y CatBoost necesita otro preprocesamiento (categóricas crudas). Además, el modelo final ya es un Voting *soft* 50/50, y combinarlo con otros modelos ya se probó en las Fases D, H, I y N (voting, ponderado, Hill Climber, Stacking, redes, LightGBM) sin mejoras.

### Paso 8 — Entrega con BETO (2026-10-27)

- **`texto_beto.py` (nuevo):** fine-tuning de BETO con la configuración de la Fase Q, en funciones: `fijar_semillas` (Python, numpy, torch, MPS y transformers en 42, al principio de cada fold), `entrenar_beto`, `predecir_beto`, `oof_beto` (5 modelos con `StratifiedKFold(5, shuffle=True, random_state=42)`; guarda pesos, tokenizer y OOF en `modelos_beto/fold_k/`; saltea los folds ya guardados) y `predecir_beto_promedio` (promedio de los 5 sobre el test).
- **`features.py`:** `PromedioConBeto` (= (logística `PrediccionOOF` + `prob_beto_new`) / 2, salida `prob_texto_new`) y `con_beto=False` en `armar_pipeline_final` y `armar_pipeline_histgb_combos` (con `False`, idénticos a antes).
- **Controles antes de entrenar** (con el BETO de la Fase Q): los pipelines con `con_beto=True` reproducen las OOF de la variante (a) (HistGB y CatBoost, diferencia 1e-16) y `submission_texto_combinado.csv` (0 diferencias); con `con_beto=False`, P7a sin cambios. Guardar y cargar un BETO da la misma predicción (diferencia 0).
- **`train.ipynb`:** sección 3 nueva (`oof_beto`), pipelines con `con_beto=True`. **103,7 min** (5 BETO de ~20 min en MPS, enchufada, sin Chrome ni Spotify). Pesos: `modelos_beto/` (2,0 GB, en `.gitignore`). El modelo anterior quedó en `modelo_final_p7c_duracion.joblib`.
- **`predict.ipynb`:** calcula `prob_beto_new` del test con los 5 BETO. **7,3 min.** Dos corridas: **idénticas byte a byte** (MD5 6d5ea02b…). Copia: `submission/submission_beto_entrega.csv`.
- **BETO reentrenado vs Fase Q:** OOF del train con diferencia mediana 2e-7 y máxima 0,002 (accuracy 0,8497 en los dos, 0% de discrepancia); test con diferencia máxima 0,0003. Las semillas funcionan: en MPS no es bit a bit, pero casi.
- **La submission de la entrega difiere de la de prueba en 140 predicciones (0,47%)** y de `submission_p7c_duracion.csv` en 382 (1,27%). Diagnóstico: con los modelos nuevos y el BETO de test **viejo** salen las mismas 140 diferencias, y 0 contra la entrega. Las 140 vienen **del lado del train**: diferencias de hasta 0,002 en la probabilidad de BETO del train mueven algunos cortes de los árboles y cambian predicciones cerca de 0,5. Es variabilidad de reentrenamiento, no un error.
- **Leaderboard público: 0,94438** (`submission_beto_entrega.csv`, subida el 2026-10-27 con `05_submit`), contra 0,94247 de P7c + duración: **+0,0019**, en línea con la CV (+0,0015). Leaderboard público: Base HistGB 0,93476 → CatBoost P6b 0,93695 → P7c + duración 0,94247 → **+ BETO 0,94438**.
- **Pendiente:** confirmación con la semilla 7 (15 ajustes de BETO, ~4,4 h).

### Paso 9 — Semana 3: `13_comparacion_modelos` corrida (2026-11-09)

- **Qué se corrió:** la celda `MODELOS` con regresión logística, Random Forest, Gradient Boosting (HistGB con grilla), Voting (soft), Stacking y el modelo final (el árbol de decisión y el resto quedaron en `False`; la tabla de los 14 está en la Fase A). Misma `StratifiedKFold(5, shuffle=True, random_state=42)` para todos. **60,7 min** con la máquina libre (Chrome cerrado), lanzada con `ejecutar.py` + `nohup` + `caffeinate`.
- **Cambios en la notebook antes de correrla** (no tocan `features.py`, `train` ni `predict`):
  - `ajustar_y_registrar` y `medir_ensamble` imprimen cuántos minutos tardó cada modelo.
  - Celda nueva después del modelo final: suma **"CatBoost + BETO", "HistGB con combinaciones + BETO" y "Entrega actual: promedio 50/50 + BETO"**. No reentrena BETO: lee las OOF de la Fase T (`oof_texto_combinado.csv`, variante (a) exacta), con un `assert` de que los folds y la `y` coinciden con los de la notebook.
  - Conclusión escrita en la última celda (texto, sin reejecutar).
- **Tabla (confirmada, outputs guardados en la notebook):**

| Modelo | Accuracy | Desvío | ROC AUC | Mejores parámetros | Tiempo |
|---|---|---|---|---|---|
| **Entrega actual: promedio 50/50 + BETO** | **0,9435** | 0,0008 | 0,9864 | (logística + BETO)/2 | OOF de la Fase T |
| CatBoost + BETO | 0,9431 | 0,0007 | 0,9859 | — | OOF de la Fase T |
| HistGB con combinaciones + BETO | 0,9425 | 0,0014 | 0,9859 | — | OOF de la Fase T |
| Promedio 50/50 sin BETO (P7c + duración) | 0,9420 | 0,0017 | 0,9860 | 50/50 | 3,0 min (los dos) |
| CatBoost | 0,9414 | 0,0013 | 0,9854 | por defecto | |
| HistGB con combinaciones | 0,9412 | 0,0016 | 0,9855 | | |
| Stacking (GB + RF + logística) | 0,9384 | 0,0017 | 0,9831 | | 9,0 min |
| Gradient Boosting (HistGB) | 0,9384 | 0,0014 | 0,9845 | `learning_rate=0.1`, `max_depth=None`, `max_iter=200` | 3,3 min |
| Voting (soft: logística + RF + GB) | 0,9357 | 0,0021 | 0,9823 | | 3,7 min |
| Random Forest | 0,9291 | 0,0017 | 0,9787 | `max_depth=20`, `n_estimators=200` | 1,7 min |
| Regresión logística | 0,9218 | 0,0021 | 0,9765 | `C=1`, `penalty="l1"` | 39,9 min |

- **Comparación pareada de la entrega** (diferencia por fold, B − A):
  - contra CatBoost + BETO: [+0,0009, −0,0003, +0,0012, −0,0001, +0,0003] → +0,0004, 3/5;
  - contra HistGB + BETO: [−0,0001, +0,0022, 0, +0,0010, +0,0019] → +0,0010, 3/5;
  - contra el promedio sin BETO: [+0,0001, +0,0034, +0,0019, +0,0015, +0,0009] → **+0,0015, 5/5** (los mismos números de la Fase T);
  - contra el HistGB de la grilla: **+0,0052, 5/5**.
- **Lectura:**
  - Con las mismas features, boosting > Random Forest > logística. Combinar modelos más flojos no suma (Voting empeora, Stacking empata con el GB solo): lo mismo que en las Fases A y D.
  - Lo que separa al modelo final del GB de la grilla (+0,005): la codificación del vendedor y la categoría (CatBoost crudo y target encoding de combinaciones), el promedio de los dos y la probabilidad de texto con BETO.
  - La entrega no le gana con claridad a CatBoost + BETO solo (3/5). Se mantiene el 50/50 porque es la configuración que se eligió por CV en la Fase F (5/5 con las dos semillas) y la que se validó en el leaderboard (0,94438); además tiene el menor desvío entre folds después de CatBoost + BETO. **Interpretación, no medido aparte:** cambiar ahora a CatBoost + BETO solo sería elegir con los mismos folds que se están reportando.
  - La regresión logística tardó 40 min (grilla de 5 valores de `C` × L1/L2 = 10 combinaciones × 5 folds, con `liblinear`, que es lento con C alto); el resto, minutos.
- **Para la presentación:** esta tabla es la diapositiva de "selección de modelos".
- **Submissions finales (elegidas el 2026-11-10):**
  1. `submission/submission_beto_entrega.csv`: la entrega con BETO (CV 0,9435, público 0,94438; la reproducen `train` + `predict`);
  2. `submission/submission_p7c_duracion.csv`: P7c + duración sin BETO (CV 0,9420 con la semilla 42 y 0,9422 con la 7, público 0,94247), como respaldo más simple y confirmado con dos semillas. Difieren en 382 predicciones.
  - Por qué estas dos: son las dos mejores por CV que pasaron el criterio, y una no depende de BETO, así que si BETO generalizara peor en el privado queda la otra.
- **EDA nueva (`eda.ipynb`, raíz):** 11 secciones (carga, target, faltantes, outliers, numéricas, categóricas, vendedor, texto, duración, train vs test, conclusión que une cada hallazgo con `features.py`). Datos nuevos medidos ahí: predecir la clase mayoritaria de las otras publicaciones del vendedor (*leave-one-out*) da 0,9234 en el 71,5% de las filas que tienen vendedor repetido; palabras del título más asociadas a cada clase.
- **Cierre de la entrega (2026-11-10, sin cambiar lógica):**
  - `train.ipynb`: `import transformers` + `transformers.logging.set_verbosity_error()` en la celda de imports (oculta los LOAD REPORT; probado aparte: desaparecen, las barras de tqdm `Loading weights` quedan). Outputs de la celda de `oof_beto` (la 7) limpiados; **no se volvió a correr**. Copia previa en el scratchpad de la sesión.
  - `trabajo_realizado/figuras/shap_ejemplo_duracion.png` regenerada (`21_shap`, 1,5 min): la etiqueta dice "60 días + 8 s". Solo cambió esa función de dibujo; las otras 4 figuras salieron idénticas byte a byte. **Ojo:** la `21_shap` explica P7c + duración **sin BETO** (`con_beto=False`).
  - `README.md` reescrito para la entrega con BETO (tiempos, archivos grandes con lugar para el link de Drive, reproducibilidad, dependencias).
  - **Para correr solo `predict.ipynb`** hacen falta, además del repo: `modelo_final.joblib` (40 MB) + `modelos_beto/fold_1..5/` (2,2 GB; de cada fold alcanzan `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`; `oof.npy` solo lo usa `train` para saltear el ajuste) + `data/test_data.jsonlines` (81 MB, de Kaggle).
- **Reorganización del repo (2026-11-09):** en la raíz quedan solo la entrega (`eda.ipynb`: EDA limpia de 11 secciones, generada con `.cache_laboratorio/armar_eda.py`, `train.ipynb`, `predict.ipynb`, `features.py`, `texto_beto.py`, `modelos_beto/`, `modelo_final.joblib`, `data/`, `submission/`). `notebooks/` se renombró a `trabajo_realizado/` (misma profundidad: los caminos relativos de las notebooks viejas siguen andando) y ahí se movieron esta bitácora y los `.joblib` viejos. Las referencias `notebooks/` se cambiaron en el texto (también en los comentarios de `features.py`; el código quedó idéntico). `ejecutar.py` ahora corre cada notebook en su propia carpeta.
- **Verificación corta de la entrega (2026-11-09):** `predict.ipynb` corrido desde la estructura nueva (7,3 min, a batería) da `submission/submission.csv` **idéntica byte a byte** a `submission_beto_entrega.csv` (MD5 6d5ea02b…). No se reentrenó `train.ipynb` (descarté la verificación completa de ~2 h).

## 5. Configuración final (P7c + duración; la entrega actual le suma BETO, ver al final de la sección)

```
train/test crudos
  └─ aplanar_anidadas → agregar_features_contenido (incluye duracion_dias) → agregar_categoricas_crudas → agregar_combos
     (determinístico, afuera de los Pipeline)

Modelo 1 — CatBoost (armar_pipeline_final):
  CapeoPorMoneda → agregar_logs → ColumnTransformer:
       - 24 numéricas (COLUMNAS_NUMERICAS) + duracion_dias
       - TF-IDF del título (500 palabras)
       - prob_texto_new = PrediccionOOF(regresión logística sobre título + garantía)
       - 6 categóricas crudas (seller_id, category_id, listing_type_id, buying_mode, shipping_mode, seller_state)
  → CatBoostClassifier(random_state=42, verbose=0)        fit(..., modelo__cat_features=INDICES_CAT)  (INDICES_CAT = 526–531)

Modelo 2 — HistGB con combinaciones (armar_pipeline_histgb_combos):
  CapeoPorMoneda → agregar_logs → ColumnTransformer:
       - one-hot de 4 categóricas + las 24 numéricas + TF-IDF 500 + prob_texto_new
       - TargetEncoder(target_type="binary") sobre seller_id, category_id y las 19 COLUMNAS_COMBOS
       - VendedorEncoder(incluir_target=False): conteo y precio promedio del vendedor
       - duracion_dias (última columna)
  → HistGradientBoostingClassifier(learning_rate=0.1, max_depth=None, max_iter=200, random_state=42)

Predicción final: P(new) = 0,5 · P_CatBoost + 0,5 · P_HistGB → "new" si ≥ 0,5
```

| | CV semilla 42 | CV semilla 7 | Público Kaggle |
|---|---|---|---|
| CatBoost + duración | 0,9414 | 0,9406 | — |
| HistGB con combinaciones + duración | 0,9412 | 0,9412 | — |
| **Promedio 50/50 (modelo final)** | **0,9420** | **0,9422** | **0,94247** |

- Tiempos: CV de los dos modelos ~2 min por modelo con la máquina libre; `train.ipynb` ~2 min; `predict.ipynb` segundos.
- Detalle técnico: `cat_features` se pasa en el `fit` y no en el constructor. Con lista en el constructor, `sklearn.base.clone` falla (incompatibilidad conocida de CatBoost), y CatBoost no acepta tupla.
- Las notebooks viejas que llaman a `armar_pipeline_final()` o `armar_pipeline_histgb_combos()` (15, 17, verificaciones del laboratorio) darían ahora números con duración si se re-corren; sus outputs guardados son de antes.

### Entrega actual: + BETO (Paso 8, `con_beto=True`)

Es lo mismo que el esquema de arriba, con un solo cambio: `prob_texto_new` = `PromedioConBeto` = (logística `PrediccionOOF` + `prob_beto_new`) / 2.
- **Train:** `prob_beto_new` = OOF de 5 BETO (`texto_beto.oof_beto`, `StratifiedKFold(5, shuffle=True, random_state=42)`, pesos en `modelos_beto/fold_k/`).
- **Test:** promedio de los 5 BETO.

| | CV semilla 42 | CV semilla 7 | Público Kaggle |
|---|---|---|---|
| CatBoost + BETO | 0,9431 | pendiente | — |
| HistGB con combinaciones + BETO | 0,9425 | pendiente | — |
| **Promedio 50/50 (entrega actual)** | **0,9435** | **pendiente** (15 ajustes de BETO, ~4,4 h) | **0,94438** |

- Tiempos: `train.ipynb` ~104 min (5 BETO de ~20 min en MPS), `predict.ipynb` ~7 min.

---

## 6. Qué está y qué no está en el material

**Del material:** XGBoost y LightGBM **solo nombrados** (`Árboles.pdf`, slides 64-65: título y logo, sin contenido ni notebook), Histogram Gradient Boosting (slide 67: histogramas en vez del dato puntual), SHAP (`material/SHAP_Values.ipynb`, agregado después de la Fase L), Hill Climber (`Árboles.pdf`, slide 75), regresión logística y regularización L1/L2, LDA/QDA/Naive Bayes, kNN, SVM, árboles (`max_depth`, `min_samples_leaf`, `max_leaf_nodes`, `ccp_alpha`), Random Forest, Extra Trees, AdaBoost, Gradient Boosting (incluye HistGB), **CatBoost**, Voting, `cross_val_predict`, `GridSearchCV`, curva ROC/AUC, target encoder y frequency encoder (versión naive), imputación por mediana con indicador, *feature engineering*.

**Material nuevo (después del 13/10):** SIA `S12 - Redes Neuronales - NLP` (tokens, embeddings, atención, transformers, pre-entrenamiento), S11 CNN, `material/Kmeans/AP_KMeans.ipynb`.

**No está en el material (explicado aparte en las notebooks):** fine-tuning de un transformer (BETO, AdamW, *scheduler* con *warmup*; Fase Q), voto ponderado (el parámetro `weights` de `VotingClassifier` solo aparece en una documentación impresa en S8), TF-IDF y n-gramas, *cross-fitting* y *smoothing* de `TargetEncoder`, *leave-one-out*, `PrediccionOOF`, Stacking (`S8_Ensambles.ipynb` dice que "no forma parte de la versión actual de la clase"; decidí mantenerlo en la comparación), comparación pareada fold por fold, `RandomizedSearchCV`, *early stopping*, `l2_regularization`, `Pipeline(memory=…)`, *ordered target statistics* de CatBoost, elección de umbral, embeddings de oraciones / *transfer learning* (Fase G), Optuna/TPE (Fase L), categóricas nativas de LightGBM (Fase N), *confident learning* (limpieza de etiquetas, Fase P), propagación de predicciones por vendedor (Fase P), conteos transductivos sobre train + test (Fases B, P).

---

## 7. Lo que no funcionó (útil para "limitaciones y mejoras")

- **Hiperparámetros:** ni la grilla de `train.ipynb`, ni 30 combinaciones al azar, ni promediar semillas mejoran al HistGB fijo.
- **Umbral:** 0,5 ya es óptimo (modelo calibrado).
- **Relaciones entre filas:** `parent_item_id` apunta afuera de la muestra; los títulos repetidos son muy pocos.
- **Más features de vendedor:** el perfil sin target y las correcciones de conteo no agregan nada sobre lo que ya codifica `seller_pct_new` (o CatBoost).
- **Límites del problema:**
  - Ruido de etiqueta: ≈ 1,4% de las filas son errores "seguros" (la etiqueta contradice el título).
  - Vendedores sin historial: concentran la mitad de los errores, y solo los ayuda el contenido de la publicación.
- **No aplicado por decisión:** V2 (conteo del vendedor en train + test) no serviría para una publicación nueva que llega sola.
- **Resto de las fechas al segundo** (Fase M): diferencias, sobrantes, componentes, milisegundos, coincidencias y ráfagas de publicación por vendedor no suman nada sobre la duración.
- **La duración es un artefacto de la plataforma** (60 días + unos segundos ≈ publicación creada por primera vez), no una propiedad del producto. Sirve en esta competencia porque el test tiene el mismo campo; en un sistema real habría que verificar que se siga generando igual.
- **Zona de duda:** fuera de P(new) entre 0,2 y 0,8 el modelo acierta el 97,2%; el 60% de los errores está en el 11% de filas donde duda, y ningún especialista mejoró ahí (información ambigua y ruido de etiqueta).
- **Ruido de etiqueta:** limpiar las filas "sospechosas" empeora (Fase P). Son casos ambiguos (libros viejos `new`, "sin usar") más que etiquetas mal puestas: es un límite del problema, no algo que se pueda sacar.
- **Precio compartido entre vendedores** (Fase P): la feature con más señal sola del TP (0,705) no suma sobre el modelo, porque lo que dice ya lo dicen el vendedor, el tipo de publicación y el texto.
- **Features derivadas** (Fase O): cantidades relativas y precio relativo a la categoría o al vendedor ya están implícitos en `sold_quantity`, `log_price` + categoría y las combinaciones.
- **LightGBM** (Fase N): con las mismas features es igual al HistGB; con `seller_id` como categórica nativa sobreajusta (0,923-0,932).
- **Atipicidad respecto de productos parecidos (Fase R):** solo el 18% de las publicaciones tiene "parecidos" de otros vendedores en su categoría, y donde los hay, la información ya está en cantidad, garantía, precio + categoría y texto. **Umbral por segmento:** el modelo está calibrado también en vendedores sin historial y en `free` (lo que pasó con la semilla 42 desapareció con la 7).
- **Especialista ponderado para vendedores sin historial (Fase S):** darles más peso a los vendedores con 1 publicación (en lugar de entrenar solo con ellos, como I4) no mejora al especialista en su grupo, y el promedio con el modelo final pasó con la semilla 42 (+0,0003, 5/5) pero no con la 7 (−0,0005, 0/5). Con ~30% del test sin historial, ese grupo es la principal limitación (0,893 contra ~0,964 del resto) y no se encontró cómo mejorarlo.
- **Transformer con fine-tuning (BETO, Fase Q):** es el mejor modelo de texto (0,8497 contra 0,8414). Como feature **extra** suma +0,0013 en 4/5 folds y no pasa el criterio. **Promediado con la logística** (Fase T) sí pasa y es la entrega actual. Limitaciones:
  - la confirmación con la semilla 7 cuesta ~4,4 h de GPU y no se corrió (decisión mía, 2026-11-04);
  - `train.ipynb` pasa de ~2 min a ~104 min.
- **Redes neuronales y TabNet** quedan entre 0,6 y 1,2 puntos por debajo de los árboles (coherente con la slide 3 del PDF de redes de AP).

---

## 8. Mapa de archivos

| Archivo | Qué es | ¿Se entrega? |
|---|---|---|
| `features.py` | Funciones y transformers compartidos, incluidos los dos pipelines del modelo final (sección 3 y 5) | Sí (lo importan train/predict) |
| `train.ipynb` | Fine-tuning de 5 BETO (`modelos_beto/`) + entrena CatBoost + HistGB (con duración, `con_beto=True`) y guarda `modelo_final.joblib` (~104 min) | Sí |
| `predict.ipynb` | Predice con los 5 BETO, promedia 50/50 y genera `submission/submission.csv` (idéntica a `submission_beto_entrega.csv`, público 0,94438; ~7 min) | Sí |
| `texto_beto.py` | Fine-tuning y predicción de BETO para la entrega (Paso 8) | Sí (lo importan train/predict) |
| `modelos_beto/` | Pesos de los 5 BETO de la entrega (2,0 GB, en `.gitignore`) | Lo genera `train` |
| `modelo_final_p7c_duracion.joblib` | Modelo anterior, sin BETO (público 0,94247) | No |
| `trabajo_realizado/33_artefactos` (armada con `.cache_laboratorio/armar_33.py`) | Búsqueda de artefactos de formato en todos los campos del JSON, ranking en la zona de duda, 5 elegidas sumadas al modelo (Fase W, no pasa) | No |
| `trabajo_realizado/34_beto_dapt` + `trabajo_realizado/bert_dapt.py` | BETO con preentrenamiento MLM sobre títulos + garantía de train + test, y fine-tuning de siempre (Fase X: texto solo +0,006 5/5, en el modelo completo −0,0000 contra la entrega → no pasa) | No |
| `trabajo_realizado/32_beto_contexto` + `trabajo_realizado/bert_contexto.py` | BETO con contexto tabular como texto (Fase V: texto solo 0,908, pero en el modelo completo +0,0008 4/5 sesgada y −0,0002 2/5 en el promedio de 3 → no pasa) | No |
| `trabajo_realizado/anteriores/` | Versiones viejas de `train`/`predict` (comparación de 14 modelos) | No |
| `trabajo_realizado/01_eda` … `07_nuevas_features` | Fase A | No |
| `trabajo_realizado/08_errores` … `12_tuning`, `11b_marginales` | Fase B (plan de 7 pasos) | No |
| `trabajo_realizado/13_comparacion_modelos` | Comparación de modelos para la presentación (**corrida el 2026-11-09**, tabla en el Paso 9) | No (la tabla va a la presentación) |
| `eda.ipynb` (raíz) | EDA limpia de 11 secciones (generada con `.cache_laboratorio/armar_eda.py` + `eda_conclusiones.json`) | Sí (junto a train y predict) |
| `trabajo_realizado/14_submission_base`, `15_submission_final`, `17_submission_p7c`, `19_submission_duracion` | Generan submissions intermedias (Base, solo CatBoost, P7c, **P7c + duración**) | No |
| `trabajo_realizado/16_ensambles` | OOF de 6 modelos y ensambles (Fase D) | No |
| `trabajo_realizado/18_barrido_campos` | Barrido de todos los campos (encontró la duración) | No |
| `trabajo_realizado/31_texto_ensamble` | Ensamble de modelos de texto: NB multinomial/Bernoulli, logística de caracteres/palabras, SVM calibrado, promediados con logística + BETO (Fase U, todo empeora contra la variante a) | No |
| `trabajo_realizado/30_submission_texto_combinado` | Submission de prueba de la variante (a) de la Fase T (sin subir; control idéntico a P7c + duración; cambian 366 predicciones) | No |
| `trabajo_realizado/29_texto_combinado` | Probabilidad de texto = combinación de logística + BETO guardado, en lugar de `prob_texto_new` (Fase T, **pasa con la semilla 42: 0,9435, +0,0015 5/5**; es la entrega actual, Paso 8; falta la semilla 7) | No |
| `trabajo_realizado/28_especialista_ponderado` | Especialista HistGB P7a con más peso en vendedores de 1 publicación + ablación por grupos + promedio en los sin historial (Fase S, pasa con la semilla 42 y no con la 7) | No |
| `trabajo_realizado/27_atipicidad_producto` | Atipicidad respecto de productos parecidos (vecinos por título), umbral por segmento, TE del precio exacto del vendedor (Fase R, nada pasa) | No |
| `trabajo_realizado/26_bert_texto` + `trabajo_realizado/bert_texto.py` | Fine-tuning de BETO sobre título + garantía, cross-fitting con 15 ajustes (Fase Q, +0,0013 4/5, no pasa). Modelos y probabilidades en `.cache_laboratorio/bert/` | No |
| `trabajo_realizado/25_ideas_finales` | R0 precio compartido, R1 limpieza de etiquetas, R2 propagación por vendedor, R3 conteos train + test (Fase P, nada pasa; cierre de experimentos) | No |
| `trabajo_realizado/24_features_derivadas` | Cantidades relativas y precio relativo a la categoría y al vendedor (Fase O, no suma) | No |
| `trabajo_realizado/23_lightgbm` | LightGBM L1 (features de P7a), L2 (categóricas nativas) y promedio de 3 modelos (Fase N, no suma) | No |
| `trabajo_realizado/22_barrido_timestamps` | Barrido fino de las 4 fechas al segundo + cargas por vendedor (Fase M, no suma) | No |
| `trabajo_realizado/20_tuning_catboost_optuna` | Tuning de CatBoost con Optuna (no mejora) | No |
| `trabajo_realizado/21_shap` + `trabajo_realizado/figuras/` | SHAP: importancia global de los dos modelos y 3 ejemplos (gráficos para la presentación) | No (las figuras van a la presentación) |
| `trabajo_realizado/laboratorio` | Laboratorio de features y rondas C, P, N, I, cascada | No |
| `trabajo_realizado/05_submit` | Utilidad para subir submissions (la corro a mano) | No |
| `trabajo_realizado/.cache_laboratorio/` | Caché del modelo de texto, registro `resultados.csv`, OOF guardadas, base de Optuna (en `.gitignore`) | — |
| `submission/` | CSVs generados (en `.gitignore`); **entrega actual: `submission_beto_entrega.csv` = `submission.csv`** (antes: `submission_p7c_duracion.csv`) | — |
| `requirements.txt` | Dependencias (`catboost` para el modelo final; `torch`, `sentence-transformers` (incluye `transformers`, usado por BETO), `tensorflow`, `pytorch-tabnet`, `optuna`, `shap`, `lightgbm` solo para experimentos). `pypdf` se instaló solo para leer los PDFs del material y no está en `requirements.txt` | — |

---

## 9. Plan de las 4 semanas que quedan (la competencia cierra a mediados de noviembre)

**Semanas 1–2: experimentos** (con el criterio de siempre: comparación pareada contra el modelo final, robustez con `random_state=7`, submission solo si pasa).
1. ~~**Barrido fino de timestamps**~~ **HECHO (Fase M, `22_barrido_timestamps.ipynb`): nada pasa, sin cambios.** Plan original: barrer `start_time`, `stop_time`, `date_created` y `last_updated` a nivel de segundos:
   - diferencias entre los 6 pares, en segundos exactos y en la parte "sobrante" respecto de días redondos;
   - componentes (segundos, minutos, hora, día de la semana, día del mes);
   - coincidencias exactas como flags.

   Cada candidata: señal sola + aporte sobre el HistGB P7a con duración; lo que pase, confirmado en el promedio final con las semillas 42 y 7. (Se sumaron milisegundos/valores redondos y cargas masivas por vendedor.)
2. ~~**LightGBM**~~ **HECHO (Fase N, `23_lightgbm.ipynb`): nada pasa, sin cambios.** (Después del barrido; está en `material/Arboles/Árboles.pdf`: citarlo y explicar en qué se parece y en qué se diferencia del HistGB).
   - L1: con las mismas features del HistGB P7a (comparación pareada contra P7a).
   - L2: con `seller_id`, `category_id` y las categóricas como categóricas nativas (`categorical_feature`), sin target encoding, subiendo `min_data_per_group` y `cat_smooth` (~28.000 vendedores).
   - Si alguno queda parecido: promedio de 3 modelos contra el modelo final (pesos iguales, y pesos elegidos con 4 folds y evaluados en el quinto).

3. ~~**Features derivadas**~~ **HECHO (Fase O, `24_features_derivadas.ipynb`): nada pasa.** (Agregado al plan después de revisar "qué columnas no se usan": todos los campos crudos ya estaban usados o medidos; quedaban solo features derivadas.)
4. ~~**Ronda final de ideas R0–R3**~~ **HECHO (Fase P, `25_ideas_finales.ipynb`): nada pasa.** (Ideas de otra IA a partir de un resumen del TP, más R0, idea propia.)

**Experimentos cerrados (decisión mía, 2026-10-13).** Excepciones posteriores, que decidí probar igual:
- **Fase Q** (BETO como feature extra): no pasó.
- **Fase T** (logística + BETO): pasó → **entrega actual** (Paso 8, 2026-10-27, público 0,94438).
- Fases R, S y U: no pasaron.
- **Fase V** (BETO con contexto, 2026-11-04): no pasó.
- **Fase W** (artefactos de formato, 2026-11-04): no pasó.
- **Fase X** (BETO con preentrenamiento adaptado, 2026-11-09): no pasó.

**Pendientes al 2026-11-04** (la máquina corre de a una cosa por vez; el orden lo decido sobre la marcha):
1. ~~Semilla 7 de la Fase T~~ **No se corre** (decisión mía, 2026-11-04).
2. ~~**Fase V**~~ **HECHO (2026-11-04): no pasa.**
3. ~~**Fase W** (artefactos de formato)~~ **HECHO (2026-11-04): no pasa.**
4. ~~**Fase X** (BETO con preentrenamiento adaptado)~~ **HECHO (2026-11-09): no pasa.**

**Semana 3: congelar.**
- Congelar features y modelo.
- ~~Correr **una vez** `13_comparacion_modelos.ipynb`~~ **HECHO (2026-11-09, 60,7 min):** tabla en el Paso 9.
- Dejar `train.ipynb` / `predict.ipynb` definitivos (verificar que reproduzcan la submission elegida).
- ~~Elegir las 2 submissions finales~~ **HECHO (2026-11-10):** `submission_beto_entrega.csv` (CV 0,9435, público 0,94438) y `submission_p7c_duracion.csv` (CV 0,9420 / 0,9422, público 0,94247).

**Semana 4: presentación y práctica de la defensa.**
- Presentación de 5 min: baseline (0,822) → selección de modelos (tabla de la 13) → modelo final (CatBoost + HistGB con combinaciones + texto + duración; gráficos SHAP de `trabajo_realizado/figuras/`) → limitaciones y mejoras (sección 7).
- Practicar la explicación del código (tengo que poder explicar todo): `PrediccionOOF`, target encoding con sus resguardos, *ordered target statistics*, comparación pareada, por qué la duración no es leakage.

---

## 10. Tropiezos del proceso (para transparencia)

- En `12_tuning`, la primera corrida (30 min) se perdió por un error en los nombres de columnas de `cv_results_`. Se corrigió, se probó con una búsqueda de juguete y se volvió a correr.
- La corrida completa de la 13 se cortó a los 51 min por decisión: las features todavía iban a cambiar. Al cortarla se mataron todos los kernels de Jupyter de la compu, no solo el de la corrida.
- La primera corrida de P6b falló porque CatBoost no acepta `cat_features` como tupla. Se resolvió pasándolo en el `fit`.
- En la `18_barrido_campos`, 4 campos figuraban "pasa" con diferencia +0,0000: un artefacto de redondeo (diferencias de 1e-17 contadas como "gana"). Se corrigió redondeando.
- La primera versión del MLP de texto (sklearn `MLPClassifier`, ronda N) se lanzó sin medir un fold y no terminó en 2 horas. Desde entonces se mide un fold antes de cada corrida larga; la versión en PyTorch con capa sparse tardó 0,4 min por ajuste.
- Se concluyó que `shap` "se trababa" con el HistGB; era la máquina saturada. Corregido.
- **La máquina (18 GB de RAM) se satura** con corridas pesadas en paralelo y otras apps abiertas (Spotify, Chrome): el tuning de Optuna se cerró en 24 pruebas y su confirmación tardó ~36 h (con la máquina libre, ~30 min). Correr de a una cosa por vez.
- En la `22_barrido_timestamps`, la primera versión de "segundos desde la publicación anterior del vendedor" dependía del orden de las filas en los empates y daba distinta distribución en train y en test (14,6% contra 30% de "0 s"). Se detectó en la EDA, antes de medir, y se corrigió usando instantes distintos.
- Los scripts de `.venv/bin` (`jupyter`, etc.) apuntan a la carpeta vieja "kaggle-ap-2026 2" y no andan. Para ejecutar notebooks desde la terminal se usó `nbclient` con `.venv/bin/python3` (`nbconvert` no está instalado).
- En la `25_ideas_finales`, la primera corrida con R0 falló en una celda de EDA (`pd.cut` sobre un array de numpy devuelve un `Categorical`, que no acepta `value_counts(normalize=True)`). Se corrigió y se volvió a correr. R1–R3 dieron idéntico en las dos corridas (todo es determinístico y está cacheado).
- **Fase R:** la terminal perdió el permiso de lectura sobre `material/` (Google Drive): `Operation not permitted`, incluso fuera del sandbox. Hay que darle a VS Code acceso total al disco (o a la carpeta de Google Drive) en Privacidad y seguridad.
- **Fase Q:** el ajuste de BETO `sin_5` estuvo **~11 h** casi sin avanzar (menos de 3 min de procesador), con la máquina a batería y el swap lleno (17,8 de 19 GB). `sin_4` ya había tardado 69 min en lugar de ~21. Se cortó solo esa cola (script, `caffeinate` y Python) y se relanzó con la compu enchufada: los modelos ya guardados no se pierden porque el script los saltea. **Lección: entrenar en GPU solo enchufada, con la tapa abierta y sin apps pesadas.**
- La carpeta temporal de scripts se borró entre sesiones (con los scripts que armaban las notebooks 22-26). Las notebooks están en el repo, así que no se perdió nada entregable.
- La `22_barrido_timestamps` tardó 55 min en lugar de ~30: algunas candidatas tardaron 4-5 min por carga de la máquina.
- En la Fase N, el plan decía "subir `min_data_per_group` y `cat_smooth`" para las categóricas nativas de LightGBM. Subirlos empeoró, así que se agregaron configuraciones en el sentido contrario (también peores). Quedó documentado como cambio de plan, no como resultado elegido a posteriori.
- **LightGBM**: la librería busca `libomp` solo en rutas de Homebrew/MacPorts, y no hay Homebrew. Se resolvió dentro del `.venv`: enlace a la `libomp` de sklearn + `install_name_tool -add_rpath @loader_path` y firma local (`codesign -s -`). Si se reinstala lightgbm hay que repetirlo. Se instaló también `pypdf` (solo para leer el texto de los PDFs del material).
- En `08_errores` se describió mal un ejemplo ("antigüedades a estrenar" como ruido de etiqueta). "Departamento a estrenar" está bien etiquetado `new`: el error es del modelo. Corregido en las tres trabajo_realizado/documentos.


**Verificación de réplica (2026-11-10):** se copió el repo a una carpeta temporal con `modelos_beto/` y sin `modelo_final.joblib`. `train.ipynb` salteó el fine-tuning de los 5 BETO (1,7 min) y `predict.ipynb` tardó 7,1 min. La submission es idéntica byte a byte a `submission_beto_entrega.csv`. El `modelo_final.joblib` nuevo difiere del original solo en metadatos (`model_guid` y `train_finish_time` de CatBoost, `_stop_words_id` de `TfidfVectorizer`), y `predict_proba` del CatBoost y del HistGB es idéntico bit a bit sobre el train. Se agregó la sección "Replicar el modelo final" al README.

**Respaldo reproducible (2026-11-10):** se agregó la constante `CON_BETO` a `train.ipynb` y `predict.ipynb`; con `True` el comportamiento es el mismo de antes. Se verificó en copias limpias: con `False`, la submission es idéntica byte a byte a `submission_p7c_duracion.csv`; con `True` y los pesos de BETO, idéntica a `submission_beto_entrega.csv`. `modelo_final_p7c_duracion.joblib` se movió a la raíz y se sube al repo.
