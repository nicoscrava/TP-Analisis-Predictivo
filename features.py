"""Transformaciones de features compartidas entre train.ipynb y predict.ipynb.

Decisiones tomadas y justificadas en trabajo_realizado/01_eda.ipynb (aplanado de
anidadas), trabajo_realizado/02_features.ipynb y trabajo_realizado/03_titulo.ipynb (qué
columnas suman), trabajo_realizado/04_vendedor.ipynb (VendedorEncoder),
trabajo_realizado/09_texto.ipynb (PrediccionOOF, modelo de texto) y
trabajo_realizado/11b_marginales.ipynb (fotos, fechas y garantía como texto) y
trabajo_realizado/laboratorio.ipynb (modelo final P7c: promedio 50/50 de CatBoost con
categóricas crudas, armar_pipeline_final, y HistGB con combinaciones,
armar_pipeline_histgb_combos + agregar_combos), con la duración de la
publicación (trabajo_realizado/18_barrido_campos.ipynb) y, con con_beto=True, la
probabilidad de texto promediada con BETO (PromedioConBeto,
trabajo_realizado/29_texto_combinado.ipynb; el fine-tuning está en texto_beto.py).
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, TargetEncoder

COLUMNAS_CONTENIDO = [
    "foto_log_px_primera", "foto_pct_cuadradas", "foto_pct_camara", "foto_n_tamanios",
    "hora_inicio", "gap_actualizacion",
]


COLUMNAS_NUMERICAS = [
    "log_price", "log_available_quantity", "automatic_relist",
    "sold_quantity", "initial_quantity",
    "n_fotos", "n_atributos", "n_variaciones", "tiene_video", "tiene_garantia",
    "free_shipping", "local_pick_up", "n_metodos_pago", "accepts_mercadopago",
    "es_tienda_oficial", "tiene_descuento", "tiene_oferta", "tiene_parent_item",
] + COLUMNAS_CONTENIDO

COLUMNAS_CATEGORICAS = ["seller_id", "category_id", "listing_type_id", "buying_mode",
                        "shipping_mode", "seller_state"]
COLUMNAS_CAT_CRUDAS = [f"cat_{c}" for c in COLUMNAS_CATEGORICAS]

# Posición de las categóricas en la salida del ColumnTransformer de armar_pipeline_final:
# numéricas + duracion_dias -> TF-IDF (500) -> prob_texto_new (1) -> categóricas crudas (6)
INDICES_CAT = list(range(len(COLUMNAS_NUMERICAS) + 1 + 501,
                         len(COLUMNAS_NUMERICAS) + 1 + 501 + len(COLUMNAS_CATEGORICAS)))

COLUMNAS_COMBOS = [
    "c_seller_cat", "c_seller_lt", "c_seller_bm", "c_seller_sm", "c_cat_lt", "c_cat_bm", "c_cat_price",
    "c_seller_price", "c_cat_sm", "c_cat_state", "c_seller_cat_lt", "c_cat_garantia", "c_seller_garantia",
    "c_currency", "c_status", "c_store", "c_tags", "c_pagos", "c_attr",
]

def aplanar_anidadas(df):
    """Aplana columnas dict/list y crea flags de nulos. No aprende nada de los
    datos (sin fit), por eso se aplica igual a train y test, afuera del Pipeline.
    """
    df = df.copy()

    df["seller_state"] = df["seller_address"].map(lambda d: d.get("state", {}).get("name"))
    df["shipping_mode"] = df["shipping"].map(lambda d: d.get("mode"))
    df["free_shipping"] = df["shipping"].map(lambda d: d.get("free_shipping"))
    df["local_pick_up"] = df["shipping"].map(lambda d: d.get("local_pick_up"))

    columnas_conteo = {
        "non_mercado_pago_payment_methods": "n_metodos_pago",
        "attributes": "n_atributos",
        "variations": "n_variaciones",
        "pictures": "n_fotos",
    }
    for col_original, col_nueva in columnas_conteo.items():
        df[col_nueva] = df[col_original].map(len)

    columnas_flag_lista = {"sub_status": "tiene_sub_status", "deal_ids": "tiene_oferta"}
    for col_original, col_nueva in columnas_flag_lista.items():
        df[col_nueva] = df[col_original].map(len) > 0

    tags_posibles = [
        "dragged_bids_and_visits", "dragged_visits", "free_relist",
        "good_quality_thumbnail", "poor_quality_thumbnail",
    ]
    for tag in tags_posibles:
        df[f"tag_{tag}"] = df["tags"].map(lambda lista, t=tag: t in lista)

    columnas_flag_nulos = {
        "original_price": "tiene_descuento",
        "official_store_id": "es_tienda_oficial",
        "video_id": "tiene_video",
        "warranty": "tiene_garantia",
        "parent_item_id": "tiene_parent_item",
    }
    for col_original, col_nueva in columnas_flag_nulos.items():
        df[col_nueva] = df[col_original].notna()

    return df



def _medidas_fotos(lista_fotos):
    """Ancho y alto (en píxeles) de cada foto, a partir de max_size "ANCHOxALTO"."""
    medidas = []
    for foto in lista_fotos:
        try:
            ancho, alto = foto["max_size"].split("x")
            medidas.append((int(ancho), int(alto)))
        except (KeyError, AttributeError, ValueError):
            continue
    return medidas


def _es_proporcion_camara(ancho, alto):
    return abs(ancho / alto - 4 / 3) < 0.01 or abs(ancho / alto - 3 / 4) < 0.01


def agregar_features_contenido(df):
    """Fotos (bloque N, trabajo_realizado/11_anidadas.ipynb), fechas (bloque L,
    trabajo_realizado/07_nuevas_features.ipynb) y warranty como texto (T3,
    trabajo_realizado/09_texto.ipynb), elegidas en trabajo_realizado/11b_marginales.ipynb; y
    duracion_dias = stop_time - start_time en días (trabajo_realizado/18_barrido_campos.ipynb:
    las publicaciones que duran 60 días + unos segundos son casi todas used).
    Determinística (no aprende nada de los datos): va afuera del Pipeline.
    """
    df = df.copy()

    df["warranty_texto"] = df["warranty"].fillna("")

    medidas = df["pictures"].map(_medidas_fotos)
    df["foto_log_px_primera"] = medidas.map(lambda m: np.log(m[0][0] * m[0][1]) if m else np.nan)
    df["foto_pct_cuadradas"] = medidas.map(lambda m: np.mean([a == h for a, h in m]) if m else np.nan)
    df["foto_pct_camara"] = medidas.map(
        lambda m: np.mean([_es_proporcion_camara(a, h) for a, h in m]) if m else np.nan)
    df["foto_n_tamanios"] = medidas.map(lambda m: len(set(m)))

    ultima_actualizacion = pd.to_datetime(df["last_updated"]).dt.tz_localize(None)
    df["hora_inicio"] = df["start_time"].dt.hour
    df["gap_actualizacion"] = (ultima_actualizacion - df["start_time"]).dt.total_seconds() / 86400
    df["duracion_dias"] = (pd.to_datetime(df["stop_time"]) - pd.to_datetime(df["start_time"])).dt.total_seconds() / 86400

    return df


def agregar_logs(df):
    """Agrega log_price (log, price capeado nunca es 0) y log_available_quantity
    (log1p, esta sí puede ser 0). Determinística, no aprende nada de los datos.
    """
    df = df.copy()
    df["log_price"] = np.log(df["price"])
    df["log_available_quantity"] = np.log1p(df["available_quantity"])
    return df


class VendedorEncoder(BaseEstimator, TransformerMixin):
    """Agrega features de vendedor a partir de seller_id + log_price.

    seller_n_listings y seller_avg_log_price NO usan el target: se calculan
    igual en fit/transform, sin riesgo de leakage.

    seller_pct_new SÍ usa el target (% de las publicaciones de ese vendedor
    que son 'new'). Por eso fit_transform (usado sobre las filas de train)
    hace leave-one-out: para la fila i, excluye su propio y al promediar.
    transform (usado sobre filas que el transformer nunca vio, ej. el fold
    de validación o el test real) usa el promedio completo del vendedor sin
    excluir nada, porque esas filas nunca participaron en ese promedio.

    Vendedores no vistos en fit (o con 0 publicaciones restantes tras
    excluirse a sí mismos) caen a un promedio global de respaldo.
    """

    def __init__(self, incluir_target=True):
        self.incluir_target = incluir_target

    def fit(self, X, y=None):
        seller = X["seller_id"]
        self.conteo_ = seller.value_counts()
        self.media_log_price_vendedor_ = X.groupby(seller)["log_price"].mean()
        self.media_log_price_global_ = X["log_price"].mean()
        if self.incluir_target:
            if y is None:
                raise ValueError("VendedorEncoder necesita y para calcular seller_pct_new")
            y_serie = pd.Series(np.asarray(y), index=X.index)
            self.suma_y_vendedor_ = y_serie.groupby(seller).sum()
            self.media_y_global_ = y_serie.mean()
        return self

    def _agregados_simples(self, X):
        seller = X["seller_id"]
        conteo = seller.map(self.conteo_).fillna(0)
        n_listings = np.log1p(conteo)
        avg_log_price = seller.map(self.media_log_price_vendedor_).fillna(self.media_log_price_global_)
        return seller, pd.DataFrame({
            "seller_n_listings": n_listings.to_numpy(),
            "seller_avg_log_price": avg_log_price.to_numpy(),
        }, index=X.index)

    def transform(self, X):
        seller, salida = self._agregados_simples(X)
        if self.incluir_target:
            media_y_vendedor = self.suma_y_vendedor_ / self.conteo_
            salida["seller_pct_new"] = seller.map(media_y_vendedor).fillna(self.media_y_global_).to_numpy()
        return salida

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        seller, salida = self._agregados_simples(X)
        if self.incluir_target:
            y_serie = pd.Series(np.asarray(y), index=X.index)
            conteo_fila = seller.map(self.conteo_)
            suma_fila = seller.map(self.suma_y_vendedor_)
            denominador = conteo_fila - 1
            pct_new = np.where(
                denominador > 0,
                (suma_fila - y_serie) / denominador,
                self.media_y_global_,
            )
            salida["seller_pct_new"] = pct_new
        return salida


class CapeoPorMoneda(BaseEstimator, TransformerMixin):
    """Capea (winsoriza) columnas numéricas al percentil `percentil`, calculado
    por separado para cada categoría de `columna_grupo` (ej. moneda).

    El techo se aprende en .fit() (solo con los datos que se le pasan, ej. el
    train de cada fold de la CV) y se aplica igual en .transform(). Si aparece
    una categoría de columna_grupo no vista en fit, usa un techo global de
    respaldo en vez de romper.
    """

    def __init__(self, columnas, columna_grupo, percentil=0.995):
        self.columnas = columnas
        self.columna_grupo = columna_grupo
        self.percentil = percentil

    def fit(self, X, y=None):
        self.techos_ = {}
        self.techos_globales_ = {}
        for col in self.columnas:
            self.techos_[col] = X.groupby(self.columna_grupo)[col].quantile(self.percentil)
            self.techos_globales_[col] = X[col].quantile(self.percentil)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columnas:
            techo = X[self.columna_grupo].map(self.techos_[col]).fillna(self.techos_globales_[col])
            X[col] = np.minimum(X[col], techo)
        return X


class PrediccionOOF(BaseEstimator, TransformerMixin):
    """Envuelve un clasificador y devuelve su probabilidad de clase 1 como una feature.

    fit_transform (filas de train): probabilidad out-of-fold con una CV interna,
    cada fila predicha por un modelo que no la vio. Sin esto, el modelo envuelto
    casi memoriza el train y el modelo siguiente aprendería a confiar en
    probabilidades demasiado buenas, que en test no se repiten.
    transform (filas nuevas: validación o test): modelo ajustado con todo el train.
    Misma lógica de cross-fitting que sklearn.preprocessing.TargetEncoder.
    """

    def __init__(self, estimador, n_splits=5, random_state=42):
        self.estimador = estimador
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        self.estimador_ = clone(self.estimador).fit(X, y)
        return self

    def transform(self, X):
        return self.estimador_.predict_proba(X)[:, [1]]

    def fit_transform(self, X, y):
        cv_interna = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        prob_oof = cross_val_predict(clone(self.estimador), X, y, cv=cv_interna,
                                     method="predict_proba")[:, [1]]
        self.fit(X, y)
        return prob_oof

    def get_feature_names_out(self, input_features=None):
        return np.array(["prob_texto_new"])


class PromedioConBeto(BaseEstimator, TransformerMixin):
    """Probabilidad de texto = (logística de texto + BETO) / 2 (trabajo_realizado/29_texto_combinado.ipynb,
    variante a de la Fase T). La logística (modelo_texto, un PrediccionOOF) se aprende adentro del
    Pipeline, como siempre. BETO no se puede reentrenar dentro de cada fold, así que entra ya
    calculado en la columna columna_beto, con la misma regla: en el train, la probabilidad de un
    modelo que no vio la fila (texto_beto.oof_beto); en el test, el promedio de los 5 modelos.
    """

    def __init__(self, modelo_texto, columna_beto="prob_beto_new"):
        self.modelo_texto = modelo_texto
        self.columna_beto = columna_beto

    def fit(self, X, y):
        self.modelo_texto_ = clone(self.modelo_texto).fit(X.drop(columns=[self.columna_beto]), y)
        return self

    def transform(self, X):
        logistica = self.modelo_texto_.transform(X.drop(columns=[self.columna_beto]))
        return (logistica + X[[self.columna_beto]].to_numpy()) / 2

    def fit_transform(self, X, y):
        self.modelo_texto_ = clone(self.modelo_texto)
        logistica = self.modelo_texto_.fit_transform(X.drop(columns=[self.columna_beto]), y)
        return (logistica + X[[self.columna_beto]].to_numpy()) / 2

    def get_feature_names_out(self, input_features=None):
        return np.array(["prob_texto_new"])


def _transformer_texto(con_beto):
    if con_beto:
        return ("texto", PromedioConBeto(PrediccionOOF(armar_modelo_texto(), random_state=42)),
                ["title", "warranty_texto", "prob_beto_new"])
    return ("texto", PrediccionOOF(armar_modelo_texto(), random_state=42), ["title", "warranty_texto"])


def armar_modelo_texto(C=2):
    """Modelo de texto que va adentro de PrediccionOOF (trabajo_realizado/09_texto.ipynb):
    TF-IDF del título (palabras 1-2 + caracteres 3-5) y de la garantía
    (palabras 1-2) -> regresión logística. Necesita las columnas
    title y warranty_texto (esta última la crea agregar_features_contenido).
    """
    vectores = ColumnTransformer([
        ("palabras", TfidfVectorizer(ngram_range=(1, 2), min_df=2, strip_accents="unicode"), "title"),
        ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                       strip_accents="unicode"), "title"),
        ("garantia", TfidfVectorizer(ngram_range=(1, 2), min_df=2, strip_accents="unicode"), "warranty_texto"),
    ])
    return Pipeline([
        ("vectores", vectores),
        ("modelo", LogisticRegression(C=C, max_iter=1000, random_state=42)),
    ])


def agregar_categoricas_crudas(df):
    """Copia las categóricas como texto (cat_<columna>), para que CatBoost las use
    crudas como cat_features. Determinística: va afuera del Pipeline."""
    df = df.copy()
    for columna in COLUMNAS_CATEGORICAS:
        df[f"cat_{columna}"] = df[columna].astype(str)
    return df


def armar_pipeline_final(con_beto=False):
    """CatBoost del modelo final (trabajo_realizado/laboratorio.ipynb, prueba P6b, + duracion_dias
    de trabajo_realizado/18_barrido_campos.ipynb): categóricas crudas, modelo de texto y numéricas.
    Al entrenar hay que pasarle las categóricas: pipeline.fit(X, y, modelo__cat_features=INDICES_CAT).
    con_beto=True: la probabilidad de texto es (logística + BETO) / 2 (PromedioConBeto); X necesita
    la columna prob_beto_new. Misma posición, así que INDICES_CAT no cambia.
    """
    preprocesador = ColumnTransformer([
        ("num", "passthrough", COLUMNAS_NUMERICAS + ["duracion_dias"]),
        ("tfidf", TfidfVectorizer(max_features=500), "title"),
        _transformer_texto(con_beto),
        ("cat_crudas", "passthrough", COLUMNAS_CAT_CRUDAS),
    ], sparse_threshold=0)
    return Pipeline([
        ("capeo", CapeoPorMoneda(columnas=["price", "base_price"], columna_grupo="currency_id")),
        ("logs", FunctionTransformer(agregar_logs)),
        ("preprocesador", preprocesador),
        ("modelo", CatBoostClassifier(random_state=42, verbose=0)),
    ])


def agregar_combos(df):
    """Combinaciones de categóricas como texto (columnas c_*), más algunas
    categóricas sueltas (moneda, status, tienda oficial, tags, métodos de pago,
    ids de atributos). No usa el target; determinística: va afuera del Pipeline.
    Debe aplicarse DESPUÉS de aplanar_anidadas (usa tiene_garantia).
    El bin de precio usa price crudo: floor(2 * log10(price)).
    """
    df = df.copy()
    s = lambda c: df[c].astype(str)
    pb = np.floor(2 * np.log10(df["price"].clip(lower=1))).astype(int).astype(str)  # bin de precio fijo
    df["c_seller_cat"] = s("seller_id") + "_" + s("category_id")
    df["c_seller_lt"] = s("seller_id") + "_" + s("listing_type_id")
    df["c_seller_bm"] = s("seller_id") + "_" + s("buying_mode")
    df["c_seller_sm"] = s("seller_id") + "_" + s("shipping_mode")
    df["c_cat_lt"] = s("category_id") + "_" + s("listing_type_id")
    df["c_cat_bm"] = s("category_id") + "_" + s("buying_mode")
    df["c_cat_price"] = s("category_id") + "_" + pb
    df["c_seller_price"] = s("seller_id") + "_" + pb
    df["c_cat_sm"] = s("category_id") + "_" + s("shipping_mode")
    df["c_cat_state"] = s("category_id") + "_" + s("seller_state")
    df["c_seller_cat_lt"] = df["c_seller_cat"] + "_" + s("listing_type_id")
    df["c_cat_garantia"] = s("category_id") + "_" + s("tiene_garantia")
    df["c_seller_garantia"] = s("seller_id") + "_" + s("tiene_garantia")
    df["c_currency"] = s("currency_id")
    df["c_status"] = s("status")
    df["c_store"] = df["official_store_id"].fillna(-1).astype(int).astype(str)
    df["c_tags"] = df["tags"].map(lambda l: "|".join(sorted(l)))
    df["c_pagos"] = df["non_mercado_pago_payment_methods"].map(
        lambda l: "|".join(sorted(d.get("description", "") for d in l)))
    df["c_attr"] = df["attributes"].map(lambda l: "|".join(sorted(d.get("id", "") for d in l)))
    return df


def armar_pipeline_histgb_combos(con_beto=False):
    """HistGB P7a (trabajo_realizado/laboratorio.ipynb, ronda P7) + duracion_dias al final
    (trabajo_realizado/18_barrido_campos.ipynb): mismas features que la Base
    HistGB, pero el vendedor y la categoría se codifican con TargetEncoder (cross-fitting
    + smoothing) junto con las 19 combinaciones de agregar_combos, más VendedorEncoder
    sin target (conteo y precio promedio del vendedor). Necesita las columnas que crean
    aplanar_anidadas, agregar_features_contenido y agregar_combos.
    con_beto=True: la probabilidad de texto es (logística + BETO) / 2 (PromedioConBeto); X necesita
    la columna prob_beto_new.
    """
    preprocesador = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"),
         ["listing_type_id", "buying_mode", "shipping_mode", "seller_state"]),
        ("num", "passthrough", COLUMNAS_NUMERICAS),
        ("tfidf", TfidfVectorizer(max_features=500), "title"),
        _transformer_texto(con_beto),
        ("te_combos", TargetEncoder(target_type="binary", random_state=42),
         ["seller_id", "category_id"] + COLUMNAS_COMBOS),
        ("vendedor_simple", VendedorEncoder(incluir_target=False), ["seller_id", "log_price"]),
        ("duracion", "passthrough", ["duracion_dias"]),
    ], sparse_threshold=0)
    return Pipeline([
        ("capeo", CapeoPorMoneda(columnas=["price", "base_price"], columna_grupo="currency_id")),
        ("logs", FunctionTransformer(agregar_logs)),
        ("preprocesador", preprocesador),
        ("modelo", HistGradientBoostingClassifier(learning_rate=0.1, max_depth=None, max_iter=200,
                                                  random_state=42)),
    ])
