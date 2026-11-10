"""BETO con contexto tabular serializado como texto (trabajo_realizado/32_beto_contexto.ipynb, Fase V).

Misma receta y mismo cross-fitting que bert_texto.py (Fase Q): reusa su main() y solo cambia
1) la entrada: "título [SEP] garantía [SEP] categoría: ... [SEP] precio: ... [SEP] ... envío gratis: sí/no"
   (sin seller_id ni nada que use el target);
2) max_length: 128 (con 64 se truncaba el 57% de las filas; con 128, el 0,6%);
3) la carpeta de salida: .cache_laboratorio/bert_contexto/.

Uso (desde trabajo_realizado/), igual que bert_texto.py:
    python bert_contexto.py --excluir 1        # entrena sin el fold 1: predice fold 1 + test
    python bert_contexto.py --excluir 1,2      # entrena sin los folds 1 y 2: predice folds 1 y 2
"""
import os
import random

import numpy as np
import pandas as pd
import transformers
from sklearn.model_selection import StratifiedKFold

import bert_texto as bt


def contexto(fila):
    return " [SEP] ".join([
        f"categoría: {fila['category_id']}",
        f"precio: {round(fila['price'])} {fila['currency_id']}",
        f"tipo: {fila['listing_type_id']}",
        f"compra: {fila['buying_mode']}",
        f"cantidad: {fila['initial_quantity']}",
        f"vendidas: {fila['sold_quantity']}",
        f"fotos: {len(fila['pictures'])}",
        f"envío gratis: {'sí' if fila['shipping'].get('free_shipping') else 'no'}",
    ])


def cargar_con_contexto():
    train = pd.read_json("../data/train_data.jsonlines", lines=True)
    test = pd.read_json("../data/test_data.jsonlines", lines=True)
    todo = pd.concat([train.drop(columns="condition"), test], ignore_index=True)
    titulos = todo["title"].fillna("").tolist()
    garantias = todo["warranty"].fillna("")
    # Segundo segmento: garantía (si hay) + contexto. El tokenizer reconoce "[SEP]" como separador.
    segundo = (garantias.where(garantias == "", garantias + " [SEP] ") + todo.apply(contexto, axis=1)).tolist()
    y = (train["condition"] == "new").astype(int).to_numpy()
    folds = np.empty(len(train), dtype=int)
    for k, (_, val) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=bt.SEED).split(np.zeros(len(y)), y)):
        folds[val] = k + 1
    return titulos, segundo, y, folds


if __name__ == "__main__":
    random.seed(bt.SEED)
    transformers.set_seed(bt.SEED)          # además de torch y numpy, que fija bt.main()
    bt.cargar = cargar_con_contexto
    bt.MAX_LENGTH = 128
    bt.CARPETA = os.path.join(".cache_laboratorio", "bert_contexto")
    bt.main()
