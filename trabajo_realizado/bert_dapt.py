"""Preentrenamiento de BETO adaptado al dominio (Fase X, trabajo_realizado/34_beto_dapt.ipynb) + el mismo fine-tuning de siempre.

Paso 1 (--mlm): *masked language modeling* (MLM) sobre "título [SEP] garantía" de train + test (100.000 textos).
    Se tapa al azar el 15% de los tokens y BETO aprende a adivinarlos. NO usa `condition`: solo texto.
    1 época, AdamW lr 5e-5 (weight decay 0,01), warmup lineal el 10% de los pasos, batch 32, max_length 64.
    Guarda el modelo en .cache_laboratorio/bert_dapt/mlm/.
Paso 2 (--excluir): el mismo fine-tuning de la Fase Q/T (bert_texto.py: 2 épocas, lr 2e-5, ...) con los folds de
    StratifiedKFold(5, shuffle=True, random_state=42), pero partiendo del modelo del paso 1 en lugar de BETO original.
    Guarda en .cache_laboratorio/bert_dapt/ (no pisa bert/, bert_contexto/ ni modelos_beto/).

Uso (desde trabajo_realizado/):
    python bert_dapt.py --mlm --bench 50        # solo mide 50 pasos del MLM
    python bert_dapt.py --mlm                   # paso 1 (una sola vez)
    python bert_dapt.py --excluir 1 --bench 50  # solo mide 50 pasos del fine-tuning
    python bert_dapt.py --excluir 1             # paso 2: entrena sin el fold 1, predice fold 1 + test
    python bert_dapt.py --excluir 1,2           # paso 2 "de a pares" (para la CV exacta, como la Fase Q)
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

import bert_texto as bt

CARPETA = os.path.join(".cache_laboratorio", "bert_dapt")
CARPETA_MLM = os.path.join(CARPETA, "mlm")
LR_MLM, EPOCAS_MLM, PROB_MASCARA = 5e-5, 1, 0.15


def textos_sin_etiqueta():
    # Solo title y warranty de train + test: no se lee condition
    columnas = ["title", "warranty"]
    train = pd.read_json("../data/train_data.jsonlines", lines=True)[columnas]
    test = pd.read_json("../data/test_data.jsonlines", lines=True)[columnas]
    todo = pd.concat([train, test], ignore_index=True)
    return todo["title"].fillna("").tolist(), todo["warranty"].fillna("").tolist()


def mlm(bench):
    random.seed(bt.SEED)
    np.random.seed(bt.SEED)
    torch.manual_seed(bt.SEED)
    transformers.set_seed(bt.SEED)
    dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
    titulos, garantias = textos_sin_etiqueta()

    tokenizer = AutoTokenizer.from_pretrained(bt.MODELO)
    modelo = AutoModelForMaskedLM.from_pretrained(bt.MODELO).to(dispositivo)
    # Tapa el 15% de los tokens (de esos: 80% [MASK], 10% otro token al azar, 10% sin cambio), como BERT original
    juntar = DataCollatorForLanguageModeling(tokenizer, mlm_probability=PROB_MASCARA)
    datos = bt.codificar(tokenizer, titulos, garantias, range(len(titulos)))
    cargador = DataLoader(datos, batch_size=bt.BATCH, shuffle=True, collate_fn=juntar,
                          generator=torch.Generator().manual_seed(bt.SEED))
    pasos = len(cargador) * EPOCAS_MLM
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=LR_MLM, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizador, int(0.1 * pasos), pasos)

    inicio, paso = time.time(), 0
    modelo.train()
    for epoca in range(EPOCAS_MLM):
        for lote in cargador:
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            perdida = modelo(**lote).loss
            perdida.backward()
            optimizador.step()
            scheduler.step()
            optimizador.zero_grad()
            paso += 1
            if paso % 200 == 0 or (bench and paso == bench):
                print(f"[mlm] paso {paso}/{pasos} pérdida {perdida.item():.4f} {(time.time() - inicio) / 60:.1f} min", flush=True)
            if bench and paso == bench:
                seg = (time.time() - inicio) / paso
                print(f"BENCH MLM: {seg:.3f} s/paso -> 1 época ({pasos} pasos) ≈ {seg * pasos / 60:.1f} min", flush=True)
                return
    os.makedirs(CARPETA_MLM, exist_ok=True)
    modelo.save_pretrained(CARPETA_MLM)
    tokenizer.save_pretrained(CARPETA_MLM)
    print(f"[mlm] listo: {(time.time() - inicio) / 60:.1f} min, guardado en {CARPETA_MLM}", flush=True)


if __name__ == "__main__":
    if "--mlm" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--mlm", action="store_true")
        p.add_argument("--bench", type=int, default=0)
        args = p.parse_args()
        if os.path.exists(os.path.join(CARPETA_MLM, "config.json")) and not args.bench:
            print("ya existe", CARPETA_MLM)
        else:
            mlm(args.bench)
    else:
        # Paso 2: mismo main() que la Fase Q, partiendo del modelo adaptado
        if not os.path.exists(os.path.join(CARPETA_MLM, "config.json")) and "--bench" not in sys.argv:
            sys.exit(f"Falta el paso 1: no existe {CARPETA_MLM}")
        random.seed(bt.SEED)
        transformers.set_seed(bt.SEED)
        if os.path.exists(os.path.join(CARPETA_MLM, "config.json")):
            bt.MODELO = CARPETA_MLM
        bt.CARPETA = CARPETA
        bt.main()
