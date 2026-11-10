"""Fine-tuning de BETO (BERT en español) sobre "título [SEP] garantía" para new vs used.

Lo usa trabajo_realizado/26_bert_texto.ipynb. Cada corrida entrena UN modelo con las filas de train que NO están
en los folds excluidos (StratifiedKFold(5, shuffle=True, random_state=42), la CV de siempre) y guarda la
probabilidad de new de las filas de los folds excluidos (y del test, si se excluye un solo fold).

Uso (desde trabajo_realizado/):
    python bert_texto.py --excluir 1            # entrena sin el fold 1 (56.000 filas): predice fold 1 + test
    python bert_texto.py --excluir 1,2          # entrena sin los folds 1 y 2 (42.000 filas): predice folds 1 y 2
    python bert_texto.py --excluir 1 --bench 50 # solo mide el tiempo de 50 pasos de entrenamiento
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

MODELO = "dccuchile/bert-base-spanish-wwm-cased"
CARPETA = os.path.join(".cache_laboratorio", "bert")
SEED = 42
MAX_LENGTH, EPOCAS, LR, BATCH = 64, 2, 2e-5, 32


def cargar():
    train = pd.read_json("../data/train_data.jsonlines", lines=True)
    test = pd.read_json("../data/test_data.jsonlines", lines=True)
    titulos = pd.concat([train["title"], test["title"]], ignore_index=True).fillna("").tolist()
    garantias = pd.concat([train["warranty"], test["warranty"]], ignore_index=True).fillna("").tolist()
    y = (train["condition"] == "new").astype(int).to_numpy()
    folds = np.empty(len(train), dtype=int)
    for k, (_, val) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)):
        folds[val] = k + 1
    return titulos, garantias, y, folds


def codificar(tokenizer, titulos, garantias, indices, etiquetas=None):
    # Par de textos: [CLS] título [SEP] garantía [SEP], recortado a MAX_LENGTH tokens
    enc = tokenizer([titulos[i] for i in indices], [garantias[i] for i in indices],
                    truncation="longest_first", max_length=MAX_LENGTH)
    filas = [{k: enc[k][n] for k in enc} for n in range(len(indices))]
    if etiquetas is not None:
        for fila, e in zip(filas, etiquetas):
            fila["labels"] = int(e)
    return filas


def predecir(modelo, cargador, dispositivo):
    modelo.eval()
    probs = []
    with torch.no_grad():
        for lote in cargador:
            lote = {k: v.to(dispositivo) for k, v in lote.items() if k != "labels"}
            probs.append(torch.softmax(modelo(**lote).logits, dim=-1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--excluir", required=True, help="folds excluidos del entrenamiento, ej. 1 o 1,2")
    p.add_argument("--bench", type=int, default=0, help="si > 0, solo mide el tiempo de esa cantidad de pasos")
    args = p.parse_args()
    excluidos = [int(x) for x in args.excluir.split(",")]
    nombre = "sin_" + "_".join(map(str, excluidos))
    os.makedirs(CARPETA, exist_ok=True)
    archivo = os.path.join(CARPETA, f"{nombre}.npz")
    if os.path.exists(archivo) and not args.bench:
        print("ya existe", archivo)
        return

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
    titulos, garantias, y, folds = cargar()
    filas_tr = np.where(~np.isin(folds, excluidos))[0]
    filas_pred = np.where(np.isin(folds, excluidos))[0]
    if len(excluidos) == 1:                      # los modelos "globales" también predicen el test
        filas_pred = np.r_[filas_pred, np.arange(len(y), len(titulos))]

    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO, num_labels=2).to(dispositivo)
    juntar = DataCollatorWithPadding(tokenizer)
    generador = torch.Generator().manual_seed(SEED)
    cargador_tr = DataLoader(codificar(tokenizer, titulos, garantias, filas_tr, y[filas_tr]), batch_size=BATCH,
                             shuffle=True, collate_fn=juntar, generator=generador)
    pasos = len(cargador_tr) * EPOCAS
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=0.01)
    # learning rate: sube linealmente el primer 10% de los pasos (warmup) y después baja linealmente a 0
    scheduler = get_linear_schedule_with_warmup(optimizador, int(0.1 * pasos), pasos)

    inicio, paso = time.time(), 0
    modelo.train()
    for epoca in range(EPOCAS):
        for lote in cargador_tr:
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            perdida = modelo(**lote).loss
            perdida.backward()
            optimizador.step()
            scheduler.step()
            optimizador.zero_grad()
            paso += 1
            if paso % 200 == 0 or (args.bench and paso == args.bench):
                print(f"[{nombre}] época {epoca + 1} paso {paso}/{pasos} pérdida {perdida.item():.4f} "
                      f"{(time.time() - inicio) / 60:.1f} min", flush=True)
            if args.bench and paso == args.bench:
                seg = (time.time() - inicio) / paso
                print(f"BENCH: {seg:.3f} s/paso -> entrenamiento completo ({pasos} pasos) ≈ {seg * pasos / 60:.1f} min", flush=True)
                return
    minutos_entrenamiento = (time.time() - inicio) / 60

    cargador_pred = DataLoader(codificar(tokenizer, titulos, garantias, filas_pred), batch_size=128, collate_fn=juntar)
    probs = predecir(modelo, cargador_pred, dispositivo)
    np.savez(archivo, filas=filas_pred, prob=probs, minutos=minutos_entrenamiento)
    es_train = filas_pred < len(y)
    acc = ((probs[es_train] >= 0.5) == y[filas_pred[es_train]]).mean()
    print(f"[{nombre}] listo: entrenamiento {minutos_entrenamiento:.1f} min, total {(time.time() - inicio) / 60:.1f} min, "
          f"accuracy en los folds excluidos {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
