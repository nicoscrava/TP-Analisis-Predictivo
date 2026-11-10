"""Fine-tuning de BETO (BERT en español) sobre "título [SEP] garantía" para new vs used.

Misma configuración que trabajo_realizado/bert_texto.py (Fase Q de BITACORA.md), que es con la que se
midió el modelo final (trabajo_realizado/29_texto_combinado.ipynb, Fase T):
- train.ipynb llama a oof_beto: entrena 5 modelos, cada uno sin un fold de
  StratifiedKFold(5, shuffle=True, random_state=42), y devuelve la probabilidad de new de cada
  fila del train predicha por el modelo que NO la vio (out-of-fold). Guarda los pesos de los 5.
- predict.ipynb llama a predecir_beto_promedio: carga los 5 modelos y promedia su probabilidad
  sobre el test.
"""
import os
import random

import numpy as np
import torch
import transformers
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

MODELO_BASE = "dccuchile/bert-base-spanish-wwm-cased"
SEMILLA = 42
MAX_LENGTH, EPOCAS, LR, BATCH = 64, 2, 2e-5, 32


def fijar_semillas(semilla=SEMILLA):
    """Fija todas las fuentes de azar: Python, numpy, torch (CPU y GPU de la Mac) y transformers."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(semilla)
    transformers.set_seed(semilla)


def _dispositivo():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _codificar(tokenizer, titulos, garantias, etiquetas=None):
    # Par de textos: [CLS] título [SEP] garantía [SEP], recortado a MAX_LENGTH tokens
    enc = tokenizer(list(titulos), list(garantias), truncation="longest_first", max_length=MAX_LENGTH)
    filas = [{k: enc[k][i] for k in enc} for i in range(len(titulos))]
    if etiquetas is not None:
        for fila, e in zip(filas, etiquetas):
            fila["labels"] = int(e)
    return filas


def entrenar_beto(titulos, garantias, y, semilla=SEMILLA):
    """Fine-tuning de BETO: 2 épocas, AdamW (lr 2e-5, weight decay 0,01), learning rate que sube
    linealmente el primer 10% de los pasos (warmup) y después baja linealmente a 0."""
    fijar_semillas(semilla)
    dispositivo = _dispositivo()
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_BASE, num_labels=2).to(dispositivo)
    cargador = DataLoader(_codificar(tokenizer, titulos, garantias, y), batch_size=BATCH, shuffle=True,
                          collate_fn=DataCollatorWithPadding(tokenizer),
                          generator=torch.Generator().manual_seed(semilla))
    pasos = len(cargador) * EPOCAS
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizador, int(0.1 * pasos), pasos)
    modelo.train()
    for _ in range(EPOCAS):
        for lote in cargador:
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            modelo(**lote).loss.backward()
            optimizador.step()
            scheduler.step()
            optimizador.zero_grad()
    return modelo, tokenizer


def predecir_beto(modelo, tokenizer, titulos, garantias):
    """Probabilidad de new (softmax de la clase 1)."""
    dispositivo = _dispositivo()
    modelo.to(dispositivo).eval()
    cargador = DataLoader(_codificar(tokenizer, titulos, garantias), batch_size=128,
                          collate_fn=DataCollatorWithPadding(tokenizer))
    probs = []
    with torch.no_grad():
        for lote in cargador:
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            probs.append(torch.softmax(modelo(**lote).logits, dim=-1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs)


def oof_beto(titulos, garantias, y, carpeta):
    """Entrena 5 BETO (cada uno sin un fold) y devuelve la probabilidad out-of-fold de cada fila.

    Guarda en carpeta/fold_k/ los pesos y el tokenizer del modelo k, y su predicción del fold k
    (oof.npy). Si un fold ya está guardado, no lo vuelve a entrenar.
    """
    titulos, garantias, y = np.asarray(titulos), np.asarray(garantias), np.asarray(y)
    prob = np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    for k, (filas_tr, filas_va) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        carpeta_k = os.path.join(carpeta, f"fold_{k}")
        archivo_oof = os.path.join(carpeta_k, "oof.npy")
        if not os.path.exists(archivo_oof):
            modelo, tokenizer = entrenar_beto(titulos[filas_tr], garantias[filas_tr], y[filas_tr])
            modelo.save_pretrained(carpeta_k)
            tokenizer.save_pretrained(carpeta_k)
            np.save(archivo_oof, predecir_beto(modelo, tokenizer, titulos[filas_va], garantias[filas_va]))
        prob[filas_va] = np.load(archivo_oof)
    return prob


def predecir_beto_promedio(carpeta, titulos, garantias):
    """Promedio de la probabilidad de new de los 5 BETO guardados por oof_beto."""
    probs = []
    for k in range(1, 6):
        carpeta_k = os.path.join(carpeta, f"fold_{k}")
        modelo = AutoModelForSequenceClassification.from_pretrained(carpeta_k)
        tokenizer = AutoTokenizer.from_pretrained(carpeta_k)
        probs.append(predecir_beto(modelo, tokenizer, list(titulos), list(garantias)))
    return np.mean(probs, axis=0)
