"""Métricas por janela e por sequência.

Por janela: acurácia, F1 macro e matriz de confusão, por condição.
Por sequência: atraso de detecção de cada troca verdadeira e trocas espúrias por minuto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def por_janela(df: pd.DataFrame, por=("pipeline", "condicao")) -> pd.DataFrame:
    """df com colunas classe, pred e as de `por`. Uma linha por grupo."""
    linhas = []
    for chave, g in df.groupby(list(por), dropna=False):
        chave = chave if isinstance(chave, tuple) else (chave,)
        linhas.append(dict(zip(por, chave), n=len(g),
                           acuracia=float((g["classe"] == g["pred"]).mean()),
                           f1_macro=float(f1_score(g["classe"], g["pred"], average="macro",
                                                   labels=sorted(g["classe"].unique()),
                                                   zero_division=0))))
    return pd.DataFrame(linhas)


def matriz_confusao(df: pd.DataFrame, classes) -> np.ndarray:
    return confusion_matrix(df["classe"], df["pred"], labels=list(classes))


def trocas(rotulos) -> np.ndarray:
    r = np.asarray(rotulos, dtype=object)
    return np.flatnonzero(r[1:] != r[:-1]) + 1


def sequencia(verdade, pred, passo_s: float) -> dict:
    """Atraso de detecção e trocas espúrias.

    Para cada troca verdadeira em t_k (para o estado B), o atraso é o tempo até a
    primeira janela >= t_k com pred == B, antes da próxima troca verdadeira; se não há,
    a troca é perdida. Trocas espúrias = trocas na predição que não são a primeira
    detecção de uma troca verdadeira, por minuto de sequência.
    """
    v = np.asarray(verdade, dtype=object)
    p = np.asarray(pred, dtype=object)
    tv = list(trocas(v)) + [len(v)]
    atrasos, perdidas, deteccoes = [], 0, set()
    for k in range(len(tv) - 1):
        t0, t1 = tv[k], tv[k + 1]
        alvo = v[t0]
        hit = np.flatnonzero(p[t0:t1] == alvo)
        if len(hit) == 0:
            perdidas += 1
            continue
        i = t0 + int(hit[0])
        atrasos.append(int(hit[0]) * passo_s)
        if i in set(trocas(p)) or (i == t0 and p[i - 1] != p[i]):
            deteccoes.add(i)
    tp = set(trocas(p))
    espurias = len(tp - deteccoes)
    dur_min = len(v) * passo_s / 60.0
    return {"n_trocas": len(tv) - 1, "perdidas": perdidas,
            "atraso_mediano_s": float(np.median(atrasos)) if atrasos else float("nan"),
            "atraso_medio_s": float(np.mean(atrasos)) if atrasos else float("nan"),
            "trocas_pred": len(tp), "espurias": espurias,
            "espurias_por_min": espurias / dur_min if dur_min > 0 else float("nan"),
            "acuracia": float((v == p).mean()), "duracao_min": dur_min}
