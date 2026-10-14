"""Sequências com trocas de estado, montadas a partir das janelas de validação.

Toda pipeline extrai features por janela, independentemente das vizinhas, então
concatenar as probabilidades de janelas equivale a concatenar o áudio. Uma sequência
alterna estados sem repetição; cada permanência é um trecho CONTÍGUO de janelas de
validação de uma mesma gravação/bloco, com duração sorteada em [dmin, dmax] janelas.

Decodificações comparadas sobre as mesmas probabilidades:
  argmax   — sem modelo temporal (isola o HMM: é a "p13" aplicada a qualquer pipeline)
  viterbi  — HMM do violão, offline (olha o futuro)
  online   — mesmo HMM, filtro forward causal (o que roda no MCU)
O atraso de detecção honesto é o do `online`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import hmm, metricas


def montar(df_val: pd.DataFrame, classes, n_seq=20, dmin=5, dmax=20, seed=0):
    """df_val: linhas de UMA pipeline e UM split, com grupo, idx, classe e p_<classe>.
    Devolve lista de (verdade, P) com P (T, C)."""
    rng = np.random.default_rng(seed)
    trechos = {c: [] for c in classes}
    for (_, c), g in df_val.groupby(["grupo", "classe"]):
        g = g.sort_values("idx")
        # quebra em corridas contíguas de idx
        idx = g["idx"].to_numpy()
        cortes = np.flatnonzero(np.diff(idx) != 1) + 1
        for parte in np.split(np.arange(len(g)), cortes):
            if len(parte):
                trechos[c].append(g.iloc[parte][[f"p_{k}" for k in classes]].to_numpy())
    seqs = []
    for _ in range(n_seq):
        verdade, P, ant = [], [], None
        for _ in range(int(rng.integers(4, 9))):
            c = rng.choice([k for k in classes if k != ant and trechos[k]])
            tr = trechos[c][int(rng.integers(len(trechos[c])))]
            d = int(min(len(tr), rng.integers(dmin, dmax + 1)))
            i0 = int(rng.integers(0, len(tr) - d + 1))
            P.append(tr[i0:i0 + d])
            verdade += [c] * d
            ant = c
        seqs.append((verdade, np.vstack(P)))
    return seqs


def avaliar(seqs, classes, passo_s, change_penalty=2.0, temp=1.0, n_logprob=-50.0,
            min_dur=0.0) -> pd.DataFrame:
    linhas = []
    for i, (v, P) in enumerate(seqs):
        decs = {
            "argmax": [classes[k] for k in P.argmax(axis=1)],
            "viterbi": hmm.decode(P, classes, change_penalty, temp, n_logprob),
            "online": hmm.filtro_online(P, classes, change_penalty, temp, n_logprob),
        }
        for nome, r in decs.items():
            if min_dur > 0:
                r = hmm.segmentos_para_rotulos(
                    hmm.absorb_short(hmm.rotulos_para_segmentos(r, passo_s), min_dur),
                    len(r), passo_s)
            linhas.append({"seq": i, "decodificacao": nome, "change_penalty": change_penalty,
                           "min_dur": min_dur, **metricas.sequencia(v, r, passo_s)})
    return pd.DataFrame(linhas)
