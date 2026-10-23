"""Leitor da base UOEMD-VAFCVS (Sehri & Dumond, Data in Brief 53, 2024).

Escrito do zero em 22/10/2026 (não é cópia de Artigo3). Ver docs/bases/uoemd.md.

Cada arquivo (.mat ou .csv) tem 420 000 linhas x 5 colunas, a 42 kHz:
    col 0  acelerômetro 1 (PCB 603C01, lado acionado do motor), m/s²
    col 1  MICROFONE (PCB 130F20, ~2 cm do mancal esquerdo), em volts
    col 2  acelerômetro 2 (PCB 623C01, mancal do eixo perto do motor), m/s²
    col 3  acelerômetro 3 (PCB 623C01, mancal do eixo mais distante), m/s²
    col 4  temperatura (sensor do acelerômetro 1), °C
Não há canal de tacômetro/encoder: a rotação só é conhecida pela frequência do inversor
que o artigo lista por arquivo (ver scripts/inventario_uoemd.py).

O sinal devolvido é o canal acústico CRU, em volts (sensibilidade nominal 45 mV/Pa),
com o offset DC original (~0,012 V); quem usa decide se remove a média.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 42_000          # Hz, declarado pelos autores e conferido no vetor 1_Time
COL_MICROFONE = 1
N_AMOSTRAS = 420_000


def _matriz(caminho: Path) -> np.ndarray:
    caminho = Path(caminho)
    suf = caminho.suffix.lower()
    if suf == ".mat":
        import scipy.io as sio
        m = sio.loadmat(str(caminho), variable_names=["data"])["data"]
    elif suf == ".csv":
        import pandas as pd
        m = pd.read_csv(caminho, encoding="utf-8-sig").to_numpy()
    else:
        raise ValueError(f"extensão não suportada: {caminho}")
    if m.ndim != 2 or m.shape[1] != 5:
        raise ValueError(f"esperava matriz N x 5 em {caminho}, achei {m.shape}")
    return m


def carregar(caminho_absoluto) -> tuple[np.ndarray, int]:
    """Canal acústico (coluna 2 da documentação, índice 1) como float32 mono, e sr."""
    m = _matriz(Path(caminho_absoluto))
    return np.ascontiguousarray(m[:, COL_MICROFONE], dtype=np.float32), SR


def carregar_tacometro(caminho):
    """A base não tem tacômetro nem encoder gravado; devolve sempre None."""
    return None


def carregar_todos_canais(caminho_absoluto) -> tuple[np.ndarray, int]:
    """Matriz N x 5 (float64) com os cinco canais, para conferências."""
    return _matriz(Path(caminho_absoluto)), SR
