"""Leitor da base IM-VACD (Induction Motor Vibration and Acoustic Cellular Device, Sehri,
Universidade de Ottawa; Mendeley Data doi:10.17632/yc8yhg5xjd.2).

Escrito do zero em 23/10/2026 (não é cópia de Artigo3). Ver docs/bases/imvacd.md.

Cada arquivo é um CSV com cabeçalho; a ORDEM DAS COLUNAS MUDA com o celular:
    iPhone 13:  x, y, z, acoustic, accelerometer_*, acoustic_*, device, timestamp,
    Galaxy S6/A50: acoustic, x, y, z, acoustic_*, accelerometer_*, device, timestamp,
(por isso tudo aqui é lido pelo nome da coluna).
- x, y, z: acelerômetro do celular (m/s²), só nas primeiras ~1000-5000 linhas (taxa baixa);
- acoustic: microfone do celular, "pcm16_normalized" (inteiro de 16 bit / 32768), em TODAS as
  linhas; no iPhone é a 4.ª coluna, não a 1.ª como diz a descrição do Mendeley;
- os metadados (taxas pedida e "real", duração, aparelho, carimbo de tempo) ficam só na
  primeira linha de dados.

`carregar` devolve o áudio cru (float32 mono, sem remover a média) e a taxa NOMINAL pedida
(`acoustic_sampling_frequency_set`: 48 000 Hz no iPhone, 44 100 Hz nos Galaxy; não 42 kHz). A taxa "real" do
arquivo é amostras/duração de relógio do app e fica em `carregar_meta`; ver a ficha, §3.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

COL_AUDIO = "acoustic"


def carregar_meta(caminho_absoluto) -> dict:
    """Metadados da primeira linha de dados (strings convertidas quando numéricas)."""
    with open(Path(caminho_absoluto), newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        cab = next(r)
        lin = next(r)
    meta = {}
    for k, v in zip(cab, lin):
        k = k.strip()
        if not k or k in ("x", "y", "z", COL_AUDIO):
            continue
        try:
            meta[k] = float(v)
        except ValueError:
            meta[k] = v
    return meta


def carregar(caminho_absoluto) -> tuple[np.ndarray, int]:
    """Canal acústico como float32 mono e a taxa nominal (Hz, int)."""
    caminho = Path(caminho_absoluto)
    if caminho.suffix.lower() != ".csv":
        raise ValueError(f"extensão não suportada: {caminho}")
    meta = carregar_meta(caminho)
    y = pd.read_csv(caminho, usecols=[COL_AUDIO], dtype={COL_AUDIO: np.float32},
                    encoding="utf-8-sig")[COL_AUDIO].to_numpy()
    if np.isnan(y).any():
        # linhas finais sem áudio (não observado, mas não arriscar)
        y = y[~np.isnan(y)]
    sr = int(round(meta["acoustic_sampling_frequency_set"]))
    return np.ascontiguousarray(y, dtype=np.float32), sr


def carregar_acelerometro(caminho_absoluto) -> tuple[np.ndarray, float]:
    """Acelerômetro do celular (N x 3, m/s²) e a taxa nominal pedida."""
    caminho = Path(caminho_absoluto)
    meta = carregar_meta(caminho)
    a = pd.read_csv(caminho, usecols=["x", "y", "z"], encoding="utf-8-sig").dropna().to_numpy()
    return a, float(meta["accelerometer_sampling_frequency_set"])


def carregar_tacometro(caminho):
    """A base não tem tacômetro nem encoder; devolve sempre None."""
    return None
