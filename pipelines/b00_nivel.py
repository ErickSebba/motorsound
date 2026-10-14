"""Baseline de nível: só o RMS e o pico da janela + regressão logística.

Existe para medir o atalho de nível (docs/DATASETS.md §1.5). Com `normalizar_rms=True`
no runner o RMS vira constante e esta pipeline deve cair para o acaso; sem
normalização, o quanto ela acerta é o quanto o nível sozinho explica.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from motorsound.config import PostConfig, PreConfig

NOME = "b00_nivel"
ORIGEM = "baseline"
TIPO = "baseline"
PRE = PreConfig(sr=16000)
POST = PostConfig()


def extrair(y, sr, pre):
    r = np.sqrt(np.mean(y ** 2)) + 1e-12
    return np.array([20 * np.log10(r), 20 * np.log10(np.abs(y).max() + 1e-12)])


def treinar(X, y, post, n_classes):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y), n_classes


def prever(modelo, X, post):
    m, C = modelo
    P = np.zeros((len(X), C))
    P[:, m.classes_] = m.predict_proba(X)
    return P


def custo_mcu():
    return {"params": 9, "ram_kb": 0.1, "mflops_s": 0.03, "nota": "RMS + pico por janela"}
