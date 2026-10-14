"""B1 do PLANO_PIPELINE_MOTOR.md §5: forma espectral sem rastreador + regressão logística.

Energia relativa em 32 bandas log-espaçadas (50 Hz–8 kHz), centróide, espalhamento,
rolloff 85 %, planura e ZCR, médias sobre a janela. Nenhuma informação de rotação.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from motorsound.config import PostConfig, PreConfig
from motorsound.rotacao import stft_mag

NOME = "b01_forma_espectral"
ORIGEM = "baseline"
TIPO = "baseline"
PRE = PreConfig(sr=16000)
POST = PostConfig()
_NFFT, _HOP, _NB = 1024, 512, 32


def extrair(y, sr, pre):
    S, f = stft_mag(y, sr, _NFFT, _HOP)
    P = S ** 2
    bordas = np.geomspace(50, sr / 2, _NB + 1)
    bandas = np.stack([P[(f >= a) & (f < b)].sum(axis=0) for a, b in zip(bordas[:-1], bordas[1:])])
    tot = P.sum(axis=0) + 1e-12
    rel = 10 * np.log10(bandas / tot + 1e-10).mean(axis=1)
    cent = (f[:, None] * P).sum(0) / tot
    esp = np.sqrt(((f[:, None] - cent) ** 2 * P).sum(0) / tot)
    roll = f[np.argmax(np.cumsum(P, axis=0) >= 0.85 * tot, axis=0)]
    plan = np.exp(np.log(P + 1e-12).mean(0)) / (P.mean(0) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.sign(y))) > 0)
    return np.concatenate([rel, [cent.mean(), esp.mean(), roll.mean(), plan.mean(), zcr]])


def treinar(X, y, post, n_classes):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X, y)
    return m, n_classes


def prever(modelo, X, post):
    m, C = modelo
    P = np.zeros((len(X), C))
    P[:, m.classes_] = m.predict_proba(X)
    return P


def custo_mcu():
    # FFT 1024 a cada 512 amostras a 16 kHz: ~31 FFT/s × ~5·N·log2N flops
    return {"params": 37 * 3 + 3, "ram_kb": 1024 * 4 * 2 / 1024 + 1,
            "mflops_s": 31 * 5 * 1024 * 10 / 1e6, "nota": "estimativa analítica"}
