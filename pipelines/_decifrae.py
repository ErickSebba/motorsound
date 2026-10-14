"""Núcleo comum das pipelines Decifrae (p10, p11, p12).

PRÉ (cara): rastreador de rotação do baixo V3 (`motorsound.rotacao`) e features físicas
por janela. PÓS (barata): cosseno contra protótipos de classe (violão). O HMM de estados
do violão é aplicado pelo runner de sequências sobre as probabilidades de qualquer
pipeline (motorsound/sequencias.py), então "p13 = p11 sem HMM" é a decodificação
`argmax` das sequências da p11, não um arquivo separado.

Features (13):
  0 f_rot mediana (Hz)        1 dispersão de f_rot (IQR, Hz)   2 fração de quadros SIL
  3 saliência média no caminho (clareza da família harmônica)
  4 razão harmônica: energia nos k·f_rot (±1 bin) / energia 10 Hz–4 kHz
  5–8 energia relativa por grupo de ordem harmônica (1–4, 5–12, 13–24, 25–48), dB
  9 pico de modulação do envelope (Hz, 2–100 Hz)   10 proeminência desse pico (dB)
  11 centróide espectral (Hz)   12 planura espectral
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as ss

from motorsound.classificador import CossenoPrototipos
from motorsound.config import PostConfig, PreConfig
from motorsound.rotacao import rastrear, stft_mag

N_HARM = 48
GRUPOS = [(1, 4), (5, 12), (13, 24), (25, 48)]


@dataclass(frozen=True)
class PreDecifrae(PreConfig):
    sr: int = 16000
    fmin: float = 15.0
    fmax: float = 60.0
    passo_cents: float = 10.0
    n_fft: int = 8192
    hop: int = 1024
    usar_pyin: bool = False
    usar_viterbi: bool = True
    max_salto: int = 10


@dataclass(frozen=True)
class PostDecifrae(PostConfig):
    temp_cos: float = 0.1


def _envelope_mod(y, sr):
    env = np.abs(ss.hilbert(y))
    sos = ss.butter(4, 200, fs=sr, output="sos")
    env = ss.sosfilt(sos, env)[:: sr // 1000]          # 1 kHz
    env = env - env.mean()
    f, P = ss.welch(env, 1000, nperseg=min(len(env), 1024))
    m = (f >= 2) & (f <= 100)
    if not np.any(m):
        return 0.0, 0.0
    i = np.argmax(P[m])
    return float(f[m][i]), float(10 * np.log10(P[m][i] / (np.median(P[m]) + 1e-20) + 1e-12))


def extrair(y, sr, pre: PreDecifrae):
    f_rot, e_norm, grade = rastrear(y, sr, pre.fmin, pre.fmax, pre.passo_cents,
                                    np.arange(1, N_HARM + 1), pre.n_fft, pre.hop,
                                    usar_pyin=pre.usar_pyin, usar_viterbi=pre.usar_viterbi,
                                    max_salto=pre.max_salto)
    vivos = np.isfinite(f_rot)
    fr = f_rot[vivos] if np.any(vivos) else np.array([np.nan])
    f_med = float(np.nanmedian(fr)) if np.any(vivos) else 0.0
    iqr = float(np.subtract(*np.nanpercentile(fr, [75, 25]))) if np.any(vivos) else 0.0
    # saliência no caminho
    j = np.searchsorted(grade, np.where(vivos, f_rot, grade[0]))
    j = np.clip(j, 0, len(grade) - 1)
    sal = float(np.mean(np.where(vivos, e_norm[np.arange(len(j)), j], 0.0)))
    # espectro médio da janela (n_fft maior para resolver as ordens)
    S, f = stft_mag(y, sr, pre.n_fft, pre.hop)
    P = (S ** 2).mean(axis=1)
    df = f[1] - f[0]
    banda = (f >= 10) & (f <= 4000)
    tot = P[banda].sum() + 1e-20
    if f_med > 0:
        ordens = np.arange(1, N_HARM + 1)
        pos = np.round(ordens * f_med / df).astype(int)
        ok = pos < len(P) - 1
        e_ord = np.zeros(N_HARM)
        e_ord[ok] = P[pos[ok] - 1] + P[pos[ok]] + P[pos[ok] + 1]
    else:
        e_ord = np.zeros(N_HARM)
    razao = float(e_ord.sum() / tot)
    grupos = [10 * np.log10(e_ord[a - 1:b].sum() / tot + 1e-10) for a, b in GRUPOS]
    fm, prom = _envelope_mod(y, sr)
    cent = float((f * P).sum() / (P.sum() + 1e-20))
    plan = float(np.exp(np.mean(np.log(P[banda] + 1e-20))) / (P[banda].mean() + 1e-20))
    return np.array([f_med, iqr, 1 - vivos.mean(), sal, razao, *grupos, fm, prom, cent, plan],
                    dtype=np.float64)


def treinar(X, y, post: PostDecifrae, n_classes):
    clf = CossenoPrototipos(temp=post.temp_cos).fit(X, y, [int(c) for c in np.unique(y)])
    return clf, n_classes


def prever(modelo, X, post):
    clf, C = modelo
    P = np.zeros((len(X), C))
    P[:, clf.classes] = clf.predict_proba(X)
    return P


def custo_mcu(pre: PreDecifrae):
    fps = pre.sr / pre.hop
    J = int(np.floor(1200 * np.log2(pre.fmax / pre.fmin) / pre.passo_cents)) + 1
    fft = 5 * pre.n_fft * np.log2(pre.n_fft)
    soma_harm = J * N_HARM * 4
    vit = J * (2 * pre.max_salto + 3) * 2 if pre.usar_viterbi else J
    mflops = fps * (fft + soma_harm + vit) / 1e6
    ram_kb = (pre.n_fft * 4 * 2 + J * 8 + (J * 2 * 40 if pre.usar_viterbi else 0)) / 1024
    return {"params": 13 * 3 + 26, "ram_kb": round(ram_kb, 1), "mflops_s": round(mflops, 2),
            "nota": "estimativa analítica; pYIN não embarcável" if pre.usar_pyin
            else "estimativa analítica; Viterbi com janela de 40 quadros"}
