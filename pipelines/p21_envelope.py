"""p21_envelope: espectro de modulação do envelope em bandas de portadora × bandas de modulação,
log-energia, + regressão logística.

ORIGEM = "decifrae", TIPO = "inspirada": não reproduz um artigo específico; segue a ideia geral
da análise de envelope / espectro de modulação usada em diagnóstico de máquinas (a mesma do
HARH da wheat2025 e da feature 9–10 de pipelines/_decifrae.py), com todos os detalhes escolhidos
por nós, declarados abaixo e fixados antes de olhar qualquer resultado (sem busca de
hiperparâmetros).

Desenho (todo causal, pensado para o ESP32)
  1. Bandas de portadora: 5 bandas log-espaçadas entre 100 Hz e 7,6 kHz (bordas
     geomspace(100, 7600, 6)); passa-banda Butterworth ordem 2 por borda (ordem 4 total),
     SOS, `sosfilt` (causal; sem `sosfiltfilt`).
  2. Envelope: retificação de onda completa |x| + passa-baixa Butterworth ordem 4 em 250 Hz
     (SOS, causal), decimação por 16 → 1 kHz. Escolhido no lugar de Hilbert por custo de MCU
     (Hilbert exige FFT do bloco inteiro).
  3. Espectro de modulação: envelope menos a média, janela Hann, periodograma de |rfft|² de
     toda a janela (1 s no A2212 → resolução 1 Hz; 3 s no IDMT → 0,33 Hz).
  4. Bandas de modulação: 8 bandas log entre 2 e 200 Hz (bordas geomspace(2, 200, 9)); 2 Hz é
     o mínimo resolvível com 1 s de janela. Energia somada por banda e normalizada pela
     energia DC do envelope (média²·n), isto é, profundidade de modulação, independente do
     nível da portadora; 10·log10.
  5. Mais a energia relativa de cada banda de portadora (dB em relação à soma das 5).
  Features: 5 × 8 + 5 = 45, mesma forma para 1 s e 3 s.
  6. Classificador: StandardScaler + LogisticRegression(C = 1, lbfgs, max_iter = 2000), igual
     ao b01 para comparação direta. Sem aleatoriedade (seed de POST não usada).

Suposições: número e limites das bandas, ordem dos filtros, corte de 250 Hz e uso da
retificação são escolhas nossas, sem ajuste na validação.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.signal as ss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from motorsound.config import PostConfig, PreConfig

NOME = "p21_envelope"
ORIGEM = "decifrae"
TIPO = "inspirada"


@dataclass(frozen=True)
class PreEnv(PreConfig):
    sr: int = 16000
    n_port: int = 5
    port_lo: float = 100.0
    port_hi: float = 7600.0
    n_mod: int = 8
    mod_lo: float = 2.0
    mod_hi: float = 200.0
    corte_env: float = 250.0
    decim: int = 16


@dataclass(frozen=True)
class PostEnv(PostConfig):
    C: float = 1.0


PRE = PreEnv()
POST = PostEnv()


@lru_cache(maxsize=8)
def _filtros(sr, n_port, lo, hi, corte):
    bordas = np.geomspace(lo, hi, n_port + 1)
    bp = [ss.butter(2, [a, b], btype="bandpass", fs=sr, output="sos")
          for a, b in zip(bordas[:-1], bordas[1:])]
    lp = ss.butter(4, corte, fs=sr, output="sos")
    return bp, lp


def extrair(y, sr, pre: PreEnv):
    bp, lp = _filtros(sr, pre.n_port, pre.port_lo, pre.port_hi, pre.corte_env)
    sr_env = sr / pre.decim
    bordas_mod = np.geomspace(pre.mod_lo, pre.mod_hi, pre.n_mod + 1)
    y = np.asarray(y, dtype=np.float64)
    feats, e_port = [], []
    for sos in bp:
        x = ss.sosfilt(sos, y)
        e_port.append(np.mean(x ** 2))
        env = ss.sosfilt(lp, np.abs(x))[:: pre.decim]
        n = len(env)
        dc = env.mean() ** 2 * n + 1e-20
        P = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(n))) ** 2
        f = np.fft.rfftfreq(n, 1.0 / sr_env)
        for a, b in zip(bordas_mod[:-1], bordas_mod[1:]):
            m = (f >= a) & (f < b)
            feats.append(10 * np.log10(P[m].sum() / dc + 1e-12) if np.any(m) else -120.0)
    e_port = np.asarray(e_port)
    rel = 10 * np.log10(e_port / (e_port.sum() + 1e-20) + 1e-12)
    return np.concatenate([feats, rel])


def treinar(X, y, post: PostEnv = POST, n_classes=None):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, C=post.C)).fit(X, y)
    return m, n_classes


def prever(modelo, X, post: PostEnv = POST):
    m, C = modelo
    P = np.zeros((len(X), C))
    P[:, m.classes_] = m.predict_proba(X)
    return P


def custo_mcu():
    pre = PRE
    nb, nm = pre.n_port, pre.n_mod
    # por amostra de entrada: 5 passa-bandas (2 seções biquad) + 5 passa-baixas (2 seções)
    # ~9 flops por seção biquad; retificação desprezível
    filtros = pre.sr * nb * (2 + 2) * 9
    # por segundo: uma FFT de ~1024 pontos por banda (envelope a 1 kHz, janela de 1 s)
    fft = nb * 5 * 1024 * 10
    n_feat = nb * nm + nb
    clf = 3 * n_feat * 2
    mflops = (filtros + fft + clf) / 1e6
    ram_kb = (nb * 1024 * 4 + 1024 * 4 + nb * 4 * 4 * 2) / 1024
    return {"params": 3 * n_feat + 3 + 2 * n_feat,
            "ram_kb": round(ram_kb, 1),
            "flash_kb": round((3 * n_feat + 3 + 2 * n_feat + nb * 4 * 6) * 4 / 1024, 1),
            "mflops_s": round(mflops, 2),
            "nota": "estimativa analítica, janela de 1 s (buffer de envelope 1024 floats por "
                    "banda; 3 s triplica a RAM do buffer); params = LR (3 classes) + "
                    "média/desvio do scaler"}
