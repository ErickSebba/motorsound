"""P02 — linha de base dos autores do IDMT: |STFT| linear → DNN 256-64-16-3 por quadro.

Referência: docs/literatura/grollmisch2019.md (Grollmisch et al., EUSIPCO 2019, "Sounding
Industry"). Conferido no PDF em 14/10/2026 (Seção III e Tabela I).

Seguido do artigo:
  - taxa mantida em 44,1 kHz; magnitude da STFT com janela de Hann e 50 % de sobreposição,
    fase descartada, escala linear; Tabela I, motor: Input = 2048 = metade da janela →
    janela 4096, hop 2048;
  - normalização para [0, 1] pelo máximo dos dados de treino (o mesmo valor vai para a
    validação);
  - DNN 2048 → 256 → 64 → 16 → 3, ReLU nas ocultas, softmax na saída; dropout 0,2 entre a
    entrada e a 1.ª oculta (DIn), 0,5 entre ocultas (D), 0,2 antes da camada final (DOut);
    Glorot uniforme; Adam, lr = 0,001, entropia cruzada categórica;
  - decisão por quadro, agregada sobre um trecho mais longo.

Suposições e desvios (declarados):
  1. Entrada de 2048 valores a partir de rfft(4096) = 2049 bins: descartamos o bin de
     Nyquist (bins 0..2047). Não descrito.
  2. Lote = 32 (padrão do Keras; não descrito).
  3. **Épocas: 60 em vez de 1000** (POST.epocas). 1000 épocas por fold custariam ~30–40 min
     por fold em 2 threads, fora do orçamento do briefing. Desvio de custo, não de ajuste:
     o valor foi fixado antes de olhar a validação e não foi variado.
  4. Agregação na janela do runner (3 s no IDMT, 1 s no A2212) por **média das
     probabilidades** dos quadros, como pedido pela sessão principal; o artigo usa voto
     majoritário sobre os quadros do arquivo (A_file).
  5. Sem a validação interna 90/10 do artigo nem as 5 repetições: o runner já dá os folds.
  6. Hiperparâmetros da otimização bayesiana não refeitos: usamos o ponto ótimo da Tabela I.
  7. No A2212 (gravado a 16 kHz) o runner reamostra para 44,1 kHz: os bins acima de 8 kHz
     ficam vazios. Mantido para usar a mesma pipeline nas duas bases.
  8. O runner normaliza o RMS de cada janela para 0,1 antes da STFT (não existe no artigo).
  9. Armazena as magnitudes em float16 no cache (184 MB → 92 MB no IDMT); a rede roda em float32.

Por esses itens (sobretudo 3 e 4) TIPO = "inspirada", não "reproducao".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import nn

from motorsound.config import PostConfig, PreConfig
from motorsound.rotacao import stft_mag
from pipelines._p02p05_util import glorot, n_params, probabilidades, treinar_rede

NOME = "p02_grollmisch"
ORIGEM = "grollmisch2019"
TIPO = "inspirada"


@dataclass(frozen=True)
class PreGrollmisch(PreConfig):
    sr: int = 44100
    n_fft: int = 4096
    hop: int = 2048          # 50 % de sobreposição


@dataclass(frozen=True)
class PostGrollmisch(PostConfig):
    epocas: int = 60         # artigo: 1000 (ver suposição 3)
    lote: int = 32
    lr: float = 1e-3


PRE = PreGrollmisch()
POST = PostGrollmisch()
_N_IN = PRE.n_fft // 2


def extrair(y, sr, pre):
    """(quadros, 2048) magnitudes lineares em float16."""
    S, _ = stft_mag(np.asarray(y, dtype=np.float32), sr, pre.n_fft, pre.hop)   # (F, T)
    return S[:pre.n_fft // 2].T.astype(np.float16)


def _rede(n_classes):
    m = nn.Sequential(
        nn.Dropout(0.2), nn.Linear(_N_IN, 256), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(256, 64), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(64, 16), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(16, n_classes))
    glorot(m)
    return m


def treinar(X, y, post, n_classes):
    X = np.asarray(X, dtype=np.float32)                     # (n_jan, T, 2048)
    n, T, F = X.shape
    Xq = X.reshape(n * T, F)
    yq = np.repeat(np.asarray(y), T)
    escala = float(Xq.max()) or 1.0
    rede = treinar_rede(_rede(n_classes), Xq / escala, yq, post.epocas, post.lote,
                        post.lr, post.seed)
    return {"rede": rede, "escala": escala, "classes": np.unique(y), "C": n_classes}


def prever(modelo, X, post):
    X = np.asarray(X, dtype=np.float32)
    n, T, F = X.shape
    Pq = probabilidades(modelo["rede"], X.reshape(n * T, F) / modelo["escala"])
    P = Pq.reshape(n, T, -1).mean(axis=1)                   # média por janela
    mask = np.zeros(modelo["C"], bool)
    mask[modelo["classes"]] = True
    P[:, ~mask] = 0.0                                        # classe ausente no treino
    return P / P.sum(axis=1, keepdims=True)


def custo_mcu():
    params = n_params(_rede(3))
    q_s = PRE.sr / PRE.hop                                   # ~21,5 quadros/s
    fft = 5 * PRE.n_fft * np.log2(PRE.n_fft)                 # flops por FFT real (aprox.)
    dnn = 2 * (2048 * 256 + 256 * 64 + 64 * 16 + 16 * 3)     # MAC×2 por quadro
    return {"params": params,
            "ram_kb": round((PRE.n_fft * 4 * 2 + PRE.n_fft * 4 + (256 + 64 + 16) * 4) / 1024, 1),
            "flash_kb": round(params * 4 / 1024, 1),
            "mflops_s": round(float(q_s * (fft + dnn)) / 1e6, 1),
            "nota": "estimativa analítica; pesos float32 (int8: ~1/4 do flash); "
                    "RAM = buffer de entrada com sobreposição + FFT + ativações"}
