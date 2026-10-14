"""P00 — rota STM de Strantzalis et al. (2022): log-mel 30×32 + CNN 2D de duas convoluções.

Referência: ficha docs/literatura/strantzalis2022.md (texto completo, PMC9783357, §5.1 e §4).

SEGUIDO DO ARTIGO
  - 16 kHz (librosa); FFT 1024, hop 512; 30 filtros mel triangulares; log-mel; entrada 30×32
    (32 colunas ≈ 1,024 s, uma inferência por espectrograma, como no MCU, §4).
  - Normalização z-score.
  - CNN: Conv2D(16, 3×3, ReLU, 'valid') → MaxPool 2×2 → Conv2D(16, 3×3, ReLU) → MaxPool 2×2
    → Flatten (576) → Dense(9, ReLU) → Dense(C, softmax). As formas intermediárias que o artigo
    dá (12×13 após a 2.ª conv, 6×6 após o 2.º pool, flatten 576) só fecham com 'valid' e pool
    com descarte, que é o que está aqui. 7 703 parâmetros com C=3 (≈ 7,7 kB em int8,
    coerente com a flash reportada).
  - Lote 500, 10 épocas de referência (ver passos abaixo), entropia cruzada.

SUPOSIÇÕES (não descritas no artigo) — por isso TIPO = "inspirada"
  1. Otimizador: não citado na rota STM; Adam com lr = 1e-3 (padrão do Keras).
  2. Filtros mel e log: os de librosa.feature.melspectrogram (Slaney, fmax = 8 kHz, potência)
     e log = librosa.power_to_db(ref=1, amin=1e-10, sem top_db). `center=True` (padrão da
     librosa): 1 s a 16 kHz dá exatamente 32 colunas.
  3. z-score com média e desvio ESCALARES do conjunto de treino do fold (o artigo não diz se
     é por banda ou global, nem sobre que conjunto).
  4. Janelas do runner (3 s no IDMT → 30×94; 1 s no A2212 → 30×32): `extrair` devolve o
     log-mel inteiro da janela (30×T) e a formação do 30×32 é feita no treino/predição:
       - treino: todas as sub-janelas de 32 colunas com passo de 1 coluna (63 por janela de
         3 s, 1 por janela de 1 s). Isso imita o passo implícito do artigo (≈127 600
         espectrogramas em ≈75 min ⇒ passo ≈ 35 ms ≈ 1 hop), mas aqui só DENTRO de cada
         janela de treino, sem cruzar o split;
       - predição: ceil(T/32) sub-janelas espaçadas por igual cobrindo a janela (3 no IDMT:
         colunas 0, 31, 62; 1 no A2212), i.e. as inferências a cada 1,024 s do MCU; a
         probabilidade da janela é a média das probabilidades das sub-janelas.
  5. Número de atualizações: o artigo treinou 10 épocas de lote 500 sobre ≈71 780
     espectrogramas (127 608 × 0,75 × 0,75) = 10 × 144 = 1 440 passos de gradiente. Com as
     bases daqui (centenas a ~17 mil sub-janelas) "10 épocas" dariam de 10 a ~350 passos e a
     rede ficaria subtreinada por motivo alheio ao método; fixamos então 1 440 passos de lote
     min(500, n), o mesmo número de atualizações do artigo. Decisão a priori, não ajustada
     olhando a validação.
  6. Sem quantização int8 (treino e inferência em float32); o custo int8 do artigo é usado
     só em `custo_mcu`.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from motorsound.config import PostConfig, PreConfig
from pipelines import _strantzalis as S

NOME = "p00_logmel_cnn"
ORIGEM = "strantzalis2022"
TIPO = "inspirada"


@dataclass(frozen=True)
class PreLogMel(PreConfig):
    sr: int = 16000
    n_fft: int = 1024
    hop: int = 512
    n_mels: int = 30


@dataclass(frozen=True)
class PostLogMel(PostConfig):
    colunas: int = 32          # largura da entrada do artigo
    passo_treino: int = 1      # colunas entre sub-janelas de treino
    lote: int = 500
    n_passos: int = 1440       # = 10 épocas × 144 lotes do artigo (suposição 5)
    lr: float = 1e-3


PRE = PreLogMel()
POST = PostLogMel()


def extrair(y, sr, pre):
    M = librosa.feature.melspectrogram(y=np.asarray(y, dtype=np.float32), sr=sr,
                                       n_fft=pre.n_fft, hop_length=pre.hop,
                                       n_mels=pre.n_mels, power=2.0, center=True)
    return librosa.power_to_db(M, ref=1.0, amin=1e-10, top_db=None).astype(np.float32)


class RedeSTM(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = 30, colunas: int = 32):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
        h = ((n_mels - 2) // 2 - 2) // 2
        w = ((colunas - 2) // 2 - 2) // 2
        self.c = nn.Sequential(nn.Linear(16 * h * w, 9), nn.ReLU(), nn.Linear(9, n_classes))

    def forward(self, x):                       # x: (n, 30, 32)
        return self.c(self.f(x.unsqueeze(1)))   # logits; softmax fica na perda / predição


def _subjanelas(X, L, passo):
    """(n, F, T) → (n·k, F, L), k sub-janelas por janela."""
    ini = S.inicios(X.shape[2], L, passo)
    V = sliding_window_view(X, L, axis=2)[:, :, ini, :]     # (n, F, k, L)
    return V.transpose(0, 2, 1, 3).reshape(-1, X.shape[1], L), len(ini)


def treinar(X, rotulos, post, n_classes):
    Xs, k = _subjanelas(np.asarray(X, dtype=np.float32), post.colunas, post.passo_treino)
    ys = np.repeat(np.asarray(rotulos), k)
    mu, sd = float(Xs.mean()), float(Xs.std() + 1e-8)
    rede = RedeSTM(n_classes, Xs.shape[1], post.colunas)
    S.treinar_rede(rede, (Xs - mu) / sd, ys, lr=post.lr, lote=post.lote,
                   n_passos=post.n_passos, seed=post.seed)
    presentes = np.zeros(n_classes, bool)
    presentes[np.unique(rotulos)] = True
    return {"rede": rede, "mu": mu, "sd": sd, "presentes": presentes}


def prever(modelo, X, post):
    Xs, k = _subjanelas(np.asarray(X, dtype=np.float32), post.colunas, None)
    P = S.probabilidades(modelo["rede"], (Xs - modelo["mu"]) / modelo["sd"])
    return S.media_por_janela(P, k, modelo["presentes"])


def _macc(n_classes=3, F=30, L=32):
    """MACC por inferência, contando conv, densas e (como o X-CUBE-AI) pooling/ativações."""
    h1, w1 = F - 2, L - 2                       # 28×30
    p1h, p1w = h1 // 2, w1 // 2                 # 14×15
    h2, w2 = p1h - 2, p1w - 2                   # 12×13
    p2h, p2w = h2 // 2, w2 // 2                 # 6×6
    conv = h1 * w1 * 16 * 9 + h2 * w2 * 16 * 16 * 9
    dens = 16 * p2h * p2w * 9 + 9 * n_classes
    extra = 16 * (h1 * w1 + h2 * w2) * 2        # ReLU + pooling, aproximado
    return conv + dens + extra


def custo_mcu():
    rede = RedeSTM(3)
    params = S.n_parametros(rede)
    macc = _macc()
    # Front-end: 31,25 FFT de 1024 pontos por s (~5·N·log2N flops) + 30 filtros mel sobre 513 bins
    dsp_flops_s = 16000 / 512 * (5 * 1024 * 10 + 2 * 513 * 30)
    nn_flops_s = 2 * macc / 1.024               # uma inferência a cada 1,024 s
    # ESP32: escala pela frequência a partir dos ciclos/MACC medidos no artigo
    # (STM32L475 Cortex-M4 a 80 MHz: 40,767 ms para 501 428 MACC ≈ 6,5 ciclos/MACC).
    ciclos_macc = 40.767e-3 * 80e6 / 501428
    esp32_ms = macc * ciclos_macc / 240e6 * 1e3
    return {
        "params": params,
        "ram_kb": 5.52 + 30 * 32 / 1024 + 1024 * 4 / 1024,   # ativações (artigo) + entrada int8 + buffer FFT f32
        "flash_kb": round(params / 1024 + 0.1, 2),           # pesos int8 (+ escalas); artigo 7,65–7,8 kB
        "mflops_s": round((dsp_flops_s + nn_flops_s) / 1e6, 2),
        "macc_inferencia": macc,
        "artigo_stm32": {"macc": 501428, "ms": 40.767, "ram_kb": 5.52, "flash_kb": 7.65,
                         "placa": "B-L475E-IOT01A (Cortex-M4 80 MHz), int8 X-CUBE-AI"},
        "esp32_ms_estimado": round(esp32_ms, 1),
        "nota": (f"{params} parâmetros (artigo ≈7,7 kB flash int8 coerente); MACC analítico "
                 f"{macc} vs 501 428 do X-CUBE-AI. ESP32 (LX6 240 MHz) estimado escalando "
                 f"os ≈{ciclos_macc:.1f} ciclos/MACC medidos no STM32 pela frequência: "
                 f"≈{esp32_ms:.0f} ms/inferência (ESP-NN no ESP32-S3 tende a ser mais rápido; "
                 "sem SIMD no LX6 pode ser 2× pior). RAM: 5,5 kB de ativações do artigo + "
                 "buffers de entrada e FFT. Uma inferência por 1,024 s: carga < 2 % da CPU."),
    }
