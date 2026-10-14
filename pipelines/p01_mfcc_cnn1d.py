"""P01 — rota Edge Impulse de Strantzalis et al. (2022): 13 MFCC + CNN 1D.

Referência: ficha docs/literatura/strantzalis2022.md (texto completo, PMC9783357, §5.2).

SEGUIDO DO ARTIGO
  - Janelas de 1 s com 0,5 s de sobreposição.
  - MFCC: FFT de 256, quadro de 20 ms, 32 filtros mel, 13 coeficientes.
  - Rede: reshape → Conv1D(30, k=5, ReLU) → MaxPool(pool_size=5) → Conv1D(10, k=5, ReLU)
    → MaxPool(5) → Flatten → Dense(C, softmax). Adam, entropia cruzada, 90 épocas.

SUPOSIÇÕES (o artigo usa os padrões internos do Edge Impulse e não os lista) — TIPO = "inspirada"
  1. Taxa de 16 kHz (a do microfone da placa e da rota STM).
  2. MFCC à moda do speechpy usado pelo Edge Impulse (bloco MFCC, lembrança do código público,
     não verificada contra a versão usada em 2022): pré-ênfase 0,98; quadros de 320 amostras
     com passo de 20 ms (sem sobreposição) → 50 quadros por s; espectro de potência por
     rfft(n=256), que TRUNCA o quadro de 320 amostras às primeiras 256; 32 filtros
     triangulares de 0 Hz a 8 kHz; log natural; DCT-II ortonormal, 13 primeiros coeficientes
     (c0 mantido); normalização cepstral de média e variância em janela deslizante de 101
     quadros (cmvnw, preenchimento simétrico), que com 50 quadros equivale quase a uma
     normalização sobre a janela de 1 s inteira.
  3. Convoluções com padding 'same' e pooling com stride = pool_size e 'same' (Keras/EI):
     50×13 → 50×30 → 10×30 → 10×10 → 2×10 → 20 → Dense. Sem dropout (o artigo não cita).
  4. lr = 0,005 e lote 32 (padrões do Edge Impulse); 90 épocas como no artigo.
  5. Janelas do runner: 3 s (IDMT) → 5 sub-janelas de 1 s com passo de 0,5 s; 1 s (A2212) →
     1 sub-janela. `extrair` devolve (k, 50, 13). No treino cada sub-janela é uma amostra
     com o rótulo da janela; na predição a probabilidade da janela é a média das k
     sub-janelas.
  6. Sem quantização (float32); o custo int8 do artigo é usado só em `custo_mcu`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct
from torch import nn

from motorsound.config import PostConfig, PreConfig
from pipelines import _strantzalis as S

NOME = "p01_mfcc_cnn1d"
ORIGEM = "strantzalis2022"
TIPO = "inspirada"


@dataclass(frozen=True)
class PreMFCC(PreConfig):
    sr: int = 16000
    janela_s: float = 1.0
    passo_s: float = 0.5
    quadro_s: float = 0.02
    passo_quadro_s: float = 0.02
    n_fft: int = 256
    n_filtros: int = 32
    n_ceps: int = 13
    pre_enfase: float = 0.98
    fmin: float = 0.0
    cmvn_janela: int = 101


@dataclass(frozen=True)
class PostMFCC(PostConfig):
    epocas: int = 90
    lote: int = 32
    lr: float = 0.005


PRE = PreMFCC()
POST = PostMFCC()


def _hz2mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)


def _mel2hz(m):
    return 700.0 * (np.exp(np.asarray(m) / 1127.0) - 1.0)


def _banco_mel(n_filtros, n_fft, sr, fmin, fmax):
    """Banco triangular no estilo speechpy (bordas arredondadas para bins da FFT)."""
    pts = _mel2hz(np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_filtros + 2))
    nb = n_fft // 2 + 1
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    H = np.zeros((n_filtros, nb))
    for i in range(n_filtros):
        a, c, b = bins[i], bins[i + 1], bins[i + 2]
        for k in range(a, c):
            H[i, k] = (k - a) / max(c - a, 1)
        for k in range(c, min(b, nb - 1) + 1):
            H[i, k] = (b - k) / max(b - c, 1)
    return H


def _cmvnw(F, win):
    """speechpy.processing.cmvnw com variance_normalization=True."""
    n = F.shape[0]
    pad = (win - 1) // 2
    P = np.pad(F, ((pad, pad), (0, 0)), mode="symmetric")
    out = np.empty_like(F)
    for i in range(n):
        w = P[i:i + win]
        out[i] = (F[i] - w.mean(axis=0)) / (w.std(axis=0) + 1e-10)
    return out


def _mfcc(sinal, pre):
    sr = pre.sr
    x = np.append(sinal[0], sinal[1:] - pre.pre_enfase * sinal[:-1])
    L = int(round(pre.quadro_s * sr))
    H = int(round(pre.passo_quadro_s * sr))
    n_q = int(np.ceil((len(x) - L) / H)) + 1
    x = np.pad(x, (0, max(0, (n_q - 1) * H + L - len(x))))
    Q = np.stack([x[i * H:i * H + L] for i in range(n_q)])
    pot = np.abs(np.fft.rfft(Q, n=pre.n_fft, axis=1)) ** 2 / pre.n_fft   # trunca 320 → 256
    E = pot @ _banco_mel(pre.n_filtros, pre.n_fft, sr, pre.fmin, sr / 2).T
    E = np.log(np.where(E == 0, np.finfo(float).eps, E))
    C = dct(E, type=2, axis=1, norm="ortho")[:, :pre.n_ceps]
    return _cmvnw(C, pre.cmvn_janela)


def extrair(y, sr, pre):
    y = np.asarray(y, dtype=np.float64)
    J, P = int(round(pre.janela_s * sr)), int(round(pre.passo_s * sr))
    ini = range(0, len(y) - J + 1, P) if len(y) >= J else [0]
    return np.stack([_mfcc(np.pad(y[i:i + J], (0, max(0, J - len(y)))), pre)
                     for i in ini]).astype(np.float32)


class RedeEI(nn.Module):
    def __init__(self, n_classes: int, n_ceps: int = 13, n_quadros: int = 50):
        super().__init__()
        t = -(-(-(-n_quadros // 5)) // 5)          # ceil duas vezes ('same')
        self.f = nn.Sequential(
            nn.Conv1d(n_ceps, 30, 5, padding=2), nn.ReLU(), nn.MaxPool1d(5, ceil_mode=True),
            nn.Conv1d(30, 10, 5, padding=2), nn.ReLU(), nn.MaxPool1d(5, ceil_mode=True),
            nn.Flatten(), nn.Linear(10 * t, n_classes))

    def forward(self, x):                      # x: (n, 50 quadros, 13 coef)
        return self.f(x.transpose(1, 2))


def treinar(X, rotulos, post, n_classes):
    X = np.asarray(X, dtype=np.float32)
    n, k, T, C = X.shape
    rede = RedeEI(n_classes, C, T)
    S.treinar_rede(rede, X.reshape(n * k, T, C), np.repeat(np.asarray(rotulos), k),
                   lr=post.lr, lote=post.lote, epocas=post.epocas, seed=post.seed)
    presentes = np.zeros(n_classes, bool)
    presentes[np.unique(rotulos)] = True
    return {"rede": rede, "presentes": presentes}


def prever(modelo, X, post):
    X = np.asarray(X, dtype=np.float32)
    n, k, T, C = X.shape
    P = S.probabilidades(modelo["rede"], X.reshape(n * k, T, C))
    return S.media_por_janela(P, k, modelo["presentes"])


def custo_mcu():
    rede = RedeEI(3)
    params = S.n_parametros(rede)
    macc = 50 * 30 * 13 * 5 + 10 * 10 * 30 * 5 + 20 * 3     # conv1 + conv2 + densa
    # DSP por janela de 1 s: 50 quadros × (FFT 256 ≈ 5·N·log2N + mel 2·129·32 + DCT 2·32·13)
    dsp_flops_jan = 50 * (5 * 256 * 8 + 2 * 129 * 32 + 2 * 32 * 13)
    # duas janelas por s (passo 0,5 s), cada uma refeita do zero como no EI
    flops_s = 2 * (dsp_flops_jan + 2 * macc)
    ciclos_macc = 40.767e-3 * 80e6 / 501428     # medido na rota STM (mesma placa)
    esp32_nn_ms = macc * ciclos_macc / 240e6 * 1e3
    return {
        "params": params,
        "ram_kb": 7.0,                              # pico reportado pelo EI (inclui DSP)
        "flash_kb": 38.2,                           # ROM reportada pelo EI (rede + DSP + runtime)
        "mflops_s": round(flops_s / 1e6, 2),
        "macc_inferencia": macc,
        "artigo_ei": {"ram_kb": 7.0, "rom_kb": 38.2, "nn_ms": 16, "dsp_ms": 337,
                      "classificacao_ms": 36},
        "artigo_stm_ref": {"ram_kb": 5.52, "flash_kb": 7.65, "ms": 40.767},
        "esp32_nn_ms_estimado": round(esp32_nn_ms, 1),
        "nota": (f"{params} parâmetros (pesos int8 ≈ {params / 1024:.1f} kB; os 38,2 kB do EI "
                 f"incluem DSP e runtime). {macc} MACC por inferência, ~4,4× menos que a rota "
                 "STM (501 428). ESP32: rede ≈ "
                 f"{esp32_nn_ms:.1f} ms escalando os ciclos/MACC do STM32 pela frequência; o "
                 "gargalo reportado é o DSP (337 ms por janela no MCU do artigo, que com passo "
                 "de 0,5 s ocuparia > 60 % da CPU), não a rede. Comparação: rota STM 5,5 kB "
                 "RAM, 7,7 kB flash, 40,8 ms."),
    }
