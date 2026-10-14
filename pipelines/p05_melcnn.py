"""P05 — mel-espectrograma 128 bandas redimensionado para 32×32 → CNN pequena (~2,4 M parâmetros).

Referência: docs/literatura/melcnn2025.md (Uzel et al., Scientific Reports 16:3379, 2025).
A ficha classifica o artigo como "só inspiração" (split por recorte com vazamento; camadas
finais da CNN cortadas no texto; taxa de amostragem incerta). Portanto TIPO = "inspirada".

Seguido do artigo:
  - mel-espectrograma com FFT 2048, hop 512, 128 bandas;
  - Resize para 32 × 32 → Normalization → Conv(32, 3×3) → Conv(64, 3×3) → MaxPool 2×2 →
    Conv(128, 3×3) → Dropout → … → Dense → softmax;
  - Adam, entropia cruzada esparsa, 10 épocas, lote 32.

Suposições (o que o texto não dá ou dá de forma inconsistente):
  1. Taxa 22 050 Hz (padrão do `librosa.load`; o artigo usa librosa nas linhas de base). As
     figuras falam em eixo de 0 a 16 000 "amostras", o que não define a taxa.
  2. Escala em dB: `power_to_db(ref=1.0)` sobre a potência mel (não descrito; ref fixo para
     não apagar o nível entre janelas — o runner já normaliza o RMS para 0,1).
  3. Resize bilinear sem antialias (padrão do `tf.image.resize`), feito na extração porque é
     determinístico; equivale à camada Resize do modelo.
  4. Normalization: média e desvio globais (um canal) estimados no treino de cada fold,
     como `keras.layers.Normalization.adapt` com eixo de canal único.
  5. Convoluções 'valid' com ReLU; Dropout 0,25 após a Conv(128); depois Flatten →
     Dense(128, ReLU) → Dropout 0,5 → Dense(C). O trecho "…" do artigo foi preenchido assim
     (2 452 483 parâmetros contra os 2 403 861 declarados; nenhuma combinação simples de
     camadas que testamos reproduz o número exato — declarado, não ajustado).
  6. lr = 0,001 (padrão do Adam no Keras); inicialização Glorot uniforme (padrão do Keras).
  7. Sem early stopping (paciência 5): com só 10 épocas e sem validação interna agrupada por
     gravação, ele exigiria separar parte do treino; preferimos 10 épocas fixas.
  8. A janela do runner (3 s no IDMT, 1 s no A2212) é a amostra da CNN; o artigo usa os
     recortes de 3 s. No A2212 há menos quadros (32 → 32 por interpolação).
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

from motorsound.config import PostConfig, PreConfig
from pipelines._p02p05_util import glorot, n_params, probabilidades, treinar_rede

NOME = "p05_melcnn"
ORIGEM = "melcnn2025"
TIPO = "inspirada"


@dataclass(frozen=True)
class PreMelCNN(PreConfig):
    sr: int = 22050
    n_fft: int = 2048
    hop: int = 512
    n_mels: int = 128
    lado: int = 32


@dataclass(frozen=True)
class PostMelCNN(PostConfig):
    epocas: int = 10
    lote: int = 32
    lr: float = 1e-3


PRE = PreMelCNN()
POST = PostMelCNN()


def extrair(y, sr, pre):
    """(32, 32) log-mel redimensionado."""
    M = librosa.feature.melspectrogram(y=np.asarray(y, dtype=np.float32), sr=sr,
                                       n_fft=pre.n_fft, hop_length=pre.hop,
                                       n_mels=pre.n_mels, power=2.0)
    L = librosa.power_to_db(M, ref=1.0).astype(np.float32)             # (128, T)
    t = torch.from_numpy(L)[None, None]
    R = Fn.interpolate(t, size=(pre.lado, pre.lado), mode="bilinear",
                       align_corners=False, antialias=False)
    return R[0, 0].numpy()


class _CNN(nn.Module):
    def __init__(self, n_classes, lado=32):
        super().__init__()
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sd", torch.ones(1))
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(),
            nn.Dropout(0.25))
        s = (lado - 4) // 2 - 2                                         # 32 → 12
        self.dense = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * s * s, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, n_classes))

    def forward(self, x):
        x = (x[:, None] - self.mu) / self.sd
        return self.dense(self.conv(x))


def _rede(n_classes):
    m = _CNN(n_classes, PRE.lado)
    glorot(m)
    return m


def treinar(X, y, post, n_classes):
    X = np.asarray(X, dtype=np.float32)
    rede = _rede(n_classes)
    rede.mu.fill_(float(X.mean()))
    rede.sd.fill_(float(X.std()) + 1e-6)
    rede = treinar_rede(rede, X, y, post.epocas, post.lote, post.lr, post.seed)
    return {"rede": rede, "classes": np.unique(y), "C": n_classes}


def prever(modelo, X, post):
    P = probabilidades(modelo["rede"], np.asarray(X, dtype=np.float32))
    mask = np.zeros(modelo["C"], bool)
    mask[modelo["classes"]] = True
    P[:, ~mask] = 0.0
    return P / P.sum(axis=1, keepdims=True)


def custo_mcu():
    params = n_params(_rede(3))
    # MACs por inferência (entrada 32×32): conv1 30²·32·9, conv2 28²·64·32·9, conv3 12²·128·64·9
    macs = 30 * 30 * 32 * 9 + 28 * 28 * 64 * 32 * 9 + 12 * 12 * 128 * 64 * 9 \
        + 12 * 12 * 128 * 128 + 128 * 3
    q_s = PRE.sr / PRE.hop                                              # ~43 FFT/s
    mel = q_s * (5 * PRE.n_fft * np.log2(PRE.n_fft) + 2 * PRE.n_mels * (PRE.n_fft // 2 + 1))
    inf_s = 1 / 3.0                                                     # 1 inferência por janela de 3 s
    return {"params": params,
            "ram_kb": round((30 * 30 * 32 + 28 * 28 * 64) * 4 / 1024
                            + PRE.n_mels * 130 * 4 / 1024, 1),
            "flash_kb": round(params * 4 / 1024, 1),
            "mflops_s": round(float(mel + 2 * macs * inf_s) / 1e6, 1),
            "nota": "estimativa analítica; flash float32 ~9,4 MB não cabe no ESP32 (int8 ~2,4 MB); "
                    "RAM = duas maiores ativações + mel de 3 s; 1 inferência por janela de 3 s"}
