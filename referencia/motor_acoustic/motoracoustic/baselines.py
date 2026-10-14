"""Baselines para a tabela de comparação.

1. pYIN (librosa) como rastreador de f0 alternativo ao ridge harmônico.
2. log-mel + classificador, que reproduz o tipo de entrada usado pelos
   trabalhos com CNN (Strantzalis 2022 usa log-mel 30x32 a 16 kHz).
   A CNN em si fica para o notebook de comparação; aqui só a feature.
"""
from __future__ import annotations

import numpy as np

from .config import PipelineConfig


def pyin_track(x: np.ndarray, sr: int, cfg: PipelineConfig):
    import librosa
    # pYIN precisa de janela com pelo menos 2 períodos de f0_min
    frame = int(2 ** np.ceil(np.log2(max(2048, 4 * sr / cfg.f0_min))))
    f0, voiced, vprob = librosa.pyin(
        x, fmin=cfg.f0_min, fmax=min(cfg.f0_max, sr / 4), sr=sr,
        frame_length=frame, hop_length=cfg.stft_hop)
    return f0, voiced, vprob


def logmel_patch(x: np.ndarray, sr: int, n_mels: int = 30, n_fft: int = 1024,
                 hop: int = 512) -> np.ndarray:
    import librosa
    M = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, power=2.0)
    return np.log(M + 1e-10).astype(np.float32)


def logmel_summary(x: np.ndarray, sr: int) -> np.ndarray:
    """Média e desvio por banda: baseline 'saco de features' para classificador raso."""
    L = logmel_patch(x, sr)
    return np.concatenate([L.mean(axis=1), L.std(axis=1)])
