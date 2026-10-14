"""Pré-processamento. Tudo aqui precisa ter equivalente barato em C."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .config import PipelineConfig


def highpass(x: np.ndarray, sr: int, fc: float) -> np.ndarray:
    # Butterworth de 2ª ordem = 1 biquad no firmware.
    sos = butter(2, fc, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def normalize_rms(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalização por RMS do trecho: invariância ao ganho do microfone."""
    return (x / (np.sqrt(np.mean(x * x)) + eps)).astype(np.float32)


def preprocess(x: np.ndarray, sr: int, cfg: PipelineConfig) -> np.ndarray:
    x = x - np.mean(x)
    x = highpass(x, sr, cfg.highpass_hz)
    return normalize_rms(x)


def frame_windows(n_samples: int, sr: int, win_s: float, hop_s: float):
    """Índices (início, fim) das janelas de decisão."""
    w, h = int(win_s * sr), int(hop_s * sr)
    if n_samples < w:
        return [(0, n_samples)]
    return [(s, s + w) for s in range(0, n_samples - w + 1, h)]
