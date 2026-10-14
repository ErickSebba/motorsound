"""Rastreamento da rotação: saliência por soma harmônica + Viterbi.

Equivale ao pYIN do Decifrae em estrutura (observação por quadro -> HMM ->
caminho mais provável), mas troca o YIN por soma harmônica no espectro, que:
  * não depende da fundamental ser forte (som de motor costuma ter
    harmônicos de comutação mais altos que f0);
  * reutiliza a mesma FFT das outras features (no ESP32 é a mesma chamada).

Saliência de um candidato f:
    S(f) = sum_h w_h A(h f)  -  lambda * sum_h w_h A((h - 1/2) f)
O termo inter-harmônico é o que impede o erro de sub-oitava: em f/2 os
pontos (h - 1/2)·f/2 caem exatamente sobre os harmônicos reais.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

from .config import PipelineConfig


def magnitude_spectrogram(x: np.ndarray, sr: int, cfg: PipelineConfig):
    f, t, Z = stft(x, fs=sr, window="hann", nperseg=cfg.n_fft,
                   noverlap=cfg.n_fft - cfg.stft_hop, boundary=None, padded=False)
    A = np.abs(Z).astype(np.float32)
    A = np.sqrt(A)                     # compressão: harmônicos fracos contam
    return f, t, A


def f0_grid(cfg: PipelineConfig) -> np.ndarray:
    return np.arange(cfg.f0_min, cfg.f0_max + 1e-9, cfg.f0_step, dtype=np.float32)


def harmonic_salience(A: np.ndarray, freqs: np.ndarray, grid: np.ndarray,
                      cfg: PipelineConfig) -> np.ndarray:
    """Retorna S com forma (n_candidatos, n_quadros), normalizada por quadro."""
    df = freqs[1] - freqs[0]
    nb = A.shape[0]
    S = np.zeros((len(grid), A.shape[1]), dtype=np.float32)

    def sample(fq: np.ndarray) -> np.ndarray:
        # interpolação linear entre bins (barata no MCU)
        pos = fq / df
        i0 = np.clip(np.floor(pos).astype(int), 0, nb - 2)
        a = (pos - i0)[:, None]
        return (1 - a) * A[i0] + a * A[i0 + 1]

    for h in range(1, cfg.n_harmonics + 1):
        w = cfg.harmonic_decay ** (h - 1)
        valid = h * grid < freqs[-1]
        if not valid.any():
            break
        S[valid] += w * sample(h * grid[valid])
        S[valid] -= cfg.interharmonic_penalty * w * sample((h - 0.5) * grid[valid])

    S = np.maximum(S, 0)
    S /= S.max(axis=0, keepdims=True) + 1e-9
    return S


def viterbi_ridge(S: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Caminho de máxima saliência acumulada com custo linear por salto.

    Maximiza  sum_t S[k_t, t]  -  c * sum_t |f(k_t) - f(k_{t-1})|
    com |salto| <= max_jump. Custo O(K·J) por quadro, J = saltos permitidos.
    """
    K, T = S.shape
    J = max(1, int(round(cfg.max_jump_hz / cfg.f0_step)))
    offsets = np.arange(-J, J + 1)
    pen = cfg.jump_cost * np.abs(offsets) * cfg.f0_step

    score = S[:, 0].copy()
    back = np.zeros((K, T), dtype=np.int32)
    idx = np.arange(K)
    for t in range(1, T):
        best = np.full(K, -np.inf, dtype=np.float32)
        arg = np.zeros(K, dtype=np.int32)
        for o, p in zip(offsets, pen):
            src = idx - o
            ok = (src >= 0) & (src < K)
            cand = np.full(K, -np.inf, dtype=np.float32)
            cand[ok] = score[src[ok]] - p
            upd = cand > best
            best[upd] = cand[upd]
            arg[upd] = src[upd]
        score = best + S[:, t]
        back[:, t] = arg

    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return path


def parabolic_refine(S: np.ndarray, path: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Refino sub-grade por parábola nos 3 pontos em volta do pico."""
    K = S.shape[0]
    f = grid[path].astype(np.float64)
    step = grid[1] - grid[0]
    for t, k in enumerate(path):
        if 0 < k < K - 1:
            y0, y1, y2 = S[k - 1, t], S[k, t], S[k + 1, t]
            den = y0 - 2 * y1 + y2
            if den < 0:
                f[t] += 0.5 * (y0 - y2) / den * step
    return f


def track_rotation(x: np.ndarray, sr: int, cfg: PipelineConfig):
    """Pipeline completo do ridge.

    Retorna dict com tempos, f0 (Hz), saliência no caminho e a matriz S.
    """
    freqs, times, A = magnitude_spectrogram(x, sr, cfg)
    grid = f0_grid(cfg)
    S = harmonic_salience(A, freqs, grid, cfg)
    path = viterbi_ridge(S, cfg)
    f0 = parabolic_refine(S, path, grid)
    sal = S[path, np.arange(S.shape[1])]
    # "contraste": quão destacado o pico está do resto do quadro
    contrast = sal - np.median(S, axis=0)
    return {"t": times, "f0": f0, "salience": sal, "contrast": contrast,
            "S": S, "grid": grid, "A": A, "freqs": freqs}


def track_rotation_framewise(x: np.ndarray, sr: int, cfg: PipelineConfig) -> np.ndarray:
    """Ablação: argmax por quadro, sem Viterbi. Serve para medir o ganho do Viterbi."""
    freqs, _, A = magnitude_spectrogram(x, sr, cfg)
    grid = f0_grid(cfg)
    S = harmonic_salience(A, freqs, grid, cfg)
    return grid[np.argmax(S, axis=0)]
