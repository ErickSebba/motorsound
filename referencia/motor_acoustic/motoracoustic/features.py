"""Vetor de features por janela de decisão.

Princípio: cada feature tem significado físico e custo conhecido no MCU.
Nenhuma depende de rede neural.

Grupos
------
rotação  : f0 mediana, dispersão de f0, saliência e contraste do ridge
harmônico: razão de energia nos harmônicos 1..4 sobre a energia total
modulação: pico do espectro de modulação do envelope (frequência e força)
espectro : centróide, planicidade, energia em 4 bandas largas
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .config import PipelineConfig
from .preprocess import frame_windows
from .ridge import track_rotation

FEATURE_NAMES = [
    "f0_med", "f0_iqr", "f0_slope",
    "ridge_sal", "ridge_contrast",
    "harm_r1", "harm_r2", "harm_r3", "harm_r4", "harm_total",
    "mod_freq", "mod_strength",
    "spec_centroid", "spec_flatness",
    "band_0_250", "band_250_1k", "band_1k_3k", "band_3k_8k",
]
GROUPS = {
    "rotacao": ["f0_med", "f0_iqr", "f0_slope", "ridge_sal", "ridge_contrast"],
    "harmonico": ["harm_r1", "harm_r2", "harm_r3", "harm_r4", "harm_total"],
    "modulacao": ["mod_freq", "mod_strength"],
    "espectro": ["spec_centroid", "spec_flatness", "band_0_250", "band_250_1k",
                 "band_1k_3k", "band_3k_8k"],
}


def envelope_modulation(x: np.ndarray, sr: int, cfg: PipelineConfig):
    """Espectro de modulação: envelope RMS a env_rate Hz -> FFT -> pico.

    É aqui que o chaveamento de tensão do "broken" (a cada 18 ms) deve
    aparecer, e onde falha de hélice/rolamento aparece como modulação
    na rotação ou em múltiplos dela.
    """
    hop = max(1, sr // cfg.env_rate)
    n = len(x) // hop
    if n < 16:
        return 0.0, 0.0
    env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    env = env - env.mean()
    E = np.abs(np.fft.rfft(env * np.hanning(n)))
    fm = np.fft.rfftfreq(n, d=hop / sr)
    band = (fm >= cfg.mod_fmin) & (fm <= cfg.mod_fmax)
    if not band.any():
        return 0.0, 0.0
    k = np.argmax(E[band])
    peak = E[band][k]
    strength = peak / (np.median(E[band]) + 1e-9)
    return float(fm[band][k]), float(np.log1p(strength))


def harmonic_ratios(A: np.ndarray, freqs: np.ndarray, f0: float, n: int = 4):
    """Energia em ±1 bin de cada harmônico, sobre a energia total."""
    P = (A ** 4).mean(axis=1)          # A já é sqrt(|X|) -> A**4 = |X|^2
    tot = P.sum() + 1e-12
    df = freqs[1] - freqs[0]
    out = []
    for h in range(1, n + 1):
        k = int(round(h * f0 / df))
        if 1 <= k < len(P) - 1:
            out.append(P[k - 1:k + 2].sum() / tot)
        else:
            out.append(0.0)
    return out, float(sum(out))


def spectral_shape(x: np.ndarray, sr: int):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-12
    f = np.fft.rfftfreq(len(x), 1 / sr)
    centroid = float((f * X).sum() / X.sum())
    flatness = float(np.exp(np.mean(np.log(X))) / np.mean(X))
    tot = X.sum()
    edges = [(0, 250), (250, 1000), (1000, 3000), (3000, sr / 2)]
    bands = [float(np.log(X[(f >= a) & (f < b)].sum() / tot + 1e-9)) for a, b in edges]
    return centroid, flatness, bands


def window_features(x: np.ndarray, sr: int, cfg: PipelineConfig) -> np.ndarray:
    """Features de uma única janela de decisão."""
    r = track_rotation(x, sr, cfg)
    f0 = r["f0"]
    f0_med = float(np.median(f0))
    q75, q25 = np.percentile(f0, [75, 25])
    slope = float(np.polyfit(np.arange(len(f0)), f0, 1)[0]) if len(f0) > 2 else 0.0
    hr, htot = harmonic_ratios(r["A"], r["freqs"], f0_med)
    mf, ms = envelope_modulation(x, sr, cfg)
    cen, flat, bands = spectral_shape(x, sr)
    return np.array([f0_med, q75 - q25, slope,
                     float(np.mean(r["salience"])), float(np.mean(r["contrast"])),
                     *hr, htot, mf, ms, cen, flat, *bands], dtype=np.float32)


def file_features(x: np.ndarray, sr: int, cfg: PipelineConfig) -> np.ndarray:
    """Matriz (n_janelas, n_features) de um arquivo inteiro."""
    rows = [window_features(x[a:b], sr, cfg)
            for a, b in frame_windows(len(x), sr, cfg.win_s, cfg.hop_s)]
    return np.vstack(rows)


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)
