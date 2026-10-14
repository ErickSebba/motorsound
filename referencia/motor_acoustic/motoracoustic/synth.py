"""Sinal de motor sintético com verdade conhecida.

Serve para testar o ridge ANTES de ter a base baixada, e para os testes
automatizados. Não entra como resultado no artigo.

Modelo: soma de harmônicos de f0(t), fundamental propositalmente fraca
(caso difícil), modulação AM opcional e ruído branco ou "fala" (ruído
modulado em sílabas).
"""
from __future__ import annotations

import numpy as np


def motor_tone(f0: np.ndarray, sr: int, amps=(0.15, 1.0, 0.6, 0.8, 0.3, 0.4, 0.2, 0.15),
               am_hz: float = 0.0, am_depth: float = 0.0, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = np.zeros_like(f0, dtype=np.float64)
    for h, a in enumerate(amps, start=1):
        x += a * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
    if am_hz > 0:
        t = np.arange(len(f0)) / sr
        x *= 1 + am_depth * np.sign(np.sin(2 * np.pi * am_hz * t))
    return x.astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, kind: str = "white", sr: int = 16000,
              rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    n = rng.standard_normal(len(x))
    if kind == "speechlike":
        t = np.arange(len(x)) / sr
        n *= 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 4 * t))   # ~4 sílabas/s
        n = np.convolve(n, np.ones(8) / 8, mode="same")     # puxa para graves
    ps, pn = np.mean(x ** 2), np.mean(n ** 2)
    n *= np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return (x + n).astype(np.float32)


def synthetic_classes(sr: int = 16000, dur: float = 10.0, rng=0):
    """Três 'estados' que imitam a lógica do IDMT:
    good      : f0 constante ~40 Hz
    heavyload : f0 menor (carga freia o motor)
    broken    : f0 médio com chaveamento de amplitude rápido
    """
    rng = np.random.default_rng(rng)
    n = int(sr * dur)
    out = {}
    base = 40 + rng.normal(0, 0.5)
    out["good"] = motor_tone(np.full(n, base), sr, rng=rng)
    out["heavyload"] = motor_tone(np.full(n, base * 0.85), sr, rng=rng)
    out["broken"] = motor_tone(np.full(n, base * 0.95), sr, am_hz=27.8,
                               am_depth=0.6, rng=rng)
    return out
