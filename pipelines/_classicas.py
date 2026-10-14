"""Núcleo comum de p04_classicas e p04c_classicas_lr (ficha: docs/literatura/vanaraj2026.md).

Features acústicas clássicas citadas no resumo de Vanaraj et al. (2026): ZCR, RMS, centróide
espectral e MFCCs. O resumo não dá nenhum parâmetro; tudo abaixo é suposição nossa:

- quadros de 32 ms (512 amostras a 16 kHz), hop de 50 % (256), janela de Hann, sem centralização
  (quadros só dentro da janela, para que a forma não dependa de padding);
- 13 MFCC com banco mel de 26 filtros (20 Hz–sr/2), log de potência, DCT-II ortonormal
  (escolha "embarcável", como sugerido na ficha; o librosa usa 128 filtros por padrão);
- agregação por janela de avaliação: média e desvio-padrão de cada uma das 16 séries
  → 32 features. Funciona igual em 1 s (A2212) e 3 s (IDMT): só muda o número de quadros.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from motorsound.config import PreConfig


@dataclass(frozen=True)
class PreClassicas(PreConfig):
    sr: int = 16000
    n_fft: int = 512          # 32 ms a 16 kHz
    hop: int = 256            # 50 % de sobreposição
    n_mfcc: int = 13
    n_mels: int = 26
    fmin_mel: float = 20.0


N_FEATURES = 2 * (3 + 13)
NOMES = [f"{e}_{n}" for e in ("media", "desvio")
         for n in ["zcr", "rms", "centroide"] + [f"mfcc{i}" for i in range(13)]]


def extrair_classicas(y, sr, pre: PreClassicas) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) < pre.n_fft:
        y = np.pad(y, (0, pre.n_fft - len(y)))
    quadros = librosa.util.frame(y, frame_length=pre.n_fft, hop_length=pre.hop)  # (N, T)
    zcr = np.mean(np.abs(np.diff(np.signbit(quadros).astype(np.int8), axis=0)), axis=0)
    rms = np.sqrt(np.mean(quadros.astype(np.float64) ** 2, axis=0))
    w = np.hanning(pre.n_fft)
    S = np.abs(np.fft.rfft(quadros.T * w, axis=1)).T                             # (F, T)
    f = np.fft.rfftfreq(pre.n_fft, 1.0 / sr)
    P = S ** 2
    cent = (f[:, None] * S).sum(0) / (S.sum(0) + 1e-12)
    mel = librosa.filters.mel(sr=sr, n_fft=pre.n_fft, n_mels=pre.n_mels,
                              fmin=pre.fmin_mel, fmax=sr / 2)
    logmel = librosa.power_to_db(mel @ P, ref=1.0, amin=1e-10, top_db=None)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=pre.n_mfcc, dct_type=2, norm="ortho")
    series = np.vstack([zcr, rms, cent, mfcc])                                   # (16, T)
    return np.concatenate([series.mean(1), series.std(1)]).astype(np.float32)


def custo_features(pre: PreClassicas) -> dict:
    """Custo por segundo de áudio das features (estimativa analítica)."""
    q = pre.sr / pre.hop                                   # quadros por segundo
    F = pre.n_fft // 2 + 1
    fft = 5 * pre.n_fft * np.log2(pre.n_fft)               # FFT real ~ 5 N log2 N / 2, folgado
    mel = pre.n_mels * F                                   # banco mel denso (na prática esparso)
    dct = pre.n_mfcc * pre.n_mels
    outros = 4 * pre.n_fft                                 # janela, ZCR, RMS, centróide
    mflops = q * (fft + mel + dct + outros) / 1e6
    ram_kb = (2 * pre.n_fft + F + pre.n_mels + 2 * 16) * 4 / 1024   # buffer, espectro, mel, acumuladores
    flash_kb = pre.n_mels * F * 4 / 1024                   # matriz mel (denso; esparsa é ~10x menor)
    return {"mflops_s": float(mflops), "ram_kb": float(ram_kb), "flash_kb": float(flash_kb)}


def prever_proba(m, C, X):
    P = np.zeros((len(X), C))
    P[:, m.classes_] = m.predict_proba(X)
    return P
