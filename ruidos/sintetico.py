"""Gerador sintético de tom de motor com rotação conhecida (bancada da Etapa 5).

Modelo (todas as escolhas são suposições declaradas, não medidas):

* tensão v(t) de `ruidos.lee_shen.tensao_perturbada` (1,0 p.u. = condição "good");
* rotação de 1ª ordem   τ·df/dt = K·v(t) − f,   τ = 0,10 s (TAU_MEC; constante mecânica
  do 42BLF01 não consultada), K = f_nom em Hz/p.u. (f_nom = 25 Hz por padrão, dentro da
  faixa de rotação ~24,5–28 Hz observada no IDMT), f(0) = K·v(0);
* áudio = Σ_h A_h · a(t) · sen(φ_h + 2π·h·∫f dt), h = 1..48, com fase integrada;
  a(t) = (f/f_nom)·(1 − m_am + m_am·v(t)): nível proporcional à rotação mais uma parcela
  (m_am = 0,3) que segue a tensão instantânea (torque eletromagnético);
* amplitudes A_h inspiradas no espectro do IDMT (docs/DATASETS.md, exploração de 14/10):
  fundo fraco ∝ 1/√h em todos os harmônicos; múltiplos de 4 mais fortes; 12×, 16×, 24× e
  48× os mais fortes, com o 24º dominante. Cada semente aplica ±3 dB aleatórios por
  harmônico e fases iniciais aleatórias;
* ruído rosa (ruidos.noise.add_pink_noise) ao SNR pedido, sobre a potência do trecho todo.

A verdade é devolvida por amostra (f_rot na taxa de áudio); `verdade_por_quadro` a
converte para os quadros do rastreador (média ponderada pela janela de Hann de cada
quadro, a mesma janela do `motorsound.rotacao.stft_mag`).
"""
from __future__ import annotations

import numpy as np

from .lee_shen import TAU_MEC, rotacao_relativa, tensao_perturbada
from .noise import add_pink_noise

SR = 16000
F_NOM = 25.0
N_HARM = 48
# harmônico → amplitude relativa (linear), sobre o fundo 0,05/√h
FORTES = {4: 0.30, 8: 0.35, 12: 0.60, 16: 0.50, 20: 0.30, 24: 1.00, 28: 0.25,
          32: 0.30, 36: 0.25, 40: 0.20, 44: 0.20, 48: 0.50}


def amplitudes_harmonicas(n_harm=N_HARM, rng=None, jitter_db=3.0):
    h = np.arange(1, n_harm + 1)
    A = 0.05 / np.sqrt(h)
    for k, a in FORTES.items():
        if k <= n_harm:
            A[k - 1] = a
    if rng is not None and jitter_db > 0:
        A = A * 10 ** (rng.uniform(-jitter_db, jitter_db, n_harm) / 20)
    return A


def gerar(dur_s=15.0, sr=SR, tipo="nominal", snr_db=None, seed=None, f_nom=F_NOM,
          tau_s=TAU_MEC, m_am=0.3, n_harm=N_HARM, devolver_v=False, **params_disturbio):
    """Devolve (y, f_rot_verdade por amostra) [, v, params].

    snr_db=None → sem ruído. `params_disturbio` vão para tensao_perturbada (parâmetros
    omitidos são sorteados com a semente).
    """
    rng = np.random.default_rng(seed)
    seed_v, seed_r = (int(s) for s in rng.integers(0, 2**31 - 1, 2))
    v, p = tensao_perturbada(dur_s, sr, tipo, seed=seed_v, devolver_params=True,
                             **params_disturbio)
    f_rot = f_nom * rotacao_relativa(v, sr, tau_s)
    fase = 2 * np.pi * np.concatenate([[0.0], np.cumsum(f_rot[:-1])]) / sr
    a = (f_rot / f_nom) * np.clip(1.0 - m_am + m_am * v, 0.0, None)
    A = amplitudes_harmonicas(n_harm, rng)
    phi0 = rng.uniform(0, 2 * np.pi, n_harm)
    y = np.zeros(len(v))
    for h in range(1, n_harm + 1):
        y += A[h - 1] * np.sin(phi0[h - 1] + h * fase)
    y *= a
    y /= np.sqrt(np.mean(y ** 2)) + 1e-12
    y *= 0.1
    if snr_db is not None:
        y = add_pink_noise(y, snr_db=snr_db, seed=seed_r)
    y = y.astype(np.float32)
    return (y, f_rot, v, p) if devolver_v else (y, f_rot)


def tempos_quadro(n_amostras, sr, n_fft=8192, hop=1024):
    """Centro de cada quadro do `rotacao.stft_mag` (sem centralização)."""
    n = 1 + (max(n_amostras, n_fft) - n_fft) // hop
    return (np.arange(n) * hop + n_fft / 2) / sr


def verdade_por_quadro(f_rot, sr, n_fft=8192, hop=1024):
    """f_rot média de cada quadro, ponderada pela janela de Hann do STFT."""
    w = np.hanning(n_fft)
    f = np.asarray(f_rot, dtype=float)
    if len(f) < n_fft:
        f = np.pad(f, (0, n_fft - len(f)), mode="edge")
    n = 1 + (len(f) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return (f[idx] * w).sum(axis=1) / w.sum()
