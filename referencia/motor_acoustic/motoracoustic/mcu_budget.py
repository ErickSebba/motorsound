"""Orçamento de memória e operações para o ESP32 (Xtensa LX6, 240 MHz, FPU simples).

Estimativa analítica para a seção de viabilidade do artigo. Deve ser
confirmada com medição real (esp_timer_get_time) assim que o C existir.
Contagens em operações de ponto flutuante aproximadas (mult+soma = 2).
"""
from __future__ import annotations

import math

from .config import PipelineConfig


def budget(cfg: PipelineConfig, n_features: int = 18, n_classes: int = 3,
           cpu_hz: float = 240e6, flops_per_cycle: float = 0.5) -> dict:
    """flops_per_cycle=0.5 é conservador para float32 sem SIMD no LX6."""
    N = cfg.n_fft
    frames_per_s = cfg.sr / cfg.stft_hop
    K = int((cfg.f0_max - cfg.f0_min) / cfg.f0_step) + 1
    J = 2 * max(1, round(cfg.max_jump_hz / cfg.f0_step)) + 1
    H = cfg.n_harmonics

    fft = 2.5 * N * math.log2(N)            # FFT real ~ (5/2) N log2 N
    window_mag = 4 * N                       # janela + |X| + sqrt
    salience = K * H * 2 * 6                 # 2 amostras interpoladas por harmônico
    viterbi = K * J * 3                      # soma, comparação, seleção
    per_frame = fft + window_mag + salience + viterbi

    decision_per_s = 1 / cfg.hop_s
    env = cfg.sr * 2 * 1                     # RMS do envelope
    env_n = int(cfg.win_s * cfg.env_rate)
    env_fft = 2.5 * env_n * math.log2(max(env_n, 2))
    clf = 2 * n_features * n_classes + 10 * n_classes
    hmm = 3 * n_classes ** 2
    per_decision = env_fft + clf + hmm + 4 * cfg.win_s * cfg.sr  # + forma espectral

    flops_s = per_frame * frames_per_s + env + per_decision * decision_per_s
    cpu_load = flops_s / (cpu_hz * flops_per_cycle)

    ram = {
        "buffer_audio_win (float32)": 4 * int(cfg.win_s * cfg.sr),
        "fft_in+out": 4 * N * 2,
        "saliencia (1 quadro)": 4 * K,
        "viterbi score": 4 * K * 2,
        "viterbi backpointers (1 janela)": 2 * K * int(cfg.win_s * frames_per_s + 1),
        "envelope": 4 * env_n,
    }
    flash = {
        "janela Hann": 4 * N,
        "classificador (log. reg.)": 4 * (n_features * n_classes + n_classes + 2 * n_features),
    }
    return {
        "K_candidatos": K, "saltos_viterbi": J,
        "MFLOP_por_segundo": flops_s / 1e6,
        "carga_cpu_estimada": cpu_load,
        "ram_bytes": ram, "ram_total_kB": sum(ram.values()) / 1024,
        "flash_bytes": flash, "flash_total_kB": sum(flash.values()) / 1024,
    }


if __name__ == "__main__":
    from .config import IDMT_CFG, BLDC_CFG
    import json
    for name, c in (("IDMT", IDMT_CFG), ("BLDC", BLDC_CFG)):
        print(name, json.dumps(budget(c), indent=2, ensure_ascii=False))
