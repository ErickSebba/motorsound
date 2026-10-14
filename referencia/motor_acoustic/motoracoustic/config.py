"""Parâmetros centrais do pipeline.

Tudo que o firmware vai precisar replicar mora aqui, num lugar só.
Quando um valor mudar, muda aqui e o mcu_budget recalcula o custo.
"""
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # --- aquisição -------------------------------------------------------
    sr: int = 16_000            # taxa alvo (microfone MEMS / I2S no ESP32)
    highpass_hz: float = 15.0   # remove DC e deriva de baixa frequência

    # --- janelamento para decisão ---------------------------------------
    win_s: float = 1.0          # janela de decisão (1 feature vector)
    hop_s: float = 0.5          # 50 % de sobreposição

    # --- STFT do rastreador de rotação ----------------------------------
    n_fft: int = 4096           # 256 ms a 16 kHz -> 3,9 Hz por bin
    stft_hop: int = 1024        # 64 ms entre quadros do ridge

    # --- grade de candidatos de f0 (Hz) ---------------------------------
    # Ajuste por base depois do script explore_spectrum.py.
    f0_min: float = 20.0
    f0_max: float = 400.0
    f0_step: float = 0.5
    n_harmonics: int = 8
    harmonic_decay: float = 0.85     # peso do h-ésimo harmônico = decay**(h-1)
    interharmonic_penalty: float = 0.5  # penaliza energia em (h-1/2)·f -> mata sub-harmônico

    # --- Viterbi do ridge -----------------------------------------------
    max_jump_hz: float = 4.0    # salto máximo de f0 entre quadros de 64 ms
    jump_cost: float = 0.02     # custo por Hz de salto (em unidades de saliência normalizada)

    # --- espectro de modulação do envelope ------------------------------
    env_rate: int = 1000        # taxa do envelope (Hz)
    mod_fmin: float = 2.0
    mod_fmax: float = 200.0

    # --- HMM de estados --------------------------------------------------
    p_stay: float = 0.98        # probabilidade de permanecer no mesmo estado por janela

    extra: dict = field(default_factory=dict)


IDMT_CFG = PipelineConfig()

# A2212 1000KV a ~11 V gira perto de 11 000 RPM (~180 Hz mecânicos). Os
# arquivos têm velocidades variadas, então a grade é mais larga. Confirmar
# no explore_spectrum.py antes de congelar.
BLDC_CFG = PipelineConfig(f0_min=40.0, f0_max=1200.0, f0_step=1.0,
                          max_jump_hz=10.0)
