# Origem: Artigo2(Acordes)/Decifra-Violão/config.py (commit 93609f7)
# Mudanças: campos de chroma trocados pelos do motor; cada pipeline pode estender
#   PreConfig/PostConfig com campos próprios (dataclass filha). A separação é a mesma:
#   Pre = caro, entra na chave do cache; Post = barato, varrível sobre o cache.
"""Configurações comuns. Folha da árvore de dependências."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreConfig:
    """Extração (cara). Todo campo entra no hash do cache."""
    sr: int = 16000


@dataclass(frozen=True)
class PostConfig:
    """Decisão (barata): HMM de estados do violão e duração mínima."""
    temp: float = 1.0              # temperatura sobre as probabilidades da pipeline
    change_penalty: float = 2.0    # custo (log) de trocar de estado
    n_logprob: float = -2.5        # log-prob base do estado N
    rel_gate: float = 0.2          # N forçado se RMS < 20 % da mediana
    abs_floor: float = 5e-4
    min_dur: float = 0.0           # s; absorve estados mais curtos que isso
    seed: int = 20261014


@dataclass(frozen=True)
class AvaliacaoConfig:
    """Como o runner recorta e prepara o áudio. Igual para todas as pipelines."""
    seg_s: float = 3.0             # janela de avaliação (IDMT: 3 s, como os *_cut)
    normalizar_rms: bool = True    # tira o atalho de nível (docs/DATASETS.md §1.5)
    rms_alvo: float = 0.1


AVAL_IDMT = AvaliacaoConfig(seg_s=3.0)
AVAL_A2212 = AvaliacaoConfig(seg_s=1.0)
