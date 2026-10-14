"""p10_decifrae_pyin_viterbi: pYIN + soma harmônica na emissão do baixo V3, Viterbi de rotação, cosseno do violão.

Ver pipelines/_decifrae.py para as features e o desenho PRÉ/PÓS.
"""
from pipelines import _decifrae as _d

NOME = "p10_decifrae_pyin_viterbi"
ORIGEM = "decifrae"
TIPO = "decifrae"
PRE = _d.PreDecifrae(usar_pyin=True, usar_viterbi=True)
POST = _d.PostDecifrae()
extrair = _d.extrair
treinar = _d.treinar
prever = _d.prever


def custo_mcu():
    return _d.custo_mcu(PRE)
