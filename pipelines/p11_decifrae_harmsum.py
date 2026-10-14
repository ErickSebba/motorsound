"""p11_decifrae_harmsum: só soma harmônica na emissão (sem pYIN; versão embarcável), Viterbi de rotação, cosseno do violão.

Ver pipelines/_decifrae.py para as features e o desenho PRÉ/PÓS.
"""
from pipelines import _decifrae as _d

NOME = "p11_decifrae_harmsum"
ORIGEM = "decifrae"
TIPO = "decifrae"
PRE = _d.PreDecifrae(usar_pyin=False, usar_viterbi=True)
POST = _d.PostDecifrae()
extrair = _d.extrair
treinar = _d.treinar
prever = _d.prever


def custo_mcu():
    return _d.custo_mcu(PRE)
