"""p12_decifrae_sem_viterbi: p11 com argmax por quadro no lugar do Viterbi de rotação (isola o Viterbi).

Ver pipelines/_decifrae.py para as features e o desenho PRÉ/PÓS.
"""
from pipelines import _decifrae as _d

NOME = "p12_decifrae_sem_viterbi"
ORIGEM = "decifrae"
TIPO = "decifrae"
PRE = _d.PreDecifrae(usar_pyin=False, usar_viterbi=False)
POST = _d.PostDecifrae()
extrair = _d.extrair
treinar = _d.treinar
prever = _d.prever


def custo_mcu():
    return _d.custo_mcu(PRE)
