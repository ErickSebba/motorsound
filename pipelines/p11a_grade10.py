"""p11a_grade10: a p11 com a grade original de 10 cents (antes da correção de 14/10/2026).

Mantida para registro: com 10 cents o rastreador trava em 4/3·f no `good` do IDMT
(ver motorsound/rotacao.py::rastrear). Comparar p11a × p11 isola o efeito da grade.
"""
from pipelines import _decifrae as _d

NOME = "p11a_grade10"
ORIGEM = "decifrae"
TIPO = "decifrae"
PRE = _d.PreDecifrae(passo_cents=10.0, max_salto=10)
POST = _d.PostDecifrae()
extrair = _d.extrair
treinar = _d.treinar
prever = _d.prever


def custo_mcu():
    return _d.custo_mcu(PRE)
