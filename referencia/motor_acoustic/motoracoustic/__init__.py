"""Diagnóstico acústico de motores: adaptação do pipeline do Decifrae.

Camada 1 (contínua): ridge de rotação por soma harmônica + Viterbi.
Camada 2 (discreta): estado de operação por classificador raso + HMM.
"""
from .config import PipelineConfig, IDMT_CFG, BLDC_CFG  # noqa: F401
