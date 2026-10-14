"""p04c_classicas_lr — mesmas 32 features da p04_classicas, só regressão logística.

Referência: docs/literatura/vanaraj2026.md (só o resumo) → TIPO "inspirada".
Variante leve para o custo no MCU, sugerida na própria ficha (§8). O artigo não reporta essa
variante; ela é nossa. Features: pipelines/_classicas.py (mesmas suposições da p04).
Classificador: StandardScaler + LogisticRegression(C=1, lbfgs, multinomial), sem busca.
No MCU, a padronização é dobrada nos pesos: 32 médias + 32 desvios + 33×C pesos.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from motorsound.config import PostConfig
from pipelines._classicas import (N_FEATURES, PreClassicas, custo_features,
                                  extrair_classicas, prever_proba)

NOME = "p04c_classicas_lr"
ORIGEM = "vanaraj2026"
TIPO = "inspirada"
PRE = PreClassicas()
POST = PostConfig()


def extrair(y, sr, pre):
    return extrair_classicas(y, sr, pre)


def treinar(X, y, post, n_classes):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X, y)
    return m, n_classes


def prever(modelo, X, post):
    m, C = modelo
    return prever_proba(m, C, X)


def custo_mcu():
    c = custo_features(PRE)
    params = (N_FEATURES + 1) * 3 + 2 * N_FEATURES      # 3 classes (A2212/IDMT)
    return {"params": params, "ram_kb": c["ram_kb"], "flash_kb": c["flash_kb"] + params * 4 / 1024,
            "mflops_s": c["mflops_s"] + 2 * params / 1e6,
            "nota": "estimativa analítica; custo dominado pela FFT de 512 pts (62,5 quadros/s)"}
