"""p04_classicas — features clássicas + Voting Classifier suave.

Referência: docs/literatura/vanaraj2026.md (Vanaraj et al., Proc. IMechE Part A, 2026).
Leitura: SÓ O RESUMO → TIPO "inspirada" (não é reprodução).

Seguido do resumo: features ZCR, RMS, centróide espectral e MFCCs; classificador final é um
Voting Classifier, apontado como o melhor dentre "vários modelos supervisionados".

Suposto (nada disso está no resumo):
- features: ver pipelines/_classicas.py (32 ms / hop 50 %, 13 MFCC com 26 mels, média+desvio
  por janela → 32 features, 16 kHz);
- padronização (StandardScaler) antes do voto;
- composição do voto: SVM-RBF (C=1, gamma="scale", probability=True), Random Forest
  (100 árvores), kNN (k=5, uniforme) e regressão logística (C=1); voto SUAVE com pesos iguais;
  hiperparâmetros = padrões do scikit-learn 1.9, sem busca. Sementes de SVM e RF = post.seed;
- protocolo de split: o do runner (o artigo não declara o seu; ver risco de vazamento na ficha).
"""
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from motorsound.config import PostConfig
from pipelines._classicas import (N_FEATURES, PreClassicas, custo_features,
                                  extrair_classicas, prever_proba)

NOME = "p04_classicas"
ORIGEM = "vanaraj2026"
TIPO = "inspirada"
PRE = PreClassicas()
POST = PostConfig()
_N_ARVORES = 100


def extrair(y, sr, pre):
    return extrair_classicas(y, sr, pre)


def treinar(X, y, post, n_classes):
    k = min(5, len(X))
    voto = VotingClassifier(
        estimators=[
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                        random_state=post.seed)),
            ("rf", RandomForestClassifier(n_estimators=_N_ARVORES, random_state=post.seed,
                                          n_jobs=1)),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
            ("lr", LogisticRegression(max_iter=2000, C=1.0)),
        ],
        voting="soft",
    )
    m = make_pipeline(StandardScaler(), voto).fit(X, y)
    return m, n_classes


def prever(modelo, X, post):
    m, C = modelo
    return prever_proba(m, C, X)


def custo_mcu():
    c = custo_features(PRE)
    # Classificador (estimativa, depende do treino): SVM com ~N_sv vetores de suporte × 32,
    # RF de 100 árvores (~2·folhas nós cada), kNN guarda TODO o treino, LR 33×3.
    n_treino = 1000          # ordem de grandeza das janelas de treino (A2212 1 s ≈ 300–400; IDMT mais)
    params = (n_treino * N_FEATURES            # kNN (memória do treino)
              + 0.5 * n_treino * N_FEATURES   # SVM (~metade como vetores de suporte, suposição)
              + _N_ARVORES * 2 * 50 * 2       # RF: ~50 folhas/árvore, (limiar, índice) por nó
              + (N_FEATURES + 1) * 3)          # LR
    return {"params": int(params), "ram_kb": c["ram_kb"] + 2,
            "flash_kb": c["flash_kb"] + params * 4 / 1024,
            "mflops_s": c["mflops_s"] + 1e-3 * params / 1e3,
            "nota": "estimativa analítica; classificador dominado por kNN/SVM, que guardam o "
                    "treino (n_treino≈1000 suposto); a variante embarcável é p04c_classicas_lr"}
