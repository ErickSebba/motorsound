"""P03b — ETNeXt, variante Fine kNN (Celik et al., Sci. Rep. 2026). Ficha: docs/literatura/etnext2026.md.

Mesmo extrator, mesma seleção (dentro do fold de treino) e mesmas suposições de `p03_etnext.py`
(ler o cabeçalho de lá). Só muda o classificador: "Fine kNN" do Classification Learner → k = 1,
distância euclidiana, padronização → StandardScaler + KNeighborsClassifier(n_neighbors=1).
A escolha do candidato (NCA / Chi2 / voto) no CV interno também usa o kNN.
Probabilidade = one-hot do vizinho mais próximo. TIPO = "inspirada".
Cache próprio (cache/p03b_etnext_knn/), com a mesma PRE.
"""
from sklearn.neighbors import KNeighborsClassifier  # noqa: F401  (documenta o classificador)

from pipelines import p03_etnext as _e

NOME = "p03b_etnext_knn"
ORIGEM = "etnext2026"
TIPO = "inspirada"
PRE = _e.PRE
POST = _e.POST

extrair = _e.extrair


def treinar(X, rotulos, post, n_classes):
    return _e.treinar_com(_e._classificador_knn, X, rotulos, post, n_classes)


def prever(modelo, X, post):
    return _e.prever_com(modelo, X)


def custo_mcu():
    c = dict(_e.custo_mcu())
    # kNN guarda o treino inteiro: n_treino × 35 floats. Com ~2 000 janelas ≈ 270 KB (o artigo cita
    # ~280 KB). Predição: n_treino × 35 × 3 flops por janela.
    c.update({"params": 2000 * 35, "flash_kb": 2000 * 35 * 4 / 1024,
              "nota": "estimativa analítica; n_treino ~2000 suposto (depende da base)"})
    return c
