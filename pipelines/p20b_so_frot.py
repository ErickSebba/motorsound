"""p20b_so_frot: controle (iv) da p20_hfs: só a rotação estimada (f_o e sua dispersão).

Referência: docs/literatura/wheat2025.md §7(c), controle (iv). TIPO = "inspirada" (o controle
não está no artigo; é exigido pela ficha para separar o que vem da rotação do que vem da forma
harmônica).

Features (3): f_o = mediana de f_rot na janela (rastreador do projeto, mesma configuração da
p11 e da p20), IQR de f_rot, fração de quadros SIL. Classificador: StandardScaler → kNN com k*
escolhido como na p20 (validação interna em blocos contíguos estratificados, só no treino).
Sem ajuste de condição e sem PCA (não fazem sentido em 3 dimensões dominadas por f_o).

Leitura esperada: se este controle já separa as classes do IDMT, o IDMT mede rotação (classes
produzidas por tensão e carga), e o HFS, que remove a rotação por construção, deve piorar.
Achado do rastreador (vale aqui também): no IDMT a família forte é múltipla de 102,8 Hz e em
`good` o rastreador às vezes escolhe 34,3 Hz em vez de 25,7 Hz (ambiguidade 3:4), o que torna
f_o bimodal dentro de uma classe; o IQR e a fração SIL ajudam a capturar essa instabilidade.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from motorsound.config import PostConfig
from pipelines._decifrae import PreDecifrae
from pipelines._decifrae import custo_mcu as _custo_rastreador
from pipelines.p20_hfs import fo_janela, k_estrela

NOME = "p20b_so_frot"
ORIGEM = "wheat2025"
TIPO = "inspirada"


@dataclass(frozen=True)
class PostFo(PostConfig):
    k_folds: int = 5
    k_max: int = 25


PRE = PreDecifrae()
POST = PostFo()


def extrair(y, sr, pre):
    return np.array(fo_janela(y, sr, pre), dtype=np.float64)


def treinar(X, y, post: PostFo = POST, n_classes=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    k = k_estrela(Z, y, post.k_folds, post.k_max)
    return {"sc": sc, "knn": KNeighborsClassifier(n_neighbors=min(k, len(y))).fit(Z, y),
            "k": k, "C": n_classes}


def prever(modelo, X, post: PostFo = POST):
    knn = modelo["knn"]
    P = np.zeros((len(X), modelo["C"]))
    P[:, knn.classes_] = knn.predict_proba(modelo["sc"].transform(np.asarray(X, dtype=float)))
    return P


def custo_mcu():
    r = _custo_rastreador(PRE)
    return {"params": 6, "ram_kb": r["ram_kb"], "flash_kb": 0.1, "mflops_s": float(r["mflops_s"]),
            "nota": "custo = rastreador (como p11); kNN guarda os pontos de treino em 3-D "
                    "(12 B por janela de treino, fora de 'params')"}
