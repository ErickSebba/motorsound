"""Splits de validação.

Regra geral (CLAUDE.md, "Dados", regra 1): segmentos do mesmo grupo nunca ficam dos dois
lados de um fold. O que é "grupo" depende da base:

* IDMT: o `train/` tem uma única gravação por classe (docs/DATASETS.md §1.5), então o
  agrupamento por gravação é impossível. Decisão do Erick em 14/10/2026: cada gravação de
  treino é cortada em `n_blocos` blocos contíguos, alinhados em posição relativa entre as
  classes; o grupo é (gravação, bloco); a validação cruzada deixa um bloco de fora. O
  primeiro segmento de cada bloco (exceto o bloco 0) é descartado como guarda, para que
  nenhum segmento de treino seja vizinho imediato de um de validação. A validação é otimista
  (mesma sessão, só `pure`) e isso é declarado em toda tabela.
* A2212: grupo = arquivo. `a2212_por_arquivo` usa StratifiedGroupKFold;
  `a2212_por_motor` deixa um motor de fora (expõe a confusão rolamento = M4).
* `aleatorio_por_segmento` ignora os grupos de propósito. Existe só para a auditoria de
  vazamento e marca o fold com `vazamento_intencional=True`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .io import Segmento

SEED = 20261014


@dataclass(frozen=True)
class Fold:
    nome: str
    treino: np.ndarray        # índices na lista de segmentos
    val: np.ndarray
    vazamento_intencional: bool = False


class VazamentoDeGrupo(AssertionError):
    pass


def grupos_idmt_blocos(segs: list[Segmento], n_blocos: int = 5, guarda: int = 1
                       ) -> np.ndarray:
    """Bloco de cada segmento (0..n_blocos-1) ou -1 se for guarda."""
    n_por_rec: dict[str, int] = {}
    for s in segs:
        n_por_rec[s.rec_id] = max(n_por_rec.get(s.rec_id, 0), s.idx + 1)
    bloco = np.empty(len(segs), dtype=int)
    for i, s in enumerate(segs):
        n = n_por_rec[s.rec_id]
        b = min(s.idx * n_blocos // n, n_blocos - 1)
        inicio = -(-b * n // n_blocos)              # primeiro idx do bloco b (teto)
        bloco[i] = -1 if (b > 0 and s.idx < inicio + guarda) else b
    return bloco


def grupo_de(segs: list[Segmento], bloco: np.ndarray | None = None) -> np.ndarray:
    if bloco is None:
        return np.array([s.rec_id for s in segs])
    return np.array([f"{s.rec_id}#b{b}" for s, b in zip(segs, bloco)])


def idmt_blocos(segs: list[Segmento], n_blocos: int = 5, guarda: int = 1) -> list[Fold]:
    if any(s.base != "idmt" or s.split != "train" for s in segs):
        raise ValueError("idmt_blocos só aceita segmentos do train/ do IDMT")
    bloco = grupos_idmt_blocos(segs, n_blocos, guarda)
    folds = []
    for b in range(n_blocos):
        val = np.flatnonzero(bloco == b)
        tr = np.flatnonzero((bloco != b) & (bloco >= 0))
        folds.append(Fold(f"idmt_bloco{b}", tr, val))
    verificar(segs, folds, grupo_de(segs, bloco))
    return folds


def a2212_por_arquivo(segs: list[Segmento], n_folds: int = 5, seed: int = SEED
                      ) -> list[Fold]:
    y = np.array([s.classe for s in segs])
    g = grupo_de(segs)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [Fold(f"a2212_arq{k}", tr, va)
             for k, (tr, va) in enumerate(cv.split(np.zeros(len(segs)), y, g))]
    verificar(segs, folds, g)
    return folds


def a2212_por_motor(segs: list[Segmento]) -> list[Fold]:
    m = np.array([s.motor for s in segs])
    folds = [Fold(f"a2212_motor_{mo}", np.flatnonzero(m != mo), np.flatnonzero(m == mo))
             for mo in sorted(set(m))]
    verificar(segs, folds, grupo_de(segs))
    return folds


def aleatorio_por_segmento(segs: list[Segmento], n_folds: int = 5, seed: int = SEED
                           ) -> list[Fold]:
    """Split errado de propósito (auditoria de vazamento)."""
    y = np.array([s.classe for s in segs])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [Fold(f"aleatorio{k}", tr, va, vazamento_intencional=True)
            for k, (tr, va) in enumerate(cv.split(np.zeros(len(segs)), y))]


def verificar(segs: list[Segmento], folds: list[Fold], grupos: np.ndarray) -> None:
    """Levanta VazamentoDeGrupo se algum grupo aparece no treino e na validação de um fold
    que não seja de vazamento intencional."""
    for f in folds:
        if f.vazamento_intencional:
            continue
        comum = set(grupos[f.treino]) & set(grupos[f.val])
        if comum:
            raise VazamentoDeGrupo(f"{f.nome}: grupos dos dois lados: {sorted(comum)[:5]}")
        if len(set(f.treino) & set(f.val)):
            raise VazamentoDeGrupo(f"{f.nome}: segmento nos dois lados")
