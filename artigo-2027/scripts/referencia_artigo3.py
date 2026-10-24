"""Linhas de referência IDMT e A2212, lidas (sem alterar) dos CSVs do trabalho do ICIT.

Fonte: `../out/experiments/gz/20261014_185859_validacao.csv.gz` (predições por janela de
b00_nivel e b01_forma_espectral, validação; o teste do IDMT nunca foi lido). As bases não são
reanalisadas aqui. O protocolo é o do Artigo3, não o desta auditoria: IDMT em janelas de 3 s
com normalização RMS e validação por blocos temporais; A2212 em janelas de 1 s, splits por
arquivo, aleatório e motor fora (ver `../docs/DATASETS.md` §3). Por isso as linhas entram no
comparativo como referência, não como medida equivalente.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from lib.comum import RAIZ, salvar

FONTE = RAIZ.parent / "out" / "experiments" / "gz" / "20261014_185859_validacao.csv.gz"
# split do Artigo3 → split desta auditoria
MAPA = {"aleatorio": "aleatorio", "arquivo": "gravacao", "blocos": "gravacao", "motor": "unidade"}


def main():
    t0 = time.time()
    d = pd.read_csv(FONTE, usecols=["pipeline", "base", "split", "classe", "pred", "normalizar_rms"])
    d = d[d["pipeline"].isin(["b00_nivel", "b01_forma_espectral"])]
    linhas = []
    for (pl, base, split, norm), g in d.groupby(["pipeline", "base", "split", "normalizar_rms"]):
        linhas.append(dict(base=base, pipeline=pl, split_artigo3=split, split=MAPA.get(split, split),
                           normalizar_rms=norm, n=len(g), acc=float((g.classe == g.pred).mean()),
                           bacc=float(balanced_accuracy_score(g.classe, g.pred)),
                           acaso=1.0 / g.classe.nunique()))
    df = pd.DataFrame(linhas)
    salvar(df, "ref_artigo3", "idmt_a2212", {"fonte": str(FONTE.relative_to(RAIZ.parent))}, t0)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
