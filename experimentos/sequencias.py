"""Etapa 4.4: atraso de detecção e trocas espúrias em sequências concatenadas.

Lê um CSV de janelas (validacao ou robustez), monta as mesmas sequências para todas as
pipelines (motorsound.sequencias.montar com semente fixa) e decodifica com argmax, HMM
offline (Viterbi) e HMM online (filtro causal), varrendo o change_penalty.

    python experimentos/sequencias.py --csv out/experiments/<ts>_robustez_idmt.csv \
        --condicao '{"snr_db": 10.0, "tipo": "rosa"}'
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from motorsound import runner, sequencias  # noqa: E402

PENALIDADES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, nargs="+")
    ap.add_argument("--base", default="idmt")
    ap.add_argument("--split", default=None)
    ap.add_argument("--condicao", default=None,
                    help="valor da coluna condicao_sintetica (CSV de robustez)")
    ap.add_argument("--n-seq", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    df = pd.concat([pd.read_csv(c) for c in a.csv], ignore_index=True)
    df = df[df.base == a.base]
    split = a.split or ("blocos" if a.base == "idmt" else "arquivo")
    df = df[df.split == split]
    if "normalizar_rms" in df:
        df = df[df.normalizar_rms.astype(str) == "True"]
    if a.condicao:
        alvo = json.dumps(json.loads(a.condicao), sort_keys=True)
        df = df[df.condicao_sintetica == alvo]
    classes = runner.classes_de(a.base)
    passo = json.loads(df.aval.iloc[0])["seg_s"]
    dmin, dmax = (5, 20) if a.base == "idmt" else (3, 10)
    linhas = []
    for p, g in df.groupby("pipeline"):
        seqs = sequencias.montar(g, classes, n_seq=a.n_seq, dmin=dmin, dmax=dmax, seed=a.seed)
        for cp in PENALIDADES:
            r = sequencias.avaliar(seqs, classes, passo, change_penalty=cp)
            r["pipeline"] = p
            linhas.append(r)
        print(p, "ok", flush=True)
    out = pd.concat(linhas, ignore_index=True)
    out["condicao_sintetica"] = a.condicao or "limpo"
    caminho = runner.escrever(out, f"sequencias_{a.base}", {
        "argv": sys.argv, "entradas": [Path(c).name for c in a.csv], "split": split,
        "condicao": a.condicao, "n_seq": a.n_seq, "seed": a.seed, "passo_s": passo,
        "permanencia_janelas": [dmin, dmax], "penalidades": PENALIDADES,
        "segundos": round(time.perf_counter() - t0, 1)})
    resumo = (out.groupby(["pipeline", "decodificacao", "change_penalty"])
                 [["acuracia", "atraso_mediano_s", "espurias_por_min", "perdidas"]].mean())
    print(resumo.round(3).to_string())
    print(f"escrito {caminho.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
