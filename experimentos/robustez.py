"""Etapa 5.1: robustez. Treino sempre limpo; validação com cada perturbação de
`ruidos.CONDICOES_VAL` (e uma curva de SNR branco/rosa). Só validação.

Guardado separado dos resultados principais (pode virar artigo próprio).
Grava out/experiments/<ts>_robustez_<base>.csv + _meta.json.

    python experimentos/robustez.py --pipelines p11_decifrae_harmsum b01_forma_espectral --base idmt
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

from motorsound import runner, splits  # noqa: E402
from motorsound.config import AVAL_A2212, AVAL_IDMT  # noqa: E402
from ruidos import CONDICOES_VAL  # noqa: E402

CURVA_SNR = [{"tipo": t, "snr_db": float(s)} for t in ("branco", "rosa")
             for s in (30, 20, 15, 10, 5, 0, -5)]


def condicoes():
    vistos, out = set(), []
    for c in [{"tipo": "limpo"}] + CURVA_SNR + CONDICOES_VAL:
        k = json.dumps(c, sort_keys=True)
        if k not in vistos:
            vistos.add(k)
            out.append(c)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", nargs="+", required=True)
    ap.add_argument("--base", choices=["idmt", "a2212"], default="idmt")
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args(argv)
    split = "blocos" if a.base == "idmt" else "arquivo"
    aval = AVAL_IDMT if a.base == "idmt" else AVAL_A2212
    segs = runner.segmentos_de(a.base, aval)
    t0 = time.perf_counter()
    dfs, falhas = [], []
    for p in a.pipelines:
        # um modelo por fold, treinado no limpo, previsto em todas as condições
        try:
            X = runner.extrair(p, segs, aval, None, a.n_jobs)
            Xc = {}
            for c in condicoes():
                pert = None if c["tipo"] == "limpo" else c
                Xc[json.dumps(c, sort_keys=True)] = (
                    X if pert is None else runner.extrair(p, segs, aval, pert, a.n_jobs))
            df = runner.avaliar_condicoes(p, a.base, split, aval, X, Xc)
            dfs.append(df)
            print(df.groupby("condicao_sintetica").acerto.mean().round(3).to_string(),
                  flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            falhas.append(f"{p}: {type(exc).__name__}: {exc}")
    df = pd.concat(dfs, ignore_index=True)
    caminho = runner.escrever(df, f"robustez_{a.base}", {
        "argv": sys.argv, "pipelines": a.pipelines, "base": a.base, "split": split,
        "condicoes": condicoes(), "seed_splits": splits.SEED, "falhas": falhas,
        "segundos": round(time.perf_counter() - t0, 1),
        "nota": "treino limpo, validação perturbada; semente da perturbação = crc32(rec#idx)"})
    print(f"escrito {caminho.relative_to(RAIZ)}; falhas: {falhas}")


if __name__ == "__main__":
    main()
