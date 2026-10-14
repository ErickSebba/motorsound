"""Etapa 4: matriz de validação (nunca lê o teste).

Para cada pipeline, um bloco de execuções com o mesmo split e as mesmas janelas:
  idmt  / blocos      limpo, RMS normalizado            (referência)
  idmt  / aleatorio   limpo                             (auditoria de vazamento)
  idmt  / blocos      sem normalizar o RMS              (auditoria do atalho de nível)
  a2212 / arquivo     limpo                             (referência)
  a2212 / motor       limpo                             (confusão classe × motor)
  a2212 / aleatorio   limpo                             (auditoria de vazamento)
  a2212 / arquivo     sem normalizar
Grava out/experiments/<ts>_validacao.csv + _meta.json (uma linha por janela).

    python experimentos/validacao.py --pipelines b00_nivel b01_forma_espectral ...
"""
import argparse
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from motorsound import runner, splits  # noqa: E402
from motorsound.config import AVAL_A2212, AVAL_IDMT  # noqa: E402

CENARIOS = [
    ("idmt", "blocos", True), ("idmt", "aleatorio", True), ("idmt", "blocos", False),
    ("a2212", "arquivo", True), ("a2212", "motor", True), ("a2212", "aleatorio", True),
    ("a2212", "arquivo", False),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", nargs="+", required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--tag", default="validacao")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    dfs, falhas, custos, tempos = [], [], {}, {}
    for p in a.pipelines:
        for base, sp, norm in CENARIOS:
            aval = replace(AVAL_IDMT if base == "idmt" else AVAL_A2212, normalizar_rms=norm)
            t1 = time.perf_counter()
            try:
                df = runner.avaliar(p, base, sp, aval, n_jobs=a.n_jobs, verbose=False)
                df["normalizar_rms"] = norm
                dfs.append(df)
                print(f"{p:28s} {base:5s} {sp:9s} norm={norm!s:5s} acc={df.acerto.mean():.3f} "
                      f"({time.perf_counter() - t1:.0f}s)", flush=True)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                falhas.append(f"{p}/{base}/{sp}/{norm}: {type(exc).__name__}: {exc}")
            tempos[f"{p}/{base}/{sp}/{norm}"] = round(time.perf_counter() - t1, 1)
        try:
            custos[p] = runner.carregar_pipeline(p).custo_mcu()
        except Exception as exc:
            custos[p] = f"erro: {exc}"
    df = pd.concat(dfs, ignore_index=True)
    caminho = runner.escrever(df, a.tag, {
        "argv": sys.argv, "pipelines": a.pipelines, "cenarios": CENARIOS,
        "seed_splits": splits.SEED, "falhas": falhas, "custo_mcu": custos,
        "tempos_s": tempos, "segundos": round(time.perf_counter() - t0, 1),
        "nota": "validação apenas; test/ do IDMT trancado por motorsound.io"})
    print(f"escrito {caminho.relative_to(RAIZ)}; falhas: {falhas}")


if __name__ == "__main__":
    main()
