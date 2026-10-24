"""Medição NO PC (categoria 2 do CLAUDE.md): p11 Python de referência e p11 em C (build float
do PC), nas mesmas janelas que vão para o ESP32 (bancada_comum.casos_latencia).

    .venv\\Scripts\\python.exe embarcados\\src\\python\\bench_pc.py --n 200 --aquecimento 5

Python: tempo de parede de `_preparar` + `_d.extrair` + `_d.prever` por janela (uma execução
por medição, perf_counter). C: `p11_host bench <caso> <modelo> 1` por medição (o tempo é medido
dentro do executável, em torno das chamadas do núcleo, sem o início do processo). Saída:
results/csv/bench_pc_<ts>.csv (uma linha por medição) + _meta.json, e
results/metrics/bench_pc_<ts>.json (mediana, p95, máx, fator de tempo real).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd

import bancada_comum as bc

bc.garantir_raiz_no_path()
from motorsound.classificador import CossenoPrototipos  # noqa: E402
from motorsound.runner import _preparar  # noqa: E402
from motorsound.config import AVAL_A2212, AVAL_IDMT  # noqa: E402
from pipelines import p11_decifrae_harmsum as p11  # noqa: E402
from pipelines import _decifrae as _d  # noqa: E402

ETAPAS_C = ["preparar", "rastrear_quadros", "viterbi", "saliencia_caminho", "envelope_mod",
            "features", "cosseno", "total_sem_init"]


def modelo_python(base):
    """CossenoPrototipos da referência com os parâmetros exportados (mesmo objeto de `fit`)."""
    j = bc.modelo_json(base)
    clf = CossenoPrototipos(temp=j["temp"])
    clf.classes = list(range(j["n_classes"]))
    clf.mu = np.asarray(j["mu"], dtype=float)
    clf.sd = np.asarray(j["sd"], dtype=float)
    clf.M = np.asarray(j["M"], dtype=float)
    return (clf, j["n_classes"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--aquecimento", type=int, default=5)
    ap.add_argument("--build", default="float")
    a = ap.parse_args(argv)
    exe = bc.EMB / f"build-{a.build}" / "p11_host.exe"
    ts = bc.carimbo()
    linhas = []
    t_ini = time.perf_counter()
    for base in ("idmt", "a2212"):
        aval = AVAL_IDMT if base == "idmt" else AVAL_A2212
        ids, n_dist = bc.casos_latencia(base, a.n, a.aquecimento)
        mod = modelo_python(base)
        for k, cid in enumerate(ids):
            y = bc.audio(cid)
            t0 = time.perf_counter()
            x = p11.extrair(_preparar(y, aval), p11.PRE.sr, p11.PRE)
            P = p11.prever(mod, x[None, :], p11.POST)
            t1 = time.perf_counter()
            linhas.append({"impl": "python_ref", "base": base, "janela": cid, "k": k,
                           "aquecimento": k < a.aquecimento, "n_janelas_distintas": n_dist,
                           "total_ms": 1e3 * (t1 - t0), "pred": int(P.argmax())})
            r = subprocess.run([str(exe), "bench", str(bc.GOLD / "casos" / cid),
                                str(bc.modelo_bin(base)), "1"], capture_output=True, text=True,
                               check=True)
            b = json.loads(r.stdout.strip().splitlines()[-1])
            lin = {"impl": f"c_{a.build}_pc", "base": base, "janela": cid, "k": k,
                   "aquecimento": k < a.aquecimento, "n_janelas_distintas": n_dist,
                   "ws_bytes": b["ws_bytes"]}
            for e in ETAPAS_C:
                lin[f"{e}_ms"] = b["etapas_ms"][e]["mediana"]
            lin["total_ms"] = lin["total_sem_init_ms"]
            linhas.append(lin)
    df = pd.DataFrame(linhas)
    out = bc.EMB / "results" / "csv" / f"bench_pc_{ts}.csv"
    if out.exists():
        raise FileExistsError(out)
    df.to_csv(out, index=False)
    maquina = {"plataforma": platform.platform(), "processador": platform.processor(),
               "n_cpu": os.cpu_count(), "python": platform.python_version(),
               "numpy": np.__version__}
    meta = {"categoria": "medicao_no_PC", "gerado_em": ts, "projeto": bc.commit(),
            "argv": vars(a), "maquina": maquina, "executavel": str(exe),
            "compilador": "MSVC 19.44 x64 /O2 /fp:precise (embarcados/CMakeLists.txt)",
            "segundos": round(time.perf_counter() - t_ini, 1),
            "nota": "janelas = casos nao-borda do ouro percorridos em ciclo; aquecimento excluido"}
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    res = {"meta": meta, "resumo": {}}
    m = df[~df.aquecimento]
    for (impl, base), g in m.groupby(["impl", "base"]):
        d = {"total_ms": bc.resumo(g.total_ms),
             "fator_tempo_real_mediana": float(np.median(g.total_ms) / 1e3 / bc.DUR_S[base]),
             "duracao_janela_s": bc.DUR_S[base]}
        for e in ETAPAS_C:
            if f"{e}_ms" in g:
                d[f"{e}_ms"] = bc.resumo(g[f"{e}_ms"])
        res["resumo"][f"{impl}/{base}"] = d
    (bc.EMB / "results" / "metrics" / f"bench_pc_{ts}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in res["resumo"].items():
        print(f"{k}: mediana {v['total_ms']['mediana']:.2f} ms, p95 {v['total_ms']['p95']:.2f}, "
              f"máx {v['total_ms']['max']:.2f}, RTF {v['fator_tempo_real_mediana']:.4f}")
    print(f"escrito {out}")


if __name__ == "__main__":
    main()
