"""Gera as saídas da variante p11_c_lowmem no PC para os casos de 1 s do ouro e chama o
comparador do Agente D com as tolerâncias congeladas do build correspondente.

    .venv\\Scripts\\python.exe embarcados\\tests\\equivalence\\gerar_lowmem.py --build float

Os casos de 3 s (IDMT) ficam de fora: a variante só cabe no ESP32 sem PSRAM para 1 s
(o Hilbert de 48 000 pontos sozinho pede 384 000 B). O pool imita uma RAM fragmentada.
As etapas stft_S e sal não existem na variante (nunca são guardadas inteiras) e aparecem
como AUSENTE no comparador; o resumo filtra só os casos de 1 s.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EMB = Path(__file__).resolve().parents[2]
POOL = "110592:0,98304:0,65536:0,40960:0,16384:0"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="float", choices=["float", "double"])
    ap.add_argument("--pool", default=None,
                    help="padrão: RAM fragmentada no float; um bloco grande no double (só PC)")
    a = ap.parse_args(argv)
    if a.pool is None:
        a.pool = POOL if a.build == "float" else "4000000:0"
    exe = EMB / f"build-{a.build}" / "p11_host.exe"
    man = json.loads((EMB / "gold" / "manifest.json").read_text(encoding="utf-8"))
    nome = f"{a.build}-lowmem"
    cout = EMB / "results" / "c_out" / nome
    picos = []
    ids = []
    for c in man["casos"]:
        if c["n_amostras"] != 16000:
            continue
        r = subprocess.run([str(exe), "lowmem", str(EMB / "gold" / "casos" / c["id"]),
                            str(EMB / "gold" / "modelos" / f"{c['modelo']}.bin"),
                            str(cout / "casos" / c["id"]), a.pool],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{c['id']}: {r.stderr}")
        picos.append(json.loads(r.stdout.strip().splitlines()[-1])["pico_total"])
        ids.append(c["id"])
    print(f"{len(ids)} casos de 1 s; pico do pool {max(picos)} B")
    sys.path.insert(0, str(Path(__file__).parent))
    import comparar
    tol_build = a.build
    df, _ = comparar.rodar(nome, EMB / "gold", cout, EMB / "configs" / "embedded" / "tolerancias.json",
                           tol_build, gerar=False, saida_csv=EMB / "results" / "csv", verbose=True)
    # resumo só do que a variante cobre: casos de 1 s, sem stft_S/sal (nunca guardados)
    d = df[(df.tipo_item == "caso") & df.id.isin(ids) & ~df.etapa.isin(["stft_S", "sal"])]
    t = d.groupby("etapa", sort=False).status.value_counts().unstack(fill_value=0)
    t["razao_tol_max"] = d.groupby("etapa", sort=False).razao_tol.max()
    print(f"== {nome}: {d.id.nunique()} casos de 1 s")
    print(t.to_string())
    x = d[~d.status.isin(["PASS"])]
    if len(x):
        print(x[["id", "etapa", "status", "razao_tol", "detalhe"]].to_string())
    return d


if __name__ == "__main__":
    main()
