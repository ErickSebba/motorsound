"""Fluxo contínuo NO PC (categoria 2): a mesma sequência good -> broken -> heavyload que vai
para a flash do ESP32 (pil_esp32.py seq-imagem), já desquantizada, processada janela a janela
pela p11 Python de referência e pela p11 em C (build float do PC), com o filtro online do HMM.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\seq_pc.py --imagem <seq_<base>_<ts>.bin> --cp 2

Python: probabilidades por `_d.extrair` + `_d.prever`, decisões por `hmm.filtro_online`
(referência). C: `p11_host caso` por janela (probs) e `p11_host seq` (filtro online em C).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import bancada_comum as bc
from bench_pc import modelo_python
from pil_esp32 import deteccao

bc.garantir_raiz_no_path()
from motorsound import hmm  # noqa: E402
from motorsound.config import AVAL_A2212, AVAL_IDMT  # noqa: E402
from motorsound.runner import _preparar  # noqa: E402
from pipelines import p11_decifrae_harmsum as p11  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagem", required=True)
    ap.add_argument("--cp", type=float, default=2.0)
    ap.add_argument("--n-logprob", type=float, default=-50.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--build", default="float")
    a = ap.parse_args(argv)
    img = Path(a.imagem)
    info = json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))
    x = np.fromfile(img.with_name(img.stem + "_audio.f32"), dtype="<f4").reshape(info["n_jan"], -1)
    verdade = info["rotulos"]
    exe = bc.EMB / f"build-{a.build}" / "p11_host.exe"
    base = info.get("base", "idmt")
    aval = AVAL_IDMT if base == "idmt" else AVAL_A2212
    passo = str(info.get("passo_s", 3.0))
    mod = modelo_python(base)
    C = mod[1]
    P_py, P_c = [], []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for j in range(info["n_jan"]):
            feat = p11.extrair(_preparar(x[j], aval), p11.PRE.sr, p11.PRE)
            P_py.append(p11.prever(mod, feat[None, :], p11.POST)[0])
            caso, saida = td / f"j{j}", td / f"o{j}"
            caso.mkdir()
            x[j].astype("<f4").tofile(caso / "audio.f32")
            subprocess.run([str(exe), "caso", str(caso), str(bc.modelo_bin(base)), str(saida)],
                           check=True, capture_output=True)
            ext = "f64" if a.build == "double" else "f32"
            P_c.append(np.fromfile(saida / f"probs.{ext}", dtype="<f8" if ext == "f64" else "<f4"))
        P_py, P_c = np.asarray(P_py), np.asarray(P_c, dtype=float)
        nomes = list(range(C))
        on_py = [x if x != hmm.ESTADO_N else C for x in
                 hmm.filtro_online(P_py, nomes, a.cp, a.temp, a.n_logprob)]
        # filtro online em C sobre as probabilidades do C
        P_c.astype("<f8").tofile(td / "probs.f64")
        subprocess.run([str(exe), "seq", str(td / "probs.f64"), str(len(P_c)), str(C), str(a.cp),
                        str(a.temp), str(a.n_logprob), "0", passo, str(td / "seqout")],
                       check=True, capture_output=True)
        on_c = np.fromfile(td / "seqout" / "hmm_online.i32", dtype="<i4").tolist()
    df = pd.DataFrame({"jan": range(len(verdade)), "verdade": verdade,
                       "argmax_python": P_py.argmax(1), "online_python": on_py,
                       "argmax_c": P_c.argmax(1), "online_c": on_c,
                       **{f"p{k}_python": P_py[:, k] for k in range(C)},
                       **{f"p{k}_c": P_c[:, k] for k in range(C)}})
    ts = bc.carimbo()
    out = bc.EMB / "results" / "csv" / f"seq_pc_{ts}.csv"
    df.to_csv(out, index=False)
    meta = {"categoria": "medicao_no_PC", "imagem": str(img), "cp": a.cp, "n_logprob": a.n_logprob,
            "temp": a.temp, "passo_s": float(passo), "base": base, "projeto": bc.commit(),
            "deteccao_online_python": deteccao(verdade, on_py),
            "deteccao_online_c": deteccao(verdade, on_c),
            "decisoes_iguais_python_c": bool(on_py == on_c),
            "max_abs_dif_probs": float(np.abs(P_py - P_c).max())}
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("deteccao_online_python", "deteccao_online_c",
                                           "decisoes_iguais_python_c", "max_abs_dif_probs")}))


if __name__ == "__main__":
    main()
