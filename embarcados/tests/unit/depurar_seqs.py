"""Uso: python depurar_<x>.py <float|double|float-sosd>. Depuração: p11_host seq contra gold/seqs (todos os cp e min_dur)."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
EMB = Path(__file__).resolve().parents[2]
build = sys.argv[1]
exe = EMB / f"build-{build}" / "p11_host.exe"
def ler(d, k):
    e = json.load(open(d / "etapas.json"))[k]
    dt = {"float32": np.float32, "float64": np.float64, "int32": np.int32}[e["dtype"]]
    return np.fromfile(d / e["arquivo"], dtype=dt).reshape(e["shape"])
tot = dif = 0; err_le = err_al = 0.0
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    for sd in sorted((EMB / "gold" / "seqs").iterdir()):
        p = json.load(open(sd / "params.json"))
        T, C = p["T"], p["C"]
        g_v, g_o, g_a, g_al = ler(sd, "hmm_viterbi"), ler(sd, "hmm_online"), ler(sd, "absorb"), ler(sd, "hmm_online_alpha")
        for ic, cp in enumerate(p["change_penalties"]):
            for im, md in enumerate(p["min_durs_s"]):
                o = td / f"{ic}_{im}"
                subprocess.run([str(exe), "seq", str(sd / "probs.f64"), str(T), str(C), repr(cp), repr(p["temp"]),
                                repr(p["n_logprob"]), repr(md), repr(p["passo_s"]), str(o)], check=True, capture_output=True)
                le = ler(o, "hmm_log_em").astype(float)
                err_le = max(err_le, np.max(np.abs(le - ler(sd, "hmm_log_em"))))
                err_al = max(err_al, np.max(np.abs(ler(o, "hmm_online_alpha") - g_al[ic])))
                ok = (np.array_equal(ler(o, "hmm_viterbi"), g_v[ic]) and np.array_equal(ler(o, "hmm_online"), g_o[ic])
                      and np.array_equal(ler(o, "absorb"), g_a[:, ic, im, :]) and np.array_equal(ler(o, "argmax"), ler(sd, "argmax")))
                tot += 1; dif += (not ok)
                if not ok: print("difere", sd.name, cp, md)
print(f"build {build}: {tot} combinações, {dif} com diferença; max|dlog_em|={err_le:.2e} max|dalpha|={err_al:.2e}")
