"""Emulador "C-free" de um build do porte: reimplementa as etapas do contrato em numpy com
float64 ("emul_double") ou float32 com somas sequenciais e FFT radix-2 ("emul_float"), e grava
as saídas no MESMO formato de results/c_out/<build>/ (reports/CONTRATO_INTERFACES.md §2).

Serve para (1) calibrar as tolerâncias sem olhar a saída do C e (2) testar o comparador.
Nunca escreve em results/c_out/: a saída vai para o diretório pedido (padrão: scratch).

    .venv\\Scripts\\python.exe embarcados\\tests\\equivalence\\emular_build.py <emul_double|emul_float> <dir_saida>
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import estudo_tolerancias as E   # noqa: E402  (insere a raiz no sys.path)

GOLD = E.EMB / "gold"


def ler_modelo(p: Path):
    b = p.read_bytes()
    C, nf = struct.unpack_from("<ii", b, 0)
    temp = struct.unpack_from("<d", b, 8)[0]
    v = np.frombuffer(b, "<f8", offset=16)
    mu, sd, M = v[:nf], v[nf:2 * nf], v[2 * nf:].reshape(C, nf)
    return SimpleNamespace(C=C, nf=nf, temp=temp, mu=mu.copy(), sd=sd.copy(), M=M.copy())


def gravar(pasta: Path, desc: dict, nome: str, arr, real_dt):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        ext, dt, nd = "i32", "<i4", "int32"
    else:
        ext, dt, nd = ("f32", "<f4", "float32") if real_dt == np.float32 else ("f64", "<f8", "float64")
    a = np.ascontiguousarray(arr.astype(dt))
    a.tofile(pasta / f"{nome}.{ext}")
    desc[nome] = {"arquivo": f"{nome}.{ext}", "dtype": nd, "shape": list(a.shape)}


# ---------------------------------------------------------------- sequências (HMM)
def seq_emulada(P, C, cp, temp, n_logprob, min_dur, passo, dt):
    """Mesma conta de motorsound/hmm.py no tipo dt (temp = 1, emissão já é probabilidade)."""
    P = P.astype(dt)
    T = len(P)
    S = C + 1
    le = np.full((T, S), dt(-50.0), dtype=dt)
    le[:, :C] = np.log(P + dt(1e-12))
    le[:, C] = dt(n_logprob)
    lT = np.full((S, S), dt(-cp), dtype=dt)
    np.fill_diagonal(lT, dt(0))
    # Viterbi (argmax primeiro índice)
    delta = le[0].copy()
    psi = np.zeros((T, S), dtype=np.int32)
    for t in range(1, T):
        cand = delta[:, None] + lT
        psi[t] = np.argmax(cand, axis=0)
        delta = cand[psi[t], np.arange(S)] + le[t]
    vit = np.empty(T, dtype=np.int32)
    vit[-1] = int(np.argmax(delta))
    for t in range(T - 2, -1, -1):
        vit[t] = psi[t + 1, vit[t + 1]]

    def lse(x, axis=None):
        m = np.max(x, axis=axis, keepdims=True)
        r = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True, dtype=dt))
        return np.squeeze(r, axis=axis) if axis is not None else r.item()
    alpha = le[0] - dt(lse(le[0]))
    al = [alpha]
    for t in range(1, T):
        a = lse(alpha[:, None] + lT, axis=0) + le[t]
        alpha = (a - dt(lse(a))).astype(dt)
        al.append(alpha)
    al = np.array(al, dtype=dt)
    onl = np.argmax(al, axis=1).astype(np.int32)
    amx = np.argmax(P, axis=1).astype(np.int32)
    from motorsound import hmm
    nomes = list(range(C)) + [C]
    ab = []
    for r in (amx, vit, onl):
        rr = [int(x) for x in r]
        if min_dur > 0:
            seg = hmm.absorb_short(hmm.rotulos_para_segmentos(rr, passo), min_dur)
            rr = hmm.segmentos_para_rotulos(seg, T, passo)
            rr = [C if x == hmm.ESTADO_N else int(x) for x in rr]
        ab.append(rr)
    return dict(hmm_log_em=le, hmm_viterbi=vit, hmm_online=onl, hmm_online_alpha=al,
                argmax=amx, absorb=np.array(ab, dtype=np.int32))


def main(build: str, saida: Path):
    dt, seq = {"emul_double": (np.float64, False), "emul_float": (np.float32, True)}[build]
    man = json.loads((GOLD / "manifest.json").read_text("utf-8"))
    modelos = {}
    for c in man["casos"]:
        mod = modelos.setdefault(c["modelo"], ler_modelo(GOLD / "modelos" / f"{c['modelo']}.bin"))
        audio = np.fromfile(GOLD / "casos" / c["id"] / "audio.f32", "<f4")
        em = E.etapas_emuladas(audio, mod, dt, seq)
        pasta = saida / "casos" / c["id"]
        pasta.mkdir(parents=True, exist_ok=True)
        desc = {}
        for k, v in em.items():
            if k == "vit_path":
                v = np.asarray(v, dtype=np.int32)
            gravar(pasta, desc, k, v, dt)
        (pasta / "etapas.json").write_text(json.dumps(desc, indent=1), "utf-8")
    for s in man["seqs"]:
        gp = GOLD / "seqs" / s["id"]
        par = json.loads((gp / "params.json").read_text("utf-8"))
        T, C = par["T"], par["C"]
        P = np.fromfile(gp / "probs.f64", "<f8").reshape(T, C)
        for cp in par["change_penalties"]:
            for md in par["min_durs_s"]:
                o = seq_emulada(P, C, cp, par["temp"], par["n_logprob"], md, par["passo_s"], dt)
                pasta = saida / "seqs" / s["id"] / f"cp{cp:g}_md{md:g}"
                pasta.mkdir(parents=True, exist_ok=True)
                desc = {}
                for k, v in o.items():
                    gravar(pasta, desc, k, v, dt)
                (pasta / "etapas.json").write_text(json.dumps(desc, indent=1), "utf-8")
    print(f"{build}: {len(man['casos'])} casos, {len(man['seqs'])} seqs -> {saida}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
