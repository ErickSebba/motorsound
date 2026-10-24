"""Comparador de equivalência referência Python (embarcados/gold/) × porte C (um build).

Lê gold/manifest.json, as etapas de cada caso e de cada sequência, as saídas do build em
results/c_out/<build>/ (ou em --c-out) e as tolerâncias congeladas em
configs/embedded/tolerancias.json. Para cada (caso, etapa) calcula erro máximo, médio, RMS,
erro relativo ao máximo da referência, razão erro/tolerância, concordância (discretos) e
PASS / PASS_EMPATE / FAIL / FAIL_JUSANTE / AUSENTE; identifica a primeira etapa que diverge.

Regras de tolerância e a justificativa de cada número: reports/c_equivalence.md.

Uso (raiz Artigo3(MotorSound)):
    .venv\\Scripts\\python.exe embarcados\\tests\\equivalence\\comparar.py --build float
    .venv\\Scripts\\python.exe embarcados\\tests\\equivalence\\comparar.py --build double --gerar
    ... --c-out <dir> --tol-build float   (comparar uma árvore arbitrária no formato do contrato)

--gerar chama `p11_host caso` / `p11_host seq` do build para os casos/sequências cujas
saídas faltam. Escreve results/csv/equivalencia_<build>_<timestamp>.csv, _resumo.csv e
_meta.json (nunca sobrescreve).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
EMB = AQUI.parents[1]
RAIZ = EMB.parent
TOL_PADRAO = EMB / "configs" / "embedded" / "tolerancias.json"

ORDEM_CASO = ["y_prep", "stft_S", "grade", "sal", "e_norm", "custo", "vit_path", "vit_score",
              "f_rot", "P_mean", "env_abs", "env_dec", "welch_f", "welch_P", "env_mod",
              "features", "z", "cos", "probs", "decisao"]
ORDEM_SEQ = ["hmm_log_em", "argmax", "hmm_viterbi", "hmm_online_alpha", "hmm_online", "absorb"]
DECODS = ["argmax", "hmm_viterbi", "hmm_online"]          # eixo 0 de gold absorb
NOMES_FEAT = ["f_med", "iqr", "frac_sil", "sal_caminho", "razao_harm", "g1_4_dB", "g5_12_dB",
              "g13_24_dB", "g25_48_dB", "f_mod", "prom_mod_dB", "centroide", "planura"]


# ====================================================================== E/S
def ler_etapas(pasta: Path) -> dict:
    p = pasta / "etapas.json"
    if not p.exists():
        return {}
    desc = json.loads(p.read_text("utf-8"))
    out = {}
    for k, d in desc.items():
        arq = pasta / d["arquivo"]
        if not arq.exists():
            continue
        dt = {"float64": "<f8", "float32": "<f4", "int32": "<i4"}[d["dtype"]]
        a = np.fromfile(arq, dt)
        shape = tuple(d["shape"])
        if int(np.prod(shape)) != a.size:
            out[k] = ("forma", a, shape)
            continue
        out[k] = a.reshape(shape)
    return out


def ler_modelo(p: Path) -> dict:
    b = p.read_bytes()
    C, nf = struct.unpack_from("<ii", b, 0)
    temp = struct.unpack_from("<d", b, 8)[0]
    v = np.frombuffer(b, "<f8", offset=16)
    return {"C": C, "nf": nf, "temp": temp, "mu": v[:nf], "sd": v[nf:2 * nf],
            "M": v[2 * nf:2 * nf + C * nf].reshape(C, nf)}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def achar_host(build: str) -> Path | None:
    for base in (EMB / f"build-{build}", EMB / "build" / build):
        if base.exists():
            for nome in ("p11_host.exe", "p11_host"):
                c = sorted(base.rglob(nome))
                if c:
                    return c[0]
    return None


def gerar_saidas(build, gold, man, cout, host):
    """Chama p11_host para os casos/sequências sem saída (interface de p11_host.c::main)."""
    for c in man["casos"]:
        dst = cout / "casos" / c["id"]
        if (dst / "etapas.json").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(host), "caso", str(gold / "casos" / c["id"]),
                        str(gold / "modelos" / f"{c['modelo']}.bin"), str(dst)], check=True)
    for s in man["seqs"]:
        gp = gold / "seqs" / s["id"]
        par = json.loads((gp / "params.json").read_text("utf-8"))
        for cp in par["change_penalties"]:
            for md in par["min_durs_s"]:
                dst = cout / "seqs" / s["id"] / f"cp{cp:g}_md{md:g}"
                if (dst / "etapas.json").exists():
                    continue
                dst.mkdir(parents=True, exist_ok=True)
                subprocess.run([str(host), "seq", str(gp / "probs.f64"), str(par["T"]),
                                str(par["C"]), repr(float(cp)), repr(float(par["temp"])),
                                repr(float(par["n_logprob"])), repr(float(md)),
                                repr(float(par["passo_s"])), str(dst)], check=True)


# ====================================================================== métricas
def metricas(a, b, tol):
    """a = C, b = referência, tol = escalar ou array (broadcast). NaN deve coincidir."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    r = {"n": int(b.size)}
    if a.shape != b.shape:
        return {**r, "status": "FAIL", "detalhe": f"forma C {a.shape} != ref {b.shape}"}
    tol = np.broadcast_to(np.asarray(tol, dtype=np.float64), b.shape)
    fa, fb = np.isfinite(a), np.isfinite(b)
    nan_dif = int(np.sum(fa != fb) + np.sum(~fa & ~fb & (np.isnan(a) != np.isnan(b))))
    inf_dif = int(np.sum(~fa & ~fb & ~np.isnan(a) & ~np.isnan(b) & (a != b)))
    k = fa & fb
    d = np.abs(a[k] - b[k])
    t = tol[k]
    escala = float(np.max(np.abs(b[k]))) if k.any() else 0.0
    viol = d > t
    r.update(err_max=float(d.max()) if d.size else 0.0,
             err_medio=float(d.mean()) if d.size else 0.0,
             err_rms=float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0,
             err_rel_max=(float(d.max()) / escala if escala > 0 else (0.0 if not d.size or d.max() == 0 else np.inf)) if d.size else 0.0,
             tol_max=float(t.max()) if t.size else 0.0,
             razao_tol=float(np.max(np.where(t > 0, d / np.where(t > 0, t, 1), np.where(d > 0, np.inf, 0.0)))) if d.size else 0.0,
             n_viol=int(viol.sum()) + nan_dif + inf_dif)
    ok = r["n_viol"] == 0
    r["status"] = "PASS" if ok else "FAIL"
    if nan_dif or inf_dif:
        r["detalhe"] = f"posições NaN/inf diferentes: {nan_dif + inf_dif}"
    return r


def tol_comb(regra, b):
    return regra.get("atol", 0.0) + regra.get("rtol", 0.0) * np.abs(np.asarray(b, dtype=float))


def tol_relmax(regra, b):
    b = np.asarray(b, dtype=float)
    m = float(np.nanmax(np.abs(b))) if b.size and np.isfinite(b).any() else 0.0
    return regra.get("atol", 0.0) + regra["rtol_max"] * m


# ---- cadeia da rotação: propagação de primeira ordem a partir do erro da STFT (§2.3)
def escalas_S(ref):
    smax = float(np.max(ref["stft_S"]))
    sal = ref["sal"]
    smt = sal.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(smt > 0, smax / smt, 0.0)                         # (T,)
    q_sil = smax / float(sal.max()) if sal.max() > 0 else 0.0
    return smax, q, q_sil


def tol_sal(regra, ref):
    smax, _, _ = escalas_S(ref)
    return regra.get("atol", 0.0) + regra["rho"] * smax


def tol_enorm(regra, ref):
    _, q, _ = escalas_S(ref)
    return regra.get("atol", 0.0) + 2 * regra["rho"] * q[:, None] * np.ones_like(ref["e_norm"])


def tol_custo(regra, ref):
    _, q, q_sil = escalas_S(ref)
    c = ref["custo"]
    J = c.shape[1] - 1
    qq = np.empty_like(c)
    qq[:, :J] = q[:, None]
    qq[:, J] = q_sil
    return regra.get("atol", 0.0) + regra.get("rtol", 0.0) * np.abs(c) + 2 * regra["rho"] * np.exp(c) * qq


def score_caminho(custo, path, passo, max_salto, w_step, w_oct, w_sil):
    """Contrato §4, em float64 (mesma ordem de soma do wrapper do agente B não é exigida:
    a diferença é O(T·eps64), irrelevante frente às margens)."""
    J = custo.shape[1] - 1
    k_oct = int(round(1200 / passo))
    p = np.where(np.asarray(path) < 0, J, np.asarray(path)).astype(int)
    if np.any(p > J) or np.any(p < 0):
        return float("inf")
    tot = float(np.sum(custo[np.arange(len(p)), p]))
    for a, b in zip(p[:-1], p[1:]):
        if a == b:
            continue
        if a == J or b == J:
            tot += w_sil
        elif abs(b - a) <= max_salto:
            tot += w_step * abs(b - a) * passo / 100.0
        elif abs(b - a) == k_oct:
            tot += w_oct
        else:
            return float("inf")
    return tot


# ====================================================================== caso
def comparar_caso(ref, cc, tb, viterbi_cfg, modelo):
    """Devolve lista de linhas (uma por etapa, mais uma por feature)."""
    linhas = []
    st = {}

    def add(et, r, **extra):
        r = {"etapa": et, **r, **extra}
        linhas.append(r)
        st[et] = r["status"]
        return r

    def falta(et):
        return et not in cc or (isinstance(cc[et], tuple))

    u = tb["_u"]
    for et in ["y_prep", "stft_S", "grade"]:
        if falta(et):
            add(et, {"status": "AUSENTE"})
            continue
        regra = tb[et]
        tol = tol_comb(regra, ref[et]) if regra["tipo"] == "comb" else tol_relmax(regra, ref[et])
        add(et, metricas(cc[et], ref[et], tol))
    for et, f in (("sal", tol_sal), ("e_norm", tol_enorm), ("custo", tol_custo)):
        if falta(et):
            add(et, {"status": "AUSENTE"})
            continue
        add(et, metricas(cc[et], ref[et], f(tb[et], ref)))

    # ---- Viterbi de rotação: caminho e score
    vc = viterbi_cfg
    sc = lambda cst, p: score_caminho(cst, p, vc["passo_cents"], vc["max_salto"],  # noqa: E731
                                      vc["w_step"], vc["w_oct"], vc["w_sil"])
    p_ref = np.asarray(ref["vit_path"]).astype(int)
    tc = tol_custo(tb["custo"], ref)
    T = len(p_ref)
    J = ref["custo"].shape[1] - 1
    s_ref = sc(ref["custo"], p_ref)
    acc = 2 * T * u * (abs(s_ref) + 1.0)
    margem_score = None
    if falta("vit_path"):
        add("vit_path", {"status": "AUSENTE"})
    else:
        p_c = np.asarray(cc["vit_path"]).astype(int).ravel()
        if p_c.shape != p_ref.shape:
            add("vit_path", {"n": T, "status": "FAIL", "detalhe": f"forma {p_c.shape}"})
        else:
            ndif = int(np.sum(p_c != p_ref))
            r = {"n": T, "concordancia": 1 - ndif / T, "n_viol": ndif}
            idx = lambda p: np.where(p < 0, J, p)                          # noqa: E731
            tau_ref = float(np.sum(tc[np.arange(T), idx(p_ref)]))
            if ndif == 0:
                r["status"] = "PASS"
                margem_score = tau_ref + acc
            else:
                gap = sc(ref["custo"], p_c) - s_ref
                tau_c = float(np.sum(tc[np.arange(T), idx(p_c)]))
                if not falta("custo") and np.shape(cc["custo"]) == ref["custo"].shape:
                    cC = np.asarray(cc["custo"], dtype=float)
                    E_ref = sc(cC, p_ref) - s_ref
                    E_c = sc(cC, p_c) - sc(ref["custo"], p_c)
                    margem = E_ref - E_c + acc
                    otimo_c = sc(cC, p_c) <= sc(cC, p_ref) + acc
                    origem = "dados (custo do C)"
                else:
                    margem = tau_ref + tau_c + acc
                    otimo_c = True
                    origem = "tolerância de custo"
                ok = np.isfinite(gap) and gap <= margem and otimo_c and st.get("custo") == "PASS"
                r["status"] = "PASS_EMPATE" if ok else "FAIL"
                r["gap_score_ref"] = gap
                r["margem"] = margem
                r["detalhe"] = (f"{ndif} quadros diferentes (t={np.flatnonzero(p_c != p_ref).tolist()[:10]}); "
                                f"gap pelo custo da ref = {gap:.3e}, margem = {margem:.3e} [{origem}]"
                                + ("" if otimo_c else "; caminho do C NÃO é ótimo para o custo do próprio C")
                                + ("" if st.get("custo") == "PASS" else "; custo fora da tolerância"))
                margem_score = max(tau_ref, tau_c) + acc + max(gap, 0.0)
            add("vit_path", r)
    if falta("vit_score"):
        add("vit_score", {"status": "AUSENTE"})
    else:
        tol = margem_score if margem_score is not None else np.inf
        add("vit_score", metricas(np.ravel(cc["vit_score"])[:1], np.ravel(ref["vit_score"])[:1], tol))
    if falta("f_rot"):
        add("f_rot", {"status": "AUSENTE"})
    else:
        add("f_rot", metricas(cc["f_rot"], ref["f_rot"], tol_comb(tb["f_rot"], ref["f_rot"])))

    # ---- espectro médio e envelope
    for et in ["P_mean", "env_abs", "env_dec", "welch_f", "welch_P"]:
        if falta(et):
            add(et, {"status": "AUSENTE"})
            continue
        regra = tb[et]
        tol = tol_comb(regra, ref[et]) if regra["tipo"] == "comb" else tol_relmax(regra, ref[et])
        add(et, metricas(cc[et], ref[et], tol))

    tw = tol_relmax(tb["welch_P"], ref["welch_P"])

    def regra_fmod(f_c, f_ref):
        """f_mod é a frequência do argmax de welch_P em 2–100 Hz: igual, ou empate real."""
        if f_c == f_ref:
            return "PASS", ""
        wf, wP = ref["welch_f"], ref["welch_P"]
        i_c, i_r = int(np.argmin(np.abs(wf - f_c))), int(np.argmin(np.abs(wf - f_ref)))
        m = (wf >= 2) & (wf <= 100)
        if abs(wf[i_c] - f_c) > 1e-6 * max(1.0, abs(f_c)) or not m[i_c]:
            return "FAIL", f"f_mod do C ({f_c}) não é um bin de welch_f em 2–100 Hz"
        dif = wP[i_r] - wP[i_c]
        return ("PASS_EMPATE" if dif <= 2 * tw else "FAIL"), \
            f"argmax de welch_P: C bin {i_c}, ref bin {i_r}; P_ref[ref]-P_ref[C] = {dif:.3e} (2·tol = {2 * tw:.3e})"

    pr = tb["env_mod"]
    if falta("env_mod"):
        add("env_mod", {"status": "AUSENTE"})
    else:
        a, b = np.ravel(cc["env_mod"]).astype(float), np.ravel(ref["env_mod"])
        s0, d0 = regra_fmod(a[0], b[0])
        r = metricas(a[1:2], b[1:2], pr["prom_atol"] + pr["prom_rtol"] * abs(b[1]))
        if s0 == "FAIL" or r["status"] == "FAIL":
            r["status"] = "FAIL"
        elif s0 == "PASS_EMPATE":
            r["status"] = "PASS_EMPATE"
        r["detalhe"] = "; ".join(x for x in (d0, r.get("detalhe", "")) if x)
        add("env_mod", r)

    # ---- features (por feature)
    fr = tb["features"]
    atol_x, rtol_x = np.asarray(fr["atol"], float), np.asarray(fr["rtol"], float)
    tol_x = atol_x + rtol_x * np.abs(ref["features"])
    if falta("features"):
        add("features", {"status": "AUSENTE"})
    else:
        a, b = np.ravel(cc["features"]).astype(float), ref["features"]
        r = metricas(a, b, tol_x)
        sub = []
        for k in range(len(b)):
            rk = metricas(a[k:k + 1], b[k:k + 1], tol_x[k])
            if k == 9:
                s9, d9 = regra_fmod(a[9], b[9])
                rk["status"], rk["detalhe"] = s9, d9
            rk["valor_ref"], rk["valor_c"] = float(b[k]), float(a[k])
            sub.append((f"features[{k}]:{NOMES_FEAT[k]}", rk))
        sts = [rk["status"] for _, rk in sub]
        r["status"] = "FAIL" if "FAIL" in sts else ("PASS_EMPATE" if "PASS_EMPATE" in sts else "PASS")
        r["detalhe"] = ",".join(n for n, rk in sub if rk["status"] != "PASS")
        add("features", r)
        for n, rk in sub:
            linhas.append({"etapa": n, **rk})

    # ---- classificador: tolerâncias propagadas do erro admitido nas features (§2.7)
    mu, sd, M, temp = modelo["mu"], modelo["sd"], modelo["M"], modelo["temp"]
    zr = tb["z"]
    # erro admitido em x propagado por (x - mu)/sd, mais o arredondamento da subtração e da
    # divisão no build (k_u·u·(|x|+|mu|)/sd: cobre o cancelamento quando x ≈ mu)
    tol_z = (tol_x + zr["k_u"] * u * (np.abs(ref["features"]) + np.abs(mu))) / sd + zr.get("atol", 0.0)
    if falta("z"):
        add("z", {"status": "AUSENTE"})
    else:
        add("z", metricas(cc["z"], ref["z"], tol_z))
    nz = float(np.linalg.norm(ref["z"]))
    tcos = min(2.0, 2 * float(np.linalg.norm(tol_z)) / nz) if nz > 0 else 2.0
    tcos += tb["cos"].get("atol", 0.0)
    if falta("cos"):
        add("cos", {"status": "AUSENTE"})
    else:
        add("cos", metricas(cc["cos"], ref["cos"], tcos))
    p = np.ravel(ref["probs"])
    tol_p = tb["probs"].get("atol", 0.0) + np.minimum(1.0, 2 * p * tcos / max(temp, 1e-6))
    if falta("probs"):
        add("probs", {"status": "AUSENTE"})
        add("decisao", {"status": "AUSENTE"})
    else:
        pc = np.ravel(cc["probs"]).astype(float)
        add("probs", metricas(pc, p, tol_p))
        if pc.shape != p.shape:
            add("decisao", {"status": "FAIL", "detalhe": "forma"})
        else:
            ic, ir = int(np.argmax(pc)), int(np.argmax(p))
            r = {"n": 1, "concordancia": float(ic == ir), "n_viol": int(ic != ir)}
            if ic == ir:
                r["status"] = "PASS"
            else:
                dif = p[ir] - p[ic]
                lim = tol_p[ir] + tol_p[ic]
                r["status"] = "PASS_EMPATE" if dif <= lim else "FAIL"
                r["detalhe"] = f"argmax C={ic} ref={ir}; p_ref[top1]-p_ref[C] = {dif:.3e} (limite {lim:.3e})"
            add("decisao", r)
    return linhas


# ====================================================================== sequências
def comparar_seq(gref, par, cc, tb, i_cp, i_md):
    """Uma (sequência, change_penalty, min_dur). gref: etapas do ouro; cc: saída do C."""
    linhas, st = [], {}
    u = tb["_u"]
    cp = par["change_penalties"][i_cp]
    C = par["C"]

    def add(et, r):
        linhas.append({"etapa": et, **r})
        st[et] = r["status"]

    P = gref["probs"]
    le_ref = gref["hmm_log_em"]
    tle = tol_comb(tb["hmm_log_em"], le_ref)
    if "hmm_log_em" in cc:
        add("hmm_log_em", metricas(cc["hmm_log_em"], le_ref, tle))
    else:
        add("hmm_log_em", {"status": "AUSENTE"})

    # argmax sem HMM
    ref_am = gref["argmax"].ravel()
    if "argmax" in cc:
        a = np.ravel(cc["argmax"]).astype(int)
        dif = np.flatnonzero(a != ref_am)
        r = {"n": len(ref_am), "concordancia": 1 - len(dif) / len(ref_am), "n_viol": len(dif)}
        emp = all(P[t, ref_am[t]] - P[t, a[t]] <= 4 * u * P[t, ref_am[t]] for t in dif)
        r["status"] = "PASS" if not len(dif) else ("PASS_EMPATE" if emp else "FAIL")
        if len(dif):
            r["detalhe"] = f"t={dif.tolist()[:10]}"
        add("argmax", r)
    else:
        add("argmax", {"status": "AUSENTE"})

    # Viterbi do HMM
    def score_hmm(le, p):
        p = np.asarray(p, int)
        return float(np.sum(le[np.arange(len(p)), p]) - cp * np.sum(p[1:] != p[:-1]))
    ref_v = gref["hmm_viterbi"][i_cp]
    if "hmm_viterbi" in cc:
        a = np.ravel(cc["hmm_viterbi"]).astype(int)
        dif = np.flatnonzero(a != ref_v)
        r = {"n": len(ref_v), "concordancia": 1 - len(dif) / len(ref_v), "n_viol": len(dif)}
        if not len(dif):
            r["status"] = "PASS"
        else:
            s_r = score_hmm(le_ref, ref_v)
            gap = s_r - score_hmm(le_ref, a)
            accv = 2 * len(a) * u * (abs(s_r) + 1.0)
            if "hmm_log_em" in cc and np.shape(cc["hmm_log_em"]) == le_ref.shape:
                lC = np.asarray(cc["hmm_log_em"], float)
                E_ref = score_hmm(lC, ref_v) - s_r
                E_c = score_hmm(lC, a) - score_hmm(le_ref, a)
                margem = E_c - E_ref + accv
                otimo = score_hmm(lC, a) >= score_hmm(lC, ref_v) - accv
            else:
                margem = 2 * float(np.max(tle)) * len(a) + accv
                otimo = True
            ok = gap <= margem and otimo and st.get("hmm_log_em") in ("PASS", "AUSENTE")
            r["status"] = "PASS_EMPATE" if ok else "FAIL"
            r["gap_score_ref"], r["margem"] = gap, margem
            r["detalhe"] = f"{len(dif)} janelas; gap={gap:.3e} margem={margem:.3e}" + ("" if otimo else "; C não ótimo")
        add("hmm_viterbi", r)
    else:
        add("hmm_viterbi", {"status": "AUSENTE"})

    # filtro online: alpha (se o C o grava) e decisão
    tal = None
    if "hmm_online_alpha" in gref:
        al_ref = gref["hmm_online_alpha"][i_cp]
        tal = tol_comb(tb["hmm_online_alpha"], al_ref)
        if "hmm_online_alpha" in cc:
            add("hmm_online_alpha", metricas(cc["hmm_online_alpha"], al_ref, tal))
        else:
            add("hmm_online_alpha", {"status": "AUSENTE"})
    ref_o = gref["hmm_online"][i_cp]
    if "hmm_online" in cc:
        a = np.ravel(cc["hmm_online"]).astype(int)
        dif = np.flatnonzero(a != ref_o)
        r = {"n": len(ref_o), "concordancia": 1 - len(dif) / len(ref_o), "n_viol": len(dif)}
        if not len(dif):
            r["status"] = "PASS"
        elif tal is not None:
            ok = all(al_ref[t, ref_o[t]] - al_ref[t, a[t]] <= tal[t, ref_o[t]] + tal[t, a[t]] for t in dif)
            r["status"] = "PASS_EMPATE" if ok else "FAIL"
            r["detalhe"] = f"t={dif.tolist()[:10]}"
        else:
            r["status"] = "FAIL"
        add("hmm_online", r)
    else:
        add("hmm_online", {"status": "AUSENTE"})

    # absorb_short: determinístico dado o rótulo de entrada
    ab_ref = gref["absorb"][:, i_cp, i_md, :]
    if "absorb" in cc:
        a = np.asarray(cc["absorb"]).astype(int).reshape(ab_ref.shape) if np.size(cc["absorb"]) == ab_ref.size else None
        if a is None:
            add("absorb", {"status": "FAIL", "detalhe": "forma"})
        else:
            dif = int(np.sum(a != ab_ref))
            r = {"n": int(ab_ref.size), "concordancia": 1 - dif / ab_ref.size, "n_viol": dif}
            if not dif:
                r["status"] = "PASS"
            else:
                ruins = [DECODS[d] for d in range(3) if np.any(a[d] != ab_ref[d])]
                entrada_ok = all(st.get(d) == "PASS" for d in ruins)
                r["status"] = "FAIL"
                r["detalhe"] = f"decodificações divergentes: {ruins}" + ("" if entrada_ok else " (entrada já divergia)")
            add("absorb", r)
    else:
        add("absorb", {"status": "AUSENTE"})
    return linhas


# ====================================================================== cascata
def marcar_cascata(linhas, ordem):
    """Primeira etapa que diverge = primeira, na ordem do contrato, com status FAIL.
    Toda etapa posterior com FAIL vira FAIL_JUSANTE (causa = a primeira)."""
    primeira = None
    for et in ordem:
        for r in linhas:
            if r["etapa"] == et and r["status"] == "FAIL":
                primeira = et
                break
        if primeira:
            break
    if primeira:
        i0 = ordem.index(primeira)
        for r in linhas:
            base = r["etapa"].split("[")[0]
            pos = ordem.index(base) if base in ordem else None
            if r["status"] == "FAIL" and pos is not None and pos > i0:
                r["status"] = "FAIL_JUSANTE"
    for r in linhas:
        r["primeira_divergencia"] = primeira or ""
    return primeira


# ====================================================================== principal
def preparar_tolerancias(tol_json, build, tier="normal"):
    """Regras do build; para casos de borda, as regras de `borda` substituem as normais."""
    b = tol_json["builds"][build]
    tb = dict(b["etapas"])
    if tier == "borda":
        tb.update(b.get("borda", {}))
    tb["_u"] = float(b["u"])
    return tb


def rodar(build, gold, cout, tol_path, tol_build=None, gerar=False, saida_csv=None, verbose=True):
    tol_json = json.loads(Path(tol_path).read_text("utf-8"))
    tb = preparar_tolerancias(tol_json, tol_build or build)
    tb_borda = preparar_tolerancias(tol_json, tol_build or build, "borda")
    man = json.loads((gold / "manifest.json").read_text("utf-8"))
    if gerar:
        host = achar_host(build)
        if host is None:
            raise SystemExit(f"p11_host do build '{build}' não encontrado em embarcados/build-{build}")
        gerar_saidas(build, gold, man, cout, host)
    vcfg = tol_json["viterbi_rotacao"]
    modelos = {}
    todas = []
    for c in man["casos"]:
        mod = modelos.setdefault(c["modelo"], ler_modelo(gold / "modelos" / f"{c['modelo']}.bin"))
        ref = ler_etapas(gold / "casos" / c["id"])
        cc = ler_etapas(cout / "casos" / c["id"])
        if not cc:
            linhas = [{"etapa": et, "status": "AUSENTE"} for et in ORDEM_CASO]
        else:
            linhas = comparar_caso(ref, cc, tb_borda if c.get("tipo") == "borda" else tb, vcfg, mod)
        marcar_cascata(linhas, ORDEM_CASO)
        for r in linhas:
            r.update(tipo_item="caso", id=c["id"], tipo_caso=c.get("tipo"), base=c.get("base"),
                     modelo=c["modelo"], change_penalty=np.nan, min_dur=np.nan)
        todas += linhas
    for s in man["seqs"]:
        gp = gold / "seqs" / s["id"]
        par = json.loads((gp / "params.json").read_text("utf-8"))
        gref = ler_etapas(gp)
        for i_cp, cp in enumerate(par["change_penalties"]):
            for i_md, md in enumerate(par["min_durs_s"]):
                cc = ler_etapas(cout / "seqs" / s["id"] / f"cp{cp:g}_md{md:g}")
                if not cc:
                    linhas = [{"etapa": et, "status": "AUSENTE"} for et in ORDEM_SEQ]
                else:
                    linhas = comparar_seq(gref, par, cc, tb, i_cp, i_md)
                marcar_cascata(linhas, ORDEM_SEQ)
                for r in linhas:
                    r.update(tipo_item="seq", id=s["id"], tipo_caso="seq", base=par["base"],
                             modelo="", change_penalty=cp, min_dur=md)
                todas += linhas
    df = pd.DataFrame(todas)
    cols = ["tipo_item", "id", "tipo_caso", "base", "modelo", "change_penalty", "min_dur", "etapa",
            "status", "primeira_divergencia", "n", "err_max", "err_medio", "err_rms", "err_rel_max",
            "tol_max", "razao_tol", "n_viol", "concordancia", "gap_score_ref", "margem",
            "valor_ref", "valor_c", "detalhe"]
    for k in cols:
        if k not in df:
            df[k] = np.nan
    df = df[cols]
    df.insert(0, "build", build)
    df.insert(1, "tol_build", tol_build or build)
    df.insert(2, "sha256_tolerancias", sha256(Path(tol_path)))
    resumo = resumir(df)
    if saida_csv is not None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        saida_csv.mkdir(parents=True, exist_ok=True)
        base = saida_csv / f"equivalencia_{build}_{ts}"
        k = 1
        while Path(str(base) + ".csv").exists():
            base = saida_csv / f"equivalencia_{build}_{ts}_{k}"
            k += 1
        df.to_csv(str(base) + ".csv", index=False)
        resumo.to_csv(str(base) + "_resumo.csv", index=False)
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
                                    text=True).stdout.strip()
            sujo = bool(subprocess.run(["git", "status", "--porcelain"], cwd=RAIZ,
                                       capture_output=True, text=True).stdout.strip())
        except Exception:
            commit, sujo = "", None
        meta = {"script": "embarcados/tests/equivalence/comparar.py", "commit": commit,
                "arvore_suja": sujo, "build": build, "tol_build": tol_build or build,
                "tolerancias": str(Path(tol_path)), "sha256_tolerancias": sha256(Path(tol_path)),
                "gold_manifest_sha256": sha256(gold / "manifest.json"),
                "gold_gerado_em": man.get("gerado_em"), "c_out": str(cout),
                "data": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_linhas": len(df),
                "contagem_status": df["status"].value_counts().to_dict()}
        Path(str(base) + "_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
        if verbose:
            print(f"CSV: {base}.csv")
    if verbose:
        pd.set_option("display.width", 220)
        print(resumo.to_string(index=False))
    return df, resumo


def resumir(df):
    d = df[~df["etapa"].str.contains(r"\[")]
    g = d.groupby(["tipo_item", "etapa"], sort=False)
    res = g.agg(n_itens=("status", "size"),
                n_pass=("status", lambda s: int((s == "PASS").sum())),
                n_pass_empate=("status", lambda s: int((s == "PASS_EMPATE").sum())),
                n_fail=("status", lambda s: int((s == "FAIL").sum())),
                n_fail_jusante=("status", lambda s: int((s == "FAIL_JUSANTE").sum())),
                n_ausente=("status", lambda s: int((s == "AUSENTE").sum())),
                err_max=("err_max", "max"), err_rel_max=("err_rel_max", "max"),
                razao_tol_max=("razao_tol", "max"), concordancia_min=("concordancia", "min")).reset_index()
    prim = df.drop_duplicates(["tipo_item", "id", "change_penalty", "min_dur"])
    cnt = prim["primeira_divergencia"].replace("", np.nan).dropna().value_counts()
    res["n_primeira_divergencia"] = res["etapa"].map(cnt).fillna(0).astype(int)
    feats = df[df["etapa"].str.startswith("features[")].groupby("etapa", sort=False).agg(
        n_itens=("status", "size"), n_fail=("status", lambda s: int(s.isin(["FAIL", "FAIL_JUSANTE"]).sum())),
        n_pass_empate=("status", lambda s: int((s == "PASS_EMPATE").sum())),
        err_max=("err_max", "max"), razao_tol_max=("razao_tol", "max")).reset_index()
    feats.insert(0, "tipo_item", "feature")
    return pd.concat([res, feats], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="float | double (ou nome livre com --c-out)")
    ap.add_argument("--tol-build", default=None, help="conjunto de tolerâncias (padrão = --build)")
    ap.add_argument("--gold", default=str(EMB / "gold"))
    ap.add_argument("--c-out", default=None)
    ap.add_argument("--tolerancias", default=str(TOL_PADRAO))
    ap.add_argument("--gerar", action="store_true", help="chamar p11_host para saídas ausentes")
    ap.add_argument("--saida", default=str(EMB / "results" / "csv"))
    ap.add_argument("--sem-csv", action="store_true")
    a = ap.parse_args()
    cout = Path(a.c_out) if a.c_out else EMB / "results" / "c_out" / a.build
    _, res = rodar(a.build, Path(a.gold), cout, a.tolerancias, a.tol_build, a.gerar,
                   None if a.sem_csv else Path(a.saida))
    ruins = res[(res["tipo_item"] != "feature")]
    sys.exit(0 if int(ruins["n_fail"].sum()) == 0 and int(ruins["n_ausente"].sum()) == 0 else 1)


if __name__ == "__main__":
    main()
