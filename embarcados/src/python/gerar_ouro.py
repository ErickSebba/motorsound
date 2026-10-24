"""Gera os vetores de ouro da p11_decifrae_harmsum para o porte C.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_ouro.py

Saída (gitignored, regerável): embarcados/gold/
    manifest.json, modelos/<base>_<fold>.bin|.json, casos/<id>/..., seqs/<id>/...
Formatos: embarcados/reports/CONTRATO_INTERFACES.md. Só lê o train/ do IDMT e o A2212
(validação dos folds idmt_bloco0 e a2212_arq0); nunca o test/. Não escreve em cache/ nem
out/ da raiz (extração com usar_cache=False; o cache existente só é LIDO para conferência).
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import platform
import shutil
import struct
import subprocess
import sys
import time
import warnings
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
warnings.filterwarnings("ignore")

import ouro_p11 as O  # noqa: E402  (insere a raiz no sys.path)
from motorsound import cache, hmm, io, runner, sequencias  # noqa: E402
from motorsound.config import AVAL_A2212, AVAL_IDMT  # noqa: E402
from ruidos import aplicar  # noqa: E402

RAIZ = O.RAIZ
GOLD = RAIZ / "embarcados" / "gold"
NOME_PIPE = "p11_decifrae_harmsum"
P11 = O.P11

# ---------------------------------------------------------------- sementes e conjunto
SEED_SELECAO = 20261023        # ordem das gravações na escolha das janelas
SEED_BORDAS = 20261024         # sinais sintéticos de borda
SEED_SEQ = 0                   # = default de experimentos/sequencias.py:36
N_LIMPAS = 10                  # por classe por base
N_RUIDOSAS = 3                 # por classe por base (as 3 primeiras das limpas)
N_SEQ = 10

BASES = {
    "idmt": {"aval": AVAL_IDMT, "split": "blocos", "fold": "idmt_bloco0"},
    "a2212": {"aval": AVAL_A2212, "split": "arquivo", "fold": "a2212_arq0"},
}

# Ruído: mesmo formato de dict de experimentos/robustez.py (CURVA_SNR, linha 26) e
# ruidos/noise.py::aplicar. Os distúrbios Lee & Shen NÃO aparecem em nenhum experimento
# de validação (só na bancada sintética); o formato é o de ruidos/lee_shen.py::aplicar_disturbio
# via noise.aplicar ({"tipo": "lee_shen", "disturbio": ...}), parâmetros sorteados pela semente.
PERTURBACOES = [
    ("branco20", {"tipo": "branco", "snr_db": 20.0}),
    ("branco5", {"tipo": "branco", "snr_db": 5.0}),
    ("branco0", {"tipo": "branco", "snr_db": 0.0}),
    ("leeshen_sag", {"tipo": "lee_shen", "disturbio": "sag"}),
    ("leeshen_flicker", {"tipo": "lee_shen", "disturbio": "flicker"}),
]


# ---------------------------------------------------------------- utilidades
def _commit():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=RAIZ, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": git("rev-parse", "--short", "HEAD"),
            "sujo": bool(git("status", "--porcelain"))}


def sha256_arq(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gravar(pasta: Path, nome: str, arr, dtype_ref=None) -> dict:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        ext, dt = "i32", "<i4"
    else:
        ext, dt = "f64", "<f8"
    a = np.ascontiguousarray(arr.astype(dt))
    arq = f"{nome}.{ext}"
    a.tofile(pasta / arq)
    return {"arquivo": arq, "dtype": {"i32": "int32", "f64": "float64"}[ext],
            "shape": list(a.shape), "dtype_ref": str(dtype_ref or arr.dtype)}


def _json(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# ---------------------------------------------------------------- modelos
def exportar_modelo(nome, modelo, classes, fold, segs, treino):
    clf, C = modelo
    assert clf.classes == list(range(C)), "classes do treino != todas as classes da base"
    d = GOLD / "modelos"
    d.mkdir(parents=True, exist_ok=True)
    nf = clf.mu.shape[0]
    buf = struct.pack("<ii", C, nf) + struct.pack("<d", float(clf.temp))
    buf += np.ascontiguousarray(clf.mu, "<f8").tobytes()
    buf += np.ascontiguousarray(clf.sd, "<f8").tobytes()
    buf += np.ascontiguousarray(clf.M, "<f8").tobytes()
    (d / f"{nome}.bin").write_bytes(buf)
    h = sha256_arq(d / f"{nome}.bin")
    _json(d / f"{nome}.json", {
        "nome": nome, "pipeline": NOME_PIPE, "fold": fold, "n_classes": C, "n_feat": nf,
        "temp": clf.temp, "classes": [classes[c] for c in clf.classes],
        "mu": clf.mu.tolist(), "sd": clf.sd.tolist(), "M": clf.M.tolist(),
        "nota": "sd inclui +1e-9 de CossenoPrototipos.fit; M já normalizado por linha",
        "layout_bin": "int32 n_classes, int32 n_feat, float64 temp, float64 mu[n_feat], "
                      "float64 sd[n_feat], float64 M[n_classes][n_feat], little-endian",
        "n_treino": int(len(treino)),
        "indices_treino": [int(i) for i in treino],
        "treino": [{"rec_id": segs[i].rec_id, "idx": segs[i].idx} for i in treino],
        "rec_ids_treino": sorted({segs[i].rec_id for i in treino}),
        "sha256_bin": h,
    })
    return h


# ---------------------------------------------------------------- seleção de janelas
def selecionar(segs, val, classes, rng, n):
    """n janelas por classe, round-robin sobre as gravações (ordem sorteada) e espaçadas
    uniformemente dentro de cada gravação. A ordem devolvida é a do round-robin, então as
    primeiras janelas de cada classe vêm de gravações diferentes quando possível."""
    out = {}
    for c in classes:
        por_rec = {}
        for i in val:
            if segs[i].classe == c:
                por_rec.setdefault(segs[i].rec_id, []).append(i)
        recs = sorted(por_rec)
        recs = [recs[k] for k in rng.permutation(len(recs))]
        total = sum(len(v) for v in por_rec.values())
        alvo = min(n, total)
        cota = {r: 0 for r in recs}
        k = 0
        while sum(cota.values()) < alvo:
            r = recs[k % len(recs)]
            if cota[r] < len(por_rec[r]):
                cota[r] += 1
            k += 1
        escolhidos = {}
        for r in recs:
            ii = sorted(por_rec[r], key=lambda i: segs[i].idx)
            if cota[r]:
                pos = np.unique(np.round(np.linspace(0, len(ii) - 1, cota[r])).astype(int))
                escolhidos[r] = [ii[p] for p in pos]
        ordem, k = [], 0
        while len(ordem) < alvo:
            for r in recs:
                if k < len(escolhidos.get(r, [])):
                    ordem.append(escolhidos[r][k])
            k += 1
        out[c] = ordem
    return out


def audio_seg(s, sr):
    y = io.carregar(io._PATHS[s.rec_id], sr)
    i0, i1 = int(round(s.t0 * sr)), int(round(s.t1 * sr))
    return np.array(y[i0:i1], dtype=np.float32)


def semente_runner(rec_id, idx, pert):
    # = motorsound/runner.py:94-95
    return (zlib.crc32(f"{rec_id}#{idx}".encode()) & 0xFFFFFF) + int(pert.get("seed", 0))


# ---------------------------------------------------------------- casos
FALHAS: list[str] = []


def gravar_caso(cid, audio, modelo, aval, meta, x_esperado=None):
    pasta = GOLD / "casos" / cid
    pasta.mkdir(parents=True, exist_ok=True)
    audio = np.ascontiguousarray(audio, dtype="<f4")
    audio.tofile(pasta / "audio.f32")
    et, chk = O.etapas_p11(audio, modelo, aval)
    if x_esperado is not None:
        chk["features==runner"] = bool(np.array_equal(et["features"], x_esperado,
                                                      equal_nan=True))
    desc = {"audio": {"arquivo": "audio.f32", "dtype": "float32", "shape": [len(audio)]}}
    for k, v in et.items():
        desc[k] = gravar(pasta, k, v)
    _json(pasta / "etapas.json", desc)
    ruins = [k for k, v in chk.items() if not v and k != "vit_score==min(delta)_exato"]
    if ruins:
        FALHAS.append(f"{cid}: {ruins}")
    return {**meta, "id": cid, "n_amostras": int(len(audio)), "sr": 16000,
            "sha256_audio": sha256_arq(pasta / "audio.f32"),
            "pred": int(np.argmax(et["probs"])), "checagens": chk}


def sinal_harmonico(f0, dur, sr, rng, n_harm=48):
    t = np.arange(int(dur * sr)) / sr
    fase = rng.uniform(0, 2 * np.pi, n_harm)
    y = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t + fase[k - 1]) for k in range(1, n_harm + 1))
    return (0.1 * y / np.max(np.abs(y))).astype(np.float32)


# ---------------------------------------------------------------- sequências
def _carregar_exp_sequencias():
    spec = importlib.util.spec_from_file_location(
        "_exp_seq", RAIZ / "experimentos" / "sequencias.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gerar_seqs(base, classes, df_val, passo, info_seq):
    exp = _carregar_exp_sequencias()
    penal = list(exp.PENALIDADES)
    sig = inspect.signature(sequencias.avaliar).parameters
    temp, n_logprob = sig["temp"].default, sig["n_logprob"].default
    dmin, dmax = (5, 20) if base == "idmt" else (3, 10)       # experimentos/sequencias.py:50
    min_durs = [2 * passo, 5 * passo]                          # escolha do ouro (ver relatório)
    seqs = sequencias.montar(df_val, classes, n_seq=N_SEQ, dmin=dmin, dmax=dmax, seed=SEED_SEQ)
    C = len(classes)
    nomes_est = list(classes) + [hmm.ESTADO_N]
    ix = {n: i for i, n in enumerate(nomes_est)}
    regs = []
    for k, (verdade, P) in enumerate(seqs):
        sid = f"{base}_seq{k:02d}"
        pasta = GOLD / "seqs" / sid
        pasta.mkdir(parents=True, exist_ok=True)
        desc = {"probs": gravar(pasta, "probs", P),
                "verdade": gravar(pasta, "verdade", np.array([ix[v] for v in verdade]))}
        log_em = hmm._log_emissoes(P, temp, n_logprob, None, True)
        desc["hmm_log_em"] = gravar(pasta, "hmm_log_em", log_em)
        am = [classes[i] for i in P.argmax(axis=1)]
        desc["argmax"] = gravar(pasta, "argmax", np.array([ix[r] for r in am]))
        V, ON, AL, AF, AB = [], [], [], [], []
        for cp in penal:
            v = hmm.decode(P, classes, cp, temp, n_logprob)
            o = hmm.filtro_online(P, classes, cp, temp, n_logprob)
            o2, alphas = O.filtro_online_alpha(P, classes, cp, temp, n_logprob)
            if o2 != o:
                FALHAS.append(f"{sid} cp={cp}: filtro_online_alpha != hmm.filtro_online")
            if [nomes_est[int(i)] for i in alphas.argmax(axis=1)] != o:
                FALHAS.append(f"{sid} cp={cp}: argmax(alpha) != filtro_online")
            V.append([ix[r] for r in v])
            ON.append([ix[r] for r in o])
            AL.append(alphas)
            AF.append(alphas[-1])
            ab_cp = []
            for r in (am, v, o):
                ab_md = []
                for md in min_durs:
                    rr = hmm.segmentos_para_rotulos(
                        hmm.absorb_short(hmm.rotulos_para_segmentos(r, passo), md),
                        len(r), passo)                       # = motorsound/sequencias.py:60-63
                    ab_md.append([ix[x] for x in rr])
                ab_cp.append(ab_md)
            AB.append(ab_cp)
        desc["hmm_viterbi"] = gravar(pasta, "hmm_viterbi", np.array(V))
        desc["hmm_online"] = gravar(pasta, "hmm_online", np.array(ON))
        desc["hmm_online_alpha"] = gravar(pasta, "hmm_online_alpha", np.array(AL))
        desc["hmm_online_alpha_final"] = gravar(pasta, "hmm_online_alpha_final", np.array(AF))
        # absorb: (decodificação ∈ [argmax, viterbi, online], change_penalty, min_dur, T)
        desc["absorb"] = gravar(pasta, "absorb", np.transpose(np.array(AB), (1, 0, 2, 3)))
        _json(pasta / "etapas.json", desc)
        params = {
            "id": sid, "base": base, "fold": BASES[base]["fold"], "T": int(len(P)), "C": C,
            "estados": nomes_est, "estado_N": C, "passo_s": passo,
            "change_penalties": penal, "temp": temp, "n_logprob": n_logprob,
            "emissao_eh_prob": True, "force_n": None, "min_durs_s": min_durs,
            "log_T": {str(cp): (np.where(np.eye(C + 1, dtype=bool), 0.0, -cp)).tolist()
                      for cp in penal},
            "eixos": {"hmm_viterbi": ["change_penalty", "t"], "hmm_online": ["change_penalty", "t"],
                      "hmm_online_alpha": ["change_penalty", "t", "estado"],
                      "hmm_online_alpha_final": ["change_penalty", "estado"],
                      "absorb": ["decodificacao(argmax,viterbi,online)", "change_penalty",
                                 "min_dur", "t"]},
            "alpha": "log, normalizado por logaddexp a cada passo (hmm.filtro_online)",
            "montagem": {"funcao": "motorsound.sequencias.montar", "n_seq": N_SEQ,
                         "seed": SEED_SEQ, "dmin": dmin, "dmax": dmax, "indice": k},
            "origem_parametros": info_seq,
        }
        _json(pasta / "params.json", params)
        regs.append({"id": sid, "T": int(len(P))})
    return regs


# ---------------------------------------------------------------- principal
def main():
    t_ini = time.perf_counter()
    if GOLD.exists():
        shutil.rmtree(GOLD)          # saída 100 % regerável; nunca contém dado original
    GOLD.mkdir(parents=True)
    rng_sel = np.random.default_rng(SEED_SELECAO)
    casos, seqs_reg, modelos, info_base = [], [], {}, {}
    info_seq = {
        "change_penalties": "experimentos/sequencias.py:25 (PENALIDADES, varridos na linha 54)",
        "temp": "motorsound/sequencias.py:50 (default de avaliar; a chamada em "
                "experimentos/sequencias.py:55 não o passa) — difere de PostConfig.temp só no nome, ambos 1.0",
        "n_logprob": "motorsound/sequencias.py:50 (default -50.0 de avaliar) — difere de "
                     "PostConfig.n_logprob=-2.5 (motorsound/config.py)",
        "min_dur": "motorsound/sequencias.py:51 default 0.0, experimentos/sequencias.py nunca "
                   "passa outro valor; os min_dur>0 do ouro são escolha deste gerador "
                   "(2 e 5 passos) só para exercitar absorb_short",
        "passo_s": "experimentos/sequencias.py:49 (seg_s do aval: IDMT 3.0, A2212 1.0)",
        "dmin_dmax": "experimentos/sequencias.py:50 ((5,20) IDMT, (3,10) A2212)",
        "seed": "experimentos/sequencias.py:36 (default 0)",
        "n_seq": "experimentos/sequencias.py:35 usa 30; aqui 10 (as 10 primeiras sequências "
                 "com a mesma semente, mas sobre a validação de UM fold, não de todos)",
    }
    modelos_obj = {}
    for base, cfg in BASES.items():
        aval = cfg["aval"]
        classes = runner.classes_de(base)
        segs = runner.segmentos_de(base, aval)
        io.registrar(io.gravacoes(base, "train"))
        folds, grupos = runner.folds_de(base, cfg["split"], segs)
        f = next(ff for ff in folds if ff.nome == cfg["fold"])
        y = np.array([classes.index(s.classe) for s in segs])
        t0 = time.perf_counter()
        X = runner.extrair(NOME_PIPE, segs, aval, None, n_jobs=1, usar_cache=False)
        t_ext = time.perf_counter() - t0
        # conferência contra o cache existente (só leitura)
        iguais = difs = faltam = 0
        por_rec = {}
        for i, s in enumerate(segs):
            por_rec.setdefault(s.rec_id, []).append(i)
        for r, ii in por_rec.items():
            k = cache.chave(NOME_PIPE, P11.PRE, aval, None, r, [segs[i].idx for i in ii])
            Xc = cache.carregar(k)
            if Xc is None:
                faltam += 1
            elif np.array_equal(Xc, X[ii], equal_nan=True):
                iguais += 1
            else:
                difs += 1
        modelo = P11.treinar(X[f.treino], y[f.treino], P11.POST, n_classes=len(classes))
        nome_mod = f"{base}_{cfg['fold'].split('_', 1)[1]}"
        h = exportar_modelo(nome_mod, modelo, classes, cfg["fold"], segs, f.treino)
        modelos[nome_mod] = {"sha256_bin": h, "fold": cfg["fold"], "n_treino": int(len(f.treino))}
        modelos_obj[base] = (nome_mod, modelo, aval)
        Pval = np.asarray(P11.prever(modelo, X[f.val], P11.POST), dtype=float)
        acc = float(np.mean(Pval.argmax(axis=1) == y[f.val]))
        info_base[base] = {"n_segmentos": len(segs), "n_treino": int(len(f.treino)),
                           "n_val": int(len(f.val)), "acc_val_fold": acc,
                           "t_extracao_s": round(t_ext, 1),
                           "cache_gravacoes": {"iguais": iguais, "diferentes": difs,
                                               "ausentes": faltam}}
        # janelas
        sel = selecionar(segs, f.val, classes, rng_sel, N_LIMPAS)
        for c in classes:
            for k, i in enumerate(sel[c]):
                s = segs[i]
                a = audio_seg(s, 16000)
                assert np.array_equal(a, io.audio_segmento(s, 16000))
                meta = {"tipo": "limpo", "base": base, "rec_id": s.rec_id, "idx": s.idx,
                        "t0": s.t0, "t1": s.t1, "classe": s.classe, "classe_idx": int(y[i]),
                        "fold": cfg["fold"], "perturbacao": None, "modelo": nome_mod,
                        "indice_segmento": int(i)}
                casos.append(gravar_caso(f"{base}_limpo_{c}_{k:02d}", a, modelo, aval, meta,
                                         x_esperado=X[i]))
            for tag, pert in PERTURBACOES:
                for k, i in enumerate(sel[c][:N_RUIDOSAS]):
                    s = segs[i]
                    sem = semente_runner(s.rec_id, s.idx, pert)
                    a = aplicar(audio_seg(s, 16000), 16000, pert, seed=sem).astype(np.float32)
                    x_run = runner._extrair_gravacao(NOME_PIPE, s.rec_id, io._PATHS[s.rec_id],
                                                     [s], aval, pert, usar_cache=False)[0]
                    meta = {"tipo": "ruidoso", "base": base, "rec_id": s.rec_id, "idx": s.idx,
                            "t0": s.t0, "t1": s.t1, "classe": s.classe,
                            "classe_idx": int(y[i]), "fold": cfg["fold"], "perturbacao": pert,
                            "semente_perturbacao": int(sem), "modelo": nome_mod,
                            "indice_segmento": int(i)}
                    casos.append(gravar_caso(f"{base}_{tag}_{c}_{k:02d}", a, modelo, aval,
                                             meta, x_esperado=x_run))
        # sequências sobre TODAS as janelas de validação do fold
        df_val = pd.DataFrame([{"grupo": grupos[i], "idx": segs[i].idx, "classe": segs[i].classe,
                                **{f"p_{cc}": float(Pval[j, kk]) for kk, cc in enumerate(classes)}}
                               for j, i in enumerate(f.val)])
        seqs_reg += gerar_seqs(base, classes, df_val, aval.seg_s, info_seq)
        info_base[base]["primeira_limpa"] = int(sel[classes[0]][0])
        info_base[base]["_segs"] = segs

    # bordas
    rng_b = np.random.default_rng(SEED_BORDAS)
    bordas = {
        "silencio3s": (np.zeros(48000, np.float32), "zeros, 3 s"),
        "silencio1s": (np.zeros(16000, np.float32), "zeros, 1 s"),
    }
    for base in BASES:
        segs = info_base[base]["_segs"]
        s = segs[info_base[base]["primeira_limpa"]]
        a = np.clip(audio_seg(s, 16000) * 20.0, -1.0, 1.0).astype(np.float32)
        bordas[f"clip_{base}"] = (a, f"{s.rec_id}#{s.idx} ×20, clip ±1 antes de _preparar")
    bordas["rot90hz3s"] = (sinal_harmonico(90.0, 3.0, 16000, rng_b),
                           "f0=90 Hz, 48 harmônicos 1/k, fases sorteadas, 3 s (acima da faixa)")
    bordas["rot8hz3s"] = (sinal_harmonico(8.0, 3.0, 16000, rng_b),
                          "f0=8 Hz, 48 harmônicos 1/k, fases sorteadas, 3 s (abaixo da faixa)")
    bordas["dc_ruido3s"] = ((0.5 + 1e-5 * rng_b.standard_normal(48000)).astype(np.float32),
                            "DC 0,5 + ruído gaussiano σ=1e-5, 3 s")
    for nome, (a, descr) in bordas.items():
        for base, (nome_mod, modelo, aval) in modelos_obj.items():
            meta = {"tipo": "borda", "base": None, "rec_id": None, "idx": None,
                    "classe": None, "fold": BASES[base]["fold"], "perturbacao": None,
                    "descricao": descr, "modelo": nome_mod, "aval": base,
                    "semente": SEED_BORDAS}
            casos.append(gravar_caso(f"borda_{nome}_m{base}", a, modelo, aval, meta))

    for b in info_base.values():
        b.pop("_segs", None)
    from collections import Counter
    manifest = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "gerador": "embarcados/src/python/gerar_ouro.py",
        "projeto": _commit(), "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "plataforma": platform.platform(),
        "pipeline": NOME_PIPE, "pre": vars(P11.PRE) if hasattr(P11.PRE, "__dict__") else str(P11.PRE),
        "post": str(P11.POST),
        "sementes": {"selecao": SEED_SELECAO, "bordas": SEED_BORDAS, "sequencias": SEED_SEQ,
                     "perturbacao": "crc32(rec_id#idx)&0xFFFFFF + pert.get('seed',0) "
                                    "(motorsound/runner.py:94-95)"},
        "perturbacoes": dict(PERTURBACOES),
        "modelos": modelos, "bases": info_base,
        "contagens": dict(Counter(c["tipo"] for c in casos)), "n_casos": len(casos),
        "n_seqs": len(seqs_reg), "falhas_asserts": FALHAS,
        "segundos": round(time.perf_counter() - t_ini, 1),
        "casos": casos, "seqs": seqs_reg,
    }
    _json(GOLD / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("contagens", "n_casos", "n_seqs", "modelos",
                                                "bases", "segundos")}, indent=1, default=str))
    exatos = sum(c["checagens"]["vit_score==min(delta)_exato"] for c in casos)
    print(f"vit_score == min(delta) bit a bit em {exatos}/{len(casos)} casos")
    if FALHAS:
        print("FALHAS:\n" + "\n".join(FALHAS))
        sys.exit(1)
    print("todos os asserts passaram")


if __name__ == "__main__":
    main()
