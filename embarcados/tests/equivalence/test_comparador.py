"""Autotestes do comparador SEM o porte C.

(a) ouro contra ele mesmo → tudo PASS, nos dois conjuntos de tolerâncias;
(b) perturbações controladas acima da tolerância numa etapa → FAIL nessa etapa, etapas
    anteriores PASS e primeira divergência correta; etapas posteriores afetadas → FAIL_JUSANTE;
(c) sensibilidade a convenção: STFT com Hann periódico → FAIL em stft_S no double;
(d) empate real na decisão → PASS_EMPATE; troca de rótulo sem empate → FAIL.

Usa um subconjunto do ouro copiado para um diretório temporário (o gold/ não é tocado).
    .venv\\Scripts\\python.exe -m pytest embarcados\\tests\\equivalence\\test_comparador.py -q
ou  .venv\\Scripts\\python.exe embarcados\\tests\\equivalence\\test_comparador.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import comparar as K  # noqa: E402

GOLD = K.EMB / "gold"
TOL = K.TOL_PADRAO
IDS = ["idmt_limpo_good_00", "idmt_branco0_broken_00", "a2212_limpo_Bearing_00",
       "a2212_leeshen_sag_Healthy_00", "borda_silencio3s_midmt", "borda_dc_ruido3s_midmt"]
SEQS = ["idmt_seq00", "a2212_seq00"]


def _ambiente():
    tmp = Path(tempfile.mkdtemp(prefix="eqv_"))
    g, c = tmp / "gold", tmp / "c_out"
    man = json.loads((GOLD / "manifest.json").read_text("utf-8"))
    man["casos"] = [x for x in man["casos"] if x["id"] in IDS]
    man["seqs"] = [x for x in man["seqs"] if x["id"] in SEQS]
    assert len(man["casos"]) == len(IDS) and len(man["seqs"]) == len(SEQS)
    (g).mkdir(parents=True)
    (g / "manifest.json").write_text(json.dumps(man), "utf-8")
    shutil.copytree(GOLD / "modelos", g / "modelos")
    for i in IDS:
        shutil.copytree(GOLD / "casos" / i, g / "casos" / i)
        shutil.copytree(GOLD / "casos" / i, c / "casos" / i)
    for s in SEQS:
        shutil.copytree(GOLD / "seqs" / s, g / "seqs" / s)
        par = json.loads((GOLD / "seqs" / s / "params.json").read_text("utf-8"))
        ref = K.ler_etapas(GOLD / "seqs" / s)
        for i_cp, cp in enumerate(par["change_penalties"]):
            for i_md, md in enumerate(par["min_durs_s"]):
                d = c / "seqs" / s / f"cp{cp:g}_md{md:g}"
                d.mkdir(parents=True)
                arrs = {"hmm_log_em": ref["hmm_log_em"], "argmax": ref["argmax"],
                        "hmm_viterbi": ref["hmm_viterbi"][i_cp], "hmm_online": ref["hmm_online"][i_cp],
                        "hmm_online_alpha": ref["hmm_online_alpha"][i_cp],
                        "absorb": ref["absorb"][:, i_cp, i_md, :]}
                desc = {}
                for k, a in arrs.items():
                    _gravar(d, desc, k, a)
                (d / "etapas.json").write_text(json.dumps(desc), "utf-8")
    return tmp, g, c


def _gravar(d, desc, k, a):
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.integer):
        ext, dt, nd = "i32", "<i4", "int32"
    else:
        ext, dt, nd = "f64", "<f8", "float64"
    np.ascontiguousarray(a.astype(dt)).tofile(d / f"{k}.{ext}")
    desc[k] = {"arquivo": f"{k}.{ext}", "dtype": nd, "shape": list(a.shape)}


def _reescrever(pasta: Path, etapa: str, a):
    desc = json.loads((pasta / "etapas.json").read_text("utf-8"))
    _gravar(pasta, desc, etapa, a)
    (pasta / "etapas.json").write_text(json.dumps(desc), "utf-8")


def _rodar(g, c, build):
    df, res = K.rodar(build, g, c, TOL, None, False, None, verbose=False)
    return df


def _st(df, item, etapa, cp=None):
    d = df[(df["id"] == item) & (df["etapa"] == etapa)]
    if cp is not None:
        d = d[d["change_penalty"] == cp]
    return d["status"].tolist(), d["primeira_divergencia"].tolist()


def test_a_ouro_contra_ouro():
    tmp, g, c = _ambiente()
    try:
        for b in ("double", "float"):
            df = _rodar(g, c, b)
            assert set(df["status"]) == {"PASS"}, df[df["status"] != "PASS"][["id", "etapa", "status"]]
            assert (df["primeira_divergencia"] == "").all()
    finally:
        shutil.rmtree(tmp)


def test_b_perturbacoes():
    tmp, g, c = _ambiente()
    try:
        tol = json.loads(TOL.read_text("utf-8"))
        cid = "idmt_limpo_good_00"
        pc = c / "casos" / cid
        ref = K.ler_etapas(g / "casos" / cid)
        # b1: um elemento do custo 10× acima da sua tolerância (double)
        tc = K.tol_custo(tol["builds"]["double"]["etapas"]["custo"], ref)
        cu = ref["custo"].copy()
        t, j = 5, int(np.argmin(cu[5, :-1]))
        cu[t, j] += 10 * tc[t, j]
        _reescrever(pc, "custo", cu)
        df = _rodar(g, c, "double")
        s, p = _st(df, cid, "custo")
        assert s == ["FAIL"] and p == ["custo"], (s, p)
        for et in ["y_prep", "stft_S", "grade", "sal", "e_norm"]:
            assert _st(df, cid, et)[0] == ["PASS"], et
        assert set(df[df["id"] != cid]["status"]) == {"PASS"}
        _reescrever(pc, "custo", ref["custo"])

        # b2: caminho do Viterbi deslocado num quadro sem empate → FAIL em vit_path,
        #     f_rot e features a jusante
        vp = ref["vit_path"].copy()
        vp[10] = vp[10] + 25 if vp[10] + 25 < ref["custo"].shape[1] - 1 else vp[10] - 25
        fr = ref["f_rot"].copy()
        fr[10] = ref["grade"][vp[10]]
        _reescrever(pc, "vit_path", vp)
        _reescrever(pc, "f_rot", fr)
        df = _rodar(g, c, "double")
        assert _st(df, cid, "vit_path") == (["FAIL"], ["vit_path"])
        assert _st(df, cid, "f_rot")[0] == ["FAIL_JUSANTE"]
        assert _st(df, cid, "custo")[0] == ["PASS"]
        _reescrever(pc, "vit_path", ref["vit_path"])
        _reescrever(pc, "f_rot", ref["f_rot"])

        # b3: feature 2 (frac_sil) mudada em 1/39 → features FAIL, z/cos/probs a jusante
        x = ref["features"].copy()
        x[2] += 1 / 39
        _reescrever(pc, "features", x)
        mod = K.ler_modelo(g / "modelos" / "idmt_bloco0.bin")
        z = (x - mod["mu"]) / mod["sd"]
        _reescrever(pc, "z", z)
        df = _rodar(g, c, "float")
        assert _st(df, cid, "features") == (["FAIL"], ["features"])
        assert _st(df, cid, "features[2]:frac_sil")[0] == ["FAIL"]
        assert _st(df, cid, "z")[0] == ["FAIL_JUSANTE"]
        _reescrever(pc, "features", ref["features"])
        _reescrever(pc, "z", ref["z"])

        # b4: welch_P com erro de 2× a tolerância float no pico → FAIL em welch_P (float)
        wp = ref["welch_P"].copy()
        k = int(np.argmax(wp))
        wp[k] *= 1 + 2 * tol["builds"]["float"]["etapas"]["welch_P"]["rtol_max"]
        _reescrever(pc, "welch_P", wp)
        df = _rodar(g, c, "float")
        assert _st(df, cid, "welch_P") == (["FAIL"], ["welch_P"])
        _reescrever(pc, "welch_P", ref["welch_P"])

        # b5: sequência — um rótulo do Viterbi trocado (cp = 2) → FAIL só nesse cp
        sid = "idmt_seq00"
        par = json.loads((g / "seqs" / sid / "params.json").read_text("utf-8"))
        d = c / "seqs" / sid / f"cp2_md{par['min_durs_s'][0]:g}"
        v = K.ler_etapas(d)["hmm_viterbi"].copy()
        v[3] = (v[3] + 1) % par["C"]
        _reescrever(d, "hmm_viterbi", v)
        df = _rodar(g, c, "double")
        dd = df[(df["id"] == sid) & (df["etapa"] == "hmm_viterbi") & (df["change_penalty"] == 2)
                & (df["min_dur"] == par["min_durs_s"][0])]
        assert dd["status"].tolist() == ["FAIL"] and dd["primeira_divergencia"].tolist() == ["hmm_viterbi"]
        outros = df[(df["etapa"] == "hmm_viterbi") & ~df.index.isin(dd.index)]
        assert set(outros["status"]) == {"PASS"}
    finally:
        shutil.rmtree(tmp)


def test_c_hann_periodico_detectado_no_double():
    tmp, g, c = _ambiente()
    try:
        cid = "idmt_limpo_good_00"
        ref = K.ler_etapas(g / "casos" / cid)
        y = ref["y_prep"].astype(np.float32)
        n, hop = 8192, 1024
        w = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)
        T = 1 + (len(y) - n) // hop
        idx = np.arange(n)[None, :] + hop * np.arange(T)[:, None]
        S = np.abs(np.fft.rfft(y[idx] * w, axis=1)).T
        _reescrever(c / "casos" / cid, "stft_S", S)
        df = _rodar(g, c, "double")
        assert _st(df, cid, "stft_S") == (["FAIL"], ["stft_S"])
        dff = _rodar(g, c, "float")       # documentado: abaixo da tolerância float
        assert _st(dff, cid, "stft_S")[0] == ["PASS"]
    finally:
        shutil.rmtree(tmp)


def test_d_empate_na_decisao():
    tmp, g, c = _ambiente()
    try:
        cid = "a2212_limpo_Bearing_00"
        pg, pc = g / "casos" / cid, c / "casos" / cid
        ref = K.ler_etapas(pg)
        C = len(ref["probs"])
        p = np.full(C, 0.0)
        p[0], p[1] = 0.5 + 1e-9, 0.5 - 1e-9          # empate real na referência
        _reescrever(pg, "probs", p)
        q = p[[1, 0] + list(range(2, C))]              # C escolhe a outra classe
        _reescrever(pc, "probs", q)
        df = _rodar(g, c, "float")
        assert _st(df, cid, "probs")[0] == ["PASS"]
        assert _st(df, cid, "decisao")[0] == ["PASS_EMPATE"]
        # sem empate: mesma troca com margem grande
        p2 = np.zeros(C)
        p2[0], p2[1] = 0.9, 0.1
        _reescrever(pg, "probs", p2)
        _reescrever(pc, "probs", p2[[1, 0] + list(range(2, C))])
        df = _rodar(g, c, "float")
        assert _st(df, cid, "probs") == (["FAIL"], ["probs"])
        assert _st(df, cid, "decisao")[0] == ["FAIL_JUSANTE"]
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    for f in (test_a_ouro_contra_ouro, test_b_perturbacoes, test_c_hann_periodico_detectado_no_double,
              test_d_empate_na_decisao):
        f()
        print("OK", f.__name__)
