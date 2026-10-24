"""Testes unitários do núcleo C p11 (Agente C). NÃO são prova de equivalência: conferem cada
peça isolada contra numpy/scipy/referência com entradas sintéticas. A verificação formal
(com tolerâncias congeladas) é do Agente D.

Uso (da raiz Artigo3(MotorSound)):
    .venv\\Scripts\\python.exe -m pytest embarcados\\tests\\unit\\test_p11.py -q
    .venv\\Scripts\\python.exe embarcados\\tests\\unit\\test_p11.py      (resumo legível)
Precisa de embarcados/build-float e/ou build-double compilados (src/c/build_host.bat).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.signal as ss

RAIZ = Path(__file__).resolve().parents[3]
EMB = RAIZ / "embarcados"
sys.path.insert(0, str(RAIZ))
from motorsound import hmm, rotacao  # noqa: E402  (referência, somente leitura)

BUILDS = [b for b in ("double", "float") if (EMB / f"build-{b}" / "p11_testes.exe").exists()]
# tolerâncias deste teste unitário (erro máx. relativo ao máx. |ref|), por build
TOL = {"double": {"fft": 1e-12, "sos": 1e-10, "welch": 1e-9, "hmm": 1e-12},
       "float": {"fft": 2e-6, "sos": 2e-3, "welch": 5e-3, "hmm": 1e-5}}


def _exe(build, nome):
    return str(EMB / f"build-{build}" / f"{nome}.exe")


def _roda(build, *args, nome="p11_testes"):
    r = subprocess.run([_exe(build, nome), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300))


@pytest.fixture(scope="module")
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ------------------------------------------------------------------ FFT
@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize("n", [1000, 8192, 16000, 48000, 7 * 11 * 13])
def test_fft_complexa(build, n, tmp):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    xi = np.empty(2 * n); xi[0::2] = x.real; xi[1::2] = x.imag
    xi.tofile(tmp / "x.f64")
    for cmd, ref in (("fft", np.fft.fft(x)), ("ifft", np.fft.ifft(x) * n)):
        _roda(build, cmd, tmp / "x.f64", n, tmp / "y.f64")
        y = np.fromfile(tmp / "y.f64")
        y = y[0::2] + 1j * y[1::2]
        assert _rel(y, ref) < TOL[build]["fft"], (cmd, _rel(y, ref))


@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize("n", [1000, 1024, 8192])
def test_rfft(build, n, tmp):
    x = np.random.default_rng(n + 1).standard_normal(n)
    x.tofile(tmp / "x.f64")
    _roda(build, "rfft", tmp / "x.f64", n, tmp / "y.f64")
    y = np.fromfile(tmp / "y.f64"); y = y[0::2] + 1j * y[1::2]
    assert _rel(y, np.fft.rfft(x)) < TOL[build]["fft"]


# ------------------------------------------------------------------ sosfilt
@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize("decim", [1, 16])
def test_sosfilt(build, decim, tmp):
    rng = np.random.default_rng(7)
    x = np.abs(rng.standard_normal(48000)) * 0.1 + 0.05 * np.sin(np.arange(48000) * 0.01)
    x.tofile(tmp / "x.f64")
    _roda(build, "sosfilt", tmp / "x.f64", len(x), decim, tmp / "y.f64")
    ref = ss.sosfilt(ss.butter(4, 200, fs=16000, output="sos"), x)[::decim]
    assert _rel(np.fromfile(tmp / "y.f64"), ref) < TOL[build]["sos"]


# ------------------------------------------------------------------ welch
@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize("L", [1000, 3000, 1500, 2048])
def test_welch(build, L, tmp):
    rng = np.random.default_rng(L)
    t = np.arange(L) / 1000
    x = np.sin(2 * np.pi * 23.0 * t) + 0.3 * rng.standard_normal(L) + 0.2
    x = x - x.mean()
    x.tofile(tmp / "x.f64")
    _roda(build, "welch", tmp / "x.f64", L, tmp / "f.f64", tmp / "P.f64")
    f_ref, P_ref = ss.welch(x, 1000, nperseg=min(L, 1024))
    f = np.fromfile(tmp / "f.f64"); P = np.fromfile(tmp / "P.f64")
    assert np.array_equal(f, f_ref)
    assert _rel(P, P_ref) < TOL[build]["welch"]


# ------------------------------------------------------------------ Viterbi de rotação
def _entrada_viterbi(seed, T=39):
    rng = np.random.default_rng(seed)
    J = rotacao.grade(15.0, 60.0, 2.0).size
    sal = rng.random((T, J)) ** 4
    j = rng.integers(100, 500)
    for t in range(T):
        if rng.random() < 0.15:
            j = int(np.clip(j + rng.choice([600, -600]), 0, J - 1))
        else:
            j = int(np.clip(j + rng.integers(-40, 41), 0, J - 1))
        sal[t, j] += 3.0
    e_q = sal.max(axis=1) * np.where(rng.random(T) < 0.25, 1e-3, 1.0)   # quadros "SIL"
    e_norm = sal / (sal.max(axis=1, keepdims=True) + 1e-12)
    return e_norm, e_q


@pytest.mark.parametrize("build", BUILDS)
def test_viterbi_rotacao(build, tmp):
    grade = rotacao.grade(15.0, 60.0, 2.0)
    iguais = 0
    for seed in range(12):
        e_norm, e_q = _entrada_viterbi(seed)
        T = e_norm.shape[0]
        e_norm.tofile(tmp / "en.f64"); e_q.tofile(tmp / "eq.f64")
        _roda(build, "viterbi", tmp / "en.f64", tmp / "eq.f64", T, tmp / "p.i32", tmp / "s.f64")
        path = np.fromfile(tmp / "p.i32", dtype=np.int32)
        md, sc = np.fromfile(tmp / "s.f64")
        custo = rotacao.emissao_baixo(e_norm, grade, None, None, e_q, e_q.max())
        ref = rotacao.viterbi_rotacao(custo, 2.0, max_salto=50)
        iguais += int(np.array_equal(path, ref))
        if build == "double":
            assert np.array_equal(path, ref), seed
            assert md == sc       # min(delta) == custo recalculado (mesma ordem de soma)
        else:
            assert abs(md - sc) <= 1e-5 * abs(md)
    assert build == "float" or iguais == 12


# ------------------------------------------------------------------ HMM e absorb
def _seq(build, P, cp, temp, nlp, min_dur, passo, d):
    P = np.ascontiguousarray(P, dtype=np.float64)
    P.tofile(d / "probs.f64")
    _roda(build, "seq", d / "probs.f64", P.shape[0], P.shape[1], repr(cp), repr(temp), repr(nlp),
          repr(min_dur), repr(passo), d / "out", nome="p11_host")
    et = json.load(open(d / "out" / "etapas.json"))
    out = {}
    for k, e in et.items():
        dt = {"float32": np.float32, "float64": np.float64, "int32": np.int32}[e["dtype"]]
        out[k] = np.fromfile(d / "out" / e["arquivo"], dtype=dt).reshape(e["shape"])
    return out


def _absorb_ref(r, n, passo, min_dur):
    return hmm.segmentos_para_rotulos(
        hmm.absorb_short(hmm.rotulos_para_segmentos(r, passo), min_dur), n, passo)


@pytest.mark.parametrize("build", BUILDS)
def test_hmm_e_absorb(build, tmp):
    rng = np.random.default_rng(3)
    for caso in range(16):
        T, C = int(rng.integers(20, 80)), int(rng.integers(2, 5))
        # trechos de classe com ruído (como as sequências montadas)
        lab = np.repeat(rng.integers(0, C, size=T // 4 + 1), 4)[:T]
        P = rng.dirichlet(np.ones(C) * 0.5, size=T)
        P[np.arange(T), lab] += rng.random(T)
        P /= P.sum(axis=1, keepdims=True)
        cp = float(rng.choice([0.5, 2.0, 8.0]))
        temp = float(rng.choice([1.0, 0.5]))
        nlp = float(rng.choice([-50.0, -2.5]))
        passo = float(rng.choice([1.0, 3.0, 0.1]))
        min_dur = float(rng.choice([0.0, 2.0, 5.0])) * passo
        out = _seq(build, P, cp, temp, nlp, min_dur, passo, tmp)
        nomes = list(range(C))
        le_ref = hmm._log_emissoes(P, temp, nlp, None, True)
        assert _rel(out["hmm_log_em"].astype(float), le_ref) < TOL[build]["hmm"]
        vit = hmm.decode(P, nomes, cp, temp, nlp)
        onl = hmm.filtro_online(P, nomes, cp, temp, nlp)
        vit = [C if v == "N" else v for v in vit]
        onl = [C if v == "N" else v for v in onl]
        if build == "double":
            assert out["hmm_viterbi"].tolist() == vit, caso
            assert out["hmm_online"].tolist() == onl, caso
        for k, r in enumerate((list(P.argmax(axis=1)), vit, onl)):
            ref = _absorb_ref(r, T, passo, min_dur) if min_dur > 0 else r
            ref = [C if v == "N" else v for v in ref]
            if build == "double" or k == 0:
                assert out["absorb"][k].tolist() == ref, (caso, k)


def test_absorb_meio_para_par(tmp):
    """passo_s = 0,1: os limites dos segmentos não são exatos em binário; o C tem de seguir
    o float do Python (início/fim i*passo, round() meio-para-par)."""
    build = BUILDS[0]
    rng = np.random.default_rng(11)
    for _ in range(30):
        T, C = 50, 3
        r = list(rng.integers(0, C, size=T))
        P = np.full((T, C), 0.01); P[np.arange(T), r] = 0.98
        out = _seq(build, P, 0.5, 1.0, -50.0, 0.3, 0.1, tmp)
        ref = _absorb_ref(r, T, 0.1, 0.3)
        assert out["absorb"][0].tolist() == [int(v) for v in ref]


# ------------------------------------------------------------------ resumo legível
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-rA"]))
