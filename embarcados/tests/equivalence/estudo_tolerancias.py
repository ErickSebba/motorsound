"""Estudo empírico que fundamenta (junto com a análise de erro) as tolerâncias de
equivalência Python × C de p11_decifrae_harmsum.

Independente do porte C: reimplementa a cadeia do contrato (reports/CONTRATO_INTERFACES.md)
com um tipo de ponto flutuante escolhido ("f64": tudo em float64, como o build double do C;
"f32": tudo em float32, como o build float do C) e compara cada etapa com a referência
(pipelines/_decifrae.py e motorsound/rotacao.py, importados somente para leitura).

Não lê o conjunto de teste do IDMT: só data/idmt/train_cut e data/a2212.
Saída: results/csv/estudo_tolerancias_<timestamp>.csv + _meta.json (nunca sobrescritos).

Uso:  .venv\\Scripts\\python.exe embarcados\\tests\\equivalence\\estudo_tolerancias.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.fft as sfft
import scipy.signal as ss

RAIZ = Path(__file__).resolve().parents[3]          # Artigo3(MotorSound)
EMB = RAIZ / "embarcados"
sys.path.insert(0, str(RAIZ))

from motorsound import io as msio                    # noqa: E402
from motorsound import rotacao as rot                # noqa: E402
from motorsound.classificador import CossenoPrototipos  # noqa: E402
from pipelines import _decifrae as dec               # noqa: E402

SR = 16000
PRE = dec.PreDecifrae()
SEED = 20261023


# ---------------------------------------------------------------- referência por etapa
def etapas_referencia(audio, modelo):
    """Etapas da referência, com o mesmo código (ou cópia literal, onde a referência não
    expõe a etapa) que o wrapper do agente B usa."""
    out = {}
    y = audio.astype(np.float32)
    r = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    y = y * (0.1 / r)                                     # runner._preparar
    out["y_prep"] = y
    S, freqs = rot.stft_mag(y, SR, PRE.n_fft, PRE.hop)
    out["stft_S"] = S
    g = rot.grade(PRE.fmin, PRE.fmax, PRE.passo_cents)
    out["grade"] = g
    sal, e_norm = rot.saliencia_harmonica(S, freqs, g, np.arange(1, 49))
    out["sal"], out["e_norm"] = sal, e_norm
    e_q = sal.max(axis=1)
    custo = rot.emissao_baixo(e_norm, g, None, None, e_q, e_q.max())
    out["custo"] = custo
    path = rot.viterbi_rotacao(custo, PRE.passo_cents, max_salto=PRE.max_salto)
    out["vit_path"] = path
    out["vit_score"] = np.array([score_caminho(custo, path, PRE.passo_cents, PRE.max_salto)])
    out["f_rot"] = np.where(path >= 0, g[np.clip(path, 0, None)], np.nan)
    out["P_mean"] = (S ** 2).mean(axis=1)
    env = np.abs(ss.hilbert(y))
    out["env_abs"] = env
    sos = ss.butter(4, 200, fs=SR, output="sos")
    ed = ss.sosfilt(sos, env)[:: SR // 1000]
    out["env_dec"] = ed
    ed = ed - ed.mean()
    f, P = ss.welch(ed, 1000, nperseg=min(len(ed), 1024))
    out["welch_f"], out["welch_P"] = f, P
    out["env_mod"] = np.array(dec._envelope_mod(y, SR))
    out["features"] = dec.extrair(y, SR, PRE)
    if modelo is not None:
        clf = modelo
        x = out["features"][None, :]
        out["z"] = ((x - clf.mu) / clf.sd)[0]
        out["cos"] = clf.emissao(x)[0]
        out["probs"] = clf.predict_proba(x)[0]
    return out


def score_caminho(custo, path, passo, max_salto, w_step=rot.VIT_W_STEP,
                  w_oct=rot.VIT_W_OCTAVE, w_sil=rot.VIT_W_SIL):
    """Custo total de um caminho (contrato §4), sempre em float64."""
    J = custo.shape[1] - 1
    k_oct = int(round(1200 / passo))
    p = np.where(np.asarray(path) < 0, J, np.asarray(path))
    tot = float(sum(custo[t, p[t]] for t in range(len(p))))
    for a, b in zip(p[:-1], p[1:]):
        if a == b:
            continue
        if a == J or b == J:
            tot += w_sil
        elif abs(int(b) - int(a)) <= max_salto:
            tot += w_step * abs(int(b) - int(a)) * passo / 100.0
        elif abs(int(b) - int(a)) == k_oct:
            tot += w_oct
        else:
            return float("inf")
    return tot


# ---------------------------------------------------------------- emulação com dtype fixo
def viterbi_dt(custo, passo, max_salto, dt):
    """Mesma recursão de rotacao.viterbi_rotacao, com toda a acumulação no tipo dt."""
    custo = custo.astype(dt)
    T, S = custo.shape
    J = S - 1
    k_oct = int(round(1200 / passo))
    offs = np.arange(-max_salto, max_salto + 1)
    c_offs = (dt(rot.VIT_W_STEP) * np.abs(offs).astype(dt) * dt(passo) / dt(100.0)).astype(dt)
    w_oct, w_sil = dt(rot.VIT_W_OCTAVE), dt(rot.VIT_W_SIL)
    delta = custo[0].copy()
    psi = np.zeros((T, S), dtype=np.int32)
    jj = np.arange(J)
    for t in range(1, T):
        prev = delta
        best = np.full(J, np.inf, dtype=dt)
        arg = np.zeros(J, dtype=np.int32)
        for o, c in list(zip(offs, c_offs)) + [(k_oct, w_oct), (-k_oct, w_oct)]:
            src = jj - o
            ok = (src >= 0) & (src < J)
            cand = np.full(J, np.inf, dtype=dt)
            cand[ok] = prev[src[ok]] + c
            m = cand < best
            best[m] = cand[m]
            arg[m] = src[m]
        cs = prev[J] + w_sil
        m = cs < best
        best[m] = cs
        arg[m] = J
        novo = np.empty(S, dtype=dt)
        novo[:J] = best + custo[t, :J]
        psi[t, :J] = arg
        jb = int(np.argmin(prev[:J]))
        if prev[J] <= prev[jb] + w_sil:
            novo[J] = prev[J] + custo[t, J]
            psi[t, J] = J
        else:
            novo[J] = prev[jb] + w_sil + custo[t, J]
            psi[t, J] = jb
        delta = novo
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmin(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    path[path == J] = -1
    return path, float(delta.min())


def fft_r2(x):
    """FFT radix-2 iterativa (DIT) em complex64, twiddles em float32 por cos/sin: modelo
    de uma FFT de MCU (ESP-DSP dsps_fft2r_fc32 usa tabela de twiddles float)."""
    N = x.shape[-1]
    b = int(np.log2(N))
    rev = np.array([int(f"{i:0{b}b}"[::-1], 2) for i in range(N)])
    x = x[..., rev].astype(np.complex64)
    m = 2
    while m <= N:
        h = m // 2
        ang = np.float32(-2 * np.pi) * np.arange(h, dtype=np.float32) / np.float32(m)
        w = (np.cos(ang) + 1j * np.sin(ang)).astype(np.complex64)
        xs = x.reshape(x.shape[:-1] + (N // m, m))
        ev, od = xs[..., :h], xs[..., h:] * w
        x = np.concatenate([ev + od, ev - od], axis=-1).reshape(x.shape)
        m *= 2
    return x


def _soma(x, dt, seq, axis=None):
    """Soma no tipo dt; seq=True imita o laço sequencial de um C ingênuo."""
    if not seq:
        return np.sum(x, axis=axis, dtype=dt)
    if axis is None:
        return np.cumsum(np.ravel(x), dtype=dt)[-1]
    return np.take(np.cumsum(x, axis=axis, dtype=dt), -1, axis=axis)


def etapas_emuladas(audio, modelo, dt, seq=False):
    o = {}
    y = audio.astype(dt)
    r = np.sqrt(_soma(y * y, dt, seq) / dt(len(y))) + dt(1e-12)
    y = (y * (dt(0.1) / r)).astype(dt)
    o["y_prep"] = y
    n_fft, hop = PRE.n_fft, PRE.hop
    w = np.hanning(n_fft).astype(dt)
    n = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    if seq:
        S = np.abs(fft_r2(y[idx] * w)[:, : n_fft // 2 + 1]).T.astype(dt)
    else:
        S = np.abs(sfft.rfft(y[idx] * w, axis=1)).T.astype(dt)
    o["stft_S"] = S
    J = int(np.floor(1200 * np.log2(PRE.fmax / PRE.fmin) / PRE.passo_cents)) + 1
    g = (dt(PRE.fmin) * np.power(dt(2.0), np.arange(J).astype(dt) * dt(PRE.passo_cents)
                                 / dt(1200.0))).astype(dt)
    o["grade"] = g
    df = dt(SR) / dt(n_fft)
    F = n_fft // 2 + 1
    sal = np.zeros((S.shape[1], J), dtype=dt)
    for h in range(1, 49):
        pos = (dt(h) * g) / df
        ok = pos < F - 1
        i0 = np.floor(pos[ok]).astype(int)
        a = (pos[ok] - i0.astype(dt))[None, :]
        sal[:, ok] += (dt(1) - a) * S[i0].T + a * S[i0 + 1].T
    e_q = sal.max(axis=1)
    e_norm = sal / (e_q[:, None] + dt(1e-12))
    o["sal"], o["e_norm"] = sal, e_norm
    custo = np.empty((S.shape[1], J + 1), dtype=dt)
    custo[:, :J] = -np.log(dt(0.45) * e_norm + dt(1e-3))
    rel = e_q / (e_q.max() + dt(1e-12))
    custo[:, J] = -np.log((dt(1) - np.clip(rel, 0, 1)) + dt(1e-3))
    o["custo"] = custo
    path, dmin = viterbi_dt(custo, PRE.passo_cents, PRE.max_salto, dt)
    o["vit_path"] = path
    o["vit_score"] = np.array([dmin])
    vivos = path >= 0
    f_rot = np.where(vivos, g[np.clip(path, 0, None)], np.nan).astype(dt)
    o["f_rot"] = f_rot
    P = (_soma(S * S, dt, seq, axis=1) / dt(S.shape[1])).astype(dt)
    o["P_mean"] = P
    # envelope: hilbert sem padding, FFT no tipo dt
    Y = sfft.fft(y)
    hh = np.zeros(len(y), dtype=dt)
    N = len(y)
    if N % 2 == 0:
        hh[0] = hh[N // 2] = 1
        hh[1:N // 2] = 2
    else:
        hh[0] = 1
        hh[1:(N + 1) // 2] = 2
    env = np.abs(sfft.ifft(Y * hh)).astype(dt)
    o["env_abs"] = env
    sos = ss.butter(4, 200, fs=SR, output="sos").astype(dt)
    ed = ss.sosfilt(sos, env).astype(dt)[:: SR // 1000]
    o["env_dec"] = ed
    ed = ed - ed.mean(dtype=dt)
    nper = min(len(ed), 1024)
    f, Pw = ss.welch(ed, 1000, nperseg=nper)
    o["welch_f"], o["welch_P"] = f.astype(dt), Pw.astype(dt)
    m = (f >= 2) & (f <= 100)
    i = int(np.argmax(Pw[m]))
    fm = f[m][i]
    prom = dt(10) * np.log10(Pw[m][i] / (np.median(Pw[m]) + dt(1e-20)) + dt(1e-12))
    o["env_mod"] = np.array([fm, prom], dtype=dt)
    # features
    fr = f_rot[vivos]
    f_med = np.median(fr) if vivos.any() else dt(0)
    iqr = (np.percentile(fr, 75) - np.percentile(fr, 25)) if vivos.any() else dt(0)
    jx = np.clip(np.searchsorted(g, np.where(vivos, f_rot, g[0])), 0, J - 1)
    salc = np.mean(np.where(vivos, e_norm[np.arange(len(jx)), jx], 0)).astype(dt)
    fbin = (np.arange(F) * df).astype(dt)
    banda = (fbin >= 10) & (fbin <= 4000)
    tot = _soma(P[banda], dt, seq) + dt(1e-20)
    e_ord = np.zeros(48, dtype=dt)
    if f_med > 0:
        pos = np.round(np.arange(1, 49) * f_med / df).astype(int)
        ok = pos < F - 1
        e_ord[ok] = P[pos[ok] - 1] + P[pos[ok]] + P[pos[ok] + 1]
    razao = _soma(e_ord, dt, seq) / tot
    grupos = [dt(10) * np.log10(_soma(e_ord[a - 1:b], dt, seq) / tot + dt(1e-10))
              for a, b in dec.GRUPOS]
    cent = _soma(fbin * P, dt, seq) / (_soma(P, dt, seq) + dt(1e-20))
    nb = dt(banda.sum())
    plan = np.exp(_soma(np.log(P[banda] + dt(1e-20)), dt, seq) / nb) / (_soma(P[banda], dt, seq) / nb + dt(1e-20))
    x = np.array([f_med, iqr, dt(1) - vivos.mean(), salc, razao, *grupos, fm, prom, cent, plan],
                 dtype=dt)
    o["features"] = x
    if modelo is not None:
        mu, sd, M = (modelo.mu.astype(dt), modelo.sd.astype(dt), modelo.M.astype(dt))
        z = (x - mu) / sd
        o["z"] = z
        zn = z / (np.sqrt((z * z).sum(dtype=dt)) + dt(1e-9))
        cos = M @ zn
        o["cos"] = cos
        e = cos / dt(modelo.temp)
        e = np.exp(e - e.max())
        o["probs"] = e / e.sum(dtype=dt)
    return o


# ---------------------------------------------------------------- métricas de erro
def erros(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return dict(n=len(b), err_max=np.inf, err_rel_max=np.inf, err_rel_elem_max=np.inf)
    fa, fb = np.isfinite(a), np.isfinite(b)
    nan_ok = bool(np.array_equal(fa, fb))
    k = fa & fb
    d = np.abs(a[k] - b[k])
    escala = np.max(np.abs(b[k])) if k.any() else 1.0
    rel_elem = d / np.maximum(np.abs(b[k]), 1e-300)
    return dict(n=int(k.sum()), nan_ok=nan_ok,
                err_max=float(d.max()) if d.size else 0.0,
                err_rel_max=float(d.max() / escala) if d.size and escala > 0 else 0.0,
                err_rel_elem_max=float(rel_elem.max()) if d.size else 0.0)


def erro_condicionado(et, a, ref):
    """Erro dividido pela escala de propagação de primeira ordem a partir do erro da STFT
    (ver reports/c_equivalence.md §2.3): o número que a tolerância 'rho_S' limita."""
    smax = float(np.max(ref["stft_S"]))
    sal = ref["sal"]
    if smax == 0:
        return 0.0 if np.allclose(a, ref[et]) else np.inf
    if et == "sal":
        return float(np.max(np.abs(a - sal)) / smax)
    q = smax / np.maximum(sal.max(axis=1), 1e-300)            # (T,)
    if et == "e_norm":
        return float(np.max(np.abs(a - ref["e_norm"]).max(axis=1) / q))
    c = ref["custo"]
    J = c.shape[1] - 1
    qq = np.empty_like(c)
    qq[:, :J] = q[:, None]
    qq[:, J] = smax / max(float(sal.max()), 1e-300)
    return float(np.max(np.abs(a - c) / (np.exp(c) * qq)))


def casos():
    """Segmentos de treino (IDMT train_cut, 3 s) e A2212 (1 s), a 16 kHz, antes de _preparar."""
    rng = np.random.default_rng(SEED)
    out = []
    idmt = sorted((RAIZ / "data" / "idmt" / "train_cut").rglob("*.wav"))
    assert all("test" not in p.parts[-3] for p in idmt)
    for p in [idmt[i] for i in rng.choice(len(idmt), 24, replace=False)]:
        y = msio.carregar(str(p), SR)[: 3 * SR]
        if len(y) == 3 * SR:
            out.append(("idmt", p.parent.name, p.name, np.array(y)))
    a2 = sorted((RAIZ / "data" / "a2212").rglob("*.wav"))
    for p in [a2[i] for i in rng.choice(len(a2), 12, replace=False)]:
        y = msio.carregar(str(p), SR)
        i0 = int(rng.integers(0, len(y) - SR))
        out.append(("a2212", p.parent.parent.name, f"{p.parent.name}/{p.name}@{i0}",
                    np.array(y[i0:i0 + SR])))
    # ruído branco a 5 e 0 dB sobre 6 dos casos limpos (mesma família do ouro)
    for k, (base, cls, nome, y) in enumerate(list(out[::6])):
        for snr in (5.0, 0.0):
            ps = float(np.mean(y.astype(np.float64) ** 2))
            n = rng.standard_normal(len(y)) * np.sqrt(ps / 10 ** (snr / 10))
            out.append((base, cls, f"{nome}+branco{snr:g}", (y + n).astype(np.float32)))
    # bordas sintéticas (mesmo espírito das do ouro; não são as mesmas amostras)
    t = np.arange(48000) / SR
    def harm(f0):
        ph = rng.uniform(0, 2 * np.pi, 48)
        s = sum(np.sin(2 * np.pi * f0 * k * t + ph[k - 1]) / k for k in range(1, 49))
        return (0.1 * s / np.abs(s).max()).astype(np.float32)
    out += [("idmt", "borda", "silencio3s", np.zeros(48000, np.float32)),
            ("a2212", "borda", "silencio1s", np.zeros(16000, np.float32)),
            ("idmt", "borda", "rot90", harm(90.0)), ("idmt", "borda", "rot8", harm(8.0)),
            ("idmt", "borda", "dc", (0.5 + 1e-5 * rng.standard_normal(48000)).astype(np.float32)),
            ("idmt", "borda", "clip", np.clip(out[0][3] * 20, -1, 1).astype(np.float32))]
    return out


def main():
    t0 = time.time()
    cs = casos()
    # modelo de brinquedo (só para propagar erro até probs): protótipos por base com as
    # features de referência dos próprios casos; o modelo real vem do agente B.
    feats = {}
    for i, (base, cls, nome, y) in enumerate(cs):
        if cls != "borda":
            feats[i] = etapas_referencia(y, None)["features"]
    modelos = {}
    for base in ("idmt", "a2212"):
        ids = [i for i, c in enumerate(cs) if c[0] == base and c[1] != "borda"]
        X = np.stack([feats[i] for i in ids])
        nomes = sorted({cs[i][1] for i in ids})
        yy = np.array([nomes.index(cs[i][1]) for i in ids])
        modelos[base] = CossenoPrototipos(temp=0.1).fit(X, yy, list(range(len(nomes))))
    linhas = []
    for i, (base, cls, nome, y) in enumerate(cs):
        ref = etapas_referencia(y, modelos[base])
        g = ref["custo"]
        for build, dt, seq in (("double", np.float64, False), ("float", np.float32, False),
                               ("float_seq_r2", np.float32, True)):
            em = etapas_emuladas(y, modelos[base], dt, seq)
            for et, b in ref.items():
                a = em[et]
                lin = dict(caso=i, base=base, classe=cls, arquivo=nome, build=build, etapa=et)
                if et == "vit_path":
                    dif = int(np.sum(np.asarray(a) != np.asarray(b)))
                    sc_a = score_caminho(g, a, PRE.passo_cents, PRE.max_salto)
                    sc_b = score_caminho(g, b, PRE.passo_cents, PRE.max_salto)
                    lin.update(n=len(b), n_diferentes=dif, gap_score_ref=sc_a - sc_b)
                else:
                    lin.update(erros(a, b))
                if et in ("sal", "e_norm", "custo"):
                    lin["err_cond"] = erro_condicionado(et, a, ref)
                linhas.append(lin)
                if et == "features":
                    for k in range(13):
                        lk = dict(lin, etapa=f"features[{k}]")
                        lk.update(erros(a[k:k + 1], b[k:k + 1]))
                        lk["valor_ref"] = float(b[k])
                        linhas.append(lk)
            # decisão
            lin = dict(caso=i, base=base, classe=cls, arquivo=nome, build=build, etapa="decisao",
                       n_diferentes=int(np.argmax(em["probs"]) != np.argmax(ref["probs"])))
            ps = np.sort(ref["probs"])[::-1]
            lin["gap_top12_ref"] = float(ps[0] - ps[1])
            linhas.append(lin)
    df = pd.DataFrame(linhas)
    ts = time.strftime("%Y%m%d-%H%M%S")
    dst = EMB / "results" / "csv"
    dst.mkdir(parents=True, exist_ok=True)
    csv = dst / f"estudo_tolerancias_{ts}.csv"
    assert not csv.exists()
    df.to_csv(csv, index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    except Exception:
        commit = "desconhecido"
    meta = dict(script=str(Path(__file__).relative_to(RAIZ)), commit=commit, seed=SEED,
                data=ts, duracao_s=round(time.time() - t0, 1), n_casos=len(cs),
                numpy=np.__version__, scipy=__import__("scipy").__version__,
                sha256_csv=hashlib.sha256(csv.read_bytes()).hexdigest(),
                nota="emulação f64/f32 em numpy/scipy (pocketfft); não é o porte C")
    (dst / f"estudo_tolerancias_{ts}_meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    num = ["err_max", "err_rel_max", "err_rel_elem_max", "n_diferentes", "gap_score_ref",
           "gap_top12_ref"]
    num = [c for c in num if c in df]
    pd.set_option("display.width", 200)
    print(df.groupby(["build", "etapa"], sort=False)[num].max().to_string())
    print("gap top1-top2 mínimo na referência:", df["gap_top12_ref"].min())
    print(csv)


if __name__ == "__main__":
    main()
