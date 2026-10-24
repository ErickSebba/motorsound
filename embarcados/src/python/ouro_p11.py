"""Wrapper instrumentado da p11_decifrae_harmsum (referência dourada para o porte C).

Não altera a referência: importa `pipelines/_decifrae.py`, `motorsound/rotacao.py`,
`motorsound/classificador.py`, `motorsound/hmm.py` e `motorsound/runner.py` e chama as
funções delas sempre que existem. Onde a referência não expõe o intermediário (dentro de
`_envelope_mod`, de `CossenoPrototipos.emissao`, do laço do Viterbi e do `filtro_online`),
as MESMAS linhas são reexecutadas aqui e a coerência com a função da referência é provada
por asserts (igualdade exata, sem tolerância). Formatos: embarcados/reports/CONTRATO_INTERFACES.md.

Uso:
    from ouro_p11 import etapas_p11
    d = etapas_p11(seg_float32_antes_de_preparar, modelo, aval)   # dict {etapa: ndarray}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

RAIZ = Path(__file__).resolve().parents[3]          # .../Artigo3(MotorSound)
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from motorsound import hmm, rotacao, runner  # noqa: E402
from motorsound.rotacao import (VIT_W_OCTAVE, VIT_W_SIL, VIT_W_STEP,  # noqa: E402
                                emissao_baixo, grade, rastrear, saliencia_harmonica,
                                stft_mag, viterbi_rotacao)
from pipelines import _decifrae as _d  # noqa: E402
from pipelines import p11_decifrae_harmsum as P11  # noqa: E402

PRE = P11.PRE
POST = P11.POST


# ------------------------------------------------------------------ Viterbi
def _viterbi_com_delta(custo, passo_cents, max_salto=None, w_step=VIT_W_STEP,
                       w_oct=VIT_W_OCTAVE, w_sil=VIT_W_SIL):
    """Cópia literal de `rotacao.viterbi_rotacao` que devolve também o `delta` final.
    Só serve para provar que o score recalculado (§4) é `min(delta)`; o caminho usado
    nos vetores é sempre o da função da referência (e um assert confere que são iguais)."""
    T, S = custo.shape
    J = S - 1
    if max_salto is None:
        max_salto = J
    k_oct = int(round(1200 / passo_cents))
    offs = np.arange(-max_salto, max_salto + 1)
    c_offs = w_step * np.abs(offs) * passo_cents / 100.0
    delta = custo[0].copy()
    psi = np.zeros((T, S), dtype=np.int32)
    jj = np.arange(J)
    for t in range(1, T):
        prev = delta
        best = np.full(J, np.inf)
        arg = np.zeros(J, dtype=np.int32)
        for o, c in zip(offs, c_offs):
            src = jj - o
            ok = (src >= 0) & (src < J)
            cand = np.full(J, np.inf)
            cand[ok] = prev[src[ok]] + c
            m = cand < best
            best[m] = cand[m]
            arg[m] = src[m]
        for o in (k_oct, -k_oct):
            src = jj - o
            ok = (src >= 0) & (src < J)
            cand = np.full(J, np.inf)
            cand[ok] = prev[src[ok]] + w_oct
            m = cand < best
            best[m] = cand[m]
            arg[m] = src[m]
        cand_sil = prev[J] + w_sil
        m = cand_sil < best
        best[m] = cand_sil
        arg[m] = J
        novo = np.empty(S)
        novo[:J] = best + custo[t, :J]
        psi[t, :J] = arg
        j_best = int(np.argmin(prev[:J]))
        if prev[J] <= prev[j_best] + w_sil:
            novo[J] = prev[J] + custo[t, J]
            psi[t, J] = J
        else:
            novo[J] = prev[j_best] + w_sil + custo[t, J]
            psi[t, J] = j_best
        delta = novo
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmin(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    path[path == J] = -1
    return path, delta


def score_caminho(custo, path, passo_cents, max_salto):
    """Custo do caminho decodificado pela regra do contrato §4, acumulado na mesma ordem
    do Viterbi (delta_t = (delta_{t-1} + transição) + emissão), com as constantes de
    `motorsound.rotacao`."""
    J = custo.shape[1] - 1
    k_oct = int(round(1200 / passo_cents))
    st = np.where(path < 0, J, path)
    acc = float(custo[0, st[0]])
    for t in range(1, len(st)):
        a, b = int(st[t - 1]), int(st[t])
        if a == J and b == J:
            c = 0.0
        elif a == J or b == J:
            c = VIT_W_SIL
        else:
            d = abs(b - a)
            if d <= max_salto:
                c = VIT_W_STEP * d * passo_cents / 100.0
            elif d == k_oct:
                c = VIT_W_OCTAVE
            else:
                raise AssertionError(f"transição impossível no caminho: {a}->{b}")
        acc = acc + c
        acc = acc + float(custo[t, b])
    return acc


# ------------------------------------------------------------------ envelope
def _envelope_etapas(y, sr):
    """Mesmas linhas de `_decifrae._envelope_mod`, guardando os intermediários."""
    env_abs = np.abs(ss.hilbert(y))
    sos = ss.butter(4, 200, fs=sr, output="sos")
    env_dec = ss.sosfilt(sos, env_abs)[:: sr // 1000]          # 1 kHz, sem média
    env = env_dec - env_dec.mean()
    f, P = ss.welch(env, 1000, nperseg=min(len(env), 1024))
    m = (f >= 2) & (f <= 100)
    if not np.any(m):
        fm, prom = 0.0, 0.0
    else:
        i = np.argmax(P[m])
        fm = float(f[m][i])
        prom = float(10 * np.log10(P[m][i] / (np.median(P[m]) + 1e-20) + 1e-12))
    return env_abs, env_dec, f, P, (fm, prom)


# ------------------------------------------------------------------ features
def _features(y, sr, pre, f_rot, e_norm, f_grade, S, f, env_mod):
    """Mesmas linhas de `_decifrae.extrair` a partir dos intermediários já calculados."""
    N_HARM, GRUPOS = _d.N_HARM, _d.GRUPOS
    vivos = np.isfinite(f_rot)
    fr = f_rot[vivos] if np.any(vivos) else np.array([np.nan])
    f_med = float(np.nanmedian(fr)) if np.any(vivos) else 0.0
    iqr = float(np.subtract(*np.nanpercentile(fr, [75, 25]))) if np.any(vivos) else 0.0
    j = np.searchsorted(f_grade, np.where(vivos, f_rot, f_grade[0]))
    j = np.clip(j, 0, len(f_grade) - 1)
    sal = float(np.mean(np.where(vivos, e_norm[np.arange(len(j)), j], 0.0)))
    P = (S ** 2).mean(axis=1)
    df = f[1] - f[0]
    banda = (f >= 10) & (f <= 4000)
    tot = P[banda].sum() + 1e-20
    if f_med > 0:
        ordens = np.arange(1, N_HARM + 1)
        pos = np.round(ordens * f_med / df).astype(int)
        ok = pos < len(P) - 1
        e_ord = np.zeros(N_HARM)
        e_ord[ok] = P[pos[ok] - 1] + P[pos[ok]] + P[pos[ok] + 1]
    else:
        e_ord = np.zeros(N_HARM)
    razao = float(e_ord.sum() / tot)
    grupos = [10 * np.log10(e_ord[a - 1:b].sum() / tot + 1e-10) for a, b in GRUPOS]
    fm, prom = env_mod
    cent = float((f * P).sum() / (P.sum() + 1e-20))
    plan = float(np.exp(np.mean(np.log(P[banda] + 1e-20))) / (P[banda].mean() + 1e-20))
    x = np.array([f_med, iqr, 1 - vivos.mean(), sal, razao, *grupos, fm, prom, cent, plan],
                 dtype=np.float64)
    return x, P


def _iguais(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


# ------------------------------------------------------------------ principal
def etapas_p11(seg, modelo, aval, pre=PRE, post=POST, sr=None):
    """Roda a p11 de referência sobre UM segmento (float32, antes de `_preparar`) e devolve
    {etapa: ndarray} com todas as etapas do contrato §1, mais `checagens` (dict de bools).
    `modelo` é o objeto devolvido por `_decifrae.treinar` (clf, C)."""
    sr = sr or pre.sr
    harm = np.arange(1, _d.N_HARM + 1)
    seg = np.asarray(seg, dtype=np.float32)
    y = runner._preparar(seg, aval)                                    # y_prep

    f_grade = grade(pre.fmin, pre.fmax, pre.passo_cents)
    S, freqs = stft_mag(y, sr, pre.n_fft, pre.hop)
    sal, e_norm = saliencia_harmonica(S, freqs, f_grade, harm)
    e_q = sal.max(axis=1)
    custo = emissao_baixo(e_norm, f_grade, None, None, e_q, e_q.max())
    path = viterbi_rotacao(custo, pre.passo_cents, max_salto=pre.max_salto)
    f_rot = np.where(path >= 0, f_grade[np.clip(path, 0, None)], np.nan)

    chk = {}
    # rastrear da referência (mesmo caminho que _decifrae.extrair usa)
    f_rot_ref, e_norm_ref, grade_ref = rastrear(
        y, sr, pre.fmin, pre.fmax, pre.passo_cents, harm, pre.n_fft, pre.hop,
        usar_pyin=pre.usar_pyin, usar_viterbi=pre.usar_viterbi, max_salto=pre.max_salto)
    chk["f_rot==rastrear"] = _iguais(f_rot, f_rot_ref)
    chk["e_norm==rastrear"] = _iguais(e_norm, e_norm_ref)
    chk["grade==rastrear"] = _iguais(f_grade, grade_ref)

    # score do Viterbi (§4) e prova de que é min(delta)
    path_c, delta = _viterbi_com_delta(custo, pre.passo_cents, max_salto=pre.max_salto)
    chk["vit_path==copia"] = _iguais(path, path_c)
    score = score_caminho(custo, path, pre.passo_cents, pre.max_salto)
    chk["vit_score==min(delta)_exato"] = bool(score == float(delta.min()))
    chk["vit_score~min(delta)_1e-12"] = bool(np.isclose(score, delta.min(), rtol=1e-12, atol=0))

    # envelope
    env_abs, env_dec, welch_f, welch_P, env_mod = _envelope_etapas(y, sr)
    chk["env_mod==_envelope_mod"] = tuple(env_mod) == tuple(_d._envelope_mod(y, sr))

    # features
    x, P_mean = _features(y, sr, pre, f_rot, e_norm, f_grade, S, freqs, env_mod)
    x_ref = _d.extrair(y, sr, pre)
    chk["features==_d.extrair"] = _iguais(x, x_ref)

    # classificador
    clf, C = modelo
    X = x_ref[None, :]
    z = (np.asarray(X, dtype=float) - clf.mu) / clf.sd
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    cos = zn @ clf.M.T
    chk["cos==clf.emissao"] = _iguais(cos, clf.emissao(X))
    e = cos / max(clf.temp, 1e-6)
    e = np.exp(e - e.max(axis=1, keepdims=True))
    pp = e / e.sum(axis=1, keepdims=True)
    chk["softmax==clf.predict_proba"] = _iguais(pp, clf.predict_proba(X))
    probs = np.asarray(_d.prever(modelo, X, post), dtype=float)
    Pm = np.zeros((1, C))
    Pm[:, clf.classes] = pp
    chk["probs==_d.prever"] = _iguais(Pm, probs)

    out = {
        "y_prep": y, "stft_S": S, "grade": f_grade, "sal": sal, "e_norm": e_norm,
        "custo": custo, "vit_path": path.astype(np.int32), "vit_score": np.array([score]),
        "f_rot": f_rot, "P_mean": P_mean, "env_abs": env_abs, "env_dec": env_dec,
        "welch_f": welch_f, "welch_P": welch_P, "env_mod": np.array(env_mod, dtype=float),
        "features": x_ref, "z": z[0], "cos": cos[0], "probs": probs[0],
    }
    return out, chk


# ------------------------------------------------------------------ HMM
def filtro_online_alpha(emissoes, nomes, change_penalty=2.0, temp=1.0, n_logprob=-2.5,
                        force_n=None, emissao_eh_prob=True):
    """Mesmas linhas de `hmm.filtro_online`, devolvendo também alpha (T, S) normalizado
    em log. O assert contra a referência fica em quem chama."""
    log_em = hmm._log_emissoes(emissoes, temp, n_logprob, force_n, emissao_eh_prob)
    T, S = log_em.shape
    nomes_est = list(nomes) + [hmm.ESTADO_N]
    log_T = np.full((S, S), -change_penalty)
    np.fill_diagonal(log_T, 0.0)
    alphas = np.empty((T, S))
    alpha = log_em[0] - np.logaddexp.reduce(log_em[0])
    alphas[0] = alpha
    out = [nomes_est[int(np.argmax(alpha))]]
    for t in range(1, T):
        a = np.logaddexp.reduce(alpha[:, None] + log_T, axis=0) + log_em[t]
        alpha = a - np.logaddexp.reduce(a)
        alphas[t] = alpha
        out.append(nomes_est[int(np.argmax(alpha))])
    return out, alphas
