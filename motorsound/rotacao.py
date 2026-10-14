# Origem: Artigo1(cross-talk)/Decifra-Baixo/fase2_v1v2v3/v3_viterbi.py::_viterbi_pitch
#         e constantes _VIT_* (commit 2c9d64c); v2_polifonico.py::_filtro_passa_banda
#         (ideia do passa-banda SOS).
# Mudanças:
#   * estados = grade logarítmica de frequência de rotação (passo em cents) em vez de
#     semitons MIDI de uma corda;
#   * emissão igual à do baixo (0,55·prox·vprob + 0,45·e_norm), com e_norm vindo de uma
#     soma harmônica sobre `harmonicos` (no baixo eram só f e 0,5·2f; aqui a energia
#     está concentrada em harmônicos altos, ver docs/DATASETS.md e a exploração de
#     14/10: o 24º harmônico domina no IDMT);
#   * custo de passo proporcional à distância em semitons, como no baixo, mas só são
#     permitidos saltos de até `max_salto` estados (O(T·S·k) em vez de O(T·S²));
#     saltos de oitava exata (±1200 cents) continuam permitidos com custo W_OCT;
#   * Viterbi vetorizado (mesma recursão);
#   * passa-banda com `sosfilt` (causal) por padrão; `sosfiltfilt` só se pedido.
"""Rastreador de rotação: grade de frequências + estado SIL + custo de oitava."""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

# Constantes herdadas do baixo V3 (mesmos nomes, mesmos valores)
VIT_SIGMA_CENTS = 70.0
VIT_W_STEP = 0.55        # por semitom
VIT_W_OCTAVE = 7.0
VIT_W_SIL = 3.2
VIT_W_EMIT = 1.0


def grade(fmin: float, fmax: float, passo_cents: float) -> np.ndarray:
    n = int(np.floor(1200 * np.log2(fmax / fmin) / passo_cents)) + 1
    return fmin * 2.0 ** (np.arange(n) * passo_cents / 1200.0)


def passa_banda(y, sr, lo, hi, ordem=4, causal=True):
    sos = ss.butter(ordem, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return ss.sosfilt(sos, y) if causal else ss.sosfiltfilt(sos, y)


def stft_mag(y, sr, n_fft, hop):
    w = np.hanning(n_fft).astype(np.float32)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    S = np.abs(np.fft.rfft(y[idx] * w, axis=1)).T          # (F, T)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return S.astype(np.float32), freqs


def saliencia_harmonica(S, freqs, f_grade, harmonicos, pesos=None):
    """Soma harmônica: sal[t, j] = Σ_k w_k · S(k·f_j, t), por interpolação linear.
    Devolve (T, J) e a matriz normalizada por quadro (e_norm do baixo)."""
    harmonicos = np.asarray(harmonicos, dtype=float)
    if pesos is None:
        pesos = np.ones_like(harmonicos)
    df = freqs[1] - freqs[0]
    sal = np.zeros((S.shape[1], len(f_grade)), dtype=np.float64)
    for h, w in zip(harmonicos, pesos):
        pos = (h * f_grade) / df
        ok = pos < len(freqs) - 1
        i0 = np.floor(pos[ok]).astype(int)
        a = (pos[ok] - i0)[None, :]
        val = (1 - a) * S[i0].T + a * S[i0 + 1].T             # (T, J_ok)
        sal[:, ok] += w * val
    norm = sal / (sal.max(axis=1, keepdims=True) + 1e-12)
    return sal, norm


def emissao_baixo(e_norm, f_grade, f0=None, vprob=None, e_quadro=None, e_ref=None):
    """Custo de emissão (T, J+1), igual ao do baixo V3. Último estado = SIL.

    e_norm  : (T, J) energia harmônica normalizada no quadro
    f0, vprob : pYIN (opcionais). Sem pYIN, prox=0 e vprob=0 e só o termo de energia
                fica (versão embarcável).
    e_quadro, e_ref : energia do quadro e a de referência (máximo da gravação) para o
                custo do SIL, como `col_max / S_max` no baixo.
    """
    T, J = e_norm.shape
    if f0 is None:
        prox = np.zeros((T, J))
        vprob = np.zeros(T)
    else:
        f0 = np.asarray(f0, dtype=float)
        tem = np.isfinite(f0) & (f0 > 0)
        cents = np.where(tem[:, None],
                         np.abs(1200 * np.log2(np.where(tem, f0, 1.0)[:, None] / f_grade[None, :])),
                         np.inf)
        prox = np.exp(-(cents / VIT_SIGMA_CENTS) ** 2)
    score = VIT_W_EMIT * (0.55 * prox * vprob[:, None] + 0.45 * e_norm)
    custo = np.empty((T, J + 1))
    custo[:, :J] = -np.log(score + 1e-3)
    if e_quadro is None:
        rel = np.ones(T)
    else:
        rel = np.asarray(e_quadro) / (e_ref + 1e-12)
    sil = (1.0 - vprob) * (1.0 - np.clip(rel, 0, 1))
    custo[:, J] = -np.log(sil + 1e-3)
    return custo


def viterbi_rotacao(custo, passo_cents, max_salto=None, w_step=VIT_W_STEP,
                    w_oct=VIT_W_OCTAVE, w_sil=VIT_W_SIL):
    """Viterbi de mínimo custo sobre (T, J+1). Devolve índices (−1 = SIL).

    Transições como no baixo: ficar = 0; passo de d semitons = w_step·d (só |Δ| ≤
    max_salto estados); ±1 oitava exata = w_oct; entrar/sair de SIL = w_sil.
    """
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
    return path


def rastrear(y, sr, fmin=15.0, fmax=60.0, passo_cents=10.0, harmonicos=None,
             n_fft=8192, hop=1024, usar_pyin=False, usar_viterbi=True, max_salto=10,
             banda_pyin=None):
    """Rastreia a rotação. Devolve (f_rot por quadro, NaN = SIL; saliência normalizada;
    grade). Sem Viterbi, é o argmax por quadro da emissão (variante p12)."""
    if harmonicos is None:
        harmonicos = np.arange(1, 49)
    f_grade = grade(fmin, fmax, passo_cents)
    S, freqs = stft_mag(y, sr, n_fft, hop)
    sal, e_norm = saliencia_harmonica(S, freqs, f_grade, harmonicos)
    e_q = sal.max(axis=1)
    f0 = vprob = None
    if usar_pyin:
        import librosa
        lo, hi = banda_pyin if banda_pyin else (fmin, fmax)
        yb = passa_banda(y, sr, lo * 0.88, min(hi * 1.08, sr / 2 - 1), causal=False)
        # max_transition_rate reduzido: com hop de 64 ms o default (35,9 oct/s) gera uma
        # janela de transição maior que a grade de 2 oitavas e o librosa recusa
        f0p, vf, vp = librosa.pyin(yb, fmin=lo, fmax=hi, sr=sr, frame_length=n_fft,
                                   hop_length=hop, center=False, max_transition_rate=8.0)
        n = min(len(f0p), S.shape[1])
        vp = np.nan_to_num(vp[:n], nan=0.0)
        vprob = np.where(vf[:n], vp, 0.3 * vp)
        f0 = f0p[:n]
        # se a banda do pYIN é a de um harmônico h, f0 é dividido por h
        if banda_pyin is not None:
            h = int(round(banda_pyin[0] / fmin)) or 1
            f0 = f0 / h
        e_norm, e_q = e_norm[:n], e_q[:n]
    custo = emissao_baixo(e_norm, f_grade, f0, vprob, e_q, e_q.max())
    if usar_viterbi:
        path = viterbi_rotacao(custo, passo_cents, max_salto=max_salto)
    else:
        path = np.argmin(custo, axis=1)
        path[path == len(f_grade)] = -1
    f_rot = np.where(path >= 0, f_grade[np.clip(path, 0, None)], np.nan)
    return f_rot, e_norm, f_grade
