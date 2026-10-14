# Origem: Artigo2(Acordes)/Decifra-Violão/viterbi.py::decode e
#         pipeline.py::_merge, _absorb_short, decode (gate de RMS) (commit 93609f7)
# Mudanças: estados = estados de operação do motor em vez de acordes; a emissão
#   chega como probabilidades por janela (saída de qualquer pipeline) e não como
#   cosseno, então o softmax é opcional (`emissao_eh_prob`); laço interno do Viterbi
#   vetorizado (mesma conta); `filtro_online` acrescentado como versão causal para o
#   MCU (forward com a mesma matriz de transição, decisão por argmax do filtro).
"""HMM de estados de operação com `change_penalty` fixo e estado N.

O `change_penalty` independente do número de estados é a correção do bug do
decodificador travado do violão (CLAUDE.md, "Lições já pagas") e fica como está.
"""
from __future__ import annotations

import numpy as np

ESTADO_N = "N"


def _softmax(x, temp):
    z = (x - np.max(x)) / max(temp, 1e-6)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def _log_emissoes(emissoes, temp, n_logprob, force_n, emissao_eh_prob):
    emissoes = np.asarray(emissoes, dtype=float)
    T, C = emissoes.shape
    log_em = np.full((T, C + 1), -50.0)
    for t in range(T):
        p = emissoes[t] if emissao_eh_prob else _softmax(emissoes[t], temp)
        if emissao_eh_prob and temp != 1.0:
            # temperatura sobre probabilidades: p^(1/temp) renormalizado
            q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / max(temp, 1e-6))
            p = q / q.sum()
        log_em[t, :C] = np.log(p + 1e-12)
        log_em[t, C] = n_logprob
        if force_n is not None and force_n[t]:
            log_em[t, :] = -50.0
            log_em[t, C] = 0.0
    return log_em


def decode(emissoes, nomes, change_penalty=2.0, temp=1.0, n_logprob=-2.5,
           force_n=None, emissao_eh_prob=True):
    """Viterbi em log. Transição: 0 para ficar, -change_penalty para trocar (fixo).

    emissoes : (T, C) probabilidades (ou cossenos, com emissao_eh_prob=False)
    Devolve a lista de rótulos por janela (inclui 'N').
    """
    log_em = _log_emissoes(emissoes, temp, n_logprob, force_n, emissao_eh_prob)
    T, S = log_em.shape
    nomes_est = list(nomes) + [ESTADO_N]
    log_T = np.full((S, S), -change_penalty)
    np.fill_diagonal(log_T, 0.0)
    delta = np.empty((T, S))
    psi = np.zeros((T, S), dtype=int)
    delta[0] = log_em[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + log_T          # (de, para)
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = cand[psi[t], np.arange(S)] + log_em[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return [nomes_est[i] for i in path]


def filtro_online(emissoes, nomes, change_penalty=2.0, temp=1.0, n_logprob=-2.5,
                  force_n=None, emissao_eh_prob=True):
    """Versão causal (embarcável): filtro forward em log com a mesma transição;
    a decisão em t só usa janelas <= t."""
    log_em = _log_emissoes(emissoes, temp, n_logprob, force_n, emissao_eh_prob)
    T, S = log_em.shape
    nomes_est = list(nomes) + [ESTADO_N]
    log_T = np.full((S, S), -change_penalty)
    np.fill_diagonal(log_T, 0.0)
    alpha = log_em[0] - np.logaddexp.reduce(log_em[0])
    out = [nomes_est[int(np.argmax(alpha))]]
    for t in range(1, T):
        a = np.logaddexp.reduce(alpha[:, None] + log_T, axis=0) + log_em[t]
        alpha = a - np.logaddexp.reduce(a)
        out.append(nomes_est[int(np.argmax(alpha))])
    return out


def gate_rms(seg_rms, rel_gate=0.2, abs_floor=5e-4):
    """force_n: janela com RMS abaixo de max(abs_floor, rel_gate * mediana)."""
    seg_rms = np.asarray(seg_rms, dtype=float)
    vivos = seg_rms > abs_floor
    mediana = float(np.median(seg_rms[vivos])) if np.any(vivos) else 0.0
    return seg_rms < max(abs_floor, rel_gate * mediana)


def _merge(segmentos):
    out = []
    for s in segmentos:
        if s["end"] <= s["start"]:
            continue
        if out and out[-1]["label"] == s["label"]:
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    return out


def absorb_short(segmentos, min_dur):
    """Absorve segmentos < min_dur no vizinho mais longo (duração mínima de alarme)."""
    if min_dur <= 0:
        return segmentos
    segs = [dict(s) for s in segmentos]
    while len(segs) > 1:
        durs = [s["end"] - s["start"] for s in segs]
        idx = min(range(len(segs)), key=lambda i: durs[i])
        if durs[idx] >= min_dur:
            break
        left = segs[idx - 1] if idx > 0 else None
        right = segs[idx + 1] if idx < len(segs) - 1 else None
        if left and right:
            donor = left if (left["end"] - left["start"]) >= (right["end"] - right["start"]) else right
        else:
            donor = left or right
        segs[idx]["label"] = donor["label"]
        segs = _merge(segs)
    return segs


def rotulos_para_segmentos(rotulos, passo_s):
    return _merge([{"start": i * passo_s, "end": (i + 1) * passo_s, "label": r}
                   for i, r in enumerate(rotulos)])


def segmentos_para_rotulos(segmentos, n, passo_s):
    out = [ESTADO_N] * n
    for s in segmentos:
        i0 = int(round(s["start"] / passo_s))
        i1 = int(round(s["end"] / passo_s))
        for i in range(max(0, i0), min(n, i1)):
            out[i] = s["label"]
    return out
