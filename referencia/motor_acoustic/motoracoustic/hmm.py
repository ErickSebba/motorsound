"""HMM de estados de operação sobre as probabilidades do classificador.

Mesmo papel do Viterbi no violão do Decifrae: consolidar a decisão no tempo.
Aqui ele vira o "debounce" do alarme. Duas versões:

viterbi_states  -> offline, para o artigo (melhor caminho global)
ForwardFilter   -> online, o que roda no ESP32 (O(K^2) por janela)

Cuidado já pago no Decifrae: tudo em log, e a matriz de transição precisa
ser estocástica por LINHA, senão o decodificador trava num estado.
"""
from __future__ import annotations

import numpy as np


def sticky_transition(K: int, p_stay: float) -> np.ndarray:
    T = np.full((K, K), (1 - p_stay) / (K - 1))
    np.fill_diagonal(T, p_stay)
    assert np.allclose(T.sum(axis=1), 1.0)
    return T


def viterbi_states(post: np.ndarray, p_stay: float) -> np.ndarray:
    """post: (T, K) probabilidades por janela. Retorna rótulos (T,)."""
    Tn, K = post.shape
    logT = np.log(sticky_transition(K, p_stay))
    logE = np.log(np.clip(post, 1e-9, 1.0))
    delta = logE[0] - np.log(K)
    back = np.zeros((Tn, K), dtype=np.int32)
    for t in range(1, Tn):
        cand = delta[:, None] + logT          # (de, para)
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(K)] + logE[t]
    path = np.zeros(Tn, dtype=np.int32)
    path[-1] = int(np.argmax(delta))
    for t in range(Tn - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


class ForwardFilter:
    """Filtro HMM causal. Estado = vetor de crença (K floats)."""

    def __init__(self, K: int, p_stay: float):
        self.T = sticky_transition(K, p_stay)
        self.b = np.full(K, 1.0 / K)

    def step(self, p: np.ndarray) -> np.ndarray:
        prior = self.b @ self.T
        b = prior * np.clip(p, 1e-9, 1.0)
        self.b = b / b.sum()
        return self.b

    def run(self, post: np.ndarray) -> np.ndarray:
        return np.array([np.argmax(self.step(p)) for p in post], dtype=np.int32)
