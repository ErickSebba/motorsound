"""Núcleo comum das pipelines p00 e p01 (ficha docs/literatura/strantzalis2022.md).

Strantzalis, Gioulekas, Katsaros, Symeonidis. "Operational State Recognition of a DC Motor
Using Edge Artificial Intelligence". Sensors 22(24):9658, 2022. DOI 10.3390/s22249658.

Aqui ficam só as peças independentes do artigo: sub-janelas, laço de treino em torch (CPU,
2 threads, semente de `post.seed`), predição com média das probabilidades das sub-janelas e
zeragem de classe ausente no treino. A arquitetura e o pré-processamento de cada rota estão
nos arquivos p00 / p01, com as suposições declaradas no cabeçalho.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)


def inicios(T: int, L: int, passo: int | None) -> np.ndarray:
    """Inícios de sub-janelas de comprimento L num eixo de T quadros.

    passo=None: o mínimo de sub-janelas que cobre o eixo inteiro, espaçadas por igual
    (ceil(T/L) sub-janelas, a última alinhada ao fim). passo=k: todas as posições 0, k, 2k, …
    """
    if T < L:
        raise ValueError(f"eixo com {T} quadros < {L} exigidos")
    if passo is None:
        n = math.ceil(T / L)
        return np.round(np.linspace(0, T - L, n)).astype(int) if n > 1 else np.array([0])
    return np.arange(0, T - L + 1, passo)


def treinar_rede(rede: nn.Module, X: np.ndarray, y: np.ndarray, *, lr: float, lote: int,
                 n_passos: int | None = None, epocas: int | None = None, seed: int = 0):
    """Adam + entropia cruzada. Ou um número fixo de passos de gradiente (n_passos), ou
    `epocas` passagens completas embaralhadas. Lote = min(lote, n)."""
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    n = len(Xt)
    b = min(lote, n)
    opt = torch.optim.Adam(rede.parameters(), lr=lr)
    perda = nn.CrossEntropyLoss()
    rede.train()

    def passo(idx):
        opt.zero_grad()
        loss = perda(rede(Xt[idx]), yt[idx])
        loss.backward()
        opt.step()

    if n_passos is not None:
        ordem, pos = g.permutation(n), 0
        for _ in range(n_passos):
            if pos + b > n:
                ordem, pos = g.permutation(n), 0
            passo(torch.from_numpy(ordem[pos:pos + b]))
            pos += b
    else:
        for _ in range(epocas):
            ordem = g.permutation(n)
            for i in range(0, n, b):
                passo(torch.from_numpy(ordem[i:i + b]))
    rede.eval()
    return rede


@torch.no_grad()
def probabilidades(rede: nn.Module, X: np.ndarray, lote: int = 4096) -> np.ndarray:
    rede.eval()
    out = []
    for i in range(0, len(X), lote):
        out.append(torch.softmax(rede(torch.from_numpy(
            np.ascontiguousarray(X[i:i + lote], dtype=np.float32))), dim=1).numpy())
    return np.concatenate(out)


def media_por_janela(P_sub: np.ndarray, n_sub: int, presentes: np.ndarray) -> np.ndarray:
    """Média das probabilidades das n_sub sub-janelas de cada janela; zera classes ausentes
    no treino e renormaliza."""
    P = P_sub.reshape(-1, n_sub, P_sub.shape[1]).mean(axis=1)
    P[:, ~presentes] = 0.0
    return P / P.sum(axis=1, keepdims=True)


def n_parametros(rede: nn.Module) -> int:
    return int(sum(p.numel() for p in rede.parameters()))
