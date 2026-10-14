"""Auxiliar das pipelines p02 (Grollmisch 2019) e p05 (mel + CNN 2025): treino em torch na CPU.

Não é reaproveitado dos Artigos 1/2; escrito aqui. Só laço de treino e previsão em lote,
com toda aleatoriedade presa à semente do PostConfig.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)


def semear(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def glorot(modelo: nn.Module):
    """Glorot uniforme nos pesos e zero nos vieses (padrão do Keras)."""
    for m in modelo.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


def treinar_rede(modelo: nn.Module, X: np.ndarray, y: np.ndarray, epocas: int, lote: int,
                 lr: float, seed: int) -> nn.Module:
    """Adam + entropia cruzada, lotes embaralhados com gerador semeado; sem validação interna."""
    g = semear(seed)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    perda = nn.CrossEntropyLoss()
    n = len(Xt)
    modelo.train()
    for _ in range(epocas):
        ordem = torch.randperm(n, generator=g)
        for i in range(0, n, lote):
            idx = ordem[i:i + lote]
            opt.zero_grad()
            loss = perda(modelo(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    modelo.eval()
    return modelo


@torch.no_grad()
def probabilidades(modelo: nn.Module, X: np.ndarray, lote: int = 4096) -> np.ndarray:
    modelo.eval()
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    out = [torch.softmax(modelo(Xt[i:i + lote]), dim=1) for i in range(0, len(Xt), lote)]
    return torch.cat(out).numpy()


def n_params(modelo: nn.Module) -> int:
    return int(sum(p.numel() for p in modelo.parameters()))
