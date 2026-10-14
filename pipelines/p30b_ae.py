"""P30b: autoencoder pequeno treinado só no estado normal; escore = erro de reconstrução.

Referência: docs/literatura/purohit2019.md (arquitetura 320-64-64-8-64-64-320 e
features) e koizumi2020.md (lr, lote, validation_split, escore e protocolo de limiar).
TIPO "inspirada": as bases são outras (IDMT, A2212) e há suposições abaixo.

Seguido (purohit2019):
  * log-mel 64 bandas, n_fft 1024, hop 512, 16 kHz; 5 quadros concatenados → 320 dim;
  * AE FC(320,64,ReLU) → FC(64,64,ReLU) → FC(64,8,ReLU) → FC(8,64,ReLU) → FC(64,64,ReLU)
    → FC(64,320, linear); perda MSE; Adam; 50 épocas;
  * treino só com o estado normal (classe 0: `good` no IDMT, `Healthy` no A2212).
Seguido (koizumi2020): Adam lr 0,001; lote 512; 10 % finais das janelas normais fora do
  treino (como o `validation_split` 0,1 do Keras, que separa o fim do conjunto); escore por
  janela = média, sobre os quadros, do MSE de reconstrução; limiar no percentil 95 dos
  escores normais (FPR ≈ 5 % sem olhar anomalias).

Suposições (não estão nas fichas):
  * lr e lote em purohit2019 não são informados: usados os de koizumi2020;
  * padronização por dimensão (média/desvio dos vetores normais de treino) antes do AE;
  * a agregação do escore (média sobre quadros) é suposição em purohit2019, explícita em
    koizumi2020;
  * os 10 % finais (em janelas, contíguos, logo aproximadamente agrupados por gravação)
    servem só para calibrar o limiar (sem early stopping); se houver menos de 2 janelas
    normais para isso, usa os escores de treino;
  * probabilidade: igual a p30_anomalia (sigmoide deslocada para dar a = (C−1)/C em p95;
    coluna 0 = 1 − a; a repartida igualmente nas demais). Quando dispara, a classe
    prevista é sempre a coluna 1: a pipeline não diz qual falha é;
  * torch CPU, 2 threads, seed = post.seed.

Treino: 50 épocas, Adam lr 1e-3, lote 512, MSE.

Ressalvas que valem para toda tabela desta pipeline:
  * IDMT: cada classe é um motor físico diferente (engine1/2/3), um `pure.wav` por classe
    no train/; o AE treinado em `good` separa os outros também por identidade do motor e
    por sessão de gravação, não só pelo estado;
  * A2212: rolamento só no motor M4, sem saudável do M4; Healthy × Bearing confundido com a
    identidade do motor. Teste limpo: Healthy × Propeller em M1 e M2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motorsound.config import PostConfig, PreConfig
from pipelines.p30_anomalia import a_calibrado, logmel, probs_de, _calibracao

NOME = "p30b_ae"
ORIGEM = "purohit2019"
TIPO = "inspirada"


@dataclass(frozen=True)
class Pre(PreConfig):
    sr: int = 16000
    n_fft: int = 1024
    hop: int = 512
    n_mels: int = 64
    fmin: float = 0.0


@dataclass(frozen=True)
class Post(PostConfig):
    contexto: int = 5
    epocas: int = 50
    lr: float = 1e-3
    lote: int = 512
    frac_calib: float = 0.1
    percentil: float = 95.0


PRE = Pre()
POST = Post()


def extrair(y, sr, pre):
    """Log-mel (T, 64); T depende só do comprimento da janela."""
    return logmel(y, sr, pre.n_fft, pre.hop, pre.n_mels, pre.fmin)


def _empilhar(X, P):
    """(n, T, F) → (n, T−P+1, P·F): P quadros consecutivos concatenados."""
    X = np.asarray(X, dtype=np.float32)
    n, T, F = X.shape
    Tp = T - P + 1
    return np.concatenate([X[:, k:k + Tp, :] for k in range(P)], axis=2)


def _rede(D):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8), nn.ReLU(),
        nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, D))


def _pontuar(m, X):
    import torch
    V = _empilhar(X, m["contexto"])                    # (n, T', D)
    n, Tp, D = V.shape
    Z = ((V.reshape(-1, D) - m["mu"]) / m["sd"]).astype(np.float32)
    with torch.no_grad():
        R = m["rede"](torch.from_numpy(Z)).numpy()
    e = ((R - Z) ** 2).mean(axis=1).reshape(n, Tp)
    return e.mean(axis=1)


def treinar(X, rotulos, post, n_classes):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(post.seed)
    rng = np.random.default_rng(post.seed)
    Xn = np.asarray(X, dtype=np.float32)[np.asarray(rotulos) == 0]
    n_cal = int(round(post.frac_calib * len(Xn)))
    if n_cal < 2 or len(Xn) - n_cal < 2:
        n_cal = 0
    Xtr, Xcal = Xn[:len(Xn) - n_cal], Xn[len(Xn) - n_cal:]
    V = _empilhar(Xtr, post.contexto)
    D = V.shape[2]
    V = V.reshape(-1, D)
    mu, sd = V.mean(axis=0), V.std(axis=0) + 1e-6
    Z = torch.from_numpy(((V - mu) / sd).astype(np.float32))
    rede = _rede(D)
    opt = torch.optim.Adam(rede.parameters(), lr=post.lr)
    perda = torch.nn.MSELoss()
    rede.train()
    for _ in range(post.epocas):
        ordem = torch.from_numpy(rng.permutation(len(Z)))
        for i in range(0, len(Z), post.lote):
            zb = Z[ordem[i:i + post.lote]]
            opt.zero_grad()
            l = perda(rede(zb), zb)
            l.backward()
            opt.step()
    rede.eval()
    m = {"rede": rede, "mu": mu, "sd": sd, "contexto": post.contexto, "n_classes": n_classes}
    ref = _pontuar(m, Xcal) if n_cal else _pontuar(m, Xtr)
    m["cal"] = _calibracao(ref, post.percentil, n_classes)
    m["escores_ref"] = ref
    m["calib_fora_da_amostra"] = bool(n_cal)
    return m


def escore(modelo, X):
    """Escore bruto: MSE de reconstrução médio sobre os quadros da janela."""
    return _pontuar(modelo, X)


def prever(modelo, X, post):
    return probs_de(a_calibrado(modelo["cal"], escore(modelo, X)), modelo["n_classes"])


def custo_mcu():
    D = PRE.n_mels * POST.contexto
    camadas = [(D, 64), (64, 64), (64, 8), (8, 64), (64, 64), (64, D)]
    params = sum(i * o + o for i, o in camadas)
    macs = sum(i * o for i, o in camadas)
    quadros_s = PRE.sr / PRE.hop
    nf = PRE.n_fft // 2 + 1
    mflops = quadros_s * (5 * PRE.n_fft * np.log2(PRE.n_fft) + 2 * 2 * nf + 2 * macs) / 1e6
    ram = (PRE.n_fft * 4 * 2 + D * 4 * 3 + 64 * 4 * 2) / 1024
    return {"params": int(params), "ram_kb": round(ram, 1),
            "flash_kb": round(params / 1024, 1), "mflops_s": round(float(mflops), 2),
            "nota": "estimativa analítica; flash em int8 (×4 em float32); "
                    "AE por quadro de 32 ms com contexto de 5 quadros"}
