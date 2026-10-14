"""p04b_msaf — seleção de bins por diferença de espectros médios + média mais próxima (Manhattan).

Referência: docs/literatura/glowacz2018.md (Glowacz, Electronics 7(11):299, 2018; texto completo lido).
TIPO "inspirada": o método MSAF-15-MULTIEXPANDED-8-GROUPS não é reproduzível nem na base original
(ver ficha §7), e a adaptação abaixo muda a taxa e a regra final de seleção.

Seguido do artigo:
- espectro de magnitude de trechos de 1 s, resolução fina (FFT de 16 384 pontos);
- espectro médio por classe calculado SÓ no treino (aqui: dentro de `treinar`, logo dentro de
  cada dobra do runner; nada da validação entra na seleção);
- diferença absoluta entre os espectros médios de cada par de classes e escolha das maiores
  componentes (MSAF-15: as 15 maiores de cada par);
- vetor de features = amplitudes do espectro normalizado nos bins escolhidos;
- classificador NM (média mais próxima) com distância de Manhattan.

Suposto / adaptado (declarado):
- taxa de 16 kHz (não 44,1 kHz): o PRE é único para as duas bases e a A2212 é nativa a 16 kHz.
  Trecho de 1 s = 16 000 amostras, janela de Hann (o artigo não diz a janela), zero-padding até
  16 384 pontos → 8 193 bins de 0,977 Hz (o artigo: 32 768 pontos a 44,1 kHz, ≈1,35 Hz);
- normalização do espectro: divisão pela soma das magnitudes (L1). O artigo normaliza a amplitude
  do sinal e fala em "espectro normalizado" sem dizer a norma; o runner já normaliza o RMS;
- 1 s vs 3 s: na A2212 (janela de 1 s) é um único espectro. No IDMT (janela de 3 s) a janela é
  cortada em 3 trechos de 1 s sem sobreposição e os 3 espectros de magnitude são MEDIADOS
  (média aritmética), dando a mesma forma (8 193 bins) e a mesma resolução nos dois casos.
  Regra geral: floor(duração/1 s) trechos; janela < 1 s recebe zero-padding;
- número de bins K = 28 (o número final de frequências do artigo). Com 3 classes há 3 pares; os
  K bins são escolhidos por rodízio entre os pares, cada par contribuindo com seu próximo bin de
  maior diferença ainda não escolhido, até K. Isso substitui os passos "6 conjuntos de treino +
  limiar TCoF-TS + 8 grupos", cujas regras são ambíguas no texto. Faixa de busca: 10 Hz–sr/2;
- pseudo-probabilidade: softmax(−d / (temp·s)), com d a distância de Manhattan a cada protótipo
  e s a distância média de Manhattan de cada amostra de treino ao próprio protótipo
  (escala fixa no treino). Classe ausente no treino recebe 0.
O modelo guarda `bins` e `freqs_hz` para medir a estabilidade da seleção entre dobras
(teste crítico da ficha §8).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from motorsound.config import PostConfig, PreConfig

NOME = "p04b_msaf"
ORIGEM = "glowacz2018"
TIPO = "inspirada"


@dataclass(frozen=True)
class PreMSAF(PreConfig):
    sr: int = 16000
    trecho_s: float = 1.0
    n_fft: int = 16384


@dataclass(frozen=True)
class PostMSAF(PostConfig):
    k_bins: int = 28
    fmin_busca: float = 10.0
    temp_nm: float = 1.0


PRE = PreMSAF()
POST = PostMSAF()


def extrair(y, sr, pre):
    y = np.asarray(y, dtype=np.float64)
    L = int(round(pre.trecho_s * sr))
    n = max(1, len(y) // L)
    w = np.hanning(L)
    esp = np.zeros(pre.n_fft // 2 + 1)
    for i in range(n):
        tr = y[i * L:(i + 1) * L]
        if len(tr) < L:
            tr = np.pad(tr, (0, L - len(tr)))
        esp += np.abs(np.fft.rfft(tr * w, n=pre.n_fft))
    esp /= n
    esp /= esp.sum() + 1e-12
    return esp.astype(np.float32)


def _selecionar(medias, freqs, k, fmin):
    """Rodízio entre pares de classes, pegando os bins de maior |diferença| ainda livres."""
    validos = freqs >= fmin
    ordens = []
    for a, b in combinations(range(len(medias)), 2):
        d = np.abs(medias[a] - medias[b])
        d[~validos] = -np.inf
        ordens.append(list(np.argsort(-d)))
    escolhidos, pos = [], [0] * len(ordens)
    while len(escolhidos) < k and ordens:
        for j, o in enumerate(ordens):
            while pos[j] < len(o) and o[pos[j]] in escolhidos:
                pos[j] += 1
            if pos[j] < len(o) and len(escolhidos) < k:
                escolhidos.append(o[pos[j]])
                pos[j] += 1
    return np.array(sorted(escolhidos), dtype=int)


def treinar(X, y, post, n_classes):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    medias = np.stack([X[y == c].mean(0) for c in classes])
    freqs = np.fft.rfftfreq(PRE.n_fft, 1.0 / PRE.sr)[: X.shape[1]]
    if len(classes) < 2:
        bins = np.argsort(-medias[0])[: post.k_bins]
    else:
        bins = _selecionar(medias, freqs, post.k_bins, post.fmin_busca)
    prot = medias[:, bins]
    d_own = np.abs(X[:, bins] - prot[np.searchsorted(classes, y)]).sum(1)
    escala = float(d_own.mean()) + 1e-12
    return {"classes": classes, "bins": bins, "freqs_hz": freqs[bins], "prot": prot,
            "escala": escala, "C": n_classes}


def prever(modelo, X, post):
    Z = np.asarray(X, dtype=np.float64)[:, modelo["bins"]]
    d = np.abs(Z[:, None, :] - modelo["prot"][None]).sum(2)          # (n, C_treino)
    s = -d / (post.temp_nm * modelo["escala"])
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(1, keepdims=True)
    P = np.zeros((len(Z), modelo["C"]))
    P[:, modelo["classes"]] = p
    return P


def custo_mcu():
    N = PRE.n_fft
    k, C = POST.k_bins, 3
    # FFT real de 16 384 pts por segundo de áudio + normalização L1 + C·K distâncias.
    mflops = (2.5 * N * np.log2(N) + 3 * N + 2 * k * C) / PRE.trecho_s / 1e6
    ram_kb = (N * 4 + (N // 2 + 1) * 4) / 1024          # buffer real float32 + magnitudes
    params = k * C + k + 1                                # protótipos + índices dos bins + escala
    return {"params": params, "ram_kb": ram_kb, "flash_kb": params * 4 / 1024,
            "mflops_s": float(mflops),
            "nota": "estimativa analítica; RAM dominada pela FFT de 16 k pontos (~96 KB, cabe na "
                    "SRAM do ESP32 sem folga); alternativa: Goertzel só nos 28 bins (~28·16k MAC/s, "
                    "RAM < 1 KB, mas perde a normalização L1 sobre o espectro inteiro)"}
