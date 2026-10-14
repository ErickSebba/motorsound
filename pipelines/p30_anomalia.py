"""P30: detecção de anomalia treinada só no estado normal — Mahalanobis com encolhimento.

Referência: docs/literatura/harada2023.md §8 (variante leve sugerida lá) e
docs/literatura/_tabela_grupoC.md; protocolo de limiar e métricas de koizumi2020.md §8.
TIPO "inspirada": harada2023 aplica a Mahalanobis ao resíduo de um AE em 640 dim com dois
domínios; aqui é uma gaussiana única sobre ~32 features fixas, sem rede (a variante que a
própria ficha propõe como embarcável).

Seguido:
  * treino só com o estado normal (classe 0: `good` no IDMT, `Healthy` no A2212);
  * escore d²_M(x) = (x − μ)ᵀ Σ⁻¹ (x − μ), Σ com encolhimento Ledoit-Wolf;
  * limiar no percentil 95 de escores normais (fixa FPR ≈ 5 % sem olhar anomalias).

Suposições (não estão nas fichas):
  * features: log-mel de 16 bandas (n_fft 1024, hop 512, 16 kHz, 50 Hz–8 kHz), média e
    desvio-padrão por janela → 32 dim. A ficha cita "≈ 32 features de p04_classicas", que
    não existe no repositório; escolhi log-mel para conversar com o AE (p30b);
  * padronização por dimensão (média/desvio do treino normal) antes do Ledoit-Wolf, para o
    encolhimento não ser dominado pela escala de cada feature;
  * calibração: os escores normais de referência são fora da amostra, de uma validação
    5-fold em blocos CONTÍGUOS das linhas normais (o runner entrega as linhas na ordem das
    gravações, então blocos contíguos aproximam o agrupamento por gravação). O modelo final
    é ajustado em todas as linhas normais;
  * probabilidade: a = sigmoide((s − p95)/escala + logit((C−1)/C)), escala = p95 − p50 dos
    escores de referência. O deslocamento faz a = (C−1)/C exatamente em p95, de modo que o
    argmax da saída (coluna 0 = 1 − a, a repartida igualmente entre as C−1 demais) dispara
    "anômalo" no limiar do percentil 95. Quando dispara, a classe prevista é sempre a
    coluna 1 (empate resolvido pelo argmax): a pipeline NÃO diz qual falha é.

Ressalvas que valem para toda tabela desta pipeline:
  * IDMT: cada classe é um motor físico diferente (engine1/2/3) e o train/ tem um único
    `pure.wav` por classe; o detector treinado em `good` separa `broken`/`heavyload` também
    pela identidade do motor e pela sessão de gravação, não só pelo estado;
  * A2212: rolamento só existe no motor M4, que não tem gravação saudável; Healthy × Bearing
    está confundido com a identidade do motor. O único teste limpo é Healthy × Propeller
    restrito a M1 e M2 (experimentos/anomalia.py, cenário `a2212_m1m2`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

from motorsound.config import PostConfig, PreConfig
from motorsound.rotacao import stft_mag

NOME = "p30_anomalia"
ORIGEM = "harada2023"
TIPO = "inspirada"


@dataclass(frozen=True)
class Pre(PreConfig):
    sr: int = 16000
    n_fft: int = 1024
    hop: int = 512
    n_mels: int = 16
    fmin: float = 50.0


@dataclass(frozen=True)
class Post(PostConfig):
    percentil: float = 95.0
    k_calib: int = 5            # blocos contíguos para os escores normais fora da amostra


PRE = Pre()
POST = Post()


def _banco_mel(sr, n_fft, n_mels, fmin):
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin,
                               fmax=sr / 2).astype(np.float32)


_BANCOS: dict = {}


def logmel(y, sr, n_fft, hop, n_mels, fmin):
    """Log-mel (T, n_mels) em dB de potência."""
    chave = (sr, n_fft, n_mels, fmin)
    if chave not in _BANCOS:
        _BANCOS[chave] = _banco_mel(sr, n_fft, n_mels, fmin)
    S, _ = stft_mag(y, sr, n_fft, hop)                 # (F, T)
    M = _BANCOS[chave] @ (S ** 2)                      # (n_mels, T)
    return (10.0 * np.log10(M + 1e-10)).T.astype(np.float32)


def extrair(y, sr, pre):
    L = logmel(y, sr, pre.n_fft, pre.hop, pre.n_mels, pre.fmin)
    return np.concatenate([L.mean(axis=0), L.std(axis=0)]).astype(np.float32)


# ---------------------------------------------------------------- modelo
def _ajustar(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Z = (X - mu) / sd
    lw = LedoitWolf().fit(Z)
    return {"mu": mu, "sd": sd, "centro": lw.location_, "prec": lw.precision_,
            "encolhimento": float(lw.shrinkage_)}


def _d2(m, X):
    Z = (np.asarray(X, dtype=np.float64) - m["mu"]) / m["sd"] - m["centro"]
    return np.einsum("ij,jk,ik->i", Z, m["prec"], Z)


def _calibracao(escores_ref, percentil, n_classes):
    p_lim = float(np.percentile(escores_ref, percentil))
    p50 = float(np.percentile(escores_ref, 50))
    escala = max(p_lim - p50, 1e-6 * max(abs(p_lim), 1.0))
    alvo = (n_classes - 1) / n_classes if n_classes > 1 else 0.5
    return {"limiar": p_lim, "escala": escala, "desloc": float(np.log(alvo / (1 - alvo)))}


def a_calibrado(cal, s):
    z = (np.asarray(s, dtype=float) - cal["limiar"]) / cal["escala"] + cal["desloc"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def probs_de(a, n_classes):
    """Coluna 0 = 1 − a; a repartida igualmente entre as demais."""
    P = np.empty((len(a), n_classes))
    P[:, 0] = 1.0 - a
    if n_classes > 1:
        P[:, 1:] = (a / (n_classes - 1))[:, None]
    return P


def escores_fora_da_amostra(X, k, ajustar, pontuar):
    """Escores de cada linha por um modelo que não a viu (k blocos contíguos)."""
    n = len(X)
    k = max(2, min(k, n // 2))
    bordas = np.linspace(0, n, k + 1).astype(int)
    s = np.empty(n)
    for a, b in zip(bordas[:-1], bordas[1:]):
        tr = np.r_[0:a, b:n]
        s[a:b] = pontuar(ajustar(X[tr]), X[a:b])
    return s


def treinar(X, rotulos, post, n_classes):
    X = np.asarray(X, dtype=np.float64)
    Xn = X[np.asarray(rotulos) == 0]
    if len(Xn) < 4:
        raise ValueError("poucas janelas normais para treinar o detector")
    ref = escores_fora_da_amostra(Xn, post.k_calib, _ajustar, _d2)
    m = _ajustar(Xn)
    m["cal"] = _calibracao(ref, post.percentil, n_classes)
    m["n_classes"] = n_classes
    m["escores_ref"] = ref
    return m


def escore(modelo, X):
    """Escore bruto de anomalia (d² de Mahalanobis)."""
    return _d2(modelo, X)


def prever(modelo, X, post):
    return probs_de(a_calibrado(modelo["cal"], escore(modelo, X)), modelo["n_classes"])


def custo_mcu():
    D = 2 * PRE.n_mels
    nf = PRE.n_fft // 2 + 1
    quadros_s = PRE.sr / PRE.hop
    # FFT ~5·N·log2N; banco mel esparso ≈ 2·nf MAC; acumulação média/desvio; d² por janela
    mflops = quadros_s * (5 * PRE.n_fft * np.log2(PRE.n_fft) + 2 * 2 * nf + 4 * PRE.n_mels) / 1e6
    params = D * D + 3 * D + 2           # Σ⁻¹, μ, σ, centro, limiar/escala
    return {"params": int(params), "ram_kb": round((PRE.n_fft * 4 * 2 + D * 4 * 2) / 1024, 1),
            "flash_kb": round(params * 4 / 1024, 1), "mflops_s": round(float(mflops), 2),
            "nota": "estimativa analítica; inv(Sigma) 32x32 em float32 = 4 kB; "
                    "d2 de Mahalanobis ~2 kFLOP por janela"}
