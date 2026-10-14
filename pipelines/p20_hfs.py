"""p20_hfs: Harmonic Feature Space (HAR / HARH) normalizado pela rotação estimada + ajuste de
condição de operação + padronização, PCA e kNN.

Referência: docs/literatura/wheat2025.md (Wheat, Mohrenschildt, Habibi, Al-Ani, arXiv:2504.10661,
2025), Algoritmos 1 e 2 e §3.3; especificação em §7(c) da ficha. TIPO = "inspirada".

Seguido do artigo
  - HFS (Alg. 1): N = round(sr·d/f_o), janela Blackman, c = N/Σw², média removida DEPOIS da
    janela, HARH com h = |hilbert(z)| (HAR pula esse passo), s = c·|rfft|²/(N/2), segmentos sem
    sobreposição; bins 1 … d·H_max (d = 4, H_max = 60 → 240 features), DC descartado, dB.
  - Ajuste de condição (Alg. 2): regressão por mínimos quadrados de cada bin em monômios do
    regressor, ajustada só na classe saudável, intercepto zerado, predição subtraída de todas as
    linhas.
  - Classificador: StandardScaler → PCA(2) → kNN, probabilidade = fração de vizinhos.

Desvios (obrigatórios ou por falta de informação)
  1. f_o NÃO é o setpoint: vem do rastreador do projeto (`motorsound.rotacao.rastrear`, mesma
     configuração da p11: soma harmônica de 48 ordens, Viterbi, grade 15–60 Hz a 10 cents), um
     f_o por janela de avaliação = mediana de f_rot nos quadros não-SIL. Se todos os quadros
     forem SIL, usa o argmax da saliência média na grade (registrado na coluna 2 como
     fração SIL = 1).
  2. A carga NÃO é regressor (no IDMT, heavyload é classe: usar carga vazaria o rótulo).
     Regressor = [1, f_o, f_o²] (f_o em centenas de Hz só por condicionamento numérico; não muda
     a predição). Classe saudável = índice 0 (`good` no IDMT, `Healthy` no A2212). Sem pelo
     menos 3 janelas saudáveis no treino, o ajuste é pulado.
     Quando f_o quase não varia dentro da classe saudável (caso do IDMT: um motor, uma
     rotação), a regressão é degenerada; usa-se a solução de norma mínima do lstsq e o ajuste
     vira extrapolação para as outras classes, o que viola a suposição de interpolação do
     artigo. Isso é o comportamento do método, não um defeito a corrigir.
  3. Cada segmento HFS é uma amostra no artigo; aqui a unidade é a janela do runner (3 s IDMT,
     1 s A2212), então o espectro HFS é a MÉDIA de potência dos segmentos da janela (Welch sem
     sobreposição) antes do dB. Suposição nossa.
  4. Sem o passa-baixa Butterworth de fase zero (corte não encontrado; com sr = 16 kHz e
     H_max·f_o ≤ 60·60 = 3,6 kHz não há aliasing a temer). Sem o corte "N_min/2": como
     K = 240 < N/2 para todo f_o ≤ 60 Hz, todas as janelas têm as mesmas 240 features.
  5. k* do kNN: o artigo usa LOO por segmento (vazamento aqui). `treinar` não recebe os ids de
     gravação, então k* sai de validação cruzada INTERNA por blocos contíguos estratificados:
     as linhas de cada classe (que chegam em ordem gravação → tempo) são cortadas em
     `post.k_folds` blocos contíguos, e o bloco j de cada classe forma o fold interno j.
     Isso aproxima o agrupamento por gravação/bloco temporal (vazamento residual só nas
     fronteiras dos blocos). Critério: acurácia balanceada; empate → k maior; k ímpar até
     min(post.k_max, menor classe no treino interno). A validação externa nunca é vista.
  6. Variantes cacheadas juntas: `extrair` devolve f_o, IQR(f_rot), fração SIL, HAR(240) e
     HARH(240) (483 colunas); POST escolhe a variante (`variante`, `ajuste`, `n_pca`,
     `com_fo`), de modo que os controles (i)–(iii) da ficha saem do mesmo cache. O controle (iv)
     "só f_o" é a pipeline p20b_so_frot. Padrão = HARH + ajuste + PCA(2), a melhor
     configuração do artigo.

Achado do rastreador que afeta a normalização por ordem (registrado, não corrigido): no IDMT a
família harmônica forte é múltipla de 102,8 Hz; em `good` o rastreador às vezes escolhe
34,3 Hz (102,8/3) em vez de 25,7 Hz (102,8/4), ambiguidade 3:4. Com d = 4, a componente de
102,8 Hz cai no bin 16 (ordem 4) num caso e no bin 12 (ordem 3) no outro, ou seja, janelas da
mesma classe ficam com o eixo de ordens desalinhado. O HFS amplifica qualquer erro de f_o.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as ss
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from motorsound.config import PostConfig
from motorsound.rotacao import rastrear
from pipelines._decifrae import N_HARM, PreDecifrae
from pipelines._decifrae import custo_mcu as _custo_rastreador

NOME = "p20_hfs"
ORIGEM = "wheat2025"
TIPO = "inspirada"

N_CAB = 3          # colunas de cabeçalho: f_o, IQR, fração SIL


@dataclass(frozen=True)
class PreHFS(PreDecifrae):
    d: int = 4            # bins por ordem
    h_max: int = 60       # ordens mantidas


@dataclass(frozen=True)
class PostHFS(PostConfig):
    variante: str = "harh"     # "har" | "harh" | "ambos"
    ajuste: bool = True        # Alg. 2 com regressor [1, f_o, f_o²]
    n_pca: int = 2             # 0 = sem PCA
    com_fo: bool = False       # controle (iii): f_o e IQR como features extras
    k_folds: int = 5           # folds internos para k*
    k_max: int = 25


PRE = PreHFS()
POST = PostHFS()


# ---------------------------------------------------------------- extração
def fo_janela(y, sr, pre):
    """(f_o, IQR de f_rot, fração SIL) da janela, pelo rastreador do projeto."""
    f_rot, e_norm, grade = rastrear(y, sr, pre.fmin, pre.fmax, pre.passo_cents,
                                    np.arange(1, N_HARM + 1), pre.n_fft, pre.hop,
                                    usar_pyin=pre.usar_pyin, usar_viterbi=pre.usar_viterbi,
                                    max_salto=pre.max_salto)
    vivos = np.isfinite(f_rot)
    if np.any(vivos):
        fr = f_rot[vivos]
        return (float(np.median(fr)), float(np.subtract(*np.percentile(fr, [75, 25]))),
                float(1 - vivos.mean()))
    # nenhum quadro vivo: pico da saliência média
    return float(grade[int(np.argmax(np.asarray(e_norm).mean(axis=0)))]), 0.0, 1.0


def hfs(y, sr, f_o, d, h_max, hilbert):
    """HFS médio da janela em dB (K = d·h_max bins, bin k ↔ ordem k/d)."""
    K = d * h_max
    N = int(round(sr * d / f_o))
    n_seg = len(y) // N
    if n_seg == 0:                       # janela menor que um segmento: um segmento com zeros
        y = np.pad(y, (0, N - len(y)))
        n_seg = 1
    w = np.blackman(N)
    c = N / np.sum(w ** 2)
    z = y[: n_seg * N].reshape(n_seg, N).astype(np.float64) * w
    z = z - z.mean(axis=1, keepdims=True)
    if hilbert:
        z = np.abs(ss.hilbert(z, axis=1))
    s = c * np.abs(np.fft.rfft(z, axis=1)) ** 2 / (N / 2)
    s = s.mean(axis=0)[1: K + 1]
    if len(s) < K:                       # não ocorre com f_o ≤ 60 Hz; guarda
        s = np.pad(s, (0, K - len(s)), constant_values=1e-12)
    return 10 * np.log10(s + 1e-12)


def extrair(y, sr, pre: PreHFS):
    f_o, iqr, sil = fo_janela(y, sr, pre)
    har = hfs(y, sr, f_o, pre.d, pre.h_max, hilbert=False)
    harh = hfs(y, sr, f_o, pre.d, pre.h_max, hilbert=True)
    return np.concatenate([[f_o, iqr, sil], har, harh])


# ---------------------------------------------------------------- decisão
def _colunas(X, variante, K):
    a, b = N_CAB, N_CAB + K
    if variante == "har":
        return X[:, a:b]
    if variante == "harh":
        return X[:, b:b + K]
    if variante == "ambos":
        return X[:, a:b + K]
    raise ValueError(variante)


def _regressor(fo):
    f = np.asarray(fo, dtype=float) / 100.0
    return np.stack([np.ones_like(f), f, f ** 2], axis=1)


def _blocos_estratificados(y, n_folds):
    """Fold interno j = j-ésimo bloco contíguo de cada classe (linhas em ordem gravação→tempo)."""
    fold = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        for j, parte in enumerate(np.array_split(idx, n_folds)):
            fold[parte] = j
    return fold


def k_estrela(Z, y, n_folds=5, k_max=25):
    """k do kNN por validação interna em blocos contíguos estratificados (só treino)."""
    fold = _blocos_estratificados(y, n_folds)
    n_min = min(np.sum(y == c) for c in np.unique(y))
    lim = max(1, min(k_max, int(n_min * (n_folds - 1) / n_folds)))
    candidatos = list(range(1, lim + 1, 2))
    erros = np.zeros(len(candidatos))
    for j in range(n_folds):
        tr, va = fold != j, fold == j
        if va.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        for i, k in enumerate(candidatos):
            kk = min(k, int(tr.sum()))
            p = KNeighborsClassifier(n_neighbors=kk).fit(Z[tr], y[tr]).predict(Z[va])
            erros[i] += (1 - balanced_accuracy_score(y[va], p)) * va.sum()
    # empate → k maior
    i = len(candidatos) - 1 - int(np.argmin(erros[::-1]))
    return candidatos[i]


def preparar(X, post, A=None):
    """Seleciona a variante, aplica o ajuste de condição e o controle (iii)."""
    K = (X.shape[1] - N_CAB) // 2
    F = _colunas(X, post.variante, K)
    if A is not None:
        F = F - _regressor(X[:, 0]) @ A
    if post.com_fo:
        F = np.hstack([F, X[:, :2]])
    return F


def treinar(X, y, post: PostHFS = POST, n_classes=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    A = None
    if post.ajuste and np.sum(y == 0) >= 3:
        K = (X.shape[1] - N_CAB) // 2
        F0 = _colunas(X[y == 0], post.variante, K)
        coef, *_ = np.linalg.lstsq(_regressor(X[y == 0, 0]), F0, rcond=None)
        coef[0] = 0.0                    # intercepto zerado: preserva o nível médio
        A = coef
    F = preparar(X, post, A)
    sc = StandardScaler().fit(F)
    Z = sc.transform(F)
    pca = None
    if post.n_pca and post.n_pca < Z.shape[1]:
        pca = PCA(n_components=post.n_pca, random_state=post.seed).fit(Z)
        Z = pca.transform(Z)
    k = k_estrela(Z, y, post.k_folds, post.k_max)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(y))).fit(Z, y)
    return {"A": A, "sc": sc, "pca": pca, "knn": knn, "k": k, "C": n_classes}


def prever(modelo, X, post: PostHFS = POST):
    X = np.asarray(X, dtype=float)
    Z = modelo["sc"].transform(preparar(X, post, modelo["A"]))
    if modelo["pca"] is not None:
        Z = modelo["pca"].transform(Z)
    knn = modelo["knn"]
    P = np.zeros((len(X), modelo["C"]))
    P[:, knn.classes_] = knn.predict_proba(Z)
    return P


def custo_mcu():
    """Rastreador (como p11) + HFS por segmento + ajuste + PCA + kNN em 2-D."""
    pre = PRE
    rast = _custo_rastreador(pre)
    K = pre.d * pre.h_max
    f_o_tip = 25.0                       # rotação típica do IDMT/limite inferior pior caso
    N = int(round(pre.sr * pre.d / f_o_tip))
    seg_s = pre.sr / N
    fft = 5 * N * np.log2(N)
    por_seg = 3 * fft if POST.variante != "har" else fft   # HARH: hilbert (fft+ifft) + rfft
    ajuste = 2 * 3 * K
    mflops = rast["mflops_s"] + seg_s * (por_seg + ajuste + 2 * K * 2) / 1e6
    ram_kb = rast["ram_kb"] + (2 * N * 4 + K * 4 + 3 * K * 4 + 2 * K * 4 + 2 * K * 4) / 1024
    return {"params": 3 * K + 2 * K + 2 * K + 2,
            "ram_kb": round(ram_kb, 1),
            "flash_kb": round((3 * K + 4 * K) * 4 / 1024, 1),
            "mflops_s": float(round(mflops, 2)),
            "nota": f"estimativa analítica; inclui o rastreador ({rast['mflops_s']} MFLOP/s); "
                    f"N={N} (f_o=25 Hz, FFT não potência de 2: no MCU, Goertzel nos K={K} bins "
                    "ou FFT fixa + interpolação); kNN guarda os pontos de treino em 2-D "
                    "(8 B por janela de treino, fora de 'params')"}
