"""P03 — ETNeXt (Celik, Taskin, Akbal et al., Sci. Rep. 16, 11820, 2026), variante Cubic SVM.

Ficha: docs/literatura/etnext2026.md. Texto completo conferido no HTML da Nature
(https://www.nature.com/articles/s41598-026-37590-z), eqs. 3–23; a Tabela 2 (pseudocódigo) é imagem
e não foi lida, então os passos de Chi2 e da escolha final do vetor vêm só do texto corrido.

TIPO = "inspirada". Motivos (cada um é suposição ou desvio declarado):

O que segue o artigo
- DWT multinível sym4, 7 níveis, coeficientes de APROXIMAÇÃO de cada nível (eqs. 19–20).
  Modo de borda 'symmetric' do pywt = 'sym' padrão do MATLAB (`dwtmode`).
- Janela deslizante de 16 amostras, passo 1 (eq. 7). Em cada janela, 8 comparações
  m = wdw(i), s = wdw(i+8), i = 1..8 (eqs. 8–9) → 8 bits → código 0..255 (eqs. 12–14, peso 2^(i-1)).
  signum: bit = 1 se m − s ≥ 0 (eq. 3). Ternária: +1 se m − s > thr, −1 se m − s < −thr, 0 no meio;
  inferior = [ternária == −1], superior = [ternária == +1] (eqs. 4–6, 10–11).
- Limiar ternário = metade do desvio-padrão do sinal (texto após a eq. 11 e parágrafo da Fig. 7).
- 3 histogramas de 256 bins (contagens) por sequência → 768 atributos (eqs. 15–18).
- Normalização min–max por coluna antes da seleção (eq. 21); NCA e Chi2 como seletores (eqs. 22–23);
  5 atributos por nível × 7 níveis = 35 por subconjunto ("5 key features per level ... 35 features").
- Cubic SVM e Fine kNN do Classification Learner (MATLAB 2023b).

Suposições e desvios
1. Contagem de níveis: as eqs. 19–20 concatenam o sinal original + 7 aproximações (8 × 768 = 6 144),
   mas o texto diz 5 376 = 7 × 768 e "5 por nível → 35". Seguimos o número (7 blocos = as 7
   aproximações, sem o sinal original). `PRE.incluir_original=True` reativa o bloco extra.
2. Eq. 4 tem erro de digitação (−1 se m − s < 0); usamos −1 se m − s < −thr (ternária usual). A eq. 6/11
   ("0 se ternarval < −1") é lida como "0 se ternarval < 1".
3. "Desvio-padrão do sinal": usamos o desvio da sequência que está sendo codificada (cada nível tem o
   seu limiar). O texto não diz se é do sinal original ou de cada nível.
4. Taxa: o artigo usa o áudio nativo do IDMT (44,1/48 kHz, contraditório no texto). Aqui PRE.sr =
   16 kHz nas duas bases (o A2212 é 16 kHz nativo e uma única PRE serve as duas). Com isso as bandas
   da DWT no IDMT ficam deslocadas em relação ao artigo (aproximação de nível 7 ≈ 0–62 Hz, contra
   ≈ 0–172 Hz a 44,1 kHz).
5. NCA de seleção (`fscnca` do MATLAB) não existe no sklearn (o NCA do sklearn aprende projeção, não
   pesos por atributo). Reimplementado aqui (`_fscnca`) seguindo Yang, Wang & Zuo (2012), que é o que o
   `fscnca` implementa: distância L1 ponderada por w², kernel exp(−D/σ) com σ = 1, λ = 1/n, w0 = 1,
   L-BFGS. Desvios: (a) por custo O(n²·d), o NCA roda numa subamostra estratificada de até
   `POST.nca_n_max` janelas do fold de treino; (b) no máximo `POST.nca_iter` iterações; (c) colunas
   constantes no treino são descartadas antes (peso irrelevante).
6. "5 execuções de NCA + 5 de Chi2 → 10 subconjuntos + 28 combinações por voto = 38 candidatos,
   escolhidos por guloso no CV" não está descrito o bastante (o que muda entre execuções? Chi2 é
   determinístico). Simplificação: 1 subconjunto NCA (35), 1 subconjunto Chi2 (35) e o voto entre eles
   (atributos escolhidos pelos dois; completado pelo rank médio até 35). Escolha do candidato por CV
   estratificado de 5 dobras **dentro do fold de treino**.
7. VAZAMENTO DO ARTIGO, corrigido aqui de propósito: no artigo a seleção NCA/Chi2 e a escolha do melhor
   dos 38 vetores usaram o MESMO 10-fold CV (sobre segmentos, sem agrupar gravações) que produziu os
   100 %. Aqui min–max, NCA, Chi2, escolha de candidato e classificador são ajustados em `treinar`,
   só com o fold de treino do runner; a validação nunca participa. A diferença entre o número
   publicado e o nosso mede, em parte, esse viés. (O CV interno não é agrupado por gravação porque
   `treinar` não recebe os grupos; isso só pode piorar a escolha do candidato, nunca vazar validação.)
8. "Cubic SVM" (Classification Learner): kernel polinomial de ordem 3, BoxConstraint 1, padronização,
   kernel scale automático, ECOC um-contra-um. Tradução sklearn: StandardScaler + SVC(kernel='poly',
   degree=3, coef0=1, gamma='scale', C=1) (SVC já é um-contra-um). gamma='scale' substitui o kernel
   scale heurístico do MATLAB (suposição). Probabilidades por Platt (`probability=True`, seed = post.seed).
9. "Fine kNN": k = 1, euclidiana, padronização (padrões do preset) → StandardScaler +
   KNeighborsClassifier(1). Probabilidade = one-hot do vizinho. Ver `p03b_etnext_knn.py`.
10. Histogramas em contagens brutas (como nas eqs. 15–17). A janela do runner é 3 s (IDMT) ou 1 s
    (A2212); a forma do vetor é sempre 7 × 768 = 5 376, só a escala das contagens muda.

Sem busca de hiperparâmetros: os valores acima são os do artigo ou padrões, fixados antes de rodar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt
from scipy.optimize import minimize
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from motorsound.config import PostConfig, PreConfig

NOME = "p03_etnext"
ORIGEM = "etnext2026"
TIPO = "inspirada"


@dataclass(frozen=True)
class PreETNeXt(PreConfig):
    sr: int = 16000
    wavelet: str = "sym4"
    niveis: int = 7
    janela: int = 16              # 16 amostras → 8 pares (i, i+8) → 8 bits
    fator_limiar: float = 0.5     # limiar ternário = 0,5 · std
    incluir_original: bool = False  # True → 8 blocos (eqs. 19–20 ao pé da letra)


@dataclass(frozen=True)
class PostETNeXt(PostConfig):
    n_por_nivel: int = 5          # 5 por nível × 7 = 35
    selecao: str = "nca_chi2_cv"  # "nca_chi2_cv" | "nca" | "chi2"
    nca_n_max: int = 200          # subamostra do fold de treino para o NCA
    nca_iter: int = 40
    nca_sigma: float = 1.0
    cv_interno: int = 5


PRE = PreETNeXt()
POST = PostETNeXt()
_NB = 256


# ----------------------------------------------------------------------------- extrator
def _codigos(x, janela, thr):
    """Códigos 0..255 (signum, ternária inferior, superior) de cada janela de 16 amostras."""
    meio = janela // 2
    L = len(x) - janela + 1
    if L <= 0:
        z = np.zeros(1, dtype=np.int64)
        return z, z, z
    # d[k, t] = wdw_t(k) − wdw_t(k+8), k = 0..7  (eqs. 8–9)
    d = np.stack([x[k:k + L] - x[k + meio:k + meio + L] for k in range(meio)])
    pesos = (1 << np.arange(meio)).astype(np.int64)[:, None]   # 2^(i−1)
    sg = ((d >= 0) * pesos).sum(0)
    lo = ((d < -thr) * pesos).sum(0)
    up = ((d > thr) * pesos).sum(0)
    return sg, lo, up


def _etnext_bloco(x, pre):
    x = np.asarray(x, dtype=np.float64)
    thr = pre.fator_limiar * float(np.std(x))
    return np.concatenate([np.bincount(c, minlength=_NB)[:_NB] for c in _codigos(x, pre.janela, thr)])


def extrair(y, sr, pre):
    y = np.asarray(y, dtype=np.float64)
    seqs = [y] if pre.incluir_original else []
    a = y
    for _ in range(pre.niveis):
        a = pywt.dwt(a, pre.wavelet, mode="symmetric")[0]   # aproximação do nível
        seqs.append(a)
    return np.concatenate([_etnext_bloco(s, pre) for s in seqs]).astype(np.float32)


# ----------------------------------------------------------------------------- fscnca
def _fscnca(X, y, sigma=1.0, lam=None, maxiter=40, bloco=8):
    """Pesos de atributo do NCA de seleção (Yang, Wang & Zuo 2012; `fscnca` do MATLAB).

    Maximiza f(w) = (1/n) Σ_i p_i − λ Σ_r w_r², com p_ij ∝ exp(−Σ_r w_r²|x_ir − x_jr| / σ), p_ii = 0,
    p_i = Σ_j [y_i = y_j] p_ij. Gradiente analítico; L-BFGS-B sobre −f.
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    lam = 1.0 / n if lam is None else lam
    mesma = (y[:, None] == y[None, :])

    def fg(w):
        w2 = (w ** 2).astype(np.float32)
        soma_p = 0.0
        g = np.zeros(d, dtype=np.float64)
        for i0 in range(0, n, bloco):
            i1 = min(n, i0 + bloco)
            A = np.abs(X[i0:i1, None, :] - X[None, :, :])          # (b, n, d)
            D = A @ w2 / sigma                                       # (b, n)
            for k in range(i1 - i0):
                D[k, i0 + k] = np.inf
            D -= D.min(axis=1, keepdims=True)
            P = np.exp(-D)
            P /= P.sum(axis=1, keepdims=True)
            Y = mesma[i0:i1]
            pi = (P * Y).sum(axis=1)                                 # (b,)
            soma_p += pi.sum()
            C = P * pi[:, None] - P * Y                              # (b, n)
            g += np.einsum("bn,bnd->d", C, A, optimize=True)
        f = soma_p / n - lam * np.sum(w ** 2)
        grad = 2 * w * (g / (sigma * n) - lam)
        return -f, -grad

    r = minimize(fg, np.ones(d), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return np.abs(r.x)


def _subamostra(y, n_max, rng):
    if len(y) <= n_max:
        return np.arange(len(y))
    idx = []
    cls = np.unique(y)
    for c in cls:
        ic = np.flatnonzero(y == c)
        k = max(1, int(round(n_max * len(ic) / len(y))))
        idx.append(rng.choice(ic, size=min(k, len(ic)), replace=False))
    return np.sort(np.concatenate(idx))


def _top_por_bloco(scores, n_blocos, tam, k):
    sel = []
    for b in range(n_blocos):
        s = scores[b * tam:(b + 1) * tam]
        sel.extend((b * tam + np.argsort(-s, kind="stable")[:k]).tolist())
    return np.array(sorted(sel))


def _voto(s_nca, s_chi, n_blocos, tam, k):
    """Atributos escolhidos pelos dois seletores; completa por rank médio, k por bloco."""
    sel = []
    for b in range(n_blocos):
        sl = slice(b * tam, (b + 1) * tam)
        rn = np.argsort(np.argsort(-s_nca[sl], kind="stable"))
        rc = np.argsort(np.argsort(-s_chi[sl], kind="stable"))
        ambos = np.flatnonzero((rn < k) & (rc < k))
        resto = [j for j in np.argsort(rn + rc, kind="stable") if j not in set(ambos)]
        esc = list(ambos[np.argsort((rn + rc)[ambos], kind="stable")]) + resto
        sel.extend((b * tam + np.array(esc[:k])).tolist())
    return np.array(sorted(sel))


# ----------------------------------------------------------------------------- treino
def _classificador_svm(seed):
    return make_pipeline(StandardScaler(),
                         SVC(kernel="poly", degree=3, coef0=1.0, gamma="scale", C=1.0,
                             probability=True, random_state=seed))


def _classificador_knn(seed):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1, metric="euclidean"))


def _cv_score(fabrica, X, y, n_split, seed):
    _, cont = np.unique(y, return_counts=True)
    n_split = min(n_split, int(cont.min()))
    if n_split < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)
    acc = []
    for tr, te in skf.split(X, y):
        m = fabrica(seed).fit(X[tr], y[tr])
        acc.append(np.mean(m.predict(X[te]) == y[te]))
    return float(np.mean(acc))


def treinar_com(fabrica, X, y, post, n_classes, pre=PRE):
    """Tudo dentro do fold de treino: min–max, NCA, Chi2, escolha do candidato, classificador."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(post.seed)
    n_blocos = pre.niveis + (1 if pre.incluir_original else 0)
    tam = 3 * _NB

    lo, hi = X.min(axis=0), X.max(axis=0)
    amp = np.where(hi > lo, hi - lo, 1.0)
    Xn = (X - lo) / amp                                               # eq. 21, ajustada no treino
    vivas = hi > lo

    s_chi = np.full(X.shape[1], -np.inf)
    if post.selecao in ("chi2", "nca_chi2_cv"):
        c, _ = chi2(Xn[:, vivas], y)
        s_chi[vivas] = np.nan_to_num(c, nan=-np.inf)
    s_nca = np.full(X.shape[1], -np.inf)
    if post.selecao in ("nca", "nca_chi2_cv"):
        sub = _subamostra(y, post.nca_n_max, rng)
        w = _fscnca(Xn[np.ix_(sub, np.flatnonzero(vivas))], y[sub], sigma=post.nca_sigma,
                    maxiter=post.nca_iter)
        s_nca[vivas] = w

    cands = {}
    if post.selecao in ("nca", "nca_chi2_cv"):
        cands["nca"] = _top_por_bloco(s_nca, n_blocos, tam, post.n_por_nivel)
    if post.selecao in ("chi2", "nca_chi2_cv"):
        cands["chi2"] = _top_por_bloco(s_chi, n_blocos, tam, post.n_por_nivel)
    if post.selecao == "nca_chi2_cv":
        cands["voto"] = _voto(s_nca, s_chi, n_blocos, tam, post.n_por_nivel)

    if len(cands) > 1:
        notas = {k: _cv_score(fabrica, Xn[:, idx], y, post.cv_interno, post.seed)
                 for k, idx in cands.items()}
        escolha = max(notas, key=lambda k: (np.nan_to_num(notas[k], nan=-1), k == "nca"))
    else:
        notas, escolha = {}, next(iter(cands))
    idx = cands[escolha]
    clf = fabrica(post.seed).fit(Xn[:, idx], y)
    return {"lo": lo, "amp": amp, "idx": idx, "clf": clf, "C": n_classes,
            "escolha": escolha, "cv_interno": notas}


def prever_com(modelo, X):
    Xn = (np.asarray(X, dtype=np.float64) - modelo["lo"]) / modelo["amp"]
    clf = modelo["clf"]
    P = np.zeros((len(X), modelo["C"]))
    P[:, clf.classes_] = clf.predict_proba(Xn[:, modelo["idx"]])
    return P


def treinar(X, rotulos, post, n_classes):
    return treinar_com(_classificador_svm, X, rotulos, post, n_classes)


def prever(modelo, X, post):
    return prever_com(modelo, X)


def custo_mcu():
    # Extração a 16 kHz: DWT sym4 (8 taps, passa-baixa só: ~16 flops/amostra de saída, série
    # geométrica ≈ 16·N); códigos: 8 subtrações + 3·8 comparações por amostra de cada nível, ≈ 32·N
    # somando os níveis (N/2 + N/4 + …). Total ≈ 50 flops/amostra → 0,8 MFLOP/s.
    # Só os 35 bins escolhidos precisam existir: 35 contadores. Buffer de janela 16 amostras por nível
    # (7 × 16 floats) + filtro da DWT; streaming, sem guardar os 3 s.
    # SVM cúbico: n_SV × 35 floats; com ~200 vetores de suporte ≈ 28 KB de flash (o artigo cita
    # 50–200 KB para < 500 SV); predição ≈ n_SV·(35·2 + 4) flops por janela, desprezível.
    return {"params": 200 * 35 + 35 * 2, "ram_kb": (7 * 16 * 4 + 35 * 4 + 7 * 8 * 4) / 1024 + 1,
            "flash_kb": 200 * 35 * 4 / 1024, "mflops_s": 50 * 16000 / 1e6,
            "nota": "estimativa analítica; n_SV ~200 suposto (depende do treino)"}
