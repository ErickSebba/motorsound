# Origem: Artigo2(Acordes)/Decifra-Violão/stats.py::holm, classificar, bootstrap,
#         wilcoxon, comparar (commit 93609f7)
# Mudanças: a unidade de reamostragem deixa de ser a faixa e passa a ser o GRUPO
#   independente do split (gravação de origem no A2212; (gravação, bloco) no IDMT);
#   a métrica de cada grupo é a média do acerto por janela do grupo, ponderada pelo
#   número de janelas. Os pares deixam de ser fixos no módulo e são passados por quem
#   chama. Leitura de CSV e CLI removidas (ficam nos scripts de experimentos/).
"""Comparações pareadas entre pipelines com reamostragem por grupo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

N_BOOTSTRAP = 10_000
SEED = 0
ALFA = 0.05


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    ajustados = np.full(ps.shape, np.nan)
    validos = np.flatnonzero(~np.isnan(ps))
    if validos.size == 0:
        return ajustados
    n = validos.size
    ordem = validos[np.argsort(ps[validos], kind="stable")]
    corrente = 0.0
    for k, idx in enumerate(ordem):
        corrente = max(corrente, (n - k) * ps[idx])
        ajustados[idx] = min(1.0, corrente)
    return ajustados


def classificar(cruza_zero, p_holm, alfa=ALFA):
    if cruza_zero:
        return "empate"
    if p_holm == p_holm and p_holm < alfa:
        return "confirmado"
    return "fragil"


def bootstrap(d, w, n=N_BOOTSTRAP, seed=SEED, alfa=ALFA):
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    pesos = w[idx]
    medias = np.sum(d[idx] * pesos, axis=1) / np.sum(pesos, axis=1)
    lo, hi = np.percentile(medias, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    cauda = min(float(np.mean(medias <= 0)), float(np.mean(medias >= 0)))
    p = min(1.0, max(2.0 * cauda, 1.0 / n))
    return float(lo), float(hi), p


def bootstrap_media(x, w, n=N_BOOTSTRAP, seed=SEED, alfa=ALFA):
    """IC 95 % de uma média ponderada por reamostragem de grupos."""
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    medias = np.sum(x[idx] * w[idx], axis=1) / np.sum(w[idx], axis=1)
    lo, hi = np.percentile(medias, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return float(lo), float(hi)


def wilcoxon(d):
    if len(d) == 0 or np.allclose(d, 0):
        return float("nan"), float("nan")
    try:
        res = sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def por_grupo(df: pd.DataFrame, metrica="acerto", grupo="grupo") -> pd.DataFrame:
    """Média da métrica por (pipeline, grupo) e o peso (n de janelas)."""
    return (df.groupby(["pipeline", grupo])[metrica].agg(["mean", "size"])
              .rename(columns={"mean": "valor", "size": "peso"}).reset_index())


def comparar(df: pd.DataFrame, a: str, b: str, metrica="acerto", grupo="grupo",
             isola="") -> dict:
    """Diferença b − a da métrica, pareada por grupo."""
    g = por_grupo(df, metrica, grupo)
    pa = g[g.pipeline == a].set_index(grupo)
    pb = g[g.pipeline == b].set_index(grupo)
    comuns = sorted(set(pa.index) & set(pb.index))
    va, vb = pa.loc[comuns, "valor"].to_numpy(), pb.loc[comuns, "valor"].to_numpy()
    w = pa.loc[comuns, "peso"].to_numpy().astype(float)
    d = vb - va
    lo, hi, p = bootstrap(d, w)
    est, pw = wilcoxon(d)
    mp = lambda x: float(np.sum(x * w) / np.sum(w)) if len(x) else float("nan")
    return {"a": a, "b": b, "metrica": metrica, "isola": isola, "n_grupos": len(d),
            "media_a": mp(va), "media_b": mp(vb), "diferenca": mp(d),
            "ic95_min": lo, "ic95_max": hi, "ic_cruza_zero": bool(lo <= 0 <= hi),
            "p_bootstrap": p, "wilcoxon_p": pw,
            "b_melhor": int(np.sum(d > 1e-12)), "b_pior": int(np.sum(d < -1e-12))}


def rodar_pares(df, pares, metrica="acerto", grupo="grupo") -> pd.DataFrame:
    linhas = [comparar(df, a, b, metrica, grupo, isola) for a, b, isola in pares]
    hb = holm([l["p_bootstrap"] for l in linhas])
    hw = holm([l["wilcoxon_p"] for l in linhas])
    for l, pb, pw in zip(linhas, hb, hw):
        l["p_bootstrap_holm"] = pb
        l["wilcoxon_p_holm"] = pw
        l["classificacao"] = classificar(l["ic_cruza_zero"], pb)
    return pd.DataFrame(linhas)
