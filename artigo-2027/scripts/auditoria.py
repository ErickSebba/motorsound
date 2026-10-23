"""Experimentos baratos de auditoria (Etapa 3), iguais para todas as bases.

  exp1  controle de nível: RMS + pico → classe, splits aleatório / gravação / unidade
  exp2  rótulos permutados (por segmento e por gravação): tem de dar o acaso nos splits
        agrupados; acima do acaso no aleatório com permutação por gravação = vazamento
  exp3  espectro médio por classe e por unidade (figura) + distâncias entre espectros
  exp5  separabilidade por unidade: forma espectral (b01) + regressão logística → unidade

Uso:  .venv/Scripts/python.exe scripts/auditoria.py <base> [exp1 exp2 exp3 exp5]
          [--max-seg 30] [--forcar]

Nada disto é resultado de artigo: é diagnóstico da base. A regressão logística usa
`class_weight="balanced"` (desvio em relação a b00/b01 do Artigo3, que não balanceavam)
porque algumas bases (MIMII) têm muito mais normal que anomalia; a métrica é a acurácia
balanceada sobre as classes presentes no teste.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.comum import FIGS, FORMA_NOMES, SEED, extrair_base, ler_indice, salvar

warnings.filterwarnings("ignore", category=UserWarning)

N_DOBRAS = 5
N_PERM = 5
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ESTILOS = ["-", "--", "-.", ":"]
FEATS = {"nivel": ["rms_db", "pico_db"], "forma": FORMA_NOMES,
         "forma+nivel": FORMA_NOMES + ["rms_db", "pico_db"]}


# --------------------------------------------------------------------------- splits
def gerar_splits(tab: pd.DataFrame, y: np.ndarray, tipo: str):
    """Lista de (nome_dobra, idx_treino, idx_teste) ou (None, motivo) se não se aplica."""
    n = len(tab)
    if tipo == "aleatorio":
        cv = StratifiedKFold(N_DOBRAS, shuffle=True, random_state=SEED)
        return [(f"k{i}", a, b) for i, (a, b) in enumerate(cv.split(np.zeros(n), y))], ""
    if tipo == "gravacao":
        g = tab["gravacao_id"].to_numpy()
        por_classe = pd.Series(g).groupby(y).nunique()
        k = int(min(N_DOBRAS, por_classe.min()))
        if k < 2:
            return None, f"classe com {por_classe.min()} gravação(ões)"
        cv = StratifiedGroupKFold(k, shuffle=True, random_state=SEED)
        return [(f"k{i}", a, b) for i, (a, b) in enumerate(cv.split(np.zeros(n), y, g))], ""
    if tipo == "unidade":
        g = tab["unidade"].to_numpy()
        if len(np.unique(g)) < 2:
            return None, "uma unidade só"
        cv = LeaveOneGroupOut()
        return [(f"fora={g[b][0]}", a, b) for a, b in cv.split(np.zeros(n), y, g)], ""
    raise ValueError(tipo)


def avaliar(X, y, dobras):
    """Uma linha por dobra + linha 'total' (predições concatenadas)."""
    linhas, yt_all, yp_all = [], [], []
    for nome, a, b in dobras:
        ausentes = sorted(set(y[b]) - set(y[a]))
        if len(np.unique(y[a])) < 2:
            linhas.append(dict(dobra=nome, n_treino=len(a), n_teste=len(b), obs="treino com 1 classe"))
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, class_weight="balanced"))
        m.fit(X[a], y[a])
        p = m.predict(X[b])
        yt_all.append(y[b]); yp_all.append(p)
        linhas.append(dict(dobra=nome, n_treino=len(a), n_teste=len(b),
                           classes_teste=len(np.unique(y[b])),
                           classes_ausentes_treino="|".join(map(str, ausentes)),
                           acc=float(np.mean(p == y[b])),
                           bacc=float(balanced_accuracy_score(y[b], p)),
                           f1_macro=float(f1_score(y[b], p, average="macro")),
                           acaso=1.0 / len(np.unique(y[b]))))
    if yt_all:
        yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
        linhas.append(dict(dobra="total", n_teste=len(yt), classes_teste=len(np.unique(yt)),
                           acc=float(np.mean(yt == yp)), bacc=float(balanced_accuracy_score(yt, yp)),
                           f1_macro=float(f1_score(yt, yp, average="macro")),
                           acaso=1.0 / len(np.unique(yt))))
    return linhas


def tarefas(tab: pd.DataFrame):
    """'todas' + uma tarefa por condição e por tipo de unidade, quando houver 2 a 12 valores."""
    yield "todas", tab
    for col in ("condicao", "tipo"):
        vals = sorted(v for v in tab[col].unique() if v != "")
        if 2 <= len(vals) <= 12 and not (col == "tipo" and len(vals) == tab["unidade"].nunique()):
            for v in vals:
                sub = tab[tab[col] == v]
                if sub["classe"].nunique() >= 2:
                    yield f"{col}={v}", sub
    # dentro de cada unidade que tem ≥2 classes (cenário de modelo por máquina; só split por gravação)
    unidades = sorted(tab["unidade"].unique())
    if len(unidades) <= 16:
        for u in unidades:
            sub = tab[tab["unidade"] == u]
            if sub["classe"].nunique() >= 2:
                yield f"unidade={u}", sub


# --------------------------------------------------------------------------- experimentos
def exp1(base, tab, args):
    t0 = time.time(); linhas = []
    for nome_t, sub in tarefas(tab):
        y = sub["classe"].to_numpy(); X = sub[FEATS["nivel"]].to_numpy()
        for tipo in ("aleatorio", "gravacao", "unidade"):
            dobras, motivo = gerar_splits(sub, y, tipo)
            if dobras is None:
                linhas.append(dict(tarefa=nome_t, split=tipo, obs=f"não se aplica: {motivo}"))
                continue
            for l in avaliar(X, y, dobras):
                linhas.append(dict(tarefa=nome_t, split=tipo, **l))
    df = pd.DataFrame(linhas); df.insert(0, "features", "nivel"); df.insert(0, "base", base)
    salvar(df, "exp1_nivel", base, {"features": FEATS["nivel"], "n_dobras": N_DOBRAS,
                                    "modelo": "StandardScaler+LogReg(balanced)",
                                    "max_seg": args.max_seg, "filtro": args.filtro}, t0)
    _resumo(df, "tarefa")


def exp2(base, tab, args):
    t0 = time.time(); linhas = []
    y0 = tab["classe"].to_numpy(); grav = tab["gravacao_id"].to_numpy()
    rotulo_grav = pd.Series(y0, index=grav).groupby(level=0).first()
    for feats, modo, rep in itertools.product(("nivel", "forma"), ("segmento", "gravacao"), range(N_PERM)):
        rng = np.random.default_rng(SEED + rep)
        if modo == "segmento":
            y = rng.permutation(y0)
        else:
            perm = pd.Series(rng.permutation(rotulo_grav.to_numpy()), index=rotulo_grav.index)
            y = perm.loc[grav].to_numpy()
        X = tab[FEATS[feats]].to_numpy()
        for tipo in ("aleatorio", "gravacao", "unidade"):
            dobras, motivo = gerar_splits(tab, y, tipo)
            if dobras is None:
                linhas.append(dict(features=feats, permutacao=modo, rep=rep, split=tipo,
                                   obs=f"não se aplica: {motivo}"))
                continue
            for l in avaliar(X, y, dobras):
                if l["dobra"] == "total":
                    linhas.append(dict(features=feats, permutacao=modo, rep=rep, split=tipo, **l))
    df = pd.DataFrame(linhas); df.insert(0, "base", base)
    salvar(df, "exp2_permutado", base, {"n_perm": N_PERM, "seeds": [SEED + r for r in range(N_PERM)],
                                        "max_seg": args.max_seg, "filtro": args.filtro}, t0)
    print(df.groupby(["features", "permutacao", "split"])[["bacc", "acaso"]].mean().round(3))


def exp3(base, espectros, freqs, args):
    t0 = time.time()
    # média ponderada sobre condições → (classe, unidade)
    cel = {}
    for (c, u, _), (soma, n) in espectros.items():
        s0, n0 = cel.get((c, u), (0.0, 0))
        cel[(c, u)] = (s0 + soma, n0 + n)
    media = {k: s / n for k, (s, n) in cel.items()}
    banda = (freqs >= 50) & (freqs <= 7900)
    forma = {k: v[banda] - v[banda].mean() for k, v in media.items()}   # só forma, sem nível
    linhas = []
    for (k1, k2) in itertools.combinations(sorted(forma), 2):
        mesma_c, mesma_u = k1[0] == k2[0], k1[1] == k2[1]
        tipo = ("classe_diferente_mesma_unidade" if mesma_u else
                "unidade_diferente_mesma_classe" if mesma_c else "ambas_diferentes")
        linhas.append(dict(classe_a=k1[0], unidade_a=k1[1], classe_b=k2[0], unidade_b=k2[1],
                           tipo_par=tipo, dist_db_rms=float(np.sqrt(np.mean((forma[k1] - forma[k2]) ** 2))),
                           dif_nivel_db=float(media[k1][banda].mean() - media[k2][banda].mean())))
    df = pd.DataFrame(linhas); df.insert(0, "base", base)
    salvar(df, "exp3_espectros", base, {"welch_nperseg": 2048, "banda_hz": [50, 7900],
                                        "dist": "RMS da diferença em dB após remover a média",
                                        "max_seg": args.max_seg, "filtro": args.filtro}, t0)
    if len(df):
        print(df.groupby("tipo_par")["dist_db_rms"].agg(["median", "count"]).round(2))
    _figura_espectros(base, media, freqs)


def _figura_espectros(base, media, freqs):
    FIGS.mkdir(parents=True, exist_ok=True)
    classes = sorted({c for c, _ in media}); unidades = sorted({u for _, u in media})
    banda = (freqs >= 20) & (freqs <= 8000)
    for agrupar, linhas_por, nome in ((unidades, classes, "por_unidade"), (classes, unidades, "por_classe")):
        n = len(agrupar); nc = min(4, n); nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.4 * nr + 0.5), sharex=True, sharey=True,
                                squeeze=False)
        for i, g in enumerate(agrupar):
            ax = axs.flat[i]
            for j, l in enumerate(linhas_por):
                k = (l, g) if nome == "por_unidade" else (g, l)
                if k in media:
                    ax.semilogx(freqs[banda], media[k][banda], lw=1.2, color=CORES[j % 8],
                                ls=ESTILOS[(j // 8) % 4], label=str(l))
            ax.set_title(str(g), fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        for ax in axs.flat[n:]:
            ax.set_visible(False)
        for ax in axs[-1]:
            ax.set_xlabel("frequência (Hz)", fontsize=8)
        for ax in axs[:, 0]:
            ax.set_ylabel("PSD (dB)", fontsize=8)
        h, lab = axs.flat[0].get_legend_handles_labels()
        titulo = "uma curva por classe, um painel por unidade" if nome == "por_unidade" else \
                 "uma curva por unidade, um painel por classe"
        fig.suptitle(f"{base}: espectro médio (1 s, 16 kHz) — {titulo}", fontsize=9, y=0.995)
        H = fig.get_figheight()
        fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.22 / H),
                   ncol=min(8, len(lab)), fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / H))
        caminho = FIGS / f"exp3_espectro_{nome}_{base}.png"
        fig.savefig(caminho, dpi=150); plt.close(fig)
        print(f"-> {caminho.name}")


def exp5(base, tab, args):
    t0 = time.time(); linhas = []
    conjuntos = [("todas", tab)] + [(f"classe={c}", tab[tab["classe"] == c])
                                    for c in sorted(tab["classe"].unique())]
    for nome_t, sub in conjuntos:
        if sub["unidade"].nunique() < 2:
            linhas.append(dict(tarefa=nome_t, obs="menos de 2 unidades")); continue
        y = sub["unidade"].to_numpy()
        for feats in ("forma", "forma+nivel"):
            X = sub[FEATS[feats]].to_numpy()
            for tipo in ("aleatorio", "gravacao"):
                dobras, motivo = gerar_splits(sub, y, tipo)
                if dobras is None:
                    linhas.append(dict(tarefa=nome_t, features=feats, split=tipo,
                                       obs=f"não se aplica: {motivo}"))
                    continue
                for l in avaliar(X, y, dobras):
                    linhas.append(dict(tarefa=nome_t, features=feats, split=tipo, **l))
    df = pd.DataFrame(linhas); df.insert(0, "base", base); df.insert(1, "alvo", "unidade")
    salvar(df, "exp5_unidade", base, {"n_dobras": N_DOBRAS, "max_seg": args.max_seg, "filtro": args.filtro}, t0)
    tot = df[df.get("dobra", pd.Series(dtype=str)) == "total"]
    if len(tot):
        print(tot[["tarefa", "features", "split", "bacc", "acaso"]].round(3).to_string(index=False))


def _resumo(df, col):
    tot = df[df.get("dobra", pd.Series(dtype=str)) == "total"]
    if len(tot):
        print(tot[[col, "split", "bacc", "acaso"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("exps", nargs="*", default=["exp1", "exp2", "exp3", "exp5"])
    ap.add_argument("--max-seg", type=int, default=30, help="segmentos de 1 s por arquivo (0 = todos)")
    ap.add_argument("--forcar", action="store_true", help="refaz o cache de features")
    ap.add_argument("--filtro", default="", help="consulta pandas sobre classe/unidade/condicao")
    args = ap.parse_args()
    tab, espectros, freqs = extrair_base(args.base, args.max_seg or None, args.forcar)
    if args.filtro:
        tab = tab.merge(ler_indice(args.base)[["arquivo", "obs"]], on="arquivo", how="left")
        tab = tab.query(args.filtro)
        try:
            espectros = {k: v for k, v in espectros.items()
                         if not pd.DataFrame([dict(classe=k[0], unidade=k[1], condicao=k[2])]).query(args.filtro).empty}
        except pd.errors.UndefinedVariableError:
            # espectros são agregados por (classe, unidade, condicao); filtro por outra coluna não se aplica
            print("AVISO: filtro não se aplica aos espectros médios do exp3 (ficam sem filtro)")
    tab["tipo"] = tab["unidade"].str.split("_").str[0]
    print(f"{args.base}: {len(tab)} segmentos, {tab['arquivo'].nunique()} arquivos, "
          f"{tab['gravacao_id'].nunique()} gravações, {tab['unidade'].nunique()} unidades, "
          f"{tab['classe'].nunique()} classes")
    for e in args.exps:
        print(f"== {e}")
        if e == "exp3":
            exp3(args.base, espectros, freqs, args)
        else:
            globals()[e](args.base, tab, args)


if __name__ == "__main__":
    main()
