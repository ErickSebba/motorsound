"""Tabelas e figuras de RESULTADOS.md, lidas dos CSVs (nenhum número digitado à mão).

Pega o CSV mais recente de cada tipo em out/experiments/ (ou os passados por argumento),
escreve out/experiments/<ts>_analise_*.csv e out/figuras/*.png, e imprime tabelas em
Markdown com a origem de cada uma.

    python experimentos/analise.py
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from motorsound import stats  # noqa: E402

EXP = RAIZ / "out" / "experiments"
FIG = RAIZ / "out" / "figuras"
# paleta categórica de referência (skill dataviz), ordem fixa; cinza para o contexto
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CINZA = "#b4b2a9"
TEXTO, TEXTO2 = "#1f1e1d", "#5f5e5a"
DESTAQUE = ["p11_decifrae_harmsum", "p20b_so_frot", "p00_logmel_cnn",
            "p02_grollmisch", "p21_envelope", "b01_forma_espectral"]


def ultimo(padrao):
    c = sorted(EXP.glob(padrao))
    return c[-1] if c else None


def estilo(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(CINZA)
    ax.tick_params(colors=TEXTO2, labelsize=8)
    ax.grid(axis="y", color="#e6e4dd", lw=0.6)
    ax.set_axisbelow(True)


def md(df, floatfmt=3):
    return df.round(floatfmt).to_markdown(index=False)


def ic_por_grupo(g, metrica="acerto"):
    pg = g.groupby("grupo")[metrica].agg(["mean", "size"])
    lo, hi = stats.bootstrap_media(pg["mean"].to_numpy(), pg["size"].to_numpy().astype(float))
    return lo, hi


def tabela_validacao(csv):
    df = pd.read_csv(csv)
    linhas = []
    for (p, base, sp, norm), g in df.groupby(["pipeline", "base", "split", "normalizar_rms"]):
        lo, hi = ic_por_grupo(g)
        linhas.append({"pipeline": p, "base": base, "split": sp, "norm_rms": norm,
                       "acc": g.acerto.mean(), "ic95": f"[{lo:.3f}, {hi:.3f}]",
                       "f1_macro": __import__("sklearn.metrics", fromlist=["f1_score"])
                       .f1_score(g.classe, g.pred, average="macro"),
                       "n_janelas": len(g), "n_grupos": g.grupo.nunique()})
    t = pd.DataFrame(linhas)
    # auditoria de vazamento: aleatório − agrupado, por pipeline e base
    piv = t[t.norm_rms == True].pivot_table(index="pipeline", columns=["base", "split"],  # noqa: E712
                                             values="acc")
    return t, piv


def figura_snr(csv, base="idmt"):
    df = pd.read_csv(csv)
    df["c"] = df.condicao_sintetica.map(json.loads)
    df["tipo_p"] = df.c.map(lambda c: c.get("tipo"))
    df["snr"] = df.c.map(lambda c: c.get("snr_db", np.nan))
    limpo = df[df.tipo_p == "limpo"].groupby("pipeline").acerto.mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, tipo in zip(axes, ["branco", "rosa"]):
        g = df[df.tipo_p == tipo].groupby(["pipeline", "snr"]).acerto.mean().reset_index()
        pipes = sorted(g.pipeline.unique(), key=lambda p: (p in DESTAQUE, p))
        k = 0
        for p in pipes:
            s = g[g.pipeline == p].sort_values("snr")
            x = list(s.snr) + [40]
            yv = list(s.acerto) + [limpo.get(p, np.nan)]
            if p in DESTAQUE:
                cor = SERIES[DESTAQUE.index(p) % len(SERIES)]
                ax.plot(x, yv, color=cor, lw=2, marker="o", ms=4, label=p, zorder=3)
                k += 1
            else:
                ax.plot(x, yv, color=CINZA, lw=1, zorder=1)
        ax.axhline(1 / 3, color=TEXTO2, lw=0.8, ls=(0, (3, 3)))
        ax.text(30, 1 / 3 - 0.07, "acaso", color=TEXTO2, fontsize=7, ha="center")
        ax.set_xticks([-5, 0, 5, 10, 15, 20, 30, 40])
        ax.set_xticklabels(["−5", "0", "5", "10", "15", "20", "30", "limpo"])
        ax.set_title(f"ruído {tipo}", color=TEXTO, fontsize=9, loc="left")
        ax.set_xlabel("SNR (dB)", color=TEXTO2, fontsize=8)
        estilo(ax)
    axes[0].set_ylabel("acurácia por janela", color=TEXTO2, fontsize=8)
    axes[0].set_ylim(0, 1.03)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"snr_{base}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def tabela_robustez(csv):
    df = pd.read_csv(csv)
    t = df.groupby(["pipeline", "condicao_sintetica"]).acerto.mean().unstack()
    return t


def figura_pareto(tab_val, meta_val, csv_rob):
    """Custo (MFLOP/s estimado) × acurácia em rosa 10 dB no IDMT."""
    custos = meta_val.get("custo_mcu", {})
    rob = pd.read_csv(csv_rob)
    alvo = json.dumps({"snr_db": 10.0, "tipo": "rosa"}, sort_keys=True)
    acc = rob[rob.condicao_sintetica == alvo].groupby("pipeline").acerto.mean()
    pts = [(p, c.get("mflops_s"), acc[p]) for p, c in custos.items()
           if isinstance(c, dict) and c.get("mflops_s") and p in acc]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for p, x, y in pts:
        dest = p in DESTAQUE
        ax.scatter(x, y, s=36 if dest else 18, zorder=3,
                   color=SERIES[DESTAQUE.index(p) % len(SERIES)] if dest else CINZA,
                   edgecolor="white", linewidth=1)
        ax.annotate(p.split("_")[0], (x, y), xytext=(4, 3), textcoords="offset points",
                    fontsize=6.5, color=TEXTO2)
    ax.set_xscale("log")
    ax.set_xlabel("custo estimado (MFLOP por s de áudio)", color=TEXTO2, fontsize=8)
    ax.set_ylabel("acurácia, rosa 10 dB (IDMT val.)", color=TEXTO2, fontsize=8)
    estilo(ax)
    fig.tight_layout()
    out = FIG / "pareto_custo.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--validacao", default=None)
    ap.add_argument("--robustez", default=None)
    ap.add_argument("--sequencias", default=None)
    a = ap.parse_args(argv)
    v = Path(a.validacao) if a.validacao else ultimo("*_validacao.csv")
    r = Path(a.robustez) if a.robustez else ultimo("*_robustez_idmt.csv")
    s = Path(a.sequencias) if a.sequencias else ultimo("*_sequencias_idmt.csv")
    if v:
        t, piv = tabela_validacao(v)
        print(f"\n### Validação (origem: {v.name})\n")
        print(md(t))
        print(f"\n### Acurácia por base e split, RMS normalizado (origem: {v.name})\n")
        print(piv.round(3).to_markdown())
        t.to_csv(EXP / f"{v.stem}_analise_tabela.csv", index=False)
        meta = json.loads((EXP / f"{v.stem}_meta.json").read_text(encoding="utf-8"))
        cus = pd.DataFrame([{"pipeline": p, **(c if isinstance(c, dict) else {"nota": c})}
                            for p, c in meta.get("custo_mcu", {}).items()])
        print(f"\n### Custo MCU estimado (origem: {v.stem}_meta.json)\n")
        print(md(cus))
    if r:
        print(f"\n### Robustez IDMT (origem: {r.name})\n")
        print(tabela_robustez(r).round(3).to_markdown())
        print("figura:", figura_snr(r).relative_to(RAIZ))
        if v:
            f = figura_pareto(t, meta, r)
            if f:
                print("figura:", f.relative_to(RAIZ))
    if s:
        sq = pd.read_csv(s)
        res = (sq.groupby(["pipeline", "decodificacao", "change_penalty"])
                 [["acuracia", "atraso_mediano_s", "espurias_por_min", "perdidas"]]
                 .mean().reset_index())
        print(f"\n### Sequências (origem: {s.name})\n")
        print(md(res))


if __name__ == "__main__":
    main()
