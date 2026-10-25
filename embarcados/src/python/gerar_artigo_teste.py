"""Macros e figuras do teste oficial do IDMT (teste_idmt.py), sem tocar em numeros.tex.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_artigo_teste.py

Lê o results/csv/teste_idmt_<ts>.csv mais recente (e o _curvas_) e grava:
  paper/numeros_teste.tex / .json   macros \\tst… (fonte e coluna em cada uma)
  figures/{pdf,png}/teste_idmt_condicoes_en.*   acurácia balanceada por condição (dot plot)
  figures/{pdf,png}/teste_idmt_roc_en.*         ROC um-contra-o-resto, p11 e clássicas
Nenhum número é digitado aqui.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gerar_artigo_figuras as F  # noqa: E402  (paleta, rcParams e rótulos do artigo)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

EMB = Path(__file__).resolve().parents[2]
RAIZ = EMB.parent
VARS = {"p04_classicas": "Cls", "p11_decifrae_harmsum": "Dec", "b00_rmspico_norm": "RmsNorm",
        "b00_rmspico_bruto": "RmsRaw"}
CONDS = {"atmo_low": "AtmoLow", "atmo_medium": "AtmoMed", "atmo_high": "AtmoHigh",
         "talking_1": "TalkOne", "talking_2": "TalkTwo", "talking_3": "TalkThree",
         "talking_4": "TalkFour", "whitenoise_low": "White", "stresstest": "Stress"}
ROT_COND = {"atmo_low": "atmo low", "atmo_medium": "atmo med.", "atmo_high": "atmo high",
            "talking_1": "talking 1", "talking_2": "talking 2", "talking_3": "talking 3",
            "talking_4": "talking 4", "whitenoise_low": "white noise", "stresstest": "stress test"}


def ultimo(p):
    fs = sorted(f for f in glob.glob(str(p)) if "_meta" not in f and "_curvas" not in f)
    return Path(fs[-1])


def rel(p):
    return Path(p).resolve().relative_to(RAIZ).as_posix()


def main():
    fm = ultimo(EMB / "results" / "csv" / "teste_idmt_2*.csv")
    fc = fm.with_name(fm.name.replace("teste_idmt_", "teste_idmt_curvas_"))
    d = pd.read_csv(fm)
    M, J = [], {}

    def mac(nome, val, col, d_=3):
        txt = f"{val:.{d_}f}" if isinstance(val, float) else str(val)
        M.append(f"\\newcommand{{\\tst{nome}}}{{{txt}}}% [teste] {rel(fm)} :: {col}")
        J[f"tst{nome}"] = {"valor": txt, "fonte": rel(fm), "coluna": col}

    mac_ = d[d.classe == "_macro"]
    tot = mac_[mac_.condicao == "_todas"]
    any_v = tot[tot.variante == "p11_decifrae_harmsum"].iloc[0]
    mac("NWin", int(any_v.n_janelas), "n_janelas[_todas]")
    mac("NRec", int(any_v.n_gravacoes), "n_gravacoes[_todas]")
    mac("NCond", int(d[d.condicao != "_todas"].condicao.nunique()), "nº de condições")
    for v, s in VARS.items():
        for met, ms in (("acuracia_balanceada", "Bal"), ("roc_auc_ovr_macro", "Auc"), ("f1_macro", "Fone")):
            r = tot[(tot.variante == v) & (tot.metrica == met)].iloc[0]
            mac(f"{ms}{s}", float(r.valor), f"{met}[{v},_todas]")
            if met != "f1_macro":
                mac(f"{ms}{s}Lo", float(r.ic95_lo), f"ic95_lo[{met},{v}]")
                mac(f"{ms}{s}Hi", float(r.ic95_hi), f"ic95_hi[{met},{v}]")
        for c, cs in CONDS.items():
            r = mac_[(mac_.variante == v) & (mac_.condicao == c) & (mac_.metrica == "acuracia_balanceada")].iloc[0]
            mac(f"Bal{s}{cs}", float(r.valor), f"acuracia_balanceada[{v},{c}]")
        pc = d[(d.variante == v) & (d.condicao == "_todas") & (d.metrica.str.startswith("cm_pred_"))]
        for cl in ("good", "broken", "heavyload"):
            n_cl = int(pc[pc.classe == cl].valor.sum())
            n_good = int(pc[(pc.classe == cl) & (pc.metrica == "cm_pred_good")].valor.iloc[0])
            mac(f"PredGood{s}{cl.capitalize()}Pct", 100.0 * n_good / n_cl,
                f"cm_pred_good/suporte[{v},{cl}]", 1)
    out = EMB / "paper" / "numeros_teste.tex"
    out.write_text("% GERADO por embarcados/src/python/gerar_artigo_teste.py — não editar\n"
                   + "\n".join(M) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(J, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- figura 1: acurácia balanceada por condição (uma linha por condição)
    conds = list(CONDS)
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    off = {v: k for k, v in enumerate(VARS)}
    for v in VARS:
        y = [mac_[(mac_.variante == v) & (mac_.condicao == c) & (mac_.metrica == "acuracia_balanceada")].valor.iloc[0]
             for c in conds]
        yy = np.arange(len(conds)) + (off[v] - 1.5) * 0.17
        ax.scatter(y, yy, s=14, color=F.CORES[v], label=F.ROT[v], zorder=3,
                   marker={"p11_decifrae_harmsum": "o", "p04_classicas": "s",
                           "b00_rmspico_norm": "^", "b00_rmspico_bruto": "v"}[v],
                   edgecolors="white", linewidths=0.4)
    ax.axvline(1 / 3, color="#52514e", lw=0.6, ls=":", zorder=1)
    ax.text(1 / 3 + 0.01, len(conds) - 0.45, "chance", fontsize=6, color="#52514e")
    ax.set_yticks(range(len(conds)), [ROT_COND[c] for c in conds])
    ax.invert_yaxis()
    ax.set_xlim(0.25, 1.02)
    ax.set_xlabel("Balanced accuracy on the official IDMT test set")
    ax.grid(axis="x", color="#d9d8d4", lw=0.4, zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    F._salvar(fig, "teste_idmt_condicoes_en", [rel(fm)])

    # ---- figura 2: ROC um-contra-o-resto (p11 e clássicas)
    cv = pd.read_csv(fc)
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
    tons = {"good": "-", "broken": "--", "heavyload": ":"}
    for a, v in zip(axs, ("p11_decifrae_harmsum", "p04_classicas")):
        for cl, ls in tons.items():
            r = cv[(cv.variante == v) & (cv.classe == cl) & (cv.curva == "roc")]
            a.plot(r.x, r.y, ls, color=F.CORES[v], lw=1.0, label=cl)
        a.plot([0, 1], [0, 1], color="#d9d8d4", lw=0.5)
        a.set_title(F.ROT[v], fontsize=7)
        a.set_xlabel("False positive rate")
        a.set_aspect("equal")
    axs[0].set_ylabel("True positive rate")
    axs[1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    F._salvar(fig, "teste_idmt_roc_en", [rel(fc)])
    print(out, len(M), "macros")


if __name__ == "__main__":
    main()
