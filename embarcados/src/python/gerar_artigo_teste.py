"""Macros e figuras do teste oficial do IDMT (teste_idmt.py), sem tocar em numeros.tex.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_artigo_teste.py

Lê o results/csv/teste_idmt_<ts>.csv mais recente (e o _curvas_) e grava:
  paper/numeros_teste.tex / .json   macros \\tst… (fonte e coluna em cada uma)
  figures/{pdf,png}/teste_idmt_condicoes_en.*   acurácia balanceada por condição (dot plot)
  figures/{pdf,png}/teste_idmt_roc_en.*         ROC um-contra-o-resto, p11 e clássicas
Nenhum número é digitado aqui.
"""
from __future__ import annotations

import argparse
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


def bootstrap_diferenca_ba(predicoes, n, seed):
    modelos = ("p04_classicas", "p11_decifrae_harmsum")
    d = pd.read_csv(predicoes, usecols=["variante", "rec_id", "idx", "classe", "pred"])
    d = d[d.variante.isin(modelos)]
    if d.duplicated(["variante", "rec_id", "idx"]).any():
        raise ValueError("predições duplicadas por pipeline, gravação e janela")
    largas = d.pivot(index=["rec_id", "idx"], columns="variante", values=["classe", "pred"])
    c04, c11 = largas["classe"][modelos[0]], largas["classe"][modelos[1]]
    if c04.isna().any() or c11.isna().any() or not c04.equals(c11):
        raise ValueError("p04 e p11 não contêm os mesmos rótulos verdadeiros por janela")

    por_grupo = pd.DataFrame({
        "classe": c04,
        "acerto_p04": largas["pred"][modelos[0]].eq(c04),
        "acerto_p11": largas["pred"][modelos[1]].eq(c11),
    }).reset_index().groupby(["classe", "rec_id"], sort=True).agg(
        n=("idx", "size"),
        p04=("acerto_p04", "sum"),
        p11=("acerto_p11", "sum"),
    )
    classes = sorted(por_grupo.index.get_level_values("classe").unique())
    rng = np.random.default_rng(seed)
    diferencas = np.empty(n, dtype=float)
    observada = 0.0
    arrays = {}
    for classe in classes:
        g = por_grupo.xs(classe, level="classe").sort_index()
        arrays[classe] = (
            g["n"].to_numpy(dtype=np.int64),
            g["p04"].to_numpy(dtype=np.int64),
            g["p11"].to_numpy(dtype=np.int64),
        )
        total, acertos04, acertos11 = (v.sum() for v in arrays[classe])
        observada += (acertos04 / total - acertos11 / total) / len(classes)

    for b in range(n):
        delta = 0.0
        for classe in classes:
            total, acertos04, acertos11 = arrays[classe]
            amostra = rng.integers(0, len(total), size=len(total))
            denominador = total[amostra].sum()
            delta += (
                acertos04[amostra].sum() / denominador
                - acertos11[amostra].sum() / denominador
            ) / len(classes)
        diferencas[b] = delta
    lo, hi = np.percentile(diferencas, [2.5, 97.5])
    return observada, float(lo), float(hi), classes, arrays


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--numbers-only", action="store_true",
                        help="atualiza macros e JSON sem regenerar as figuras")
    args = parser.parse_args(argv)
    fm = ultimo(EMB / "results" / "csv" / "teste_idmt_2*.csv")
    fc = fm.with_name(fm.name.replace("teste_idmt_", "teste_idmt_curvas_"))
    ts = fm.stem[len("teste_idmt_"):]
    fp = EMB / "results" / "predictions" / f"teste_idmt_janelas_{ts}.csv"
    meta = json.loads(fm.with_name(fm.stem + "_meta.json").read_text(encoding="utf-8"))
    if not fp.exists():
        raise FileNotFoundError(f"predições do teste não encontradas: {fp}")
    d = pd.read_csv(fm)
    M, J = [], {}

    def mac(nome, val, col, d_=3):
        if isinstance(val, float):
            txt = f"{val:.{d_}f}"
        elif isinstance(val, int) and abs(val) >= 1000:
            txt = f"{val:,}".replace(",", "{,}")
        else:
            txt = str(val)
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

    delta, delta_lo, delta_hi, classes, grupos = bootstrap_diferenca_ba(
        fp, int(meta["n_boot"]), int(meta["seed_boot"]))
    fonte_delta = (
        f"bootstrap pareado estratificado por classe, grupo=rec_id, "
        f"mesmas reamostragens; {meta['n_boot']} réplicas, seed={meta['seed_boot']}"
    )
    mac("DeltaBalClassicalMinusHarmonic", delta, fonte_delta + f" :: {rel(fp)}")
    mac("DeltaBalClassicalMinusHarmonicLo", delta_lo,
        fonte_delta + f" IC95 inferior :: {rel(fp)}")
    mac("DeltaBalClassicalMinusHarmonicHi", delta_hi,
        fonte_delta + f" IC95 superior :: {rel(fp)}")
    J["paired_ba_difference"] = {
        "pipelines": ["p04_classicas", "p11_decifrae_harmsum"],
        "difference": "BA(p04) - BA(p11)",
        "valor": delta,
        "ic95": [delta_lo, delta_hi],
        "fonte_predicoes": rel(fp),
        "agrupamento": "rec_id, estratificado por classe",
        "reamostragens_compartilhadas": True,
        "n_boot": int(meta["n_boot"]),
        "seed": int(meta["seed_boot"]),
        "n_gravacoes_por_classe": {c: len(g[0]) for c, g in grupos.items()},
    }
    out = EMB / "paper" / "numeros_teste.tex"
    out.write_text("% GERADO por embarcados/src/python/gerar_artigo_teste.py — não editar\n"
                   + "\n".join(M) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(J, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.numbers_only:
        print(out, len(M), "macros; figuras mantidas")
        return

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
