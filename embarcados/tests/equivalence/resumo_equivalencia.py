"""Resumo da equivalência Python × C a partir dos CSVs do comparador (nada digitado à mão).

    .venv\\Scripts\\python.exe embarcados\\tests\\equivalence\\resumo_equivalencia.py

Lê, para cada build (double, float, float-sosd), o CSV mais recente de
`results/csv/equivalencia_<build>_<ts>.csv` (exclui as emulações) e escreve:
  reports/c_equivalence_resultados_pc.md   PASS/FAIL por etapa e lista de TODAS as discrepâncias
  tables/latex/equivalencia.tex            tabela booktabs por etapa
  figures/{pdf,png}/equivalencia_erro_etapas.*   razão erro/tolerância máxima por etapa
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EMB = Path(__file__).resolve().parents[2]
CSV = EMB / "results" / "csv"
BUILDS = [("double", "C double (PC)"), ("float", "C float32 (PC)"),
          ("float-sosd", "variante p11_c_sosd (PC)")]
ORDEM = ["y_prep", "stft_S", "grade", "sal", "e_norm", "custo", "vit_path", "vit_score",
         "f_rot", "P_mean", "env_abs", "env_dec", "welch_f", "welch_P", "env_mod", "features",
         "z", "cos", "probs", "decisao", "hmm_log_em", "argmax", "hmm_viterbi",
         "hmm_online_alpha", "hmm_online", "absorb"]


def mais_recente(build):
    cands = sorted(p for p in CSV.glob(f"equivalencia_{build}_*.csv")
                   if not p.name.endswith(("_resumo.csv",)) and "_emul_" not in p.name)
    return cands[-1] if cands else None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dados, metas = {}, {}
    for b, _ in BUILDS:
        p = mais_recente(b)
        if p is None:
            continue
        dados[b] = pd.read_csv(p)
        metas[b] = {"csv": p.name, "sha256_csv": sha(p),
                    **json.loads(p.with_name(p.stem + "_meta.json").read_text(encoding="utf-8"))}

    linhas = []
    for b, rot in BUILDS:
        if b not in dados:
            continue
        d = dados[b]
        d = d[d.tipo_item.isin(["caso", "seq"])]
        for et in ORDEM:
            g = d[d.etapa == et]
            if not len(g):
                continue
            st = g.status.value_counts()
            n_fail = int(st.get("FAIL", 0) + st.get("FAIL_JUSANTE", 0) + st.get("AUSENTE", 0))
            linhas.append({"build": b, "rotulo": rot, "etapa": et, "n": len(g),
                           "pass": int(st.get("PASS", 0)), "pass_empate": int(st.get("PASS_EMPATE", 0)),
                           "fail": n_fail, "razao_tol_max": float(g.razao_tol.max()),
                           "veredito": "PASS" if n_fail == 0 else "FAIL"})
    r = pd.DataFrame(linhas)

    # ---------------- markdown
    md = ["# Equivalência Python × C no PC: resultados", "",
          "Gerado por `tests/equivalence/resumo_equivalencia.py` a partir dos CSVs abaixo; não "
          "editar à mão. Tolerâncias congeladas no commit `447a7c2` "
          "(`configs/embedded/tolerancias.json`), antes de qualquer saída do C existir. "
          "Categoria: **medição no PC** (MSVC 19.44, x64, `/O2 /fp:precise`). Nada aqui é "
          "medição no ESP32.", ""]
    for b, rot in BUILDS:
        if b in metas:
            m = metas[b]
            md.append(f"- {rot}: `results/csv/{m['csv']}` (sha256 `{m['sha256_csv'][:16]}…`, "
                      f"tolerâncias `{str(m.get('sha256_tolerancias', ''))[:16]}…`, "
                      f"conjunto de tolerâncias `{m.get('tol_build', b)}`, commit "
                      f"`{m.get('commit', m.get('projeto', {}))}`)")
    md += ["", "## Veredito por etapa", "",
           "| etapa | " + " | ".join(rot for b, rot in BUILDS if b in dados) + " |",
           "|---|" + "---|" * len(dados)]
    for et in ORDEM:
        cel = []
        for b, _ in BUILDS:
            g = r[(r.build == b) & (r.etapa == et)]
            if not len(g):
                continue
            x = g.iloc[0]
            emp = f", {x.pass_empate} empate" if x.pass_empate else ""
            cel.append(f"{x.veredito} ({x['pass']}/{x.n}{emp}; máx {x.razao_tol_max:.2g}× tol)")
        if cel:
            md.append(f"| `{et}` | " + " | ".join(cel) + " |")
    md += ["", "`máx k× tol` é a maior razão erro/tolerância entre os itens da etapa "
           "(PASS exige ≤ 1, salvo empate demonstrado).", "",
           "## Todas as discrepâncias (status diferente de PASS)", ""]
    alguma = False
    for b, rot in BUILDS:
        if b not in dados:
            continue
        d = dados[b]
        x = d[~d.status.isin(["PASS"]) & d.status.notna()]
        x = x[x.tipo_item.isin(["caso", "seq", "feature"])]
        for _, l in x.iterrows():
            alguma = True
            det = l.get("detalhe", "")
            det = "" if (isinstance(det, float) and np.isnan(det)) else str(det)
            md.append(f"- **{rot}**, `{l.id}` ({l.tipo_caso}, modelo {l.modelo}), etapa "
                      f"`{l.etapa}`: {l.status}; erro máx {l.err_max:.4g}, razão "
                      f"{l.razao_tol:.4g}× tol. {det}")
    if not alguma:
        md.append("Nenhuma.")
    md += ["", "## Nota da auditoria independente (F)", "",
           "Os vereditos acima aplicam a regra congelada em `447a7c2`, que aceita como "
           "PASS_EMPATE um argmax do Welch diferente quando P_ref[bin da referência] − "
           "P_ref[bin do C] ≤ 2·tol absoluta. A auditoria "
           "(`reports/independent_scientific_audit.md`, M1) mostrou que nos casos de borda "
           "DC+ruído essa diferença absoluta é pequena só porque o espectro inteiro está no "
           "piso do float32: em termos relativos o pico da referência é várias vezes o bin "
           "escolhido pelo C, então não é empate real. As tolerâncias e o comparador não foram "
           "alterados depois disso; o artigo conta esses casos, de forma conservadora, como "
           "FAIL em `env_mod`, `features` e `z` (a razão exata é recalculada por "
           "`src/python/gerar_artigo_numeros.py`). A decisão de classe não muda em nenhum caso."]
    (EMB / "reports" / "c_equivalence_resultados_pc.md").write_text("\n".join(md) + "\n",
                                                                     encoding="utf-8")

    # ---------------- LaTeX
    tex = [r"% gerado por embarcados/tests/equivalence/resumo_equivalencia.py; não editar",
           r"\begin{tabular}{l" + "c" * len(dados) + "}", r"\toprule",
           "Stage & " + " & ".join({"double": "C double", "float": "C float32",
                                     "float-sosd": "C float32+SOS64"}[b]
                                    for b, _ in BUILDS if b in dados) + r" \\", r"\midrule"]
    for et in ORDEM:
        cel = []
        for b, _ in BUILDS:
            g = r[(r.build == b) & (r.etapa == et)]
            if not len(g):
                continue
            x = g.iloc[0]
            cel.append(f"{x['pass'] + x.pass_empate}/{x.n}" + ("" if x.veredito == "PASS" else r"$^\dagger$"))
        if cel:
            tex.append(r"\texttt{" + et.replace("_", r"\_") + "} & " + " & ".join(cel) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (EMB / "tables" / "latex").mkdir(parents=True, exist_ok=True)
    (EMB / "tables" / "latex" / "equivalencia.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # ---------------- figura
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    ets = [e for e in ORDEM if e in set(r.etapa) and e not in ("decisao", "argmax", "hmm_viterbi",
                                                                  "hmm_online", "absorb", "vit_path")]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    cores = {"double": "#3b6ea8", "float": "#c8553d", "float-sosd": "#6a994e"}
    w = 0.27
    for k, (b, rot) in enumerate(BUILDS):
        if b not in dados:
            continue
        v = [max(r[(r.build == b) & (r.etapa == e)].razao_tol_max.max(), 1e-6) for e in ets]
        ax.bar(np.arange(len(ets)) + (k - 1) * w, v, w, label=rot, color=cores[b])
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("max error / tolerance")
    ax.set_xticks(np.arange(len(ets)))
    ax.set_xticklabels(ets, rotation=70, ha="right", fontsize=6)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        (EMB / "figures" / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(EMB / "figures" / ext / f"equivalencia_erro_etapas.{ext}", dpi=300)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    r.to_csv(CSV / f"equivalencia_resumo_por_build_{ts}.csv", index=False)
    (CSV / f"equivalencia_resumo_por_build_{ts}_meta.json").write_text(
        json.dumps({"gerado_em": ts, "fontes": metas}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(r.groupby("build").veredito.value_counts().to_string())


if __name__ == "__main__":
    main()
