"""Gera artigo-ICIT/RESULTADOS_tabelas.md e as figuras a partir dos CSVs desta fase.

Nenhum número de RESULTADOS.md é digitado à mão: as tabelas saem daqui, cada uma com o
CSV de origem. Os CSVs usados estão listados em ENTRADAS (fase exploratória de 14/10/2026).

    python experimentos/tabelas_resultados.py
"""
import json
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(RAIZ / "experimentos"))
import analise  # noqa: E402

E = RAIZ / "out" / "experiments"
ENTRADAS = {
    "validacao": ["20261014_185859_validacao.csv", "20261014_203110_validacao_cnn.csv"],
    "robustez_idmt": ["20261014_192347_robustez_idmt.csv", "20261014_205140_robustez_idmt.csv"],
    "robustez_a2212": ["20261014_193054_robustez_a2212.csv", "20261014_210231_robustez_a2212.csv"],
    "sequencias": ["20261014_212255_sequencias_idmt.csv", "20261014_212304_sequencias_idmt.csv",
                   "20261014_212313_sequencias_idmt.csv", "20261014_212322_sequencias_idmt.csv"],
    "estatistica": ["20261014_213212_estatistica.csv"],
    "anomalia": ["20261014_184738_anomalia.csv"],
    "bancada": ["20261014_185415_bancada_sintetica.csv"],
    "tacometro": ["20261014_175806_tacometro_a2212.csv"],
}


def ler(chave):
    return pd.concat([pd.read_csv(E / f) for f in ENTRADAS[chave]], ignore_index=True)


def origem(chave):
    return "Origem: " + ", ".join(f"`out/experiments/{f}`" for f in ENTRADAS[chave])


def rot(c):
    d = json.loads(c)
    t = d.pop("tipo")
    return t if not d else f"{t} " + " ".join(f"{v:g}" for v in d.values())


def custos():
    out = {}
    for f in ["20261014_185859_validacao_meta.json", "20261014_203110_validacao_cnn_meta.json"]:
        out.update(json.loads((E / f).read_text(encoding="utf-8"))["custo_mcu"])
    return out


def main():
    md = ["# Tabelas de resultados (geradas)\n",
          "Gerado por `experimentos/tabelas_resultados.py`; não editar à mão. Todas as "
          "acurácias são por janela, na validação (o teste do IDMT não foi lido).\n"]

    # T1 validação + auditorias
    v = ler("validacao")
    t = v.groupby(["pipeline", "base", "split", "normalizar_rms"]).acerto.mean().unstack(
        ["base", "split", "normalizar_rms"])
    cols = [("idmt", "blocos", True), ("idmt", "aleatorio", True), ("idmt", "blocos", False),
            ("a2212", "arquivo", True), ("a2212", "aleatorio", True), ("a2212", "motor", True)]
    t1 = t[cols].copy()
    t1.columns = ["IDMT blocos", "IDMT aleatório", "IDMT sem norm.", "A2212 arquivo",
                  "A2212 aleatório", "A2212 motor fora"]
    t1["Δ vazamento A2212"] = t1["A2212 aleatório"] - t1["A2212 arquivo"]
    ics = []
    for p in t1.index:
        g = v[(v.pipeline == p) & (v.base == "a2212") & (v.split == "arquivo")
              & (v.normalizar_rms == True)]  # noqa: E712
        lo, hi = analise.ic_por_grupo(g)
        ics.append(f"[{lo:.2f}, {hi:.2f}]")
    t1["IC95 A2212 arquivo"] = ics
    md += ["\n## T1. Validação limpa e auditorias de vazamento e de nível\n", origem("validacao"),
           "\n\nIC95 por bootstrap de gravações (A2212). 'motor fora' = deixa-um-motor-de-fora.\n\n",
           t1.round(3).reset_index().to_markdown(index=False), "\n"]

    # T2 robustez IDMT
    r = ler("robustez_idmt")
    t2 = r.groupby(["pipeline", "condicao_sintetica"]).acerto.mean().unstack()
    t2.columns = [rot(c) for c in t2.columns]
    ordem = ["limpo", "branco 20", "branco 10", "branco 5", "branco 0", "branco -5",
             "rosa 20", "rosa 10", "rosa 5", "rosa 0", "marrom 0", "percussivo 8 1.5",
             "reverb 0.35", "ganho 6"]
    t2 = t2[[c for c in ordem if c in t2.columns]]
    md += ["\n## T2. Robustez no IDMT (treino limpo, validação perturbada)\n", origem("robustez_idmt"),
           "\n\nUm modelo por fold, treinado no limpo e aplicado a cada condição. Acaso = 0,33.\n\n",
           t2.round(2).reset_index().to_markdown(index=False), "\n"]

    # T3 robustez A2212 (resumo)
    r2 = ler("robustez_a2212")
    t3 = r2.groupby(["pipeline", "condicao_sintetica"]).acerto.mean().unstack()
    t3.columns = [rot(c) for c in t3.columns]
    t3 = t3[[c for c in ["limpo", "branco 10", "rosa 10", "rosa 5", "marrom 0", "percussivo 8 1.5"]
             if c in t3.columns]]
    md += ["\n## T3. Robustez no A2212 (por arquivo)\n", origem("robustez_a2212"), "\n\n",
           t3.round(2).reset_index().to_markdown(index=False), "\n"]

    # T4 estatística
    st = ler("estatistica")
    st["cenario"] = st.cenario.map(rot)
    t4 = st[["base", "cenario", "a", "b", "isola", "media_a", "media_b", "diferenca", "ic95_min",
             "ic95_max", "p_bootstrap_holm", "classificacao"]]
    md += ["\n## T4. Comparações pareadas (diferença = b − a)\n", origem("estatistica"),
           "\n\nBootstrap pareado de grupos (15 no IDMT, 43 no A2212), 10 000 reamostragens, Holm "
           "sobre as 28 comparações.\n\n", t4.round(3).to_markdown(index=False), "\n"]

    # T5 sequências
    s = ler("sequencias")
    s["condicao_sintetica"] = s.condicao_sintetica.map(lambda c: rot(c) if c != "limpo" else c)
    sel = ["p00_logmel_cnn", "p02_grollmisch", "p10_decifrae_pyin_viterbi",
           "p11_decifrae_harmsum", "p20b_so_frot", "p21_envelope"]
    t5 = (s[(s.pipeline.isin(sel)) & (s.change_penalty.isin([2.0, 8.0]) | (s.decodificacao == "argmax"))]
          .groupby(["condicao_sintetica", "pipeline", "decodificacao", "change_penalty"])
          [["acuracia", "atraso_mediano_s", "espurias_por_min", "perdidas"]].mean().reset_index())
    t5 = t5[~((t5.decodificacao == "argmax") & (t5.change_penalty != 2.0))]
    t5.loc[t5.decodificacao == "argmax", "change_penalty"] = np.nan
    md += ["\n## T5. Sequências: atraso, trocas espúrias e trocas perdidas\n", origem("sequencias"),
           "\n\n30 sequências por condição, permanências de 5–20 janelas de 3 s. argmax = sem HMM "
           "(a 'p13'); viterbi = HMM offline; online = filtro causal. 'perdidas' = trocas "
           "verdadeiras nunca detectadas, por sequência.\n\n",
           t5.round(2).to_markdown(index=False), "\n"]

    # T6 anomalia
    an = ler("anomalia")
    cols_an = [c for c in an.columns if c in ("pipeline", "cenario", "anomalo", "auc", "pauc")]
    if "anomalo" in an.columns:
        t6 = an[cols_an].groupby(["pipeline", "cenario", "anomalo"]).mean().reset_index()
    else:
        t6 = an[cols_an].groupby(["pipeline", "cenario"]).mean().reset_index()
    md += ["\n## T6. Detecção de anomalia treinada só no normal\n", origem("anomalia"), "\n\n",
           t6.round(3).to_markdown(index=False), "\n"]

    # T7 bancada sintética
    b = ler("bancada")
    t7 = (b[b.snr_db == 10.0].groupby(["variante", "disturbio"])
          [["erro_pct_med", "frac_ok5", "taxa_r133"]].median().unstack("disturbio"))
    md += ["\n## T7. Bancada sintética (verdade conhecida), SNR 10 dB\n", origem("bancada"),
           "\n\nMediana entre 10 sementes. erro_pct_med = erro mediano de f_rot (%); frac_ok5 = "
           "fração de quadros com erro < 5 %; taxa_r133 = fração travada em 4/3·f. Variantes "
           "sem sufixo = grade de 10 cents; `_2c` = 2 cents.\n\n",
           t7["erro_pct_med"].round(2).reset_index().to_markdown(index=False), "\n\n",
           t7["frac_ok5"].round(2).reset_index().to_markdown(index=False), "\n"]

    # T8 tacômetro A2212
    tc = ler("tacometro")
    from scipy.stats import spearmanr
    linhas = []
    for (fmin, vit, c, m), g in tc.groupby(["fmin", "viterbi", "classe", "motor"]):
        if len(g) >= 4 and vit:
            rho, p = spearmanr(g.velocidade, g.f_rot_mediana)
            linhas.append({"faixa_min_Hz": fmin, "classe": c, "motor": m, "n": len(g),
                           "spearman": rho, "p": p})
    md += ["\n## T8. Rotação rastreada × velocidade comandada (A2212, com Viterbi)\n",
           origem("tacometro"), "\n\nGrade de 10 cents (rodado antes da correção).\n\n",
           pd.DataFrame(linhas).round(3).to_markdown(index=False), "\n"]

    # T9 custo
    cu = pd.DataFrame([{"pipeline": p, **(c if isinstance(c, dict) else {"nota": c})}
                       for p, c in custos().items()])
    keep = [c for c in ["pipeline", "params", "ram_kb", "flash_kb", "mflops_s", "nota"] if c in cu]
    md += ["\n## T9. Custo estimado para o ESP32 (analítico, não medido)\n",
           "Origem: `custo_mcu` nos `_meta.json` de " + ", ".join(ENTRADAS["validacao"]) + "\n\n",
           cu[keep].to_markdown(index=False), "\n"]

    (RAIZ / "artigo-ICIT" / "RESULTADOS_tabelas.md").write_text("".join(md), encoding="utf-8")
    # figuras
    rr = r.copy()
    tmp = E / "_tmp_rob_idmt.csv"
    rr.to_csv(tmp, index=False)
    f1 = analise.figura_snr(tmp)
    meta = {"custo_mcu": custos()}
    f2 = analise.figura_pareto(None, meta, tmp)
    tmp.unlink()
    print("escrito artigo-ICIT/RESULTADOS_tabelas.md;", f1.name, f2.name if f2 else "")


if __name__ == "__main__":
    main()
