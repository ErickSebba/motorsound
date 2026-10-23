"""Consolida os CSVs da auditoria em tabelas Markdown (Etapa 4).

Lê, para cada base, o índice `out/indices/<id>.csv` e o CSV mais recente de cada
experimento em `out/experiments/`, e escreve `analise/tabelas_auditoria.md`, gerado
automaticamente. Os textos de `analise/` citam estas tabelas; nenhum número é digitado à mão.

Uso: .venv/Scripts/python.exe scripts/consolidar.py [base ...]
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from lib.comum import INDICES, RAIZ, ler_indice, ultimo_csv

SAIDA = RAIZ / "analise" / "tabelas_auditoria.md"


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _rel(p: Path | None) -> str:
    return f"`{p.relative_to(RAIZ).as_posix()}`" if p else "—"


def inventario(base):
    idx = ler_indice(base)
    m = pd.crosstab(idx["classe"], idx["unidade"])
    diag = int((m > 0).sum(axis=0).max())          # máx. de classes numa mesma unidade
    return dict(
        arquivos=len(idx), gravacoes=idx["gravacao_id"].nunique(),
        unidades=idx["unidade"].nunique(), classes=idx["classe"].nunique(),
        horas=idx["duracao_s"].sum() / 3600, sr=sorted(idx["sr"].dropna().unique().astype(int)),
        max_classes_por_unidade=diag,
        unidades_com_2_classes=int(((m > 0).sum(axis=0) >= 2).sum()),
        com_f_alim=int(idx["f_alim_hz"].notna().sum()),
        com_rot=int(idx["rot_esperada_hz"].notna().sum()), matriz=m)


def bacc_total(df, **filtro):
    d = df[df["dobra"] == "total"]
    for k, v in filtro.items():
        d = d[d[k] == v]
    return (float(d["bacc"].mean()), float(d["acaso"].mean())) if len(d) else (np.nan, np.nan)


def main():
    bases = sys.argv[1:] or sorted(p.stem for p in INDICES.glob("*.csv"))
    L = [f"# Tabelas da auditoria (geradas por `scripts/consolidar.py`)\n",
         f"Gerado em {datetime.now():%d/%m/%Y %H:%M}. Não editar à mão. Acurácia balanceada "
         "(bacc) das predições concatenadas de todas as dobras; `acaso` = 1/(classes no teste).\n"]

    L += ["## 1. Inventário (de `out/indices/<id>.csv`)\n",
          "| base | arquivos | gravações | unidades | classes | horas | taxa (Hz) | máx. classes numa unidade | unidades com ≥2 classes | arquivos com f_alim | com rotação medida |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    invs = {}
    for b in bases:
        iv = invs[b] = inventario(b)
        L.append(f"| {b} | {iv['arquivos']} | {iv['gravacoes']} | {iv['unidades']} | {iv['classes']} | "
                 f"{iv['horas']:.2f} | {', '.join(map(str, iv['sr']))} | {iv['max_classes_por_unidade']} | "
                 f"{iv['unidades_com_2_classes']} | {iv['com_f_alim']} | {iv['com_rot']} |")

    L += ["\n## 2. Exp1 — controle de nível (RMS + pico → classe), tarefa `todas`\n",
          "| base | aleatório | por gravação | por unidade (motor fora) | acaso | CSV |", "|---|---|---|---|---|---|"]
    for b in bases:
        p = ultimo_csv("exp1_nivel", b)
        if not p:
            L.append(f"| {b} | — | — | — | — | — |"); continue
        df = pd.read_csv(p); df = df[df["tarefa"] == "todas"]
        r = {s: bacc_total(df, split=s) for s in ("aleatorio", "gravacao", "unidade")}
        L.append(f"| {b} | {_f(r['aleatorio'][0])} | {_f(r['gravacao'][0])} | {_f(r['unidade'][0])} | "
                 f"{_f(r['aleatorio'][1])} | {_rel(p)} |")

    p = ultimo_csv("ref_artigo3", "idmt_a2212")
    if p:
        ref = pd.read_csv(p)
        L += [f"\nReferência IDMT e A2212 (protocolo do Artigo3, não reanalisadas; {_rel(p)}): "
              "acurácia balanceada com RMS normalizado.\n",
              "| base | pipeline | aleatório | por gravação/blocos | por unidade (motor fora) | acaso |",
              "|---|---|---|---|---|---|"]
        for (b, pl), g in ref[ref.normalizar_rms].groupby(["base", "pipeline"]):
            v = {s: g.loc[g.split == s, "bacc"].mean() for s in ("aleatorio", "gravacao", "unidade")}
            L.append(f"| {b} | {pl} | {_f(v['aleatorio'])} | {_f(v['gravacao'])} | {_f(v['unidade'])} | "
                     f"{_f(g.acaso.iloc[0])} |")

    L += ["\n## 3. Exp2 — rótulos permutados (média de 5 permutações, tarefa `todas`)\n",
          "Permutação por segmento tem de dar o acaso em todo split. Permutação por gravação "
          "acima do acaso no split aleatório significa que as features reconhecem a gravação.\n",
          "| base | features | permutação | aleatório | por gravação | por unidade | acaso |",
          "|---|---|---|---|---|---|---|"]
    for b in bases:
        p = ultimo_csv("exp2_permutado", b)
        if not p:
            continue
        df = pd.read_csv(p)
        for (fe, pm), g in df.groupby(["features", "permutacao"]):
            r = {s: bacc_total(g, split=s) for s in ("aleatorio", "gravacao", "unidade")}
            L.append(f"| {b} | {fe} | {pm} | {_f(r['aleatorio'][0])} | {_f(r['gravacao'][0])} | "
                     f"{_f(r['unidade'][0])} | {_f(r['aleatorio'][1])} |")

    L += ["\n## 4. Exp3 — distância entre espectros médios (dB RMS, só forma, 50–7900 Hz)\n",
          "| base | classe diferente, mesma unidade | unidade diferente, mesma classe | ambas diferentes | CSV |",
          "|---|---|---|---|---|"]
    for b in bases:
        p = ultimo_csv("exp3_espectros", b)
        if not p:
            continue
        df = pd.read_csv(p); med = df.groupby("tipo_par")["dist_db_rms"].agg(["median", "count"])
        cel = lambda t: (f"{med.loc[t, 'median']:.2f} (n={int(med.loc[t, 'count'])})" if t in med.index else "—")
        L.append(f"| {b} | {cel('classe_diferente_mesma_unidade')} | {cel('unidade_diferente_mesma_classe')} | "
                 f"{cel('ambas_diferentes')} | {_rel(p)} |")

    L += ["\n## 5. Exp4 — rastreador (regra pré-registrada: 0,40–1,05 × f_alim)\n",
          "`mod. oitava` é diagnóstico a posteriori: conta como acerto também f_ref/2 e 2·f_ref.\n",
          "| base | condição | arquivos | erro rel. mediano | frac. arquivos ≤2 % | frac. segmentos ≤2 % | f_alim/f_rot mediana | frac. ≤2 % mod. oitava | CSV |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b in bases:
        p = ultimo_csv("exp4_rastreador", b)
        if not p:
            continue
        df = pd.read_csv(p)
        df["oit"] = np.minimum.reduce([np.abs(df.f_rot_mediana - k * df.ref_hz) / (k * df.ref_hz) for k in (0.5, 1, 2)])
        for nome, g in [("todas", df)] + [(f"{v}", g) for v, g in df.groupby("velocidade")]:
            L.append(f"| {b} | {nome} | {len(g)} | {_f(g.erro_rel_arquivo.median())} | "
                     f"{_f((g.erro_rel_arquivo <= 0.02).mean())} | {_f(g.frac_seg_2pct.mean())} | "
                     f"{_f(g.razao_alim_rot.median())} | {_f((g.oit <= 0.02).mean())} | {_rel(p) if nome == 'todas' else ''} |")

    L += ["\n## 6. Exp5 — separabilidade por unidade (forma espectral → unidade)\n",
          "| base | tarefa | features | aleatório | por gravação | acaso | CSV |", "|---|---|---|---|---|---|---|"]
    for b in bases:
        p = ultimo_csv("exp5_unidade", b)
        if not p:
            continue
        df = pd.read_csv(p)
        if "dobra" not in df:
            continue
        for (t, fe), g in df[df["dobra"] == "total"].groupby(["tarefa", "features"]):
            r = {s: bacc_total(g, split=s) for s in ("aleatorio", "gravacao")}
            L.append(f"| {b} | {t} | {fe} | {_f(r['aleatorio'][0])} | {_f(r['gravacao'][0])} | "
                     f"{_f(r['aleatorio'][1])} | {_rel(p)} |")

    L += ["\n## 7. Matrizes classe × unidade (arquivos)\n"]
    for b in bases:
        m = invs[b]["matriz"]
        if m.shape[1] > 16:
            L.append(f"### {b}\n\n{m.shape[1]} unidades; matriz omitida (ver ficha).\n"); continue
        L.append(f"### {b}\n\n" + m.to_markdown() + "\n")

    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
