"""Gera analise/COMPARATIVO_BASES.md juntando os números da auditoria e os atributos das fichas.

Números: índices (`out/indices/<id>.csv`, com os mesmos filtros usados na auditoria) e os CSVs
mais recentes de exp1, exp4 e exp5. Atributos qualitativos (licença, login, tamanho publicado,
falhas elétricas, ruído real, adequação a cada caminho de artigo): `analise/atributos_bases.csv`,
curado à mão a partir das fichas, com a ficha de origem em cada linha. IDMT e A2212 entram como
referência: unidades, classes e taxa vêm de `../docs/DATASETS.md`; o nível vem de
`scripts/referencia_artigo3.py`.

Uso: .venv/Scripts/python.exe scripts/comparativo.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from lib.comum import RAIZ, ler_indice, ultimo_csv

ATRIB = RAIZ / "analise" / "atributos_bases.csv"
SAIDA = RAIZ / "analise" / "COMPARATIVO_BASES.md"
# mesmos filtros da auditoria (ver _meta.json dos CSVs)
FILTROS = {"mimii": "unidade.str.contains('_id_')",
           "toycar": "not unidade.str.startswith('desconhecida')"}
ORDEM = ["mimii", "toycar", "mafaulda", "uoemd", "imvacd", "kaist", "idmt", "a2212"]
# referência transcrita de ../docs/DATASETS.md (§1 e §2), sem reanálise
REF = {"idmt": dict(unidades="3", classes="3", confusao="confundida (classe = motor)",
                    taxa="44,1 kHz", rotacao="não"),
       "a2212": dict(unidades="4", classes="3", confusao="parcial (rolamento = motor 4)",
                     taxa="16 kHz (1 arquivo a 44,1 kHz)", rotacao="velocidade comandada no nome")}
LIMIAR = 0.10   # nível conta como atalho se superar o acaso em mais de 10 pontos


def _bacc(df, **f):
    d = df[df["dobra"] == "total"]
    for k, v in f.items():
        d = d[d[k] == v]
    return float(d["bacc"].mean()) if len(d) else np.nan


def numeros(b):
    idx = ler_indice(b)
    if b in FILTROS:
        idx = idx.query(FILTROS[b])
    m = pd.crosstab(idx["classe"], idx["unidade"])
    com2 = int(((m > 0).sum(axis=0) >= 2).sum())
    nu = idx["unidade"].nunique()
    confusao = ("confundida (classe = unidade)" if com2 == 0 else
                "não confundida" if com2 == nu else f"parcial ({com2} de {nu} unidades com ≥2 classes)")
    taxas = sorted(idx["sr"].dropna().unique().astype(int))
    if idx["rot_esperada_hz"].notna().any():
        rot = f"tacômetro ({int(idx['rot_esperada_hz'].notna().sum())} arquivos)"
    elif idx["f_alim_hz"].notna().any():
        rot = f"setpoint do inversor ({int(idx['f_alim_hz'].notna().sum())} arquivos)"
    else:
        rot = "não"
    out = dict(unidades=str(nu), classes=str(idx["classe"].nunique()), confusao=confusao,
               taxa=", ".join(f"{t / 1000:g} kHz" for t in taxas), rotacao=rot,
               horas=f"{idx['duracao_s'].sum() / 3600:.2f}")
    # atalho de nível (exp1)
    p = ultimo_csv("exp1_nivel", b)
    if p:
        e = pd.read_csv(p)
        t = e[e["tarefa"] == "todas"]
        g = _bacc(t, split="gravacao"); a = float(t["acaso"].dropna().iloc[0])
        u = e[e["tarefa"].str.startswith("unidade=") & (e["split"] == "gravacao") & (e["dobra"] == "total")]
        partes = [f"{g:.2f} vs acaso {a:.3f} (gravação)" if np.isfinite(g) else
                  f"{_bacc(t, split='aleatorio'):.2f} vs acaso {a:.3f} (aleatório; sem split por gravação)"]
        sim = (np.isfinite(g) and g - a > LIMIAR) or (not np.isfinite(g) and _bacc(t, split="aleatorio") - a > LIMIAR)
        if len(u):
            partes.append(f"{u['bacc'].median():.2f} dentro da unidade (acaso 0.50)")
            sim = sim or (u["bacc"].median() - 0.5 > LIMIAR)
        out["nivel"] = ("sim: " if sim else "não: ") + "; ".join(partes)
    # reconhecimento da unidade (exp5)
    p = ultimo_csv("exp5_unidade", b)
    if p:
        e = pd.read_csv(p)
        if "dobra" in e:
            t = e[(e["tarefa"] == "todas") & (e["features"] == "forma")]
            g = _bacc(t, split="gravacao")
            acaso = t["acaso"].dropna()
            if np.isfinite(g) and len(acaso):
                out["unidade_rec"] = f"{g:.2f} (acaso {acaso.iloc[0]:.3f})"
    # rastreador (exp4)
    p = ultimo_csv("exp4_rastreador", b)
    if p:
        e = pd.read_csv(p)
        out["rastreador"] = f"{(e['erro_rel_arquivo'] <= 0.02).mean():.2f} dos arquivos a ≤2 %"
    return out


def nivel_ref(b):
    p = ultimo_csv("ref_artigo3", "idmt_a2212")
    r = pd.read_csv(p)
    r = r[(r.base == b) & (r.pipeline == "b00_nivel") & r.normalizar_rms & (r.split == "gravacao")]
    g, a = float(r["bacc"].iloc[0]), float(r["acaso"].iloc[0])
    return ("sim: " if g - a > LIMIAR else "não: ") + f"{g:.2f} vs acaso {a:.3f} (protocolo do Artigo3)"


def main():
    at = pd.read_csv(ATRIB, dtype=str, keep_default_na=False).set_index("base")
    linhas = {}
    for b in ORDEM:
        n = REF[b] | {"nivel": nivel_ref(b), "horas": "—"} if b in REF else numeros(b)
        linhas[b] = n | at.loc[b].to_dict()

    L = ["# Comparativo das bases\n",
         f"Gerado por `scripts/comparativo.py` em {datetime.now():%d/%m/%Y %H:%M}; não editar à mão. "
         "Números da auditoria vêm de `out/indices/` e `out/experiments/` (ver "
         "`analise/tabelas_auditoria.md`); licença, login, tamanho publicado, tipo de falha, ruído e "
         "adequação vêm de `analise/atributos_bases.csv`, que cita a ficha de cada base. IDMT e A2212 "
         "são referência, transcritas de `../docs/DATASETS.md` e dos CSVs do trabalho do ICIT. "
         f"Na MIMII e no ToyCar as contagens usam o subconjunto auditado (`{FILTROS['mimii']}`; "
         f"`{FILTROS['toycar']}`). Atalho de nível = acurácia balanceada de RMS + pico mais de "
         f"{LIMIAR:.2f} acima do acaso, no split por gravação ou dentro da unidade.\n",
         "## Estrutura e riscos\n",
         "| base | unidades físicas | classes | classe × unidade | canal acústico | verdade de rotação | falhas elétricas/tensão | ruído real | tamanho | licença | atalho de nível | unidade reconhecível (forma, split por gravação) | rastreador |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, x in linhas.items():
        L.append(f"| **{b}** | {x['unidades']} | {x['classes']} | {x['confusao']} | {x['taxa']} | "
                 f"{x['rotacao']} | {x['falhas_eletricas_tensao']} | {x['ruido_real']} | "
                 f"{x['tamanho_publicado']} | {x['licenca']} | {x.get('nivel', '—')} | "
                 f"{x.get('unidade_rec', '—')} | {x.get('rastreador', '—')} |")
    L += ["\n## Adequação a cada caminho de artigo\n",
          "Caminhos: (2) quais distúrbios de alimentação são audíveis; (4) tacômetro acústico; "
          "(5) vários motores, um microfone; (6) auditoria de avaliação; (8) anomalia sem rótulo com "
          "features harmônicas; (ordens) features de ordem relativas à rotação rastreada. "
          "\"Caso de estudo\" em (6) quer dizer que a base não permite deixar um motor de fora, mas "
          "é exatamente o tipo de base que uma auditoria precisa examinar.\n",
          "| base | (2) | (4) | (5) | (6) | (8) | ordens |", "|---|---|---|---|---|---|---|"]
    for b, x in linhas.items():
        L.append(f"| **{b}** | {x['c2_disturbios_alimentacao']} | {x['c4_tacometro_acustico']} | "
                 f"{x['c5_varios_motores_um_microfone']} | {x['c6_auditoria_avaliacao']} | "
                 f"{x['c8_anomalia_sem_rotulo']} | {x['ordens']} |")
    L += ["\n## Fontes por base\n"] + [f"- **{b}**: {x['nome']} — {x['referencia']}; ficha `{x['fonte']}`; "
                                       f"login: {x['login']}" for b, x in linhas.items()]
    SAIDA.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
