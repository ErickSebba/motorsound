"""Gera results/manifests/manifesto.json: liga cada tabela, figura e macro do artigo ao script
gerador, aos CSV/JSON de origem (com sha256) e ao commit git atual.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_artigo_manifesto.py

Agente E (Etapa 7). Rodar por último (depois de numeros, figuras, tabelas e textos).
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import gerar_artigo_comum as C


def item(arquivo, script, fontes, extra=None):
    p = C.RAIZ / arquivo if not str(arquivo).startswith("C:") else arquivo
    d = {"arquivo": C.rel(p), "sha256": C.sha256(p), "existe": p.exists() if hasattr(p, "exists")
         else None, "script": script,
         "fontes": [{"arquivo": f, "sha256": C.sha256(C.RAIZ / f if not f.startswith(("C:", "/"))
                                                    else f)} for f in sorted(set(fontes))]}
    if extra:
        d.update(extra)
    return d


def main():
    nj = C.ler_json(C.PAPER / "numeros.json")
    fj = C.ler_json(C.PAPER / "figuras.json")
    S_NUM = "embarcados/src/python/gerar_artigo_numeros.py"
    S_TAB = "embarcados/src/python/gerar_artigo_tabelas.py"
    S_FIG = "embarcados/src/python/gerar_artigo_figuras.py"
    S_TXT = "embarcados/src/python/gerar_artigo_textos.py"
    todas = sorted(nj["fontes"])
    out = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
           "gerado_por": "embarcados/src/python/gerar_artigo_manifesto.py",
           "projeto": C.commit(), "chip_medido": nj["chip_medido"],
           "categorias": C.CAT, "tabelas": [], "figuras": [], "macros": {}, "textos": []}
    # macros
    for k, v in nj["numeros"].items():
        out["macros"][k] = {"valor": v["valor"], "status": v["status"], "categoria": v["categoria"],
                            "coluna": v["coluna"], "script": S_NUM,
                            "fontes": [{"arquivo": f, "sha256": nj["fontes"].get(f)}
                                       for f in v["fonte"]]}
    out["numeros_tex"] = item("embarcados/paper/numeros.tex", S_NUM, todas)
    # tabelas geradas aqui: dependem das macros que citam
    for nome in ("bases_protocolos", "parametros_features", "recursos", "desempenho_compacto",
                 "equivalencia_compacta"):
        p = C.TABS / f"{nome}.tex"
        txt = p.read_text(encoding="utf-8")
        cab = re.search(r"a partir de (\S+)", txt).group(1)
        fontes = [cab] if cab != "embarcados/paper/numeros.json" else todas
        out["tabelas"].append(item(C.rel(p), S_TAB, fontes + ["embarcados/paper/numeros.json"]))
    # tabelas reaproveitadas (geradas por outros scripts)
    dg = C.mais_recente(C.RES / "csv", "diagnostico_2*.csv")
    out["tabelas"].append(item("embarcados/tables/latex/desempenho.tex",
                               "embarcados/src/python/diagnostico_saidas.py", [C.rel(dg)]))
    eqr = C.mais_recente(C.RES / "csv", "equivalencia_resumo_por_build_*.csv")
    meta = C.ler_json(eqr.with_name(eqr.stem + "_meta.json"))["fontes"]
    out["tabelas"].append(item("embarcados/tables/latex/equivalencia.tex",
                               "embarcados/tests/equivalence/resumo_equivalencia.py",
                               [C.rel(eqr)] + [C.rel(C.RES / "csv" / m["csv"]) for m in meta.values()]))
    # figuras
    for nome, d in fj.items():
        fontes = [f for f in d["fontes"] if not f.startswith("data/")]
        extra = {"dados_brutos_lidos": [f for f in d["fontes"] if f.startswith("data/")]}
        out["figuras"].append(item(d["pdf"], S_FIG, fontes, extra))
        png = d["pdf"].replace("/pdf/", "/png/").replace(".pdf", ".png")
        out["figuras"].append(item(png, S_FIG, fontes, extra))
    # textos
    for nome in ("METODOLOGIA.md", "RESULTADOS.md"):
        out["textos"].append(item(f"embarcados/paper/{nome}", S_TXT, ["embarcados/paper/numeros.json"]))
    out["textos"].append(item("embarcados/paper/main.tex", "redigido à mão; números só por macro",
                              ["embarcados/paper/numeros.tex", "embarcados/paper/figuras.tex"]))
    d = C.RES / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifesto.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str),
                                      encoding="utf-8")
    sem = [t["arquivo"] for t in out["tabelas"] + out["figuras"] + out["textos"] if not t["sha256"]]
    print(f"manifesto: {len(out['tabelas'])} tabelas, {len(out['figuras'])} figuras, "
          f"{len(out['macros'])} macros; arquivos ausentes: {sem or 'nenhum'}")


if __name__ == "__main__":
    main()
