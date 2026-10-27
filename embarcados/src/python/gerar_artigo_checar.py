"""Checador do LaTeX do artigo (não há TeX instalado: BLOQUEIOS B4).

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_artigo_checar.py

Para cada versão (regular: \\wipfalse; WiP: \\wiptrue), com os \\ifchipmedido e \\iffigesp
que numeros.tex e figuras.tex definem, o script expande os \\input, resolve os ramos
\\if…\\else…\\fi e confere:
  1. todo \\input existe;                      2. todo \\ref/\\eqref tem \\label no mesmo ramo;
  3. todo \\cite está no references.bib;       4. toda macro \\V… está em numeros.tex e
     toda \\Fig… em figuras.tex;               5. todo \\includegraphics aponta para arquivo existente;
  6. \\begin/\\end e chaves balanceados;        7. nenhum dígito "solto" na prosa do main.tex
     (fora de macros, rótulos, comentários e de nomes da lista BRANCA).
Grava paper/checagem_latex.txt e sai com código 1 se houver erro (os itens 7 são avisos).
"""
from __future__ import annotations

import re
import sys

import gerar_artigo_comum as C

BRANCA = ["ESP32-S3", "ESP32", "A2212", "float32", "IEEEtran", "p11_c_sosd", "p11", "p04",
          "M1--M3", "M4", "I2S", "Cortex-M4", "IDMT-ISA-ELECTRIC-ENGINE", "p95", "IEEEkeywords",
          "ICIT", "macro-F1"]
FLAGS_IF = ("wip", "chipmedido", "figesp")


def sem_comentarios(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def expandir(texto, base, faltando, visitados):
    def sub(m):
        nome = m.group(1)
        p = (base / nome)
        if p.suffix == "":
            p = p.with_suffix(".tex")
        if not p.exists():
            faltando.append(nome)
            return ""
        visitados.append(p)
        return expandir(sem_comentarios(p.read_text(encoding="utf-8")), base, faltando, visitados)
    return re.sub(r"\\input\{([^}]+)\}", sub, texto)


def resolver_ifs(t, flags):
    """Resolve \\ifX … \\else … \\fi para X em FLAGS_IF (aninhamento suportado)."""
    tok = re.compile(r"\\if(" + "|".join(FLAGS_IF) + r")(?![A-Za-z])|\\else(?![A-Za-z])|\\fi(?![A-Za-z])")
    out, pilha, i = [], [], 0      # pilha de (ativo_pai, cond, no_else)
    ativo = True
    for m in tok.finditer(t):
        if ativo:
            out.append(t[i:m.start()])
        i = m.end()
        s = m.group(0)
        if s.startswith("\\if"):
            pilha.append((ativo, flags[m.group(1)], False))
            ativo = ativo and flags[m.group(1)]
        elif s == "\\else":
            if not pilha:
                raise ValueError("\\else sem \\if")
            pai, cond, _ = pilha[-1]
            pilha[-1] = (pai, cond, True)
            ativo = pai and not cond
        else:
            if not pilha:
                raise ValueError("\\fi sem \\if")
            pai, _, _ = pilha.pop()
            ativo = pai
    if pilha:
        raise ValueError("\\if sem \\fi")
    if ativo:
        out.append(t[i:])
    return "".join(out)


def defs(arquivo, padrao):
    t = (C.PAPER / arquivo).read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\(" + padrao + r")\}\{(.*?)\}(?:%|\n|$)", t, re.M))


def flag_de(arquivo, nome):
    t = (C.PAPER / arquivo).read_text(encoding="utf-8")
    return f"\\{nome}true" in t


def checar_versao(wip, main, macros_v, macros_fig, bib):
    erros, avisos = [], []
    flags = {"wip": wip, "chipmedido": flag_de("numeros.tex", "chipmedido"),
             "figesp": flag_de("figuras.tex", "figesp")}
    faltando, vis = [], []
    corpo = main.split(r"\begin{document}", 1)[1]
    try:
        t = resolver_ifs(expandir(sem_comentarios(corpo), C.PAPER, faltando, vis), flags)
    except ValueError as e:
        return [f"ifs: {e}"], [], flags
    for f in faltando:
        erros.append(f"\\input inexistente: {f}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", t))
    for r in re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", t):
        if r not in labels:
            erros.append(f"\\ref{{{r}}} sem \\label nesta versão")
    for grupo in re.findall(r"\\cite\{([^}]+)\}", t):
        for k in grupo.split(","):
            if k.strip() not in bib:
                erros.append(f"\\cite{{{k.strip()}}} não está no references.bib")
    for m in set(re.findall(r"\\(V[A-Z][A-Za-z]*)", t)):
        if m not in macros_v:
            erros.append(f"macro \\{m} não definida em numeros.tex")
    for m in set(re.findall(r"\\(Fig[A-Z][A-Za-z]*)", t)):
        if m not in macros_fig:
            erros.append(f"macro \\{m} não definida em figuras.tex")
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t):
        m = re.fullmatch(r"\\(Fig[A-Za-z]+)", g.strip())
        cam = macros_fig.get(m.group(1), "") if m else g
        if not cam or not (C.PAPER / cam).exists():
            erros.append(f"\\includegraphics{{{g}}} -> '{cam}' não existe")
    b = re.findall(r"\\begin\{([^}]+)\}", t)
    e = re.findall(r"\\end\{([^}]+)\}", t)
    for amb in set(b) | set(e):
        if b.count(amb) != e.count(amb) + (1 if amb == "document" else 0) - (0):
            if not (amb == "document"):
                erros.append(f"\\begin{{{amb}}} ×{b.count(amb)} vs \\end ×{e.count(amb)}")
    prof = 0
    for ch in re.sub(r"\\[{}]", "", t):
        prof += (ch == "{") - (ch == "}")
        if prof < 0:
            erros.append("chave '}' sem par")
            break
    if prof > 0:
        erros.append(f"{prof} chave(s) '{{' sem par")
    # 7. dígitos soltos na prosa do main.tex (só o arquivo principal, já com ifs resolvidos)
    prosa = resolver_ifs(sem_comentarios(corpo), flags)
    prosa = re.sub(r"\\begin\{minipage\}\[[^\]]*\]\{[^}]*\}", " ", prosa)
    prosa = re.sub(r"\\(label|ref|eqref|cite|input|includegraphics|bibliography|bibliographystyle"
                   r"|texttt)(\[[^\]]*\])?\{[^}]*\}", " ", prosa)
    prosa = re.sub(r"\$[^$]*\$", " ", prosa)        # matemática: índices, não resultados
    prosa = re.sub(r"\(\d\)", " ", prosa)           # rótulos (1)-(4) das categorias
    prosa = re.sub(r"\\[A-Za-z]+", " ", prosa)
    prosa = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", prosa)
    for w in sorted(BRANCA, key=len, reverse=True):
        prosa = prosa.replace(w, " ")
    for i, linha in enumerate(prosa.splitlines()):
        for m in re.finditer(r"[^\s]*\d[^\s]*", linha):
            avisos.append(f"dígito na prosa: '{m.group(0)}' (linha expandida {i + 1})")
    return erros, avisos, flags


def main():
    main_tex = (C.PAPER / "main.tex").read_text(encoding="utf-8")
    macros_v = defs("numeros.tex", r"V[A-Za-z]+")
    macros_fig = defs("figuras.tex", r"Fig[A-Za-z]+")
    bib = set(re.findall(r"@\w+\{([^,]+),", (C.PAPER / "references.bib").read_text(encoding="utf-8")))
    linhas, n_err = [], 0
    for wip in (False, True):
        erros, avisos, flags = checar_versao(wip, main_tex, macros_v, macros_fig, bib)
        nome = "WiP (4 pp)" if wip else "regular (6 pp)"
        linhas.append(f"== versão {nome}; flags {flags}")
        linhas += [f"ERRO  {x}" for x in erros] or ["ok: inputs, refs, cites, macros, figuras, ambientes"]
        linhas += [f"AVISO {x}" for x in avisos]
        n_err += len(erros)
    usadas = set(re.findall(r"\\(V[A-Z][A-Za-z]*)", sem_comentarios(main_tex)))
    pend = [m for m in sorted(usadas) if "pending" in macros_v.get(m, "")]
    linhas.append(f"== macros usadas no main.tex: {len(usadas)}; pending entre elas: {len(pend)}")
    linhas += [f"   pending: \\{m}" for m in pend]
    (C.PAPER / "checagem_latex.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
