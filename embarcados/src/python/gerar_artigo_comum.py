"""Funções comuns dos geradores do artigo (gerar_artigo_*.py). Agente E, Etapa 7.

Nada aqui calcula resultado: só localiza os arquivos de origem em `embarcados/results/`,
calcula sha256, lê o commit e formata números. Os geradores importam este módulo.

Regras que este módulo garante:
- o arquivo de origem é sempre o MAIS RECENTE que casa com o padrão (timestamp no nome), e o
  caminho exato vai para o numeros.json e para o manifesto;
- números do ESP32 físico só existem se os CSV/JSON da bancada existirem; caso contrário o
  valor é None e o LaTeX recebe \\textbf{[pending]} (nunca uma estimativa no lugar).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
EMB = RAIZ / "embarcados"
RES = EMB / "results"
PAPER = EMB / "paper"
TABS = EMB / "tables" / "latex"
FIGS = EMB / "figures"
ESPRESSIF = Path("C:/Espressif")
PENDENTE_TEX = r"\textbf{[pending]}"
PENDENTE_PT = "**[pendente]**"

if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# categorias do CLAUDE.md de embarcados/ (nomes usados nas macros e no texto)
CAT = {
    "codigo": "parâmetro lido do código Python da referência",
    "dados": "contagem lida das predições/CSV do experimento",
    "estimativa": "1 · estimativa analítica",
    "pc": "2 · medição no PC",
    "build": "3 · build do firmware (idf.py size)",
    "esp32": "4 · medição no ESP32 físico",
    "diagnostico": "resultado de classificação (validação, PC)",
    "equivalencia": "equivalência Python × C (PC)",
}


def rel(p: Path) -> str:
    """Caminho relativo à raiz do Artigo3 quando possível (senão absoluto, com barras /)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return p.as_posix()


def sha256(p: Path) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def commit() -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(RAIZ), *a], capture_output=True, text=True,
                              encoding="utf-8").stdout.strip()
    return {"commit": g("rev-parse", "HEAD"),
            "sujo": bool(g("status", "--porcelain", "--", "embarcados"))}


def mais_recente(pasta: Path, padrao: str, excluir=("_meta", "_resumo")) -> Path | None:
    """Arquivo mais recente pelo nome (os nomes carregam timestamp ordenável)."""
    cs = [c for c in sorted(Path(pasta).glob(padrao))
          if not any(c.stem.endswith(e) for e in excluir)]
    return cs[-1] if cs else None


def ler_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ formatação
def fmt(v, casas=3, sci=False):
    """Número em inglês (ponto decimal). None → None (o chamador decide o pending)."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if sci:
        m, e = f"{v:.{casas}e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    if isinstance(v, int) or (float(v).is_integer() and casas == 0):
        return f"{int(round(v)):,}".replace(",", "{,}")
    return f"{v:.{casas}f}"


def tex_para_pt(s: str) -> str:
    """Formato LaTeX inglês → texto corrido em português (vírgula decimal, espaço de milhar)."""
    if s is None:
        return PENDENTE_PT
    s = re.sub(r"^\\ensuremath\{(.*)\}$", r"\1", s)
    s = s.replace("{,}", "\u202f")
    s = re.sub(r"(\d)\.(\d)", r"\1,\2", s)
    s = re.sub(r"\\times10\^\{(-?\d+)\}", lambda m: f" × 10^{m.group(1)}", s)
    s = s.replace("$", "").replace("\\%", "%").replace("\\textbf{[pending]}", PENDENTE_PT)
    return s


def nome_macro(chave: str) -> str:
    if not re.fullmatch(r"V[A-Za-z]+", chave):
        raise ValueError(f"nome de macro inválido (só letras, prefixo V): {chave}")
    return "\\" + chave
