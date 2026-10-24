"""Regenera todos os artefatos do artigo na ordem certa e roda o checador do LaTeX.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_artigo_tudo.py [--sem-espectro]

Ordem: numeros → figuras → tabelas → textos → manifesto → checar.
`--sem-espectro` pula a releitura do áudio de train/ do IDMT (a figura de espectro já existente
é mantida). Agente E (Etapa 7).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PY = sys.executable


def rodar(nome, *args):
    print(f"--- {nome}", flush=True)
    r = subprocess.run([PY, str(AQUI / nome), *args], cwd=AQUI.parents[2])
    if r.returncode != 0:
        raise SystemExit(f"{nome} falhou (código {r.returncode})")


def main():
    extra = [a for a in sys.argv[1:] if a == "--sem-espectro"]
    rodar("gerar_artigo_numeros.py")
    rodar("gerar_artigo_figuras.py", *extra)
    rodar("gerar_artigo_tabelas.py")
    rodar("gerar_artigo_textos.py")
    rodar("gerar_artigo_manifesto.py")
    rodar("gerar_artigo_checar.py")


if __name__ == "__main__":
    main()
