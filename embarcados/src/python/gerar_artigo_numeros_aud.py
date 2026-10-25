"""Gera paper/numeros_aud.tex e paper/numeros_aud.json: macros \\aud… da auditoria independente.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_artigo_numeros_aud.py

Lê os CSVs MAIS RECENTES (timestamp no nome) de results/csv/:
  aud_holdout_<ts>_resumo.csv   (aud_holdout.py, M2)      → \\audHoldout…
  aud_rotacao_motor_<ts>.csv    (aud_rotacao_motor.py, M4) → \\audRot…
  aud_motor_igual_<ts>.csv      (aud_motor_igual.py, M3)   → \\audMotor…
Mesmo estilo e formatação de gerar_artigo_numeros.py (ponto decimal, {,} de milhar,
\\ensuremath{m\\times10^{e}}); cada macro registra fonte e coluna no comentário do .tex e no .json.
Nenhum número é digitado aqui. Não altera numeros.tex.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gerar_artigo_comum as C  # noqa: E402

BUILDS = {"float": "Float", "double": "Double"}
VARS = {"p04_classicas": "Cls", "p11_decifrae_harmsum": "Dec", "b00_rmspico_norm": "RmsNorm",
        "b00_rmspico_bruto": "RmsRaw"}
MOTORES = {"M1": "One", "M2": "Two", "M3": "Three", "M4": "Four"}


def nome_macro(chave):
    if not re.fullmatch(r"aud[A-Z][A-Za-z]*", chave):
        raise ValueError(f"nome de macro inválido (só letras, prefixo aud): {chave}")
    return "\\" + chave


class Registro:
    """Igual a gerar_artigo_numeros.Registro, com o prefixo \\aud."""

    def __init__(self):
        self.n = {}

    def add(self, chave, valor, tex, fonte, coluna, cat, nota=""):
        nome_macro(chave)
        if chave in self.n:
            raise KeyError(f"macro duplicada: {chave}")
        pend = valor is None or (isinstance(valor, float) and not np.isfinite(valor))
        self.n[chave] = {
            "valor": valor if not isinstance(valor, (np.floating, np.integer)) else valor.item(),
            "tex": C.PENDENTE_TEX if pend else tex,
            "pt": C.tex_para_pt(None if pend else tex),
            "status": "pending" if pend else "ok",
            "fonte": [C.rel(f) for f in (fonte if isinstance(fonte, (list, tuple)) else [fonte])],
            "coluna": coluna, "categoria": cat, "nota": nota}

    def num(self, chave, valor, fonte, coluna, cat, casas=3, sci=False, nota=""):
        if isinstance(valor, (np.integer,)):
            valor = int(valor)
        elif isinstance(valor, (np.floating,)):
            valor = float(valor)
        tex = None if valor is None else C.fmt(valor, casas, sci)
        if sci and tex is not None:
            tex = r"\ensuremath{" + tex + "}"
        self.add(chave, valor, tex, fonte, coluna, cat, nota)


def _exigir(p, nome):
    if p is None:
        raise SystemExit(f"falta {nome} em results/csv/ (rode o script correspondente)")
    return p


def holdout(R):
    f = _exigir(C.mais_recente(C.RES / "csv", "aud_holdout_*_resumo.csv", excluir=("_meta",)),
                "aud_holdout_*_resumo.csv")
    S = pd.read_csv(f)
    meta = C.ler_json(f.with_name(f.stem.replace("_resumo", "") + "_meta.json"))
    R.num("audHoldoutSeed", int(meta["seed"]), f.with_name(f.stem.replace("_resumo", "") +
                                                          "_meta.json"), "seed", "equivalencia", 0)
    t = S[(S.build == "float") & (S.base == "_todas")].iloc[0]
    R.num("audHoldoutN", int(t.n_janelas), f, "n_janelas[float,_todas]", "equivalencia", 0)
    for base, bn in (("idmt", "Idmt"), ("a2212", "Bldc")):
        r = S[(S.build == "float") & (S.base == base)].iloc[0]
        R.num(f"audHoldoutN{bn}", int(r.n_janelas), f, f"n_janelas[float,{base}]",
              "equivalencia", 0)
    for b, bn in BUILDS.items():
        r = S[(S.build == b) & (S.base == "_todas")].iloc[0]
        col = lambda k: f"{k}[{b},_todas]"  # noqa: E731
        R.num(f"audHoldout{bn}PathSame", int(r.caminho_identico), f, col("caminho_identico"),
              "equivalencia", 0)
        R.num(f"audHoldout{bn}DecSame", int(r.decisao_igual), f, col("decisao_igual"),
              "equivalencia", 0)
        R.num(f"audHoldout{bn}FeatOut", int(r.n_features_fora_tol), f,
              col("n_features_fora_tol"), "equivalencia", 0)
        R.num(f"audHoldout{bn}StageFail", int(r.n_etapas_fail), f, col("n_etapas_fail"),
              "equivalencia", 0)
        R.num(f"audHoldout{bn}RatioMax", float(r.razao_tol_max_features), f,
              col("razao_tol_max_features"), "equivalencia", 2,
              nota="max |Δx|/(atol+rtol·|x|) nas 13 features, tolerâncias congeladas")
        R.num(f"audHoldout{bn}DpMax", float(r.max_abs_dp), f, col("max_abs_dp"),
              "equivalencia", 1, sci=True, nota="max |p_C - p_ref| nas probabilidades")


def rotacao(R):
    f = _exigir(C.mais_recente(C.RES / "csv", "aud_rotacao_motor_*.csv"),
                "aud_rotacao_motor_*.csv")
    D = pd.read_csv(f).set_index("escopo")
    g = D.loc["global"]
    R.num("audRotBldcSpearman", float(g.spearman_rho), f, "spearman_rho[global]", "dados", 2,
          nota="velocidade nominal × mediana de f0 por arquivo")
    R.num("audRotBldcFiles", int(g.n_arquivos), f, "n_arquivos[global]", "dados", 0)
    R.num("audRotBldcWindows", int(g.n_janelas), f, "n_janelas[global]", "dados", 0)
    R.num("audRotBldcCeilPct", 100 * float(g.frac_janelas_teto), f,
          "100·frac_janelas_teto[global]", "dados", 1)
    R.num("audRotBldcCeilThrPct", 100 * float(g.lim_teto_rel), f, "100·lim_teto_rel", "dados", 0)
    fs = [float(D.loc[m].n_arquivos) for m in MOTORES if m in D.index]
    for m, mn in MOTORES.items():
        r = D.loc[m]
        R.num(f"audRotBldcSpearmanM{mn}", float(r.spearman_rho), f, f"spearman_rho[{m}]",
              "dados", 2)
        R.num(f"audRotBldcSpearmanPM{mn}", float(r.spearman_p), f, f"spearman_p[{m}]",
              "dados", 2)
        R.num(f"audRotBldcFilesM{mn}", int(r.n_arquivos), f, f"n_arquivos[{m}]", "dados", 0)
    rho_m = [float(D.loc[m].spearman_rho) for m in MOTORES]
    R.num("audRotBldcSpearmanMotorMin", min(rho_m), f, "min spearman_rho[M1..M4]", "dados", 2)
    R.num("audRotBldcSpearmanMotorMax", max(rho_m), f, "max spearman_rho[M1..M4]", "dados", 2)
    R.num("audRotBldcFilesMotorMin", int(min(fs)), f, "min n_arquivos[M1..M4]", "dados", 0)
    R.num("audRotBldcFilesMotorMax", int(max(fs)), f, "max n_arquivos[M1..M4]", "dados", 0)


def motor_igual(R):
    f = _exigir(C.mais_recente(C.RES / "csv", "aud_motor_igual_*.csv"), "aud_motor_igual_*.csv")
    D = pd.read_csv(f)
    for v, vn in VARS.items():
        for prot, pn in (("arquivo", "File"), ("motor", "Motor")):
            r = D[(D.linha == "protocolo") & (D.variante == v) & (D.protocolo == prot)
                  & (D.escopo == "HP_M1M3")].iloc[0]
            col = f"[protocolo,{v},{prot},HP_M1M3]"
            R.num(f"audMotorBal{vn}{pn}HP", float(r.acuracia_balanceada), f,
                  "acuracia_balanceada" + col, "diagnostico", 3)
            R.num(f"audMotorBal{vn}{pn}HPLo", float(r.ic95_lo), f, "ic95_lo" + col,
                  "diagnostico", 3)
            R.num(f"audMotorBal{vn}{pn}HPHi", float(r.ic95_hi), f, "ic95_hi" + col,
                  "diagnostico", 3)
        q = D[(D.linha == "queda") & (D.variante == v)].iloc[0]
        col = f"[queda,{v},HP_M1M3]"
        R.num(f"audMotorDrop{vn}HP", float(q.acuracia_balanceada), f,
              "acuracia_balanceada" + col, "diagnostico", 3,
              nota="Bal[arquivo,H/P,M1-M3] - Bal[motor,H/P,M1-M3]")
        R.num(f"audMotorDrop{vn}HPLo", float(q.ic95_lo), f, "ic95_lo" + col, "diagnostico", 3,
              nota="bootstrap pareado por arquivo")
        R.num(f"audMotorDrop{vn}HPHi", float(q.ic95_hi), f, "ic95_hi" + col, "diagnostico", 3)
        R.num(f"audMotorDrop{vn}Full", float(q.queda_completa), f, "queda_completa" + col,
              "diagnostico", 3, nota="Bal[arquivo] - Bal[motor], protocolos completos")
    n = D[(D.linha == "queda")].iloc[0]
    R.num("audMotorHPFiles", int(n.n_arquivos), f, "n_arquivos[queda]", "diagnostico", 0)
    R.num("audMotorHPWindows", int(n.n_janelas), f, "n_janelas[queda]", "diagnostico", 0)


def escrever(R, commit):
    linhas = ["% Gerado por embarcados/src/python/gerar_artigo_numeros_aud.py. NÃO EDITAR À MÃO.",
              f"% commit {commit['commit']} (árvore suja em embarcados/: {commit['sujo']})",
              "% Macros da auditoria independente (M2 holdout, M3 motor igual por igual, M4 rotação).",
              "% Uma macro por número. Comentário: [categoria] fonte :: coluna.", ""]
    for k, v in R.n.items():
        linhas.append(f"\\newcommand{{\\{k}}}{{{v['tex']}}}% [{v['categoria']}] "
                      f"{' + '.join(v['fonte'])} :: {v['coluna']}")
    (C.PAPER / "numeros_aud.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    fontes = sorted({f for v in R.n.values() for f in v["fonte"]})
    out = {"gerado_por": "embarcados/src/python/gerar_artigo_numeros_aud.py", "projeto": commit,
           "categorias": C.CAT,
           "fontes": {f: C.sha256(C.RAIZ / f if not f.startswith("C:") else f) for f in fontes},
           "numeros": R.n}
    (C.PAPER / "numeros_aud.json").write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                                         default=float), encoding="utf-8")


def main():
    C.PAPER.mkdir(parents=True, exist_ok=True)
    R = Registro()
    holdout(R)
    rotacao(R)
    motor_igual(R)
    escrever(R, C.commit())
    pend = [k for k, v in R.n.items() if v["status"] == "pending"]
    print(f"{len(R.n)} macros; {len(pend)} pending: {', '.join(pend)}")
    for k, v in R.n.items():
        print(f"\\{k} = {v['tex']}")


if __name__ == "__main__":
    main()
