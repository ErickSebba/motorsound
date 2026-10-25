"""Auditoria M4: rotação rastreada × rótulo de velocidade no A2212, global e por motor.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\aud_rotacao_motor.py

Torna reproduzível o cálculo do auditor (reports/independent_scientific_audit.md, M4 e m7):
Spearman entre a velocidade nominal do arquivo e a mediana de f0 (feature 0 da p11 = mediana de
f_rot na janela) do arquivo, com os 43 arquivos juntos e separado por motor (n de arquivos por
motor), e a fração de janelas no teto da busca (f0 >= LIM_TETO·fmax, fmax lido da referência).

Fontes (só leitura; as mais recentes pelo timestamp do nome):
  results/csv/p11_features_<ts>.csv                 features da p11 por janela (diagnostico.py)
  results/predictions/diagnostico_janelas_<ts>.csv  motor e velocidade de cada arquivo
`../out/experiments/` não tem features por janela da p11 (conferido por glob; registrado no meta).

Saída: results/csv/aud_rotacao_motor_<ts>.csv (+ _meta.json); uma linha por escopo
(global, M1..M4) com n_arquivos, n_janelas, Spearman, p-valor, fração no teto.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import gerar_artigo_comum as C  # noqa: E402  (insere a raiz no sys.path)
from bancada_comum import carimbo, commit  # noqa: E402
from pipelines import p11_decifrae_harmsum as p11  # noqa: E402

LIM_TETO = 0.99      # mesmo limiar de gerar_artigo_numeros.py::rotacao_estimada


def main():
    t_ini = time.perf_counter()
    ts = carimbo()
    ff = C.mais_recente(C.RES / "csv", "p11_features_*.csv")
    fpred = C.mais_recente(C.RES / "predictions", "diagnostico_janelas_*.csv")
    F = pd.read_csv(ff)
    P = pd.read_csv(fpred, usecols=["base", "rec_id", "motor", "velocidade"]).drop_duplicates()
    A = F[F.base == "a2212"].merge(P[P.base == "a2212"], on=["base", "rec_id"], how="left")
    assert A.motor.notna().all() and A.velocidade.notna().all(), "janela sem motor/velocidade"
    assert A.groupby("rec_id").velocidade.nunique().max() == 1
    fmax = float(p11.PRE.fmax)
    lim = LIM_TETO * fmax
    G = A.groupby(["rec_id", "motor", "velocidade", "classe"]).f0.median().reset_index()
    linhas = []
    for escopo, g, a in [("global", G, A)] + [(m, G[G.motor == m], A[A.motor == m])
                                               for m in sorted(G.motor.unique())]:
        r = spearmanr(g.velocidade, g.f0)
        linhas.append({
            "ts": ts, "escopo": escopo, "n_arquivos": int(len(g)), "n_janelas": int(len(a)),
            "n_velocidades": int(g.velocidade.nunique()),
            "classes": ",".join(sorted(g.classe.unique())),
            "spearman_rho": float(r.statistic), "spearman_p": float(r.pvalue),
            "f0_mediana_min": float(g.f0.min()), "f0_mediana_max": float(g.f0.max()),
            "vel_min": float(g.velocidade.min()), "vel_max": float(g.velocidade.max()),
            "n_janelas_teto": int((a.f0 >= lim).sum()),
            "frac_janelas_teto": float(np.mean(a.f0 >= lim)),
            "lim_teto_hz": lim, "lim_teto_rel": LIM_TETO, "fmax_hz": fmax,
            "unidade_f0": "mediana por arquivo da feature f0 (Hz)",
            "fonte_features": C.rel(ff), "fonte_motor_velocidade": C.rel(fpred)})
    D = pd.DataFrame(linhas)
    out = C.RES / "csv" / f"aud_rotacao_motor_{ts}.csv"
    meta_p = out.with_name(out.stem + "_meta.json")
    for p in (out, meta_p):
        if p.exists():
            raise FileExistsError(p)
    D.to_csv(out, index=False)
    alt = sorted(str(p.name) for p in (C.RAIZ / "out" / "experiments").glob("*p11*feat*"))
    meta = {"script": "embarcados/src/python/aud_rotacao_motor.py",
            "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"), "projeto": commit(),
            "seed": None, "fontes": {C.rel(ff): C.sha256(ff), C.rel(fpred): C.sha256(fpred)},
            "fonte_features_meta": C.ler_json(ff.with_name(ff.stem + "_meta.json")).get("projeto"),
            "fonte_features_gerador": "embarcados/src/python/diagnostico.py (feats_p11, "
                                      "runner.extrair usar_cache=False, normalizar_rms=True)",
            "alternativas_em_out_experiments": alt,
            "definicoes": {"spearman": "spearmanr(velocidade nominal do arquivo, mediana de f0 "
                                       "do arquivo), arquivos do A2212",
                           "teto": f"f0 >= {LIM_TETO}·PRE.fmax = {lim} Hz, por janela"},
            "nota": "arquivos do mesmo motor não são independentes; global junta níveis de "
                    "motores diferentes (efeito de Simpson possível)",
            "duracao_s": round(time.perf_counter() - t_ini, 2)}
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(D[["escopo", "n_arquivos", "n_janelas", "spearman_rho", "spearman_p",
             "frac_janelas_teto"]].to_string(index=False))
    print(f"escrito {out}")


if __name__ == "__main__":
    main()
