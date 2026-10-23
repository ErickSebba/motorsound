"""Exp4: teste do rastreador de rotação onde há verdade (frequência de alimentação conhecida).

REGRA DA FAIXA DE BUSCA, fixada em 22/10/2026 antes de rodar em qualquer base:
    fmin = 0,40 · f_alim     fmax = 1,05 · f_alim
Justificativa: cobre motores de 2 polos (rotação ≈ f_alim·(1−s)) e de 4 polos
(≈ f_alim/2·(1−s)) com escorregamento de até 20 %, e um pouco acima de f_alim para não
cortar o caso síncrono. Nenhum parâmetro do rastreador é ajustado nesta base: usa-se
`lib.rotacao.rastrear` com os padrões do Artigo3 (grade de 2 cents, harmônicos 1–48,
n_fft 8192, hop 1024, Viterbi, sem pYIN), só trocando fmin/fmax pela regra acima.

SEGUNDA REGRA (`--faixa-base`), fixada em 23/10/2026 antes de rodar em qualquer base com
tacômetro (MaFaulDa ainda não baixada): para bases com rotação medida (`rot_esperada_hz`) e
sem f_alim, a faixa é ÚNICA para a base inteira:
    fmin = 0,9 · min(rot_esperada_hz)     fmax = 1,1 · max(rot_esperada_hz)
sobre todos os arquivos do índice. O rastreador não recebe nenhuma informação do próprio
arquivo; o teste é "rastrear sem saber a velocidade, dentro da faixa de operação da bancada".

Referência para o erro, em ordem de preferência:
  1. `rot_esperada_hz` do índice (tacômetro ou outra medida direta), se houver;
  2. rotação síncrona f_alim / p, com p = pares de polos da placa (`--pares-polos`).
No caso 2 o erro inclui o escorregamento; por isso também se reporta a relação observada
f_alim / f_rot e o escorregamento aparente 1 − f_rot·p / f_alim.

Cada arquivo com f_alim conhecida é reamostrado para 16 kHz e rastreado inteiro (até
`--max-s` segundos); o erro é avaliado por segmento de 1 s (mediana dos quadros do
segmento) e por arquivo (mediana de todos os quadros não-SIL).

Uso: .venv/Scripts/python.exe scripts/exp4_rastreador.py <base> [--pares-polos 2] [--max-s 60]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from lib.comum import DATA, SR, carregador, ler_indice, para_16k, salvar
from lib.rotacao import rastrear

FMIN_FATOR, FMAX_FATOR = 0.40, 1.05
TOL = 0.02
HOP, NFFT = 1024, 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--pares-polos", type=int, default=None)
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--filtro", default="", help="consulta pandas opcional sobre o índice")
    ap.add_argument("--faixa-base", action="store_true",
                    help="segunda regra: faixa única da base a partir de rot_esperada_hz")
    args = ap.parse_args()
    t0 = time.time()
    idx = ler_indice(args.base)
    if args.faixa_base:
        idx = idx[idx["rot_esperada_hz"].notna() & (idx["rot_esperada_hz"] > 0)]
        faixa = (0.9 * idx["rot_esperada_hz"].min(), 1.1 * idx["rot_esperada_hz"].max())
    else:
        idx = idx[idx["f_alim_hz"].notna() & (idx["f_alim_hz"] > 0)]
    if args.filtro:
        idx = idx.query(args.filtro)
    ld = carregador(args.base)
    linhas = []
    for i, r in enumerate(idx.itertuples(index=False)):
        y, sr = ld.carregar(DATA / args.base / r.arquivo)
        y = para_16k(np.asarray(y, dtype=np.float32), sr)[: int(args.max_s * SR)]
        fa = float(r.f_alim_hz)
        fmin, fmax = faixa if args.faixa_base else (FMIN_FATOR * fa, FMAX_FATOR * fa)
        f_rot, _, _ = rastrear(y, SR, fmin=fmin, fmax=fmax)
        if np.isfinite(r.rot_esperada_hz):
            ref, fonte = float(r.rot_esperada_hz), "indice"
        elif args.pares_polos:
            ref, fonte = fa / args.pares_polos, f"sincrona_p{args.pares_polos}"
        else:
            ref, fonte = np.nan, ""
        # centro de cada quadro (s) → segmento de 1 s
        t_q = (np.arange(len(f_rot)) * HOP + NFFT / 2) / SR
        seg = np.floor(t_q).astype(int)
        med_seg = [np.nanmedian(f_rot[seg == s]) for s in np.unique(seg)
                   if np.isfinite(f_rot[seg == s]).any()]
        med_seg = np.array(med_seg)
        fm = float(np.nanmedian(f_rot)) if np.isfinite(f_rot).any() else np.nan
        linhas.append(dict(
            base=args.base, arquivo=r.arquivo, classe=r.classe, unidade=r.unidade,
            condicao=r.condicao, velocidade=r.velocidade, carga=r.carga, f_alim_hz=fa,
            regra="faixa_base" if args.faixa_base else "f_alim",
            fmin=fmin, fmax=fmax, ref_hz=ref, fonte_ref=fonte,
            n_quadros=len(f_rot), frac_sil=float(np.mean(~np.isfinite(f_rot))),
            f_rot_mediana=fm,
            f_rot_iqr=float(np.subtract(*np.nanpercentile(f_rot, [75, 25]))) if np.isfinite(fm) else np.nan,
            razao_alim_rot=fa / fm if np.isfinite(fm) else np.nan,
            escorr_aparente=(1 - fm * args.pares_polos / fa) if (args.pares_polos and np.isfinite(fm)) else np.nan,
            erro_rel_arquivo=abs(fm - ref) / ref if np.isfinite(ref) and np.isfinite(fm) else np.nan,
            n_seg=len(med_seg),
            erro_rel_seg_mediano=float(np.median(np.abs(med_seg - ref) / ref)) if len(med_seg) and np.isfinite(ref) else np.nan,
            frac_seg_2pct=float(np.mean(np.abs(med_seg - ref) / ref <= TOL)) if len(med_seg) and np.isfinite(ref) else np.nan,
        ))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(idx)} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(linhas)
    salvar(df, "exp4_rastreador", args.base,
           {"fmin_fator": FMIN_FATOR, "fmax_fator": FMAX_FATOR, "tol": TOL,
            "pares_polos": args.pares_polos, "max_s": args.max_s, "filtro": args.filtro,
            "faixa_base": [float(x) for x in faixa] if args.faixa_base else None,
            "rastrear": "padrões de lib.rotacao.rastrear (2 cents, h 1-48, nfft 8192, hop 1024, Viterbi)"},
           t0)
    if len(df):
        chave = "classe" if args.faixa_base else "f_alim_hz"
        print(df.groupby(chave)[["f_rot_mediana", "razao_alim_rot", "erro_rel_arquivo",
                                 "frac_seg_2pct"]].median().round(3))


if __name__ == "__main__":
    main()
