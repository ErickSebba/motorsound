"""Bancada sintética com verdade conhecida para o rastreador de rotação (Etapa 5, parte 2).

    .venv/Scripts/python.exe experimentos/bancada_sintetica.py                # out/experiments/
    .venv/Scripts/python.exe experimentos/bancada_sintetica.py --saida x.csv  # x.csv + x_meta.json

Para cada distúrbio de tensão (nominal, sag, swell, flicker, interruption, broken) × SNR de
ruído rosa (20, 10, 0 dB) × semente, gera `--dur` s de tom de motor a 16 kHz com
`ruidos.sintetico.gerar` (rotação de 1ª ordem seguindo a tensão de `ruidos.lee_shen`) e roda
`motorsound.rotacao.rastrear` nas variantes:

  viterbi      padrão (grade de 10 cents, Viterbi);
  argmax       usar_viterbi=False (p12);
  pyin         usar_pyin=True;
  viterbi_2c   diagnóstico: passo_cents=2, max_salto=50 (mesmo alcance de 100 cents por
               quadro), para medir o efeito da quantização da grade (ver RESULTADOS).

Uma linha por (distúrbio, snr, semente, variante). Métricas, só nos quadros em que a
verdade está dentro da grade [fmin, fmax] ("quadros válidos"):

  erro_hz_med, erro_pct_med  erro absoluto mediano contra a verdade do quadro (média de
                             f_rot ponderada pela janela de Hann do STFT), sobre os
                             quadros válidos com estimativa (não SIL); frac_ok5 = fração
                             desses quadros com erro relativo < 5 % (a mediana esconde a
                             bimodalidade certo/errado-por-razão);
  taxa_razao                 fração desses quadros com |f_est/f_true / r − 1| < 3 % para
                             algum r ∈ {2, 1/2, 3/4, 4/3}; taxa_r2, taxa_r05, taxa_r075,
                             taxa_r133 por razão;
  sil_indevido               fração dos quadros válidos marcados SIL;
  sil_correto                (interruption) fração dos quadros com verdade < fmin marcados SIL;
  atraso_s                   (sag e swell) tempo entre o início do degrau t1 e o primeiro
                             quadro, depois que a verdade se afastou 5 % do valor anterior,
                             que inicia 3 quadros seguidos com |erro| < 5 % contra a verdade
                             instantânea no centro do quadro, antes do fim do degrau (t2).
                             NaN = nunca acompanhou (coluna nunca_acompanhou = 1);
  atraso_verdade_s           tempo que a própria verdade leva para se afastar 5 % (τ do motor);
  atraso_causal_s            atraso_s + n_fft/(2·sr): o centro do quadro só fica disponível
                             meia janela depois (0,256 s com n_fft 8192 a 16 kHz).

Parâmetros dos distúrbios (escolha da bancada, não de Lee & Shen; sorteados por semente):
  sag          α ~ U(0,15; 0,35), t1 ~ U(0,30; 0,40)·dur, duração ~ U(4; 6) s
  swell        α ~ U(0,10; 0,30), mesma janela
  interruption α ~ U(0,90; 1,00), t1 idem, duração ~ U(1; 2) s
  flicker      α ~ U(0,10; 0,20), β ~ U(0,5; 2) Hz (o original elétrico usa 5–20 Hz, que
               o filtro mecânico de τ = 0,1 s atenua para < 2 % de variação de rotação)
  broken       chaveamento 0,25 ↔ 1,25 p.u. a cada 18 ms, fase sorteada (lee_shen.py)
  f_nom ~ U(24,5; 28) Hz por semente (faixa de rotação citada para o IDMT).
Nada aqui lê dados reais.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from motorsound import rotacao, runner  # noqa: E402
from ruidos import sintetico  # noqa: E402
from ruidos.lee_shen import TAU_MEC  # noqa: E402

SR = 16000
N_FFT, HOP = 8192, 1024
FMIN, FMAX = 15.0, 60.0
DISTURBIOS = ["nominal", "sag", "swell", "flicker", "interruption", "broken"]
SNRS = [20.0, 10.0, 0.0]
VARIANTES = {
    # explícitas: o padrão de rotacao.rastrear mudou de 10 para 2 cents em 14/10/2026
    "viterbi": {"passo_cents": 10.0, "max_salto": 10},
    "argmax": {"usar_viterbi": False, "passo_cents": 10.0, "max_salto": 10},
    "pyin": {"usar_pyin": True, "passo_cents": 10.0, "max_salto": 10},
    "viterbi_2c": {"passo_cents": 2.0, "max_salto": 50},
    "argmax_2c": {"usar_viterbi": False, "passo_cents": 2.0, "max_salto": 50},
    "pyin_2c": {"usar_pyin": True, "passo_cents": 2.0, "max_salto": 50},
}
RAZOES = {"r2": 2.0, "r05": 0.5, "r075": 0.75, "r133": 4.0 / 3.0}
TOL_RAZAO = 0.03
TOL_ATRASO = 0.05
N_SEGUIDOS = 3


def parametros_bancada(dist, dur, rng):
    if dist in ("sag", "swell", "interruption"):
        t1 = float(rng.uniform(0.30, 0.40) * dur)
        larg = float(rng.uniform(1.0, 2.0) if dist == "interruption" else rng.uniform(4.0, 6.0))
        alpha = float({"sag": rng.uniform(0.15, 0.35), "swell": rng.uniform(0.10, 0.30),
                       "interruption": rng.uniform(0.90, 1.00)}[dist])
        return {"alpha": alpha, "t1": t1, "t2": min(t1 + larg, dur - 0.5)}
    if dist == "flicker":
        return {"alpha": float(rng.uniform(0.10, 0.20)), "beta": float(rng.uniform(0.5, 2.0))}
    return {}


def _atraso(f_est, f_rot, t_q, t1, t2):
    """Atraso para acompanhar o degrau que começa em t1 (ver docstring do módulo).
    Só conta se acompanhar antes do fim do degrau (t2); senão, NaN = nunca acompanhou
    (sem esse limite, um rastreador parado no valor nominal "acompanharia" na volta)."""
    i1 = int(t1 * SR)
    f_pre = f_rot[i1]
    desvio = np.nonzero(np.abs(f_rot[i1:] / f_pre - 1) > TOL_ATRASO)[0]
    if len(desvio) == 0:
        return np.nan, np.nan
    t_dev = (i1 + desvio[0]) / SR
    f_inst = f_rot[np.clip((t_q * SR).astype(int), 0, len(f_rot) - 1)]
    ok = np.abs(f_est / f_inst - 1) < TOL_ATRASO           # NaN (SIL) → False
    for k in np.nonzero((t_q >= t_dev) & (t_q < t2))[0]:
        if k + N_SEGUIDOS <= len(ok) and ok[k:k + N_SEGUIDOS].all() \
                and t_q[k + N_SEGUIDOS - 1] < t2:
            return float(t_q[k] - t1), float(t_dev - t1)
    return np.nan, float(t_dev - t1)


def metricas(f_est, f_true_q, f_rot, t_q, dist, params):
    n = min(len(f_est), len(f_true_q))
    fe, ft, tq = f_est[:n], f_true_q[:n], t_q[:n]
    valido = (ft >= FMIN) & (ft <= FMAX)
    tem = valido & np.isfinite(fe)
    m = {"n_quadros": int(n), "n_validos": int(valido.sum())}
    if tem.any():
        e = np.abs(fe[tem] - ft[tem])
        m["erro_hz_med"] = float(np.median(e))
        m["erro_pct_med"] = float(np.median(100 * e / ft[tem]))
        m["frac_ok5"] = float((e / ft[tem] < TOL_ATRASO).mean())
        razao = fe[tem] / ft[tem]
        qualquer = np.zeros(tem.sum(), dtype=bool)
        for nome, r in RAZOES.items():
            hit = np.abs(razao / r - 1) < TOL_RAZAO
            m[f"taxa_{nome}"] = float(hit.mean())
            qualquer |= hit
        m["taxa_razao"] = float(qualquer.mean())
    else:
        m.update({"erro_hz_med": np.nan, "erro_pct_med": np.nan, "frac_ok5": np.nan,
                  "taxa_razao": np.nan,
                  **{f"taxa_{k}": np.nan for k in RAZOES}})
    m["sil_indevido"] = float(np.isnan(fe[valido]).mean()) if valido.any() else np.nan
    fora = ft < FMIN
    m["sil_correto"] = float(np.isnan(fe[fora]).mean()) if fora.any() else np.nan
    if dist in ("sag", "swell"):
        a, av = _atraso(fe, f_rot, tq, params["t1"], params["t2"])
        m["atraso_s"], m["atraso_verdade_s"] = a, av
        m["atraso_causal_s"] = a + N_FFT / (2 * SR) if np.isfinite(a) else np.nan
        m["nunca_acompanhou"] = int(not np.isfinite(a))
    else:
        m.update({"atraso_s": np.nan, "atraso_verdade_s": np.nan, "atraso_causal_s": np.nan,
                  "nunca_acompanhou": np.nan})
    return m


def rodar_caso(args):
    dist, snr, semente, dur, variantes = args
    rng = np.random.default_rng([semente, DISTURBIOS.index(dist), int(snr * 10) + 1000])
    params = parametros_bancada(dist, dur, rng)
    f_nom = float(rng.uniform(24.5, 28.0))
    seed_ger = int(rng.integers(0, 2**31 - 1))
    y, f_rot, v, p = sintetico.gerar(dur, SR, tipo=dist, snr_db=snr, seed=seed_ger,
                                     f_nom=f_nom, devolver_v=True, **params)
    f_true_q = sintetico.verdade_por_quadro(f_rot, SR, N_FFT, HOP)
    t_q = sintetico.tempos_quadro(len(y), SR, N_FFT, HOP)
    linhas = []
    for nome in variantes:
        kw = {"n_fft": N_FFT, "hop": HOP, "fmin": FMIN, "fmax": FMAX, **VARIANTES[nome]}
        t0 = time.perf_counter()
        f_est, _, _ = rotacao.rastrear(y, SR, **kw)
        dt = time.perf_counter() - t0
        linhas.append({"disturbio": dist, "snr_db": snr, "semente": semente,
                       "variante": nome, "f_nom": f_nom, "dur_s": dur,
                       "tau_s": TAU_MEC, "seed_gerador": seed_ger,
                       "params": json.dumps(p, default=float),
                       "config_rastreador": json.dumps(kw),
                       **metricas(f_est, f_true_q, f_rot, t_q, dist, params),
                       "tempo_rastreador_s": dt})
    return linhas


def _escrever_em(df, caminho: Path, meta: dict):
    caminho = Path(caminho)
    if caminho.exists():
        raise FileExistsError(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)
    meta = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
            "projeto": runner._commit(), "python": sys.version.split()[0],
            "numpy": np.__version__, **meta}
    caminho.with_name(caminho.stem + "_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return caminho


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--disturbios", nargs="+", default=DISTURBIOS, choices=DISTURBIOS)
    ap.add_argument("--snrs", nargs="+", type=float, default=SNRS)
    ap.add_argument("--sementes", type=int, default=10)
    ap.add_argument("--dur", type=float, default=15.0)
    ap.add_argument("--variantes", nargs="+", default=list(VARIANTES), choices=list(VARIANTES))
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--saida", default=None, help="caminho do CSV (padrão: out/experiments/)")
    a = ap.parse_args(argv)
    casos = [(d, s, k, a.dur, a.variantes) for d in a.disturbios for s in a.snrs
             for k in range(a.sementes)]
    t0 = time.perf_counter()
    if a.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
            res = list(ex.map(rodar_caso, casos))
    else:
        res = [rodar_caso(c) for c in casos]
    df = pd.DataFrame([lin for r in res for lin in r])
    meta = {"script": "experimentos/bancada_sintetica.py", "argv": sys.argv[1:],
            "seeds": f"np.random.default_rng([semente, idx_disturbio, snr*10+1000]), "
                     f"semente = 0..{a.sementes - 1}",
            "sr": SR, "n_fft": N_FFT, "hop": HOP, "fmin": FMIN, "fmax": FMAX,
            "tau_s": TAU_MEC, "variantes": {k: VARIANTES[k] for k in a.variantes},
            "tol_razao": TOL_RAZAO, "tol_atraso": TOL_ATRASO, "n_seguidos": N_SEGUIDOS,
            "amplitudes": {str(k): v for k, v in sintetico.FORTES.items()},
            "duracao_s": round(time.perf_counter() - t0, 1)}
    if a.saida:
        caminho = _escrever_em(df, Path(a.saida), meta)
    else:
        caminho = runner.escrever(df, "bancada_sintetica", meta)
    print(f"gravado: {caminho} ({len(df)} linhas, {meta['duracao_s']} s)")
    resumo = (df.groupby(["variante", "disturbio", "snr_db"])
                [["erro_pct_med", "taxa_razao", "sil_indevido", "atraso_s"]].median())
    print(resumo.round(3).to_string())


if __name__ == "__main__":
    main()
