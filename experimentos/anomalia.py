"""Detecção de anomalia treinada só no normal: AUC, pAUC e falso alarme por fold de validação.

    .venv/Scripts/python.exe experimentos/anomalia.py                 # grava em out/experiments/
    .venv/Scripts/python.exe experimentos/anomalia.py --saida x.csv   # grava em x.csv (+ _meta.json)

Para cada pipeline `p30*` (interface comum + `escore(modelo, X)`), e para cada cenário:

  idmt_limpo    IDMT train/, folds em blocos temporais (splits.idmt_blocos), sem perturbação;
  idmt_rosa10   idem, mas a validação recebe ruído rosa a 10 dB SNR (o treino fica limpo);
  a2212_arquivo A2212 inteiro, StratifiedGroupKFold por arquivo;
  a2212_m1m2    A2212 restrito a M1 e M2 e às classes Healthy e Propeller (único teste em que
                normal e anômalo vêm do mesmo motor), folds por arquivo dentro do recorte;

o detector é treinado só com as janelas da classe 0 do treino do fold e avaliado nas janelas
de validação. Uma linha por (pipeline, cenário, fold, conjunto anômalo), em que o conjunto
anômalo é "todas" (toda classe ≠ 0) ou cada classe anômala separada.

Métricas:
  auc            ROC AUC normal × anômalo sobre o escore bruto;
  pauc           sklearn roc_auc_score(max_fpr=0,1): pAUC com a padronização de McClish,
                 a mesma chamada do código do baseline DCASE2020 (koizumi2020);
  fa_p95_treino  fração das janelas NORMAIS de validação acima do percentil 95 dos escores
                 das janelas normais de treino (escores dentro da amostra);
  det_p95_treino fração das janelas anômalas acima desse limiar;
  fa_cal / det_cal  idem no limiar que a própria pipeline calibrou (percentil 95 de escores
                 normais fora da amostra; ver cabeçalho de cada pipeline) — é o limiar que o
                 argmax de `prever` usa.

Ressalvas obrigatórias em qualquer tabela que sair daqui:
  * IDMT: cada classe é um motor físico diferente e há um único `pure.wav` por classe no
    train/; o detector treinado em `good` separa `broken`/`heavyload` também pela identidade
    do motor e pela sessão. A validação em blocos é da mesma sessão (otimista).
  * A2212: rolamento só no M4, sem saudável do M4: Healthy × Bearing está confundido com a
    identidade do motor. Só `a2212_m1m2` compara normal e falha no mesmo motor.
Nenhuma leitura do test/ do IDMT (motorsound.io trava).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from motorsound import runner, splits  # noqa: E402
from motorsound.config import AVAL_A2212, AVAL_IDMT  # noqa: E402

PIPELINES = ["p30_anomalia", "p30b_ae"]
ROSA10 = {"tipo": "rosa", "snr_db": 10.0}
CENARIOS = {
    "idmt_limpo": ("idmt", None),
    "idmt_rosa10": ("idmt", ROSA10),
    "a2212_arquivo": ("a2212", None),
    "a2212_m1m2": ("a2212", None),
}
P_MAX_FPR = 0.1
PERCENTIL = 95.0


def _dc(o):
    return asdict(o) if is_dataclass(o) else o


def _metricas(s_norm, s_anom, lim_tr, lim_cal):
    out = {"n_normal": len(s_norm), "n_anomalo": len(s_anom)}
    if len(s_norm) and len(s_anom):
        yt = np.r_[np.zeros(len(s_norm)), np.ones(len(s_anom))]
        sc = np.r_[s_norm, s_anom]
        out["auc"] = float(roc_auc_score(yt, sc))
        out["pauc"] = float(roc_auc_score(yt, sc, max_fpr=P_MAX_FPR))
    else:
        out["auc"] = out["pauc"] = np.nan
    out["fa_p95_treino"] = float(np.mean(s_norm > lim_tr)) if len(s_norm) else np.nan
    out["det_p95_treino"] = float(np.mean(s_anom > lim_tr)) if len(s_anom) else np.nan
    out["fa_cal"] = float(np.mean(s_norm > lim_cal)) if len(s_norm) else np.nan
    out["det_cal"] = float(np.mean(s_anom > lim_cal)) if len(s_anom) else np.nan
    return out


def _dados(nome, base, n_jobs, cache_X):
    """Segmentos, rótulos e features limpas da base (memorizados por pipeline/base)."""
    k = (nome, base)
    if k not in cache_X:
        aval = AVAL_IDMT if base == "idmt" else AVAL_A2212
        segs = runner.segmentos_de(base, aval)
        X = runner.extrair(nome, segs, aval, None, n_jobs)
        cache_X[k] = (aval, segs, X)
    return cache_X[k]


def rodar_cenario(nome, cenario, n_jobs, cache_X, verbose=True):
    pipe = runner.carregar_pipeline(nome)
    post = runner._post(pipe)
    base, pert = CENARIOS[cenario]
    classes = runner.classes_de(base)
    t0 = time.perf_counter()
    aval, segs, X = _dados(nome, base, n_jobs, cache_X)
    Xval = X if not pert else runner.extrair(nome, segs, aval, pert, n_jobs)
    if cenario == "a2212_m1m2":
        sel = np.array([i for i, s in enumerate(segs)
                        if s.motor in ("M1", "M2") and s.classe in ("Healthy", "Propeller")])
        segs = [segs[i] for i in sel]
        X, Xval = X[sel], Xval[sel]
        folds = splits.a2212_por_arquivo(segs)
    else:
        folds, _ = runner.folds_de(base, "blocos" if base == "idmt" else "arquivo", segs)
    t_ext = time.perf_counter() - t0
    y = np.array([classes.index(s.classe) for s in segs])
    linhas = []
    for f in folds:
        if not np.any(y[f.treino] == 0) or len(f.val) == 0:
            continue
        t1 = time.perf_counter()
        modelo = pipe.treinar(X[f.treino], y[f.treino], post, n_classes=len(classes))
        t_tr = time.perf_counter() - t1
        s_tr = pipe.escore(modelo, X[f.treino][y[f.treino] == 0])
        s_val = pipe.escore(modelo, Xval[f.val])
        yv = y[f.val]
        lim_tr = float(np.percentile(s_tr, PERCENTIL))
        lim_cal = float(modelo["cal"]["limiar"])
        conjuntos = {"todas": yv != 0}
        for c in sorted(set(yv[yv != 0])):
            conjuntos[classes[c]] = yv == c
        for nome_conj, m_anom in conjuntos.items():
            linhas.append({
                "pipeline": pipe.NOME, "tipo": pipe.TIPO, "origem": pipe.ORIGEM,
                "cenario": cenario, "base": base, "fold": f.nome,
                "normal": classes[0], "anomalo": nome_conj,
                **_metricas(s_val[yv == 0], s_val[m_anom], lim_tr, lim_cal),
                "limiar_p95_treino": lim_tr, "limiar_cal": lim_cal,
                "n_treino_normal": int(np.sum(y[f.treino] == 0)),
                "treino_segundos": round(t_tr, 3),
                "perturbacao_val": json.dumps(pert or {}, sort_keys=True),
                "aval": json.dumps(_dc(aval), sort_keys=True),
                "pre": json.dumps(_dc(runner._pre(pipe)), sort_keys=True, default=str),
                "post": json.dumps(_dc(post), sort_keys=True, default=str),
            })
        if verbose:
            l = linhas[-len(conjuntos)]
            print(f"  {pipe.NOME} {cenario} {f.nome}: AUC={l['auc']:.3f} pAUC={l['pauc']:.3f} "
                  f"FA95={l['fa_p95_treino']:.3f} FAcal={l['fa_cal']:.3f} "
                  f"(treino {t_tr:.1f}s)", flush=True)
    return linhas, t_ext


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
    ap.add_argument("--pipelines", nargs="+", default=PIPELINES)
    ap.add_argument("--cenarios", nargs="+", default=list(CENARIOS), choices=list(CENARIOS))
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--saida", default=None,
                    help="CSV de saída; padrão out/experiments/<timestamp>_anomalia.csv")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    linhas, falhas, custos, t_ext = [], [], {}, {}
    cache_X: dict = {}
    for p in a.pipelines:
        for c in a.cenarios:
            try:
                ls, te = rodar_cenario(p, c, a.n_jobs, cache_X)
                linhas += ls
                t_ext[f"{p}/{c}"] = round(te, 1)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                falhas.append(f"{p}/{c}: {type(exc).__name__}: {exc}")
        try:
            custos[p] = runner.carregar_pipeline(p).custo_mcu()
        except Exception as exc:
            custos[p] = f"erro: {exc}"
        cache_X = {k: v for k, v in cache_X.items() if k[0] != p}
    df = pd.DataFrame(linhas)
    meta = {"argv": sys.argv, "pipelines": a.pipelines, "cenarios": a.cenarios,
            "seed_splits": splits.SEED, "percentil": PERCENTIL, "max_fpr": P_MAX_FPR,
            "falhas": falhas, "custo_mcu": custos, "t_extracao_s": t_ext,
            "segundos": round(time.perf_counter() - t0, 1)}
    if a.saida:
        caminho = _escrever_em(df, Path(a.saida), meta)
    else:
        caminho = runner.escrever(df, "anomalia", meta)
    if len(df):
        r = (df[df.anomalo == "todas"]
             .groupby(["pipeline", "cenario"])[["auc", "pauc", "fa_p95_treino", "fa_cal",
                                                "det_cal"]].mean())
        print(r.round(3).to_string())
        r2 = (df[df.anomalo != "todas"]
              .groupby(["pipeline", "cenario", "anomalo"])[["auc", "pauc"]].mean())
        print(r2.round(3).to_string())
    print(f"escrito {caminho}; falhas: {falhas}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
