"""Etapa 6 (diagnóstico): p11 × RMS/pico × clássicas nas MESMAS janelas e protocolos.

Importa a referência (`motorsound/`, `pipelines/`) sem editar. Não escreve em `cache/` nem em
`out/` da raiz: as features são extraídas UMA vez por (pipeline, base, normalização) com
`runner.extrair(..., usar_cache=False)` e passadas como `X=` para `runner.avaliar` em cada
protocolo. Nunca lê `data/idmt/test*` (io.py trava; aqui só se usa o `train/`).

Variantes:
  b00_nivel  normalizar_rms=True   (RMS normalizado: na prática só o fator de crista sobra)
  b00_nivel  normalizar_rms=False  (RMS e pico brutos: o "atalho de nível")
  p04_classicas        normalizar_rms=True  (configuração histórica de T1)
  p11_decifrae_harmsum normalizar_rms=True  (referência; configuração histórica de T1)

Protocolos: IDMT aleatorio, blocos; A2212 aleatorio, arquivo, motor (+ métrica derivada
`motor_sem_M4`, ver abaixo).

Saídas (timestamp no nome; nunca sobrescreve):
  results/predictions/diagnostico_janelas_<ts>.csv   todas as variantes, uma linha por janela
  results/predictions/p11_janelas_<ts>.csv           p11 (rec_id, idx, t0, t1, fold, classe, p_*)
  results/predictions/p11_folds_<ts>.csv             composição de cada fold (treino/val)
  results/predictions/p11_modelos_<ts>.json          modelo de cada fold (mu, sd, protótipos)
  results/csv/p11_features_<ts>.csv                  13 features da p11 por janela
  results/csv/diagnostico_<ts>.csv                   métricas em formato longo
  results/metrics/diagnostico_<ts>.json              métricas aninhadas + matrizes de confusão
  + um _meta.json por CSV/JSON

    .venv\\Scripts\\python.exe embarcados\\src\\python\\diagnostico.py --n-jobs 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
EMB = RAIZ / "embarcados"
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from motorsound import runner, splits  # noqa: E402
from motorsound.config import AVAL_A2212, AVAL_IDMT  # noqa: E402

N_BOOT = 2000
SEED_BOOT = 20261023
ALFA = 0.05

VARIANTES = [  # (rótulo, pipeline, normalizar_rms)
    ("b00_rmspico_norm", "b00_nivel", True),
    ("b00_rmspico_bruto", "b00_nivel", False),
    ("p04_classicas", "p04_classicas", True),
    ("p11_decifrae_harmsum", "p11_decifrae_harmsum", True),
]
PROTOCOLOS = {"idmt": ["aleatorio", "blocos"], "a2212": ["aleatorio", "arquivo", "motor"]}

NOTAS = {
    ("idmt", "aleatorio"): "vazamento intencional: janelas da mesma gravação nos dois lados",
    ("idmt", "blocos"): "train/ tem 1 gravação por classe: classe = motor = gravação; "
                        "blocos contíguos da mesma sessão (validação otimista, só 'pure')",
    ("a2212", "aleatorio"): "vazamento intencional: janelas do mesmo arquivo nos dois lados",
    ("a2212", "arquivo"): "StratifiedGroupKFold por arquivo; rolamento confundido com M4",
    ("a2212", "motor"): "motor fora; com M4 fora a classe Bearing não tem treino "
                        "(p_Bearing=0 por construção, sensibilidade de Bearing = 0)",
    ("a2212", "motor_sem_M4"): "folds M1-M3 do protocolo motor; só Healthy/Propeller como "
                               "classes verdadeiras (Bearing ainda pode ser predito)",
}


def _git():
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=RAIZ, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": g("rev-parse", "--short", "HEAD"), "sujo": bool(g("status", "--porcelain"))}


def _dc(o):
    return asdict(o) if is_dataclass(o) else o


def _novo(caminho: Path) -> Path:
    if caminho.exists():
        raise FileExistsError(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return caminho


def _meta(caminho: Path, meta: dict):
    m = _novo(caminho.with_name(caminho.stem + "_meta.json"))
    m.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# ------------------------------------------------------------------ métricas
def _auc(pos: np.ndarray, s: np.ndarray) -> float:
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metricas(y: np.ndarray, pred: np.ndarray, P: np.ndarray, C: int) -> dict:
    """y, pred inteiros 0..C-1; P scores contínuos (n, C)."""
    cm = np.zeros((C, C), dtype=int)
    np.add.at(cm, (y, pred), 1)
    presentes = [c for c in range(C) if cm[c].sum() > 0]
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    predn = cm.sum(0).astype(float)
    n = cm.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(sup > 0, tp / sup, np.nan)
        prec = np.where(predn > 0, tp / predn, 0.0)
        fp = predn - tp
        tn = n - sup - fp
        esp = np.where((n - sup) > 0, tn / (n - sup), np.nan)
        f1 = np.where((prec + np.nan_to_num(rec)) > 0,
                      2 * prec * np.nan_to_num(rec) / (prec + np.nan_to_num(rec)), 0.0)
    f1 = np.where(sup > 0, f1, np.nan)
    aucs = np.array([_auc(y == c, P[:, c]) for c in range(C)])
    return {"cm": cm, "presentes": presentes, "acuracia": float(tp.sum() / n),
            "acuracia_balanceada": float(np.nanmean(rec[presentes])),
            "f1_macro": float(np.nanmean(f1[presentes])),
            "roc_auc_ovr_macro": float(np.nanmean(aucs[presentes]))
            if np.any(np.isfinite(aucs[presentes])) else float("nan"),
            "precisao": prec, "sensibilidade": rec, "especificidade": esp, "f1": f1,
            "roc_auc": aucs, "suporte": sup}


def bootstrap(y, pred, P, C, grupos, n=N_BOOT, seed=SEED_BOOT):
    """IC 95 % reamostrando GRUPOS (nunca janelas), estratificado pela classe do grupo
    (cada grupo tem uma classe só nas duas bases), para que toda réplica tenha as mesmas
    classes verdadeiras."""
    ug = np.unique(grupos)
    idx_g = {g: np.flatnonzero(grupos == g) for g in ug}
    classe_g = {g: int(y[idx_g[g][0]]) for g in ug}
    assert all(np.all(y[idx_g[g]] == classe_g[g]) for g in ug), "grupo com 2 classes"
    por_c = {}
    for g in ug:
        por_c.setdefault(classe_g[g], []).append(g)
    rng = np.random.default_rng(seed)
    nomes = ["acuracia", "acuracia_balanceada", "f1_macro", "roc_auc_ovr_macro"]
    reps = {k: np.empty(n) for k in nomes}
    for b in range(n):
        sel = [g for c, gs in por_c.items() for g in rng.choice(gs, size=len(gs))]
        ii = np.concatenate([idx_g[g] for g in sel])
        m = metricas(y[ii], pred[ii], P[ii], C)
        for k in nomes:
            reps[k][b] = m[k]
    out = {}
    for k in nomes:
        v = reps[k][np.isfinite(reps[k])]
        out[k] = (tuple(float(x) for x in np.percentile(v, [100 * ALFA / 2, 100 * (1 - ALFA / 2)]))
                  if len(v) else (float("nan"), float("nan")))
    return out, {c: len(gs) for c, gs in por_c.items()}


# ------------------------------------------------------------------ execução
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_ini = time.perf_counter()
    projeto = _git()
    tempos, janelas, feats_p11, modelos, folds_rows = {}, [], [], [], []
    avals = {}
    for base in ("idmt", "a2212"):
        aval0 = AVAL_IDMT if base == "idmt" else AVAL_A2212
        classes = runner.classes_de(base)
        for rot, nome, norm in VARIANTES:
            aval = replace(aval0, normalizar_rms=norm)
            avals[(rot, base)] = _dc(aval)
            segs = runner.segmentos_de(base, aval)
            t0 = time.perf_counter()
            X = runner.extrair(nome, segs, aval, None, n_jobs=a.n_jobs, usar_cache=False)
            tempos[f"extracao/{rot}/{base}"] = round(time.perf_counter() - t0, 1)
            print(f"[{rot} {base}] extração {tempos[f'extracao/{rot}/{base}']} s", flush=True)
            pipe = runner.carregar_pipeline(nome)
            post = runner._post(pipe)
            if nome == "p11_decifrae_harmsum":
                for s, x in zip(segs, X):
                    feats_p11.append({"base": base, "rec_id": s.rec_id, "idx": s.idx,
                                      "t0": s.t0, "t1": s.t1, "classe": s.classe,
                                      **{f"f{k}": float(v) for k, v in enumerate(x)}})
            for prot in PROTOCOLOS[base]:
                t1 = time.perf_counter()
                df = runner.avaliar(nome, base, prot, aval, X=X, usar_cache=False,
                                    n_jobs=1, verbose=False)
                tempos[f"avaliacao/{rot}/{base}/{prot}"] = round(time.perf_counter() - t1, 2)
                df.insert(0, "variante", rot)
                df["normalizar_rms"] = norm
                df["t1"] = df["t0"] + aval.seg_s
                janelas.append(df)
                if nome != "p11_decifrae_harmsum":
                    continue
                # modelo de cada fold (re-treinado igual ao runner; conferido contra p_*)
                folds, _ = runner.folds_de(base, prot, segs)
                yint = np.array([classes.index(s.classe) for s in segs])
                chave = {(s.rec_id, s.idx): i for i, s in enumerate(segs)}
                for f in folds:
                    if len(set(yint[f.treino])) < 2 or len(f.val) == 0:
                        continue
                    clf, C = pipe.treinar(X[f.treino], yint[f.treino], post,
                                          n_classes=len(classes))
                    Pm = pipe.prever((clf, C), X[f.val], post)
                    dff = df[df.fold == f.nome]
                    ii = [chave[(r, k)] for r, k in zip(dff.rec_id, dff.idx)]
                    pos = {i: j for j, i in enumerate(f.val)}
                    dif = np.abs(Pm[[pos[i] for i in ii]]
                                 - dff[[f"p_{c}" for c in classes]].to_numpy()).max()
                    assert dif < 1e-12, f"modelo do fold {f.nome} não reproduz o runner ({dif})"
                    modelos.append({"base": base, "protocolo": prot, "fold": f.nome,
                                    "classes_treino": [classes[c] for c in clf.classes],
                                    "classes": classes, "temp_cos": clf.temp,
                                    "mu": clf.mu.tolist(), "sd": clf.sd.tolist(),
                                    "prototipos": clf.M.tolist(),
                                    "n_treino": int(len(f.treino)), "n_val": int(len(f.val)),
                                    "vazamento_intencional": f.vazamento_intencional})
                    for papel, ids in (("treino", f.treino), ("val", f.val)):
                        for i in ids:
                            folds_rows.append({"base": base, "protocolo": prot,
                                               "fold": f.nome, "papel": papel,
                                               "rec_id": segs[i].rec_id, "idx": segs[i].idx,
                                               "t0": segs[i].t0, "t1": segs[i].t1})
    J = pd.concat(janelas, ignore_index=True)

    # --------------------------------------------------------- métricas + bootstrap
    linhas, aninhado = [], {}
    casos = [(b, p, p) for b in PROTOCOLOS for p in PROTOCOLOS[b]] + \
            [("a2212", "motor", "motor_sem_M4")]
    for rot, nome, norm in VARIANTES:
        for base, prot_fonte, prot in casos:
            classes = runner.classes_de(base)
            C = len(classes)
            d = J[(J.variante == rot) & (J.base == base) & (J.split == prot_fonte)]
            if prot == "motor_sem_M4":
                d = d[d.fold != "a2212_motor_M4"]
            y = np.array([classes.index(c) for c in d.classe])
            pr = np.array([classes.index(c) for c in d.pred])
            P = d[[f"p_{c}" for c in classes]].to_numpy(float)
            g = d.grupo.astype(str).to_numpy()
            m = metricas(y, pr, P, C)
            ic, gpc = bootstrap(y, pr, P, C, g, n=a.n_boot)
            n_grupos = int(len(np.unique(g)))
            comum = {"ts": ts, "variante": rot, "pipeline": nome, "normalizar_rms": norm,
                     "base": base, "protocolo": prot,
                     "vazamento_intencional": prot == "aleatorio",
                     "n_janelas": int(len(d)), "n_grupos": n_grupos,
                     "grupos_por_classe": json.dumps({classes[c]: v for c, v in gpc.items()}),
                     "n_folds": int(d.fold.nunique()), "n_boot": a.n_boot,
                     "seed_boot": SEED_BOOT, "seed_splits": splits.SEED,
                     "aval": json.dumps(avals[(rot, base)], sort_keys=True),
                     "pre": d.pre.iloc[0], "post": d.post.iloc[0],
                     "nota": NOTAS[(base, prot)]}
            for k in ("acuracia", "acuracia_balanceada", "f1_macro", "roc_auc_ovr_macro"):
                linhas.append({**comum, "classe": "_macro", "metrica": k, "valor": m[k],
                               "ic95_lo": ic[k][0], "ic95_hi": ic[k][1]})
            for c in range(C):
                for k in ("precisao", "sensibilidade", "especificidade", "f1", "roc_auc",
                          "suporte"):
                    linhas.append({**comum, "classe": classes[c], "metrica": k,
                                   "valor": float(m[k][c]), "ic95_lo": np.nan,
                                   "ic95_hi": np.nan})
                for c2 in range(C):
                    linhas.append({**comum, "classe": classes[c],
                                   "metrica": f"cm_pred_{classes[c2]}",
                                   "valor": int(m["cm"][c, c2]), "ic95_lo": np.nan,
                                   "ic95_hi": np.nan})
            aninhado.setdefault(rot, {}).setdefault(base, {})[prot] = {
                **{k: m[k] for k in ("acuracia", "acuracia_balanceada", "f1_macro",
                                     "roc_auc_ovr_macro")},
                "ic95": {k: list(v) for k, v in ic.items()},
                "por_classe": {classes[c]: {k: float(m[k][c]) for k in
                                            ("precisao", "sensibilidade", "especificidade",
                                             "f1", "roc_auc", "suporte")} for c in range(C)},
                "matriz_confusao": {"linhas_verdade_colunas_pred": classes,
                                    "valores": m["cm"].tolist()},
                "n_janelas": int(len(d)), "n_grupos": n_grupos,
                "grupos_por_classe": {classes[c]: v for c, v in gpc.items()},
                "nota": NOTAS[(base, prot)]}
            print(f"{rot:22s} {base:5s} {prot:13s} bal_acc={m['acuracia_balanceada']:.3f} "
                  f"[{ic['acuracia_balanceada'][0]:.3f}, {ic['acuracia_balanceada'][1]:.3f}] "
                  f"auc={m['roc_auc_ovr_macro']:.3f} grupos={n_grupos}", flush=True)
    tempos["total"] = round(time.perf_counter() - t_ini, 1)

    # --------------------------------------------------------- escrita
    meta = {"gerado_em": datetime.now().isoformat(timespec="seconds"), "projeto": projeto,
            "script": "embarcados/src/python/diagnostico.py", "argv": sys.argv,
            "seed_boot": SEED_BOOT, "n_boot": a.n_boot, "seed_splits": splits.SEED,
            "variantes": VARIANTES, "protocolos": PROTOCOLOS, "avals": {f"{k[0]}/{k[1]}": v
                                                                      for k, v in avals.items()},
            "bootstrap": "reamostra grupos com reposição, estratificado pela classe do grupo; "
                         "grupo = (gravação, bloco) no IDMT e arquivo no A2212; IC percentil 95 %",
            "usar_cache": False, "n_jobs": a.n_jobs, "tempos_s": tempos,
            "python": sys.version.split()[0], "numpy": np.__version__,
            "nota": "validação apenas; test/ do IDMT não lido (io.py trava)"}
    P_DIR, C_DIR, M_DIR = EMB / "results" / "predictions", EMB / "results" / "csv", \
        EMB / "results" / "metrics"
    saidas = {}
    p = _novo(P_DIR / f"diagnostico_janelas_{ts}.csv")
    J.to_csv(p, index=False); _meta(p, meta); saidas["janelas"] = p
    cols = ["base", "split", "fold", "rec_id", "grupo", "idx", "t0", "t1", "classe", "pred"]
    pp = J[J.variante == "p11_decifrae_harmsum"]
    pcols = [c for c in pp.columns if c.startswith("p_") and pp[c].notna().any()]
    p = _novo(P_DIR / f"p11_janelas_{ts}.csv")
    pp[cols + pcols].rename(columns={"split": "protocolo"}).to_csv(p, index=False)
    _meta(p, meta); saidas["p11_janelas"] = p
    p = _novo(P_DIR / f"p11_folds_{ts}.csv")
    pd.DataFrame(folds_rows).to_csv(p, index=False); _meta(p, meta); saidas["p11_folds"] = p
    p = _novo(P_DIR / f"p11_modelos_{ts}.json")
    p.write_text(json.dumps({"descricao": "CossenoPrototipos por fold: z=(x-mu)/sd; z/=|z|; "
                             "e=z@prototipos.T; p=softmax(e/temp_cos) nas classes_treino; "
                             "classe ausente do treino recebe p=0",
                             "features": "pipelines/_decifrae.py::extrair (13)",
                             "modelos": modelos}, indent=1, ensure_ascii=False),
                 encoding="utf-8")
    _meta(p, meta); saidas["p11_modelos"] = p
    p = _novo(C_DIR / f"p11_features_{ts}.csv")
    pd.DataFrame(feats_p11).to_csv(p, index=False); _meta(p, meta); saidas["p11_features"] = p
    p = _novo(C_DIR / f"diagnostico_{ts}.csv")
    pd.DataFrame(linhas).to_csv(p, index=False); _meta(p, meta); saidas["metricas_csv"] = p
    p = _novo(M_DIR / f"diagnostico_{ts}.json")
    p.write_text(json.dumps({"ts": ts, "metricas": aninhado, "tempos_s": tempos},
                            indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    _meta(p, meta); saidas["metricas_json"] = p
    for k, v in saidas.items():
        print(f"escrito {k}: {v.relative_to(RAIZ)}")
    print(f"tempo total {tempos['total']} s")


if __name__ == "__main__":
    main()
