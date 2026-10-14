"""Benchmark principal do artigo.

Para cada base:
  1. extrai features por janela (cache em .npz);
  2. treina regressão logística (o que vai para o MCU) e, como teto,
     um Random Forest;
  3. reporta por janela, por condição de ruído e por grupo de features
     (ablação: tirar cada grupo e ver quanto cai);
  4. monta sequências com trocas de estado e mede atraso de detecção e
     trocas espúrias, com e sem HMM.

Splits
  IDMT : train/test oficial.
  BLDC : leave-one-file-out agrupado (GroupKFold por arquivo). Com 43
         arquivos, reportar média e desvio entre dobras.

Uso:
    python scripts/run_benchmark.py idmt /caminho/IDMT-ISA-ELECTRIC-ENGINE
    python scripts/run_benchmark.py bldc /caminho/BLDC
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motoracoustic.config import IDMT_CFG, BLDC_CFG  # noqa: E402
from motoracoustic.evaluate import (by_condition, detection_delays,  # noqa: E402
                                    spurious_switches_per_min, window_metrics)
from motoracoustic.features import FEATURE_NAMES, GROUPS, file_features  # noqa: E402
from motoracoustic.hmm import ForwardFilter, viterbi_states  # noqa: E402
from motoracoustic.io import index_bldc, index_idmt, load_wav  # noqa: E402
from motoracoustic.preprocess import preprocess  # noqa: E402


def extract(items, cfg, cache: Path):
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["file"], d["cond"], d["split"], d["group"]
    X, y, fid, cond, split, group = [], [], [], [], [], []
    for i, it in enumerate(items):
        x, sr = load_wav(it.path, cfg.sr)
        F = file_features(preprocess(x, sr, cfg), sr, cfg)
        X.append(F)
        n = len(F)
        y += [it.label] * n; fid += [i] * n; cond += [it.condition] * n
        split += [it.split] * n; group += [it.group] * n
        if i % 50 == 0:
            print(f"  {i}/{len(items)}", flush=True)
    out = (np.vstack(X), np.array(y), np.array(fid), np.array(cond),
           np.array(split), np.array(group))
    np.savez(cache, X=out[0], y=out[1], file=out[2], cond=out[3],
             split=out[4], group=out[5])
    return out


def models():
    return {
        "logreg_mcu": make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, C=1.0)),
        "rf_teto": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
    }


def sequence_eval(clf, X, y, fid, labels, cfg, rng=0, n_seq=50, files_per_seq=4):
    """Concatena arquivos de teste de classes diferentes e mede transições."""
    rng = np.random.default_rng(rng)
    lab2i = {l: i for i, l in enumerate(labels)}
    files = np.unique(fid)
    res = {"raw": [], "hmm_viterbi": [], "hmm_online": []}
    spur = {k: [] for k in res}
    for _ in range(n_seq):
        chosen, last = [], None
        for _ in range(files_per_seq):
            cand = [f for f in files if y[fid == f][0] != last]
            f = rng.choice(cand); chosen.append(f); last = y[fid == f][0]
        idx = np.concatenate([np.flatnonzero(fid == f) for f in chosen])
        yt = np.array([lab2i[v] for v in y[idx]])
        P = clf.predict_proba(X[idx])
        preds = {
            "raw": P.argmax(1),
            "hmm_viterbi": viterbi_states(P, cfg.p_stay),
            "hmm_online": ForwardFilter(len(labels), cfg.p_stay).run(P),
        }
        for k, yp in preds.items():
            res[k] += detection_delays(yt, yp, cfg.hop_s)
            spur[k].append(spurious_switches_per_min(yt, yp, cfg.hop_s))
    out = {}
    for k in res:
        d = np.array(res[k])
        out[k] = {"atraso_mediano_s": float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else None,
                  "trocas_perdidas": float(np.mean(~np.isfinite(d))),
                  "espurias_por_min": float(np.mean(spur[k]))}
    return out


def run_idmt(root):
    cfg = IDMT_CFG
    items = index_idmt(root)
    X, y, fid, cond, split, _ = extract(items, cfg, Path("cache_idmt.npz"))
    labels = sorted(set(y))
    tr, te = split == "train", split == "test"
    report = {"n_janelas": {"train": int(tr.sum()), "test": int(te.sum())}}
    for name, m in models().items():
        m.fit(X[tr], y[tr])
        yp = m.predict(X[te])
        report[name] = window_metrics(y[te], yp, labels)
        report[name]["por_condicao"] = by_condition(y[te], yp, cond[te])
        if name == "logreg_mcu":
            report[name]["sequencias"] = sequence_eval(m, X[te], y[te], fid[te], labels, cfg)
    report["ablacao_logreg"] = ablation(X, y, tr, te, labels)
    return report


def run_bldc(root):
    cfg = BLDC_CFG
    items = index_bldc(root)
    X, y, fid, _, _, group = extract(items, cfg, Path("cache_bldc.npz"))
    labels = sorted(set(y))
    report = {}
    n_splits = min(5, len(np.unique(fid)))
    for name, m in models().items():
        accs, f1s = [], []
        for tr_i, te_i in GroupKFold(n_splits=n_splits).split(X, y, groups=fid):
            m.fit(X[tr_i], y[tr_i])
            r = window_metrics(y[te_i], m.predict(X[te_i]), labels)
            accs.append(r["acc"]); f1s.append(r["f1_macro"])
        report[name] = {"acc_media": float(np.mean(accs)), "acc_dp": float(np.std(accs)),
                        "f1_media": float(np.mean(f1s))}
    return report


def ablation(X, y, tr, te, labels):
    out = {}
    for g, cols in GROUPS.items():
        keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in cols]
        m = models()["logreg_mcu"].fit(X[tr][:, keep], y[tr])
        out[f"sem_{g}"] = window_metrics(y[te], m.predict(X[te][:, keep]), labels)["acc"]
    return out


if __name__ == "__main__":
    kind, root = sys.argv[1], sys.argv[2]
    rep = run_idmt(root) if kind == "idmt" else run_bldc(root)
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"{kind}_report.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
