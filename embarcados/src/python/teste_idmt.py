"""Avaliação única no conjunto de teste oficial do IDMT (autorizada pelo Erick em 25/10/2026).

Duas fases, nesta ordem:

    .venv\\Scripts\\python.exe embarcados\\src\\python\\teste_idmt.py congelar
        grava configs/experiments/congelado_teste_idmt_<ts>.json (variantes, configs, hashes do
        código) SEM tocar no teste; o arquivo é commitado antes da fase seguinte.
    .venv\\Scripts\\python.exe embarcados\\src\\python\\teste_idmt.py avaliar --congelado <json>
        confere os hashes, abre o teste (MOTORSOUND_ABRIR_TESTE=1 só neste processo), treina cada
        variante em TODO o train/ do IDMT (3 classes, condição pure) e prevê todas as janelas de 3 s
        do test/ (9 condições). Nada é ajustado depois de ver o teste.

Saídas (nunca sobrescritas): results/predictions/teste_idmt_janelas_<ts>.csv (uma linha por janela),
results/csv/teste_idmt_<ts>.csv (métricas por variante × condição, IC95 por bootstrap de gravações
no agregado), results/csv/teste_idmt_curvas_<ts>.csv (pontos ROC e PR, um-contra-o-resto), _meta.json.

Ressalvas que valem para qualquer leitura: no IDMT cada classe é um motor físico (classe = unidade);
treino e teste são sessões diferentes do mesmo motor; o treino tem só `pure`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

EMB = Path(__file__).resolve().parents[2]
RAIZ = EMB.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import diagnostico as dg  # noqa: E402
from motorsound import io, runner  # noqa: E402
from motorsound.config import AVAL_IDMT  # noqa: E402

VARIANTES = dg.VARIANTES
SEED_BOOT = 20261025
FONTES = ["pipelines/b00_nivel.py", "pipelines/p04_classicas.py", "pipelines/p11_decifrae_harmsum.py",
          "pipelines/_decifrae.py", "motorsound/runner.py", "motorsound/io.py", "motorsound/config.py",
          "motorsound/classificador.py", "motorsound/rotacao.py", "embarcados/src/python/diagnostico.py",
          "embarcados/src/python/teste_idmt.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def commit():
    r = lambda *a: subprocess.run(["git", "-C", str(RAIZ), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": r("rev-parse", "HEAD"), "sujo": bool(r("status", "--porcelain", "--", "pipelines", "motorsound"))}


def configuracao():
    out = []
    for rot, nome, norm in VARIANTES:
        pipe = runner.carregar_pipeline(nome)
        out.append({"variante": rot, "pipeline": nome, "normalizar_rms": norm,
                    "aval": dg._dc(replace(AVAL_IDMT, normalizar_rms=norm)),
                    "pre": dg._dc(runner._pre(pipe)), "post": dg._dc(runner._post(pipe))})
    return out


def cmd_congelar(a):
    if os.environ.get("MOTORSOUND_ABRIR_TESTE") == "1":
        sys.exit("congelar não pode rodar com o teste liberado")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = {"categoria": "congelamento_antes_do_teste", "gerado_em": ts, "projeto": commit(),
         "autorizacao": "Erick, 25/10/2026, na conversa da sessão (abrir o teste oficial do IDMT)",
         "protocolo": "treino em todas as janelas de 3 s de train/ (pure, 3 classes); teste em todas "
                      "as janelas de 3 s de test/ (9 condições), uma única vez; sem ajuste posterior",
         "variantes": configuracao(), "seed_boot": SEED_BOOT, "n_boot": dg.N_BOOT,
         "hashes": {f: sha(RAIZ / f) for f in FONTES}}
    out = EMB / "configs" / "experiments" / f"congelado_teste_idmt_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(out)


def curvas(y, P, classes, rot):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
    linhas = []
    for k, c in enumerate(classes):
        pos = (y == k).astype(int)
        fpr, tpr, _ = roc_curve(pos, P[:, k])
        pr, rc, _ = precision_recall_curve(pos, P[:, k])
        ap = average_precision_score(pos, P[:, k])
        linhas += [{"variante": rot, "classe": c, "curva": "roc", "x": float(x), "y": float(v)}
                   for x, v in zip(fpr, tpr)]
        linhas += [{"variante": rot, "classe": c, "curva": "pr", "x": float(x), "y": float(v),
                    "ap": float(ap)} for x, v in zip(rc, pr)]
    return linhas


def cmd_avaliar(a):
    cong = json.loads(Path(a.congelado).read_text(encoding="utf-8"))
    dif = [f for f, h in cong["hashes"].items() if sha(RAIZ / f) != h]
    if dif:
        sys.exit(f"código mudou depois do congelamento: {dif}")
    if json.dumps(configuracao(), default=str, sort_keys=True) != \
            json.dumps(cong["variantes"], default=str, sort_keys=True):
        sys.exit("configuração difere da congelada")
    os.environ["MOTORSOUND_ABRIR_TESTE"] = "1"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    t_ini = time.perf_counter()
    classes = runner.classes_de("idmt")
    C = len(classes)
    janelas, mets, curv = [], [], []
    for rot, nome, norm in VARIANTES:
        aval = replace(AVAL_IDMT, normalizar_rms=norm)
        tr = io.segmentos(io.gravacoes("idmt", "train"), aval.seg_s)
        te = io.segmentos(io.gravacoes("idmt", "test"), aval.seg_s)
        Xtr = runner.extrair(nome, tr, aval, None, n_jobs=a.n_jobs, usar_cache=False)
        Xte = runner.extrair(nome, te, aval, None, n_jobs=a.n_jobs, usar_cache=False)
        pipe = runner.carregar_pipeline(nome)
        post = runner._post(pipe)
        ytr = np.array([classes.index(s.classe) for s in tr])
        yte = np.array([classes.index(s.classe) for s in te])
        modelo = pipe.treinar(Xtr, ytr, post, n_classes=C)
        P = np.asarray(pipe.prever(modelo, Xte, post), dtype=float)
        pred = P.argmax(1)
        print(f"[{rot}] treino {len(tr)} janelas, teste {len(te)}; acc {np.mean(pred == yte):.3f}", flush=True)
        for j, s in enumerate(te):
            janelas.append({"variante": rot, "pipeline": nome, "rec_id": s.rec_id, "idx": s.idx,
                            "t0": s.t0, "classe": s.classe, "condicao": s.condicao,
                            "pred": classes[pred[j]], "acerto": int(pred[j] == yte[j]),
                            **{f"p_{c}": float(P[j, k]) for k, c in enumerate(classes)}})
        grupos = np.array([s.rec_id for s in te])
        conds = np.array([s.condicao for s in te])
        for cond in ["_todas"] + sorted(set(conds)):
            m_ = slice(None) if cond == "_todas" else conds == cond
            yy, pp, PP, gg = yte[m_], pred[m_], P[m_], grupos[m_]
            m = dg.metricas(yy, pp, PP, C)
            ic = dg.bootstrap(yy, pp, PP, C, gg, n=dg.N_BOOT, seed=SEED_BOOT)[0] if cond == "_todas" else {}
            base_l = {"variante": rot, "pipeline": nome, "normalizar_rms": norm, "condicao": cond,
                      "n_janelas": int(len(yy)), "n_gravacoes": int(len(set(gg)))}
            # mesmo formato longo do diagnostico.py: escalares em "_macro", vetores por classe,
            # matriz de confusão como cm_pred_<classe> por linha verdadeira
            for chave in ("acuracia", "acuracia_balanceada", "f1_macro", "roc_auc_ovr_macro"):
                lo, hi = ic.get(chave, (np.nan, np.nan))
                mets.append({**base_l, "classe": "_macro", "metrica": chave, "valor": m[chave],
                             "ic95_lo": lo, "ic95_hi": hi})
            for chave in ("precisao", "sensibilidade", "especificidade", "f1", "roc_auc", "suporte"):
                for k, cl in enumerate(classes):
                    mets.append({**base_l, "classe": cl, "metrica": chave, "valor": float(m[chave][k]),
                                 "ic95_lo": np.nan, "ic95_hi": np.nan})
            for k, cl in enumerate(classes):
                for k2, cl2 in enumerate(classes):
                    mets.append({**base_l, "classe": cl, "metrica": f"cm_pred_{cl2}",
                                 "valor": int(m["cm"][k, k2]), "ic95_lo": np.nan, "ic95_hi": np.nan})
        curv += curvas(yte, P, classes, rot)
    pj = EMB / "results" / "predictions" / f"teste_idmt_janelas_{ts}.csv"
    pm = EMB / "results" / "csv" / f"teste_idmt_{ts}.csv"
    pc = EMB / "results" / "csv" / f"teste_idmt_curvas_{ts}.csv"
    pj.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(janelas).to_csv(pj, index=False)
    pd.DataFrame(mets).to_csv(pm, index=False)
    pd.DataFrame(curv).to_csv(pc, index=False)
    meta = {"categoria": "teste_oficial_IDMT", "gerado_em": ts, "congelado": str(Path(a.congelado).resolve()),
            "projeto": commit(), "segundos": round(time.perf_counter() - t_ini, 1),
            "janelas": str(pj), "curvas": str(pc), "seed_boot": SEED_BOOT, "n_boot": dg.N_BOOT,
            "nota": "teste lido uma única vez com configurações congeladas; classe = motor no IDMT"}
    for p in (pm, pj, pc):
        p.with_name(p.stem + "_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    print(pm)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("congelar")
    av = sp.add_parser("avaliar")
    av.add_argument("--congelado", required=True)
    av.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args(argv)
    {"congelar": cmd_congelar, "avaliar": cmd_avaliar}[a.cmd](a)


if __name__ == "__main__":
    main()
