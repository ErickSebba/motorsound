# Origem: molde de Artigo2(Acordes)/Decifra-Violão/runner.py (commit 93609f7):
#   um CSV por experimento com timestamp, nunca sobrescrito, `_meta.json` com commit,
#   seed, configuração e duração; `_commit()` copiado.
# Mudanças: em vez de variantes A–F de um pipeline só, roda módulos de `pipelines/`
#   com a interface do CLAUDE.md sobre os folds de `splits.py`; uma linha por janela
#   (pipeline, gravação, grupo, janela, condição), com a configuração repetida na linha.
"""Runner comum: mesmo split, mesmas janelas, mesmas métricas para toda pipeline.

    python -m motorsound.runner --pipelines p00_logmel_cnn p11_decifrae_harmsum \
        --base idmt --split blocos --tag val_idmt

Interface que cada `pipelines/<nome>.py` expõe (CLAUDE.md):
    NOME, ORIGEM, TIPO, PRE (dataclass), POST (dataclass, opcional)
    extrair(y, sr, pre) -> np.ndarray de forma fixa
    treinar(X, rotulos_int, post, n_classes) -> modelo
    prever(modelo, X, post) -> (n, n_classes) probabilidades
    custo_mcu() -> dict | None
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime

import numpy as np
import pandas as pd

from . import cache, io, splits
from .config import AVAL_A2212, AVAL_IDMT, AvaliacaoConfig, PostConfig, PreConfig

OUT = io.RAIZ / "out" / "experiments"


def _commit():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=io.RAIZ, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": git("rev-parse", "--short", "HEAD"),
            "sujo": bool(git("status", "--porcelain"))}


def carregar_pipeline(nome: str):
    if str(io.RAIZ) not in sys.path:
        sys.path.insert(0, str(io.RAIZ))
    return importlib.import_module(f"pipelines.{nome}")


def _pre(pipe):
    return getattr(pipe, "PRE", PreConfig())


def _post(pipe):
    return getattr(pipe, "POST", PostConfig())


def _dc(obj):
    return asdict(obj) if is_dataclass(obj) else obj


# ---------------------------------------------------------------- extração
def _preparar(y, aval: AvaliacaoConfig):
    y = np.asarray(y, dtype=np.float32)
    if aval.normalizar_rms:
        r = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
        y = y * (aval.rms_alvo / r)
    return y


def _extrair_gravacao(nome_pipe, rec_id, path, segs_rec, aval, perturbacao, usar_cache):
    pipe = carregar_pipeline(nome_pipe)
    pre = _pre(pipe)
    idxs = [s.idx for s in segs_rec]
    k = cache.chave(pipe.NOME, pre, aval, perturbacao, rec_id, idxs)
    if usar_cache:
        X = cache.carregar(k)
        if X is not None:
            return X
    y = io.carregar(path, pre.sr)
    out = []
    for s in segs_rec:
        i0, i1 = int(round(s.t0 * pre.sr)), int(round(s.t1 * pre.sr))
        seg = np.array(y[i0:i1], dtype=np.float32)
        if perturbacao:
            from ruidos import aplicar
            semente = (zlib.crc32(f"{rec_id}#{s.idx}".encode()) & 0xFFFFFF) \
                + int(perturbacao.get("seed", 0))
            seg = aplicar(seg, pre.sr, perturbacao, seed=semente).astype(np.float32)
        out.append(np.asarray(pipe.extrair(_preparar(seg, aval), pre.sr, pre)))
    X = np.stack(out)
    if usar_cache:
        cache.salvar(k, X)
    return X


def extrair(nome_pipe, segs, aval, perturbacao=None, n_jobs=4, usar_cache=True):
    """Features de todos os segmentos, na ordem de `segs`."""
    grs = {}
    for g in (io.gravacoes("a2212") if segs[0].base == "a2212"
              else io.gravacoes("idmt", segs[0].split)):
        grs[g.rec_id] = g.path
    por_rec: dict[str, list[int]] = {}
    for i, s in enumerate(segs):
        por_rec.setdefault(s.rec_id, []).append(i)
    tarefas = [(r, ii) for r, ii in por_rec.items()]
    if n_jobs == 1:
        res = [_extrair_gravacao(nome_pipe, r, grs[r], [segs[i] for i in ii], aval,
                                 perturbacao, usar_cache) for r, ii in tarefas]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(
            delayed(_extrair_gravacao)(nome_pipe, r, grs[r], [segs[i] for i in ii], aval,
                                       perturbacao, usar_cache) for r, ii in tarefas)
    X = None
    for (r, ii), Xr in zip(tarefas, res):
        if X is None:
            X = np.empty((len(segs),) + Xr.shape[1:], dtype=Xr.dtype)
        X[ii] = Xr
    return X


# ---------------------------------------------------------------- avaliação
def folds_de(base, split_tipo, segs):
    if base == "idmt":
        if split_tipo == "blocos":
            return splits.idmt_blocos(segs), splits.grupo_de(segs, splits.grupos_idmt_blocos(segs))
        if split_tipo == "aleatorio":
            # mesma unidade de relatório (grupo = bloco) para comparar com os blocos
            return (splits.aleatorio_por_segmento(segs),
                    splits.grupo_de(segs, splits.grupos_idmt_blocos(segs)))
    else:
        g = splits.grupo_de(segs)
        if split_tipo == "arquivo":
            return splits.a2212_por_arquivo(segs), g
        if split_tipo == "motor":
            return splits.a2212_por_motor(segs), g
        if split_tipo == "aleatorio":
            return splits.aleatorio_por_segmento(segs), g
    raise ValueError(f"split {split_tipo!r} não existe para {base}")


def classes_de(base):
    return list(io.CLASSES_IDMT if base == "idmt" else io.CLASSES_A2212)


def segmentos_de(base, aval):
    if base == "idmt":
        return io.segmentos(io.gravacoes("idmt", "train"), aval.seg_s)
    return io.segmentos(io.gravacoes("a2212"), aval.seg_s)


def avaliar(nome_pipe, base, split_tipo, aval=None, post=None, perturbacao_val=None,
            n_jobs=4, usar_cache=True, X=None, Xval=None, verbose=True):
    """Roda os folds e devolve um DataFrame com uma linha por janela de validação."""
    pipe = carregar_pipeline(nome_pipe)
    aval = aval or (AVAL_IDMT if base == "idmt" else AVAL_A2212)
    post = post or _post(pipe)
    classes = classes_de(base)
    segs = segmentos_de(base, aval)
    folds, grupos = folds_de(base, split_tipo, segs)
    y = np.array([classes.index(s.classe) for s in segs])
    t0 = time.perf_counter()
    if X is None:
        X = extrair(nome_pipe, segs, aval, None, n_jobs, usar_cache)
    if Xval is None:
        Xval = X if not perturbacao_val else extrair(nome_pipe, segs, aval, perturbacao_val,
                                                    n_jobs, usar_cache)
    t_ext = time.perf_counter() - t0
    linhas = []
    for f in folds:
        if len(set(y[f.treino])) < 2 or len(f.val) == 0:
            continue
        t1 = time.perf_counter()
        modelo = pipe.treinar(X[f.treino], y[f.treino], post, n_classes=len(classes))
        P = np.asarray(pipe.prever(modelo, Xval[f.val], post), dtype=float)
        dt = time.perf_counter() - t1
        pred = P.argmax(axis=1)
        for j, i in enumerate(f.val):
            s = segs[i]
            linhas.append({
                "pipeline": pipe.NOME, "tipo": pipe.TIPO, "base": base,
                "split": split_tipo, "fold": f.nome,
                "vazamento_intencional": f.vazamento_intencional,
                "rec_id": s.rec_id, "grupo": grupos[i], "idx": s.idx, "t0": s.t0,
                "classe": s.classe, "condicao": s.condicao, "motor": s.motor,
                "velocidade": s.velocidade,
                "perturbacao": json.dumps(perturbacao_val or {}, sort_keys=True),
                "pred": classes[pred[j]], "acerto": int(pred[j] == y[i]),
                **{f"p_{c}": float(P[j, k]) for k, c in enumerate(classes)},
                "fold_segundos": round(dt, 3),
                "aval": json.dumps(_dc(aval), sort_keys=True),
                "pre": json.dumps(_dc(_pre(pipe)), sort_keys=True, default=str),
                "post": json.dumps(_dc(post), sort_keys=True, default=str),
            })
        if verbose:
            acc = float(np.mean(pred == y[f.val]))
            print(f"  {pipe.NOME} {base}/{split_tipo} {f.nome}: acc={acc:.3f} "
                  f"(n={len(f.val)}, {dt:.1f}s)", flush=True)
    df = pd.DataFrame(linhas)
    df.attrs["t_extracao"] = t_ext
    return df


def avaliar_condicoes(nome_pipe, base, split_tipo, aval, X, Xc: dict, post=None):
    """Treina UMA vez por fold em X (limpo) e prevê cada X de `Xc` ({rótulo: X}).
    Mesmas colunas de `avaliar`, mais `condicao_sintetica`."""
    pipe = carregar_pipeline(nome_pipe)
    post = post or _post(pipe)
    classes = classes_de(base)
    segs = segmentos_de(base, aval)
    folds, grupos = folds_de(base, split_tipo, segs)
    y = np.array([classes.index(s.classe) for s in segs])
    linhas = []
    for f in folds:
        if len(set(y[f.treino])) < 2 or len(f.val) == 0:
            continue
        modelo = pipe.treinar(X[f.treino], y[f.treino], post, n_classes=len(classes))
        for rot, Xv in Xc.items():
            P = np.asarray(pipe.prever(modelo, Xv[f.val], post), dtype=float)
            pred = P.argmax(axis=1)
            for j, i in enumerate(f.val):
                s = segs[i]
                linhas.append({
                    "pipeline": pipe.NOME, "tipo": pipe.TIPO, "base": base,
                    "split": split_tipo, "fold": f.nome, "rec_id": s.rec_id,
                    "grupo": grupos[i], "idx": s.idx, "t0": s.t0, "classe": s.classe,
                    "condicao": s.condicao, "motor": s.motor, "velocidade": s.velocidade,
                    "condicao_sintetica": rot, "pred": classes[pred[j]],
                    "acerto": int(pred[j] == y[i]),
                    **{f"p_{c}": float(P[j, k]) for k, c in enumerate(classes)},
                    "aval": json.dumps(_dc(aval), sort_keys=True),
                    "pre": json.dumps(_dc(_pre(pipe)), sort_keys=True, default=str),
                    "post": json.dumps(_dc(post), sort_keys=True, default=str),
                })
    return pd.DataFrame(linhas)


def escrever(df: pd.DataFrame, tag: str, meta: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUT / f"{ts}_{tag}"
    if base.with_suffix(".csv").exists():
        raise FileExistsError(base)
    df.to_csv(base.with_suffix(".csv"), index=False)
    meta = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
            "projeto": _commit(), "python": sys.version.split()[0],
            "numpy": np.__version__, **meta}
    (OUT / f"{ts}_{tag}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return base.with_suffix(".csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", nargs="+", required=True)
    ap.add_argument("--base", choices=["idmt", "a2212"], required=True)
    ap.add_argument("--split", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--sem-normalizar", action="store_true")
    ap.add_argument("--seg-s", type=float, default=None)
    a = ap.parse_args(argv)
    aval = AVAL_IDMT if a.base == "idmt" else AVAL_A2212
    if a.sem_normalizar:
        aval = replace(aval, normalizar_rms=False)
    if a.seg_s:
        aval = replace(aval, seg_s=a.seg_s)
    t0 = time.perf_counter()
    dfs, falhas, custos = [], [], {}
    for p in a.pipelines:
        for sp in a.split:
            try:
                dfs.append(avaliar(p, a.base, sp, aval, n_jobs=a.n_jobs))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                falhas.append(f"{p}/{sp}: {type(exc).__name__}: {exc}")
        try:
            custos[p] = carregar_pipeline(p).custo_mcu()
        except Exception as exc:
            custos[p] = f"erro: {exc}"
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    caminho = escrever(df, a.tag, {"argv": sys.argv, "pipelines": a.pipelines,
                                   "base": a.base, "splits": a.split, "aval": _dc(aval),
                                   "seed_splits": splits.SEED, "falhas": falhas,
                                   "custo_mcu": custos,
                                   "segundos": round(time.perf_counter() - t0, 1)})
    if len(df):
        r = df.groupby(["pipeline", "split"]).acerto.mean().unstack()
        print(r.round(3).to_string())
    print(f"escrito {caminho.relative_to(io.RAIZ)}; falhas: {falhas}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
