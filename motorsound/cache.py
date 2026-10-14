# Origem: Artigo2(Acordes)/Decifra-Violão/cache.py (commit 93609f7)
# Mudanças: a chave deixa de ser (áudio, PreChromaConfig) e passa a ser (pipeline,
#   PRE da pipeline, AvaliacaoConfig, perturbação, conjunto de segmentos); um .npz
#   guarda as features empilhadas de todos os segmentos de uma gravação. A verificação
#   campo a campo dos metadados e a escrita atômica ficam como no violão.
"""Cache de features por (pipeline, configuração, gravação). Nunca devolve em silêncio
algo que não corresponde ao pedido."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from .io import RAIZ

CACHE_DIR = RAIZ / "cache"
_META = "meta"


class CacheMismatchError(Exception):
    pass


def _itens(obj):
    if obj is None:
        return None
    if is_dataclass(obj):
        return [(f.name, _itens(getattr(obj, f.name)) if is_dataclass(getattr(obj, f.name))
                 else getattr(obj, f.name)) for f in fields(obj)]
    if isinstance(obj, dict):
        return sorted(obj.items())
    return obj


def chave(pipeline_nome, pre, aval, perturbacao, rec_id, idxs) -> dict:
    return {"pipeline": pipeline_nome, "pre": _itens(pre), "aval": _itens(aval),
            "perturbacao": _itens(perturbacao), "rec_id": rec_id,
            "idxs": [int(i) for i in idxs]}


def caminho(k: dict) -> Path:
    s = json.dumps(k, ensure_ascii=True, separators=(",", ":"), default=str)
    h = hashlib.sha256(s.encode()).hexdigest()[:12]
    nome = k["rec_id"].replace("/", "_")
    return CACHE_DIR / k["pipeline"] / f"{nome}__{h}.npz"


def carregar(k: dict):
    p = caminho(k)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as d:
        meta = json.loads(str(d[_META].item()))
        esperado = json.loads(json.dumps(k, default=str))
        div = [c for c in esperado if meta.get(c) != esperado[c]]
        if div:
            raise CacheMismatchError(f"{p.name}: campos divergentes {div}")
        return d["X"]


def salvar(k: dict, X: np.ndarray) -> Path:
    p = caminho(k)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / (p.name + ".tmp")
    with tmp.open("wb") as fp:
        np.savez(fp, X=X, **{_META: np.array(json.dumps(k, default=str))})
    os.replace(tmp, p)
    return p
