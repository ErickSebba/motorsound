"""Peças comuns da bancada (PC e ESP32): conjunto de janelas, modelos e estatísticas.

As janelas de latência são os casos NÃO-borda do ouro (limpos + ruidosos, só validação dos
folds `idmt_bloco0` e `a2212_arq0`): 75 por base. Para chegar a >= 200 medições por base, a
lista é percorrida em ciclo, na ordem do manifesto; cada medição registra o id da janela, então
a repetição é explícita nos CSVs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

EMB = Path(__file__).resolve().parents[2]
RAIZ = EMB.parent
GOLD = EMB / "gold"
DUR_S = {"idmt": 3.0, "a2212": 1.0}
MODELO_DA_BASE = {"idmt": "idmt_bloco0", "a2212": "a2212_arq0"}


def manifesto():
    return json.loads((GOLD / "manifest.json").read_text(encoding="utf-8"))


def casos_latencia(base, n_total, aquecimento):
    """Lista de ids (aquecimento + n_total) percorrendo os casos não-borda da base em ciclo."""
    ids = [c["id"] for c in manifesto()["casos"] if c["tipo"] != "borda" and c["base"] == base]
    k = aquecimento + n_total
    return [ids[i % len(ids)] for i in range(k)], len(ids)


def audio(caso_id):
    return np.fromfile(GOLD / "casos" / caso_id / "audio.f32", dtype="<f4")


def modelo_bin(base):
    return GOLD / "modelos" / f"{MODELO_DA_BASE[base]}.bin"


def modelo_json(base):
    return json.loads((GOLD / "modelos" / f"{MODELO_DA_BASE[base]}.json").read_text(encoding="utf-8"))


def commit():
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=RAIZ, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": g("rev-parse", "HEAD"), "sujo": bool(g("status", "--porcelain", "embarcados"))}


def carimbo():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def resumo(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(v.size), "mediana": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max()), "min": float(v.min()), "media": float(v.mean())}


def garantir_raiz_no_path():
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))
