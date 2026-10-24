"""Peças comuns da Etapa 3 (auditoria das bases): índice, segmentação, features, CSV + meta.

Todas as bases passam pelo mesmo caminho: `out/indices/<id>.csv` (escrito pelo inventário de
cada base) → `lib.load_<id>.carregar(caminho)` → reamostragem para 16 kHz → segmentos de 1 s
sem sobreposição (a sobra final é descartada) → features por segmento, guardadas em cache em
`data/cache/` (nunca commitado).
"""
# Origem das features:
#   nivel_feats  ← Artigo3(MotorSound)/pipelines/b00_nivel.py::extrair (commit d761884)
#   forma_feats  ← Artigo3(MotorSound)/pipelines/b01_forma_espectral.py::extrair (commit d761884)
# Mudanças: nenhuma no cálculo; `stft_mag` vem da cópia em lib/rotacao.py.
from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

from lib.rotacao import stft_mag

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
INDICES = RAIZ / "out" / "indices"
EXPER = RAIZ / "out" / "experiments"
FIGS = RAIZ / "out" / "figuras"
CACHE = DATA / "cache"

SR = 16000          # taxa do microcontrolador
SEG_S = 1.0         # segmentos de 1 s
SEED = 20261022     # seed única de todos os experimentos da auditoria

# Welch para o espectro médio (figura do exp3): 2048 pontos a 16 kHz = 7,8 Hz por bin
WELCH_NPERSEG = 2048

FORMA_NOMES = [f"banda{i:02d}" for i in range(32)] + ["centroide", "espalhamento",
                                                       "rolloff85", "planura", "zcr"]


# --------------------------------------------------------------------------- índice
def ler_indice(base: str) -> pd.DataFrame:
    df = pd.read_csv(INDICES / f"{base}.csv", dtype=str, keep_default_na=False)
    for c in ("sr", "canais", "duracao_s", "f_alim_hz", "rot_esperada_hz"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def carregador(base: str):
    return importlib.import_module(f"lib.load_{base}")


def para_16k(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return y.astype(np.float32)
    g = gcd(int(sr), SR)
    return ss.resample_poly(y.astype(np.float64), SR // g, int(sr) // g).astype(np.float32)


def segmentos(y: np.ndarray, sr: int = SR, seg_s: float = SEG_S) -> np.ndarray:
    n = int(round(seg_s * sr))
    k = len(y) // n
    return y[: k * n].reshape(k, n)


# --------------------------------------------------------------------------- features
def nivel_feats(y):
    r = np.sqrt(np.mean(y ** 2)) + 1e-12
    return np.array([20 * np.log10(r), 20 * np.log10(np.abs(y).max() + 1e-12)])


def forma_feats(y, sr=SR):
    S, f = stft_mag(y, sr, 1024, 512)
    P = S ** 2
    bordas = np.geomspace(50, sr / 2, 33)
    bandas = np.stack([P[(f >= a) & (f < b)].sum(axis=0) for a, b in zip(bordas[:-1], bordas[1:])])
    tot = P.sum(axis=0) + 1e-12
    rel = 10 * np.log10(bandas / tot + 1e-10).mean(axis=1)
    cent = (f[:, None] * P).sum(0) / tot
    esp = np.sqrt(((f[:, None] - cent) ** 2 * P).sum(0) / tot)
    roll = f[np.argmax(np.cumsum(P, axis=0) >= 0.85 * tot, axis=0)]
    plan = np.exp(np.log(P + 1e-12).mean(0)) / (P.mean(0) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.sign(y))) > 0)
    return np.concatenate([rel, [cent.mean(), esp.mean(), roll.mean(), plan.mean(), zcr]])


def welch_db(y, sr=SR):
    f, p = ss.welch(y, fs=sr, nperseg=WELCH_NPERSEG)
    return f, 10 * np.log10(p + 1e-20)


def _escolher_segmentos(k: int, maximo: int | None) -> np.ndarray:
    """Até `maximo` segmentos igualmente espaçados ao longo da gravação (determinístico)."""
    if maximo is None or k <= maximo:
        return np.arange(k)
    return np.unique(np.linspace(0, k - 1, maximo).round().astype(int))


def extrair_base(base: str, max_seg_por_arquivo: int | None = 30, forcar: bool = False,
                 log_cada: int = 200):
    """Features por segmento de 1 s de todos os arquivos do índice, com cache.

    Devolve (tab: DataFrame com metadados + nível + forma, espectros: dict
    (classe, unidade, condicao) → (soma dB, n), freqs)."""
    idx = ler_indice(base)
    cfg = {"base": base, "sr": SR, "seg_s": SEG_S, "max_seg_por_arquivo": max_seg_por_arquivo,
           "welch": WELCH_NPERSEG, "n_arquivos": len(idx),
           "indice_sha": hashlib.sha256((INDICES / f"{base}.csv").read_bytes()).hexdigest()[:16]}
    CACHE.mkdir(parents=True, exist_ok=True)
    arq = CACHE / f"feats_{base}.pkl"
    if arq.exists() and not forcar:
        guardado = pd.read_pickle(arq)
        if guardado["cfg"] == cfg:
            return guardado["tab"], guardado["espectros"], guardado["freqs"]
    ld = carregador(base)
    linhas, espectros, freqs = [], {}, None
    t0 = time.time()
    for i, r in enumerate(idx.itertuples(index=False)):
        y, sr = ld.carregar(DATA / base / r.arquivo)
        segs = segmentos(para_16k(np.asarray(y, dtype=np.float32), sr))
        for s in _escolher_segmentos(len(segs), max_seg_por_arquivo):
            x = segs[s]
            f, pdb = welch_db(x)
            freqs = f
            chave = (r.classe, r.unidade, r.condicao)
            soma, n = espectros.get(chave, (np.zeros_like(pdb), 0))
            espectros[chave] = (soma + pdb, n + 1)
            linhas.append([r.arquivo, int(s), r.classe, r.unidade, r.condicao, r.gravacao_id,
                           *nivel_feats(x), *forma_feats(x)])
        if log_cada and (i + 1) % log_cada == 0:
            print(f"  {base}: {i + 1}/{len(idx)} arquivos, {len(linhas)} segmentos, "
                  f"{time.time() - t0:.0f} s", flush=True)
    cols = ["arquivo", "seg", "classe", "unidade", "condicao", "gravacao_id",
            "rms_db", "pico_db", *FORMA_NOMES]
    tab = pd.DataFrame(linhas, columns=cols)
    pd.to_pickle({"cfg": cfg, "tab": tab, "espectros": espectros, "freqs": freqs}, arq)
    return tab, espectros, freqs


# --------------------------------------------------------------------------- saída
def commit_atual() -> str:
    try:
        h = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
        sujo = subprocess.run(["git", "-C", str(RAIZ), "status", "--porcelain", "scripts"],
                              capture_output=True, text=True).stdout.strip()
        return h + ("+sujo" if sujo else "")
    except Exception:
        return "desconhecido"


def salvar(df: pd.DataFrame, nome: str, base: str, config: dict, t0: float) -> Path:
    """CSV com timestamp em out/experiments/ (nunca sobrescreve) + _meta.json ao lado."""
    EXPER.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{nome}_{base}_{carimbo}"
    caminho = EXPER / f"{stem}.csv"
    if caminho.exists():
        raise FileExistsError(caminho)
    df.to_csv(caminho, index=False)
    meta = {"experimento": nome, "base": base, "commit": commit_atual(), "seed": SEED,
            "config": config, "duracao_s": round(time.time() - t0, 1), "linhas": len(df),
            "python": sys.version.split()[0], "plataforma": platform.platform(),
            "gerado_em": carimbo}
    (EXPER / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"-> {caminho.relative_to(RAIZ)} ({len(df)} linhas)")
    return caminho


def ultimo_csv(nome: str, base: str | None = None) -> Path | None:
    padrao = f"{nome}_{base}_*.csv" if base else f"{nome}_*.csv"
    cands = sorted(p for p in EXPER.glob(padrao) if not p.name.endswith("_meta.json"))
    return cands[-1] if cands else None
