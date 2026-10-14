"""Leitura de áudio e indexação das duas bases.

IDMT-ISA-ELECTRIC-ENGINE (Fraunhofer, Zenodo 7551261 / Kaggle pythonafroz):
    pastas train/ test/ train_cut/ test_cut/, uma subpasta por classe.
    Usar SEMPRE o split train/test oficial. Segmentos de uma mesma gravação
    em lados diferentes do split inflam a acurácia.

BLDC A2212 (Estacio et al., Data in Brief 2023, Mendeley j4yr5fmhv4):
    43 wav de ~10 s a 16 kHz, pastas por classe (Bearing/Healthy/Propeller).
    Split por MOTOR ou por arquivo, nunca por segmento.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

IDMT_CLASSES = ("good", "broken", "heavyload")
BLDC_CLASSES = ("healthy", "bearing", "propeller")

# Condições de ruído do IDMT, inferidas do nome do arquivo.
IDMT_NOISE_TAGS = ("pure", "talking", "white_noise", "atmo_low", "atmo_medium",
                   "atmo_high", "stresstest")


@dataclass
class Item:
    path: Path
    label: str
    split: str          # "train" | "test" | "all"
    group: str          # unidade de agrupamento para validação cruzada
    condition: str      # condição de ruído (IDMT) ou "n/a"


def load_wav(path: Path | str, sr_target: int | None = None) -> tuple[np.ndarray, int]:
    """Lê wav mono em float32 [-1, 1] e reamostra se pedido."""
    sr, x = wavfile.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    if sr_target and sr != sr_target:
        g = gcd(sr, sr_target)
        x = resample_poly(x, sr_target // g, sr // g).astype(np.float32)
        sr = sr_target
    return x, sr


def _label_from_text(text: str, classes: tuple[str, ...]) -> str | None:
    t = text.lower().replace("-", "").replace(" ", "")
    for c in classes:
        if c in t:
            return c
    return None


def _idmt_condition(name: str) -> str:
    n = name.lower()
    for tag in IDMT_NOISE_TAGS:
        if tag.replace("_", "") in n.replace("_", ""):
            return tag
    if "atmo" in n:
        return "atmo"
    return "unknown"


def index_idmt(root: Path | str, use_cut: bool = False) -> list[Item]:
    """Indexa o IDMT. Classe vem do caminho; split vem da pasta train/test."""
    root = Path(root)
    items: list[Item] = []
    for split in ("train", "test"):
        folder = root / (f"{split}_cut" if use_cut else split)
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*.wav")):
            label = _label_from_text(str(p.relative_to(folder)), IDMT_CLASSES)
            if label is None:
                continue
            # grupo = gravação de origem (nome sem sufixo de recorte)
            stem = re.sub(r"_(cut_)?\d+$", "", p.stem)
            items.append(Item(p, label, split, stem, _idmt_condition(p.stem)))
    return items


def index_bldc(root: Path | str) -> list[Item]:
    """Indexa o A2212. Grupo = id do motor quando o nome permite extrair."""
    root = Path(root)
    items: list[Item] = []
    for p in sorted(root.rglob("*.wav")):
        label = _label_from_text(str(p.relative_to(root)), BLDC_CLASSES)
        if label is None:
            continue
        m = re.search(r"(?:motor|m)[ _-]?(\d+)", p.stem.lower())
        group = f"motor{m.group(1)}" if m else p.stem
        items.append(Item(p, label, "all", group, "n/a"))
    return items


def summarize(items: list[Item]) -> str:
    from collections import Counter
    c = Counter((it.split, it.label) for it in items)
    g = len({it.group for it in items})
    lines = [f"{len(items)} arquivos, {g} grupos"]
    lines += [f"  {s:5s} {l:10s} {n}" for (s, l), n in sorted(c.items())]
    return "\n".join(lines)
