"""Leitura das bases e enumeração dos segmentos.

A unidade de avaliação é o *segmento* (janela fixa, sem sobreposição) recortado da
gravação de origem. Os segmentos são recortados aqui a partir dos arquivos longos
(`train/`), não dos `*_cut`, para que a gravação de origem e a posição no tempo sejam
conhecidas sem depender de nome de arquivo (ver docs/DATASETS.md §1.4).

Trava do teste: nenhuma função lê `test/` ou `test_cut/` do IDMT a menos que a variável
de ambiente MOTORSOUND_ABRIR_TESTE=1 esteja definida, o que só acontece depois de
`congelar.py` e da aprovação do Erick (CLAUDE.md, "Dados", regra 2).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
IDMT = DATA / "idmt"
A2212 = DATA / "a2212" / "BLDC_sound_data"

CLASSES_IDMT = ("good", "broken", "heavyload")
PASTAS_IDMT = {"good": "engine1_good", "broken": "engine2_broken",
               "heavyload": "engine3_heavyload"}
CLASSES_A2212 = ("Healthy", "Propeller", "Bearing")


class TesteTrancado(RuntimeError):
    pass


def teste_liberado() -> bool:
    return os.environ.get("MOTORSOUND_ABRIR_TESTE") == "1"


def _checar_caminho(p: Path) -> None:
    partes = {x.lower() for x in p.parts}
    if ("test" in partes or "test_cut" in partes) and not teste_liberado():
        raise TesteTrancado(f"leitura do conjunto de teste bloqueada: {p}")


@dataclass(frozen=True)
class Gravacao:
    base: str          # "idmt" | "a2212"
    rec_id: str        # identificador único da gravação de origem
    path: str
    classe: str
    condicao: str      # IDMT: pure, talking_1, ...; A2212: "pure"
    motor: str         # IDMT: engine1..3 (unidade física); A2212: M1..M4
    velocidade: int    # A2212: valor do nome do arquivo; IDMT: -1
    split: str         # IDMT: train | test; A2212: all


@dataclass(frozen=True)
class Segmento:
    base: str
    rec_id: str
    classe: str
    condicao: str
    motor: str
    velocidade: int
    split: str
    idx: int           # posição do segmento dentro da gravação
    t0: float          # s
    t1: float          # s

    def as_dict(self) -> dict:
        return asdict(self)


def gravacoes_idmt(split: str = "train") -> list[Gravacao]:
    if split not in ("train", "test"):
        raise ValueError(split)
    if split == "test" and not teste_liberado():
        raise TesteTrancado("listar o test/ do IDMT exige MOTORSOUND_ABRIR_TESTE=1")
    out = []
    for classe in CLASSES_IDMT:
        pasta = IDMT / split / PASTAS_IDMT[classe]
        for f in sorted(pasta.glob("*.wav")):
            out.append(Gravacao("idmt", f"idmt/{split}/{classe}/{f.stem}", str(f), classe,
                                f.stem, PASTAS_IDMT[classe].split("_")[0], -1, split))
    return out


def gravacoes_a2212() -> list[Gravacao]:
    out = []
    for classe in CLASSES_A2212:
        for pm in sorted((A2212 / classe).iterdir()):
            for f in sorted(pm.glob("*.wav")):
                out.append(Gravacao("a2212", f"a2212/{classe}/{pm.name}/{f.stem}", str(f),
                                    classe, "pure", pm.name, int(f.stem), "all"))
    return out


def gravacoes(base: str, split: str = "train") -> list[Gravacao]:
    return gravacoes_idmt(split) if base == "idmt" else gravacoes_a2212()


def duracao(g: Gravacao) -> float:
    _checar_caminho(Path(g.path))
    return sf.info(g.path).duration


def segmentos(grs: list[Gravacao], seg_s: float) -> list[Segmento]:
    """Recorta cada gravação em janelas de `seg_s` s sem sobreposição (sobra descartada,
    como nos `*_cut` do IDMT)."""
    out = []
    for g in grs:
        n = int(np.floor(duracao(g) / seg_s + 1e-9))
        for k in range(n):
            out.append(Segmento(g.base, g.rec_id, g.classe, g.condicao, g.motor,
                                g.velocidade, g.split, k, k * seg_s, (k + 1) * seg_s))
    return out


@lru_cache(maxsize=64)
def carregar(path: str, sr: int) -> np.ndarray:
    """Áudio mono float32 reamostrado para `sr` (média dos canais quando estéreo)."""
    _checar_caminho(Path(path))
    y, sr0 = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr0 != sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr0, target_sr=sr, res_type="soxr_hq")
    y = np.ascontiguousarray(y, dtype=np.float32)
    y.setflags(write=False)
    return y


_PATHS: dict[str, str] = {}


def registrar(grs: list[Gravacao]) -> None:
    for g in grs:
        _PATHS[g.rec_id] = g.path


def audio_segmento(s: Segmento, sr: int) -> np.ndarray:
    if s.rec_id not in _PATHS:
        registrar(gravacoes(s.base, s.split if s.base == "idmt" else "train"))
    y = carregar(_PATHS[s.rec_id], sr)
    i0, i1 = int(round(s.t0 * sr)), int(round(s.t1 * sr))
    return y[i0:i1]
