"""Leitor da base MaFaulDa (Machinery Fault Database, SMT/COPPE/UFRJ).

Escrito do zero em 23/10/2026 (não é cópia de Artigo3). Ver docs/bases/mafaulda.md.

Cada CSV (sem cabeçalho, separador vírgula) tem 250 000 linhas x 8 colunas; declarados 50 kHz
(5 s), medidos 51,2 kHz (4,8828 s) — ver SR abaixo:
    col 0  tacômetro (Monarch MT-190), volts; trem de pulsos, 1 pulso por volta (verificado
           no inventário: frequência de pulsos = rotação da bancada)
    col 1-3  acelerômetro do mancal underhang (axial, radial, tangencial), segundo a página
    col 4-6  acelerômetro do mancal overhang (axial, radial, tangencial), segundo a página
    col 7  MICROFONE (Shure SM81), volts
A identificação das colunas 0 e 7 foi conferida pelo conteúdo (ver inventario_mafaulda.py).

Cache: na primeira leitura de um CSV, o microfone e o tacômetro são gravados como float32
em `data/mafaulda/_cache/<caminho relativo>.npy` (matriz N x 2: [microfone, tacômetro]).
As leituras seguintes usam o cache. O sinal devolvido é o canal CRU, com o offset DC
original; quem usa decide se remove a média.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Taxa de amostragem. A página e os artigos declaram 50 kHz, mas os arquivos indicam 51,2 kHz
# (taxa nativa do NI 9234, que não oferece 50 kHz): o zumbido da rede de 60 Hz e seus
# harmônicos só caem em 60·k Hz quando se usa 51 200 Hz, e o valor do nome do arquivo
# (múltiplo de 51 200/250 000 = 0,2048 Hz) só bate com a frequência do tacômetro com 51 200 Hz.
# Ver docs/bases/mafaulda.md §3 e §7. Os arquivos não guardam a taxa.
SR = 51_200
SR_DECLARADO = 50_000
COL_TACOMETRO = 0
COL_MICROFONE = 7
N_COLUNAS = 8


def _raiz_dados(caminho: Path) -> Path | None:
    """Pasta `data/mafaulda` que contém o arquivo (para achar o cache), ou None."""
    for p in caminho.parents:
        if p.name == "mafaulda" and p.parent.name == "data":
            return p
    return None


def ler_csv(caminho) -> np.ndarray:
    """Matriz N x 8 (float64) lida do CSV, sem cache."""
    import pandas as pd
    m = pd.read_csv(Path(caminho), header=None, engine="c", dtype=np.float64).to_numpy()
    if m.ndim != 2 or m.shape[1] != N_COLUNAS:
        raise ValueError(f"esperava N x {N_COLUNAS} em {caminho}, achei {m.shape}")
    return m


def _cache_de(caminho: Path) -> Path | None:
    raiz = _raiz_dados(caminho)
    if raiz is None:
        return None
    rel = caminho.resolve().relative_to(raiz.resolve())
    return raiz / "_cache" / rel.with_suffix(".npy")


def _mic_taco(caminho) -> np.ndarray:
    """Matriz N x 2 float32 [microfone, tacômetro], do cache quando existir."""
    caminho = Path(caminho)
    if caminho.suffix.lower() == ".npy":
        return np.load(caminho)
    c = _cache_de(caminho)
    if c is not None and c.exists():
        return np.load(c)
    m = ler_csv(caminho)
    mt = np.ascontiguousarray(m[:, [COL_MICROFONE, COL_TACOMETRO]], dtype=np.float32)
    if c is not None:
        c.parent.mkdir(parents=True, exist_ok=True)
        tmp = c.with_suffix(".tmp.npy")
        np.save(tmp, mt)
        tmp.replace(c)
    return mt


def carregar(caminho_absoluto) -> tuple[np.ndarray, int]:
    """Microfone (coluna 8 da documentação, índice 7) como float32 mono, e sr."""
    return np.ascontiguousarray(_mic_taco(caminho_absoluto)[:, 0]), SR


def carregar_tacometro(caminho) -> tuple[np.ndarray, int]:
    """Sinal do tacômetro (coluna 1, índice 0), float32 em volts, e sr."""
    return np.ascontiguousarray(_mic_taco(caminho)[:, 1]), SR


def carregar_todos_canais(caminho_absoluto) -> tuple[np.ndarray, int]:
    """Matriz N x 8 (float64) com todos os canais, sem cache, para conferências."""
    return ler_csv(caminho_absoluto), SR


def rotacao_tacometro(taco: np.ndarray, sr: int = SR) -> dict:
    """Frequência de pulsos do tacômetro (Hz) por detecção de bordas de subida.

    Limiar = ponto médio entre os percentis 5 e 95, com histerese de 10 % da faixa.
    Período refratário: bordas a menos de 0,5/f0 da anterior são descartadas, com f0 = maior
    pico do espectro do tacômetro entre 5 e 70 Hz (remove bordas espúrias de ruído).
    Devolve mediana e dispersão da frequência instantânea (1/intervalo entre bordas).
    """
    x = np.asarray(taco, dtype=np.float64)
    nf = 1 << 20
    X = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), n=nf))
    fq = np.fft.rfftfreq(nf, 1 / sr)
    mk = (fq >= 5) & (fq <= 70)
    f0 = float(fq[mk][np.argmax(X[mk])])
    lo, hi = np.percentile(x, [5, 95])
    faixa = hi - lo
    if faixa <= 0:
        return dict(f_med=np.nan, f_media=np.nan, f_p5=np.nan, f_p95=np.nan, n_bordas=0,
                    faixa=faixa)
    th_hi = lo + 0.6 * faixa
    th_lo = lo + 0.4 * faixa
    # máquina de estados com histerese, vetorizada: estado alto/baixo após cada cruzamento
    estado = np.full(len(x), np.nan)
    estado[x >= th_hi] = 1.0
    estado[x <= th_lo] = 0.0
    idx = np.where(~np.isnan(estado))[0]
    est = estado[idx]
    cand = idx[1:][(est[1:] == 1.0) & (est[:-1] == 0.0)]
    refr = 0.5 * sr / f0
    sub = []
    for b in cand:
        if not sub or b - sub[-1] >= refr:
            sub.append(b)
    n_desc = int(len(cand) - len(sub))
    sub = np.asarray(sub)
    if len(sub) < 3:
        return dict(f_med=np.nan, f_media=np.nan, f_p5=np.nan, f_p95=np.nan,
                    n_bordas=int(len(sub)), faixa=faixa)
    dt = np.diff(sub) / sr
    f = 1.0 / dt
    return dict(f_med=float(np.median(f)), f_media=float((len(sub) - 1) / ((sub[-1] - sub[0]) / sr)),
                f_p5=float(np.percentile(f, 5)), f_p95=float(np.percentile(f, 95)),
                n_bordas=int(len(sub)), n_descartadas=n_desc, f0_fft=f0, faixa=float(faixa))
