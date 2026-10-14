"""PASSO 1 com a base baixada: descobrir onde estão os ridges.

Para cada classe pega N arquivos 'pure' (ou quaisquer, no A2212), calcula o
espectro médio de janela longa e o espectrograma, e salva figuras.
Daqui saem f0_min/f0_max/n_harmonics do config.

Uso:
    python scripts/explore_spectrum.py idmt  /caminho/IDMT-ISA-ELECTRIC-ENGINE
    python scripts/explore_spectrum.py bldc  /caminho/BLDC
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motoracoustic.io import index_idmt, index_bldc, load_wav  # noqa: E402
from motoracoustic.config import IDMT_CFG, BLDC_CFG  # noqa: E402
from motoracoustic.preprocess import preprocess  # noqa: E402


def main(kind: str, root: str, n_per_class: int = 5, out: str = "figs"):
    items = index_idmt(root) if kind == "idmt" else index_bldc(root)
    cfg = IDMT_CFG if kind == "idmt" else BLDC_CFG
    if kind == "idmt":
        pure = [it for it in items if it.condition == "pure"]
        items = pure or items
    Path(out).mkdir(exist_ok=True)
    labels = sorted({it.label for it in items})

    fig, ax = plt.subplots(figsize=(10, 5))
    for lab in labels:
        sel = [it for it in items if it.label == lab][:n_per_class]
        specs = []
        for it in sel:
            x, sr = load_wav(it.path, cfg.sr)
            x = preprocess(x, sr, cfg)
            n = 2 ** 16 if len(x) >= 2 ** 16 else 2 ** int(np.log2(len(x)))
            X = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
            specs.append(20 * np.log10(X + 1e-9))
        f = np.fft.rfftfreq(n, 1 / sr)
        ax.plot(f, np.mean(specs, axis=0), label=lab, lw=0.8)
    ax.set_xlim(0, 1500)
    ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.legend()
    ax.set_title(f"{kind}: espectro médio (janela {n/sr:.1f} s, resolução {sr/n:.2f} Hz)")
    fig.tight_layout(); fig.savefig(f"{out}/{kind}_espectro_medio.png", dpi=150)

    for lab in labels:
        it = next(i for i in items if i.label == lab)
        x, sr = load_wav(it.path, cfg.sr)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.specgram(x[: sr * 10], NFFT=4096, Fs=sr, noverlap=3072, cmap="magma")
        ax.set_ylim(0, 1500); ax.set_title(f"{kind} {lab}: {it.path.name}")
        fig.tight_layout(); fig.savefig(f"{out}/{kind}_spec_{lab}.png", dpi=150)
    print(f"figuras em {out}/")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
