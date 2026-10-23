# Carregador do ToyCar do DCASE 2022 Challenge Task 2 (subconjunto do ToyADMOS2, Harada et
# al., 2021; distribuição DCASE: Dohi et al., 2022). Escrito em 23/10/2026 para a auditoria de
# bases (docs/bases/toycar.md); não é cópia de código do Artigo3.
#
# Os wav do DCASE 2022 já são mono (só o "primeiro canal" de cada gravação, segundo a página da
# task), 16 kHz, PCM 16 bit, 10 s. Se algum arquivo vier com mais de um canal, usa o canal 0,
# mesma convenção do load_mimii.
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

CANAL_ADOTADO = 0


def carregar(caminho_absoluto) -> tuple[np.ndarray, int]:
    """Lê um wav do ToyCar (DCASE 2022). Devolve (y float32 mono em [-1, 1), sr em Hz)."""
    caminho = Path(caminho_absoluto)
    y, sr = sf.read(str(caminho), dtype="float32", always_2d=True)
    return np.ascontiguousarray(y[:, CANAL_ADOTADO]), int(sr)
