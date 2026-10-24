# Carregador das bases MIMII (Purohit et al., 2019) e MIMII DUE (Tanabe et al., 2021).
# Escrito em 22/10/2026 para a auditoria de bases (docs/bases/mimii.md); não é cópia de
# código do Artigo3.
#
# Canal adotado: canal 0 (microfone "No. 1" do arranjo TAMAGO-03). É o único canal usado no
# baseline do artigo do MIMII ("We used only the first channel of microphones"), no DCASE
# 2020 Task 2 e é o único canal distribuído no MIMII DUE ("only the single-channel audio
# recorded with the first microphone of the array"). Assim os dois conjuntos ficam
# comparáveis e comparáveis com a literatura.
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

CANAL_ADOTADO = 0


def carregar(caminho_absoluto) -> tuple[np.ndarray, int]:
    """Lê um wav do MIMII (8 canais) ou do MIMII DUE (mono).

    Devolve (y, sr): y float32 mono, em escala [-1, 1), do canal 0; sr em Hz.
    """
    caminho = Path(caminho_absoluto)
    y, sr = sf.read(str(caminho), dtype="float32", always_2d=True)
    return np.ascontiguousarray(y[:, CANAL_ADOTADO]), int(sr)
