"""Gera embarcados/src/c/p11_constantes.h com as constantes de projeto que o C não recalcula.

Hoje: os coeficientes SOS de `scipy.signal.butter(4, 200, fs=16000, output="sos")`, o mesmo
filtro de `pipelines/_decifrae.py::_envelope_mod` (o sr do núcleo é sempre 16 kHz).
Os valores saem em `repr` de float64 (ida e volta exata) e também em float32 para o build
float. Uso:

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_constantes.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import scipy.signal as ss

RAIZ = Path(__file__).resolve().parents[3]          # Artigo3(MotorSound)
SAIDA = RAIZ / "embarcados" / "src" / "c" / "p11_constantes.h"

SR = 16000
ORDEM = 4
FC = 200.0


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "desconhecido"


def main() -> int:
    sos = ss.butter(ORDEM, FC, fs=SR, output="sos")
    linhas64 = []
    linhas32 = []
    for s in sos:
        linhas64.append("    {" + ", ".join(repr(float(v)) for v in s) + "},")
        linhas32.append("    {" + ", ".join(repr(float(np.float32(v))) + "f" for v in s) + "},")
    h = hashlib.sha256(sos.tobytes()).hexdigest()[:16]
    txt = f"""/* p11_constantes.h -- GERADO AUTOMATICAMENTE, não editar à mão.
 * Origem: embarcados/src/python/gerar_constantes.py (commit {_commit()}),
 *         scipy {scipy.__version__}, numpy {np.__version__},
 *         gerado em {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}.
 * Conteúdo: scipy.signal.butter({ORDEM}, {FC:g}, fs={SR}, output="sos"), o passa-baixa de
 *           pipelines/_decifrae.py::_envelope_mod. Cada linha = [b0 b1 b2 a0 a1 a2] (a0 = 1).
 * sha256(sos float64)[:16] = {h}
 */
#ifndef P11_CONSTANTES_H
#define P11_CONSTANTES_H

#define P11_SOS_N_SECOES {sos.shape[0]}

/* repr float64 (ida e volta exata) */
static const double P11_SOS_F64[P11_SOS_N_SECOES][6] = {{
{chr(10).join(linhas64)}
}};

/* os mesmos valores arredondados para float32 (build float) */
static const float P11_SOS_F32[P11_SOS_N_SECOES][6] = {{
{chr(10).join(linhas32)}
}};

#endif /* P11_CONSTANTES_H */
"""
    SAIDA.write_text(txt, encoding="utf-8", newline="\n")
    print(f"escrito {SAIDA} ({sos.shape[0]} seções, sha {h})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
