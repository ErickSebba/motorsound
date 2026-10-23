# Procedência: escrito do zero em 22/10/2026 para a auditoria da base `kaist`
# (Jung et al., Data in Brief 48:109049, 2023; Mendeley ztmf3m7h5x v6). Não copia código
# de ../motorsound/.
"""Leitura do canal acústico da base KAIST (bancada de máquina rotativa, 0 Nm).

Formato real (medido): cada `.mat` (MAT v5, exportado pelo Siemens Simcenter Test.Lab)
contém uma única struct `Signal` com
    Signal.x_values.{start_value, increment, number_of_values}   -> eixo do tempo implícito
    Signal.y_values.values (N,) float64                           -> pressão em Pa
    Signal.function_record.TL_export_properties_annotation        -> run_name, absolute_time
Não há coluna 'Time Stamp' explícita (a documentação fala em duas colunas). O arquivo já é
mono: só o microfone (canal 3 "Point1", grupo "Acoustic").

Não há tacômetro na parte 1 da base (carga variável); `carregar_tacometro` devolve None.
A corrente do motor (TDMS) foi gravada em outro dia e por outro sistema (NI FlexLogger),
portanto NÃO é síncrona com o áudio; `carregar_corrente` existe só para estimar a
frequência de alimentação do inversor naquela outra sessão.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio


def _signal(caminho):
    m = sio.loadmat(str(Path(caminho)), squeeze_me=True, struct_as_record=False)
    return m["Signal"]


def _txt(a) -> str:
    if isinstance(a, np.ndarray):
        return "".join(str(c) for c in a.ravel())
    return str(a)


def carregar(caminho_absoluto) -> tuple[np.ndarray, int]:
    """Devolve (y float32 mono em Pa, sr int) do canal acústico."""
    s = _signal(caminho_absoluto)
    y = np.asarray(s.y_values.values, dtype=np.float32).ravel()
    sr = int(round(1.0 / float(s.x_values.increment)))
    n_decl = int(s.x_values.number_of_values)
    if n_decl != y.size:
        raise ValueError(f"{caminho_absoluto}: number_of_values={n_decl} != len(y)={y.size}")
    return y, sr


def metadados(caminho_absoluto) -> dict:
    """Metadados do Test.Lab gravados dentro do .mat (sessão/gravação de origem)."""
    s = _signal(caminho_absoluto)
    fr = s.function_record
    a = fr.TL_export_properties_annotation
    return {
        "run_name": _txt(a.run_name),
        "absolute_time": _txt(a.absolute_time),
        "project_name": _txt(a.project_name),
        "section_name": _txt(a.section_name),
        "channel_group": _txt(a.channel_group),
        "orig_location": _txt(a.orig_location),
        "canal_id": int(fr.primary_channel.id),
        "canal_label": _txt(fr.primary_channel.label),
        "creation_time": _txt(fr.creation_time),
        "unidade_y": _txt(s.y_values.quantity.label),
        "t0_s": float(s.x_values.start_value),
        "dt_s": float(s.x_values.increment),
        "n": int(s.x_values.number_of_values),
    }


def carregar_tacometro(caminho):
    """Parte 1 da base (onde está o áudio) não tem tacômetro: devolve None."""
    return None


def carregar_corrente(caminho_tdms):
    """(dict nome_canal -> np.ndarray float32, sr) das fases de corrente de um .tdms.

    Canais com 1 amostra (fase não gravada, caso dos BPFO) são omitidos.
    Requer `nptdms`.
    """
    from nptdms import TdmsFile

    t = TdmsFile.read(str(Path(caminho_tdms)))
    fases, sr = {}, None
    for c in t["Log"].channels():
        if c.properties.get("DAC~Channel~Type") != "Current":
            continue
        x = c[:]
        if x.size < 2:
            continue
        fases[c.name] = np.asarray(x, dtype=np.float32)
        sr = 1.0 / float(c.properties["wf_increment"])
    return fases, sr
