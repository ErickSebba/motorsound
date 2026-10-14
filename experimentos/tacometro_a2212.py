"""Tema 4 (tacômetro acústico): a rotação rastreada acompanha a velocidade comandada?

No A2212 o nome do arquivo é a velocidade comandada (1400–2000; unidade a confirmar na
ficha estacio2023). Não é um tacômetro, mas é uma referência ordinal: se o rastreador
mede rotação, a f_rot estimada deve crescer monotonicamente com o valor do nome dentro de
cada motor. Mede Spearman por (classe, motor) e a razão f_rot/(valor/60) para três
faixas de busca, nas variantes com e sem Viterbi.

    python experimentos/tacometro_a2212.py
"""
import sys
import time
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from motorsound import io, runner  # noqa: E402
from motorsound.rotacao import rastrear  # noqa: E402

FAIXAS = [(15.0, 60.0), (60.0, 240.0), (100.0, 400.0)]


def main():
    t0 = time.perf_counter()
    linhas = []
    for g in io.gravacoes("a2212"):
        y = io.carregar(g.path, 16000)
        y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * 0.1
        for fmin, fmax in FAIXAS:
            for vit in (True, False):
                f, _, _ = rastrear(y, 16000, fmin, fmax, usar_viterbi=vit)
                linhas.append({"rec_id": g.rec_id, "classe": g.classe, "motor": g.motor,
                               "velocidade": g.velocidade, "fmin": fmin, "fmax": fmax,
                               "viterbi": vit, "f_rot_mediana": float(np.nanmedian(f)),
                               "f_rot_iqr": float(np.subtract(*np.nanpercentile(f, [75, 25]))),
                               "frac_sil": float(np.isnan(f).mean())})
    df = pd.DataFrame(linhas)
    resumo = []
    for (fmin, vit, c, m), gg in df.groupby(["fmin", "viterbi", "classe", "motor"]):
        if len(gg) >= 4:
            rho, p = spearmanr(gg.velocidade, gg.f_rot_mediana)
            resumo.append({"fmin": fmin, "viterbi": vit, "classe": c, "motor": m,
                           "n": len(gg), "spearman": rho, "p": p,
                           "razao_mediana": float(np.median(gg.f_rot_mediana / (gg.velocidade / 60)))})
    rs = pd.DataFrame(resumo)
    print(rs.round(3).to_string())
    caminho = runner.escrever(df, "tacometro_a2212", {
        "argv": sys.argv, "faixas": FAIXAS, "resumo_spearman": rs.to_dict("records"),
        "segundos": round(time.perf_counter() - t0, 1),
        "nota": "referência = velocidade comandada no nome do arquivo, não tacômetro"})
    print(f"escrito {caminho.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
