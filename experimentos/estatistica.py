"""Comparações pareadas entre pipelines (bootstrap por grupo + Wilcoxon + Holm).

Pares fixados em 14/10/2026 antes de calcular os p-valores; cada um isola uma peça.
Cenários: IDMT validação com rosa 10 dB, branco 10 dB e rosa 5 dB (treino limpo) e
A2212 por arquivo limpo. Família de Holm = todos os (par × cenário).

    python experimentos/estatistica.py --csv <robustez_idmt...> <robustez_a2212...>
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from motorsound import runner, stats  # noqa: E402

PARES = [
    ("p12_decifrae_sem_viterbi", "p11_decifrae_harmsum", "Viterbi de rotação (baixo V3)"),
    ("p11a_grade10", "p11_decifrae_harmsum", "grade de 2 cents no rastreador"),
    ("p11_decifrae_harmsum", "p10_decifrae_pyin_viterbi", "termo pYIN na emissão"),
    ("p00_logmel_cnn", "p11_decifrae_harmsum", "Decifrae embarcável × CNN de referência (Strantzalis)"),
    ("p02_grollmisch", "p11_decifrae_harmsum", "Decifrae embarcável × baseline dos autores do IDMT"),
    ("p11_decifrae_harmsum", "p20b_so_frot", "só a rotação × p11 completa"),
    ("b01_forma_espectral", "p21_envelope", "modulação do envelope × forma espectral"),
]
CENARIOS = [
    ("idmt", {"snr_db": 10.0, "tipo": "rosa"}), ("idmt", {"snr_db": 10.0, "tipo": "branco"}),
    ("idmt", {"snr_db": 5.0, "tipo": "rosa"}), ("a2212", {"tipo": "limpo"}),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    a = ap.parse_args(argv)
    df = pd.concat([pd.read_csv(c) for c in a.csv], ignore_index=True)
    linhas = []
    for base, cond in CENARIOS:
        alvo = json.dumps(cond, sort_keys=True)
        sub = df[(df.base == base) & (df.condicao_sintetica == alvo)]
        for x, y, isola in PARES:
            r = stats.comparar(sub, x, y, "acerto", "grupo", isola)
            r.update(base=base, cenario=alvo)
            linhas.append(r)
    out = pd.DataFrame(linhas)
    out["p_bootstrap_holm"] = stats.holm(out.p_bootstrap)
    out["wilcoxon_p_holm"] = stats.holm(out.wilcoxon_p)
    out["classificacao"] = [stats.classificar(c, p) for c, p in
                            zip(out.ic_cruza_zero, out.p_bootstrap_holm)]
    cols = ["base", "cenario", "a", "b", "isola", "n_grupos", "media_a", "media_b",
            "diferenca", "ic95_min", "ic95_max", "p_bootstrap_holm", "wilcoxon_p_holm",
            "classificacao", "b_melhor", "b_pior"]
    print(out[cols].round(3).to_string())
    caminho = runner.escrever(out, "estatistica", {
        "argv": sys.argv, "entradas": [Path(c).name for c in a.csv], "pares": PARES,
        "cenarios": CENARIOS, "n_bootstrap": stats.N_BOOTSTRAP, "seed": stats.SEED,
        "unidade": "grupo (gravação × bloco no IDMT; gravação no A2212)",
        "nota": "diferença = b − a; família de Holm = todos os pares × cenários"})
    print(f"escrito {caminho.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
