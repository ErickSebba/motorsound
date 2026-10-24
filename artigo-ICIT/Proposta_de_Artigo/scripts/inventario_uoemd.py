"""Inventário da base UOEMD-VAFCVS a partir dos arquivos baixados.

Uso:
    .venv\\Scripts\\python.exe -I scripts\\inventario_uoemd.py "<raiz>\\data\\uoemd" [saida.csv]

Lê:
  <dados>/3_MatLab_Data_Files/{1_Unloaded_Condition,2_Loaded_Condition}/*.mat   (128 arquivos)
  <dados>/1_Time/Time_10_seconds_420000_samples.csv                            (vetor de tempo)
  <dados>/_metadados/sehri2024_dib_PMC11636780.xml   (texto do artigo, Europe PMC; dele saem
                                                       as frequências do inversor por arquivo)
  <dados>/_metadados/mendeley_files_*_v2.json         (sha256 e tamanhos declarados pela API)
  <dados>/_conferencia_csv/...                         (2 CSV baixados só para conferir com o .mat)

Escreve out/indices/uoemd.csv (ou o caminho dado) e imprime em stdout as tabelas em
Markdown usadas em docs/bases/uoemd.md. Nenhum número da ficha é digitado à mão.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from lib.load_uoemd import SR, carregar_todos_canais  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

CLASSES = {
    "H_H": "saudavel",
    "R_U": "desequilibrio_rotor",
    "R_M": "desalinhamento_rotor",
    "S_W": "falha_enrolamento_estator",
    "V_U": "desequilibrio_tensao_e_falta_fase",
    "B_R": "rotor_empenado",
    "K_A": "barras_rotor_quebradas",
    "F_B": "rolamento_defeituoso",
}
VELOC = {1: ("const", 15, 15), 2: ("const", 30, 30), 3: ("const", 45, 45), 4: ("const", 60, 60),
         5: ("rampa_sobe", 15, 45), 6: ("rampa_sobe", 30, 60),
         7: ("rampa_desce", 45, 15), 8: ("rampa_desce", 60, 30)}
CARGA = {0: "sem_carga", 1: "com_carga"}
PASTA_CARGA = {"1_Unloaded_Condition": 0, "2_Loaded_Condition": 1}
RE_NOME = re.compile(r"^([A-Z])_([A-Z])_(\d)_(\d)$")


def freq_do_artigo(xml: Path) -> dict:
    """Frequências do inversor por arquivo, como listadas na seção 3 do artigo."""
    txt = re.sub(r"\s+", " ", " ".join(ET.parse(xml).getroot().itertext()))
    pad = (r"([A-Z])-([A-Z])-(\d)-(\d): operating frequency (of|increased from|decreased from)"
           r"\s*([\d.]+) Hz(?: to ([\d.]+) Hz)?")
    out = {}
    for a, b, v, c, tipo, f1, f2 in re.findall(pad, txt):
        out[f"{a}_{b}_{v}_{c}"] = (float(f1), float(f2) if f2 else None)
    return out


def pico_1x(x: np.ndarray, sr: int, f_ref: float, nfft: int | None = None) -> float:
    """Pico espectral do acelerômetro na faixa [0,85; 1,02]·f_ref (estimativa de 1x)."""
    x = x - x.mean()
    n = nfft or len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n))
    f = np.fft.rfftfreq(n, 1 / sr)
    m = (f >= 0.85 * f_ref) & (f <= 1.02 * f_ref)
    return float(f[m][np.argmax(X[m])])


def md(df: pd.DataFrame, idx: bool = True) -> str:
    d = df.reset_index() if idx else df
    cab = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    linhas = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([cab, sep, *linhas])


def main() -> None:
    dados = Path(sys.argv[1]).resolve()
    saida = Path(sys.argv[2]) if len(sys.argv) > 2 else AQUI.parent / "out" / "indices" / "uoemd.csv"
    meta = dados / "_metadados"
    freqs = freq_do_artigo(meta / "sehri2024_dib_PMC11636780.xml")
    print(f"frequências lidas do artigo: {len(freqs)}")

    # sha256 declarados pela API do Mendeley
    sha_api = {}
    for j in meta.glob("mendeley_files_*_v2.json"):
        for x in json.loads(j.read_text(encoding="utf-8")):
            sha_api[(j.stem, x["filename"])] = x["content_details"]["sha256_hash"]

    # vetor de tempo
    t = pd.read_csv(dados / "1_Time" / "Time_10_seconds_420000_samples.csv",
                    encoding="utf-8-sig").iloc[:, 0]
    tv = pd.to_numeric(t, errors="coerce").dropna().to_numpy()
    print(f"1_Time: {len(t)} linhas, {len(tv)} numéricas, t0={tv[0]}, tfim={tv[-1]}, "
          f"1/mediana(dt)={1 / np.median(np.diff(tv)):.1f} Hz, (N-1)/tfim={(len(tv) - 1) / tv[-1]:.1f} Hz")

    linhas, extra = [], []
    hashes_mic = defaultdict(list)
    arquivos = sorted((dados / "3_MatLab_Data_Files").glob("*/*.mat"))
    for p in arquivos:
        stem = p.stem
        a, b, v, c = RE_NOME.match(stem).groups()
        v, c = int(v), int(c)
        cod = f"{a}_{b}"
        assert PASTA_CARGA[p.parent.name] == c, f"pasta e nome divergem: {p}"
        tipo, fa, fb = VELOC[v]
        m, sr = carregar_todos_canais(p)
        mic = m[:, 1]
        hashes_mic[hashlib.sha1(mic.tobytes()).hexdigest()].append(stem)
        dc = float(mic.mean())
        ac = mic - dc
        u = np.unique(mic)
        passo = float(np.median(np.diff(u)))
        n_ext = int((mic == mic.max()).sum() + (mic == mic.min()).sum())
        f_art = freqs.get(stem)
        if tipo == "const":
            f_alim = f_art[0]
            f1x = pico_1x(m[:, 2], sr, f_alim)
            f1x_ini = f1x_fim = None
        else:
            f_alim = None
            f1x = None
            f1x_ini = pico_1x(m[:sr, 2], sr, f_art[0], nfft=8 * sr)
            f1x_fim = pico_1x(m[-sr:, 2], sr, f_art[1], nfft=8 * sr)
        obs = []
        if tipo != "const":
            obs.append(f"rampa no inversor {f_art[0]}->{f_art[1]} Hz (artigo)")
        if cod == "V_U":
            obs.append("classe junta desequilibrio de tensao e falta de fase; arquivo nao diz qual")
        obs.append("unidade inferida do artigo (1 motor por classe)")
        cond = (f"const_{fa}Hz" if tipo == "const" else f"{tipo}_{fa}-{fb}Hz") + f"_{CARGA[c]}"
        linhas.append(dict(
            base="uoemd", arquivo=p.relative_to(dados).as_posix(), classe=CLASSES[cod],
            unidade=f"motor_{cod}", condicao=cond,
            velocidade=(f"{fa}" if tipo == "const" else f"{fa}-{fb}"), carga=CARGA[c],
            gravacao_id=f"uoemd/{stem}", sr=sr, canais=1, duracao_s=round(len(mic) / sr, 6),
            f_alim_hz=f_alim, rot_esperada_hz=None, obs="; ".join(obs)))
        extra.append(dict(stem=stem, cod=cod, v=v, c=c, tipo=tipo, n=len(mic), ncol=m.shape[1],
                          dc=dc, rms=float(ac.std()), pico=float(np.abs(ac).max()),
                          vmin=float(mic.min()), vmax=float(mic.max()), niveis=len(u), passo=passo,
                          n_ext=n_ext, temp=float(m[:, 4].mean()), f_art0=f_art[0], f_art1=f_art[1],
                          f1x=f1x, f1x_ini=f1x_ini, f1x_fim=f1x_fim,
                          rms_acc2=float(m[:, 2].std())))

    idx = pd.DataFrame(linhas)
    saida.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(saida, index=False, encoding="utf-8")
    ex = pd.DataFrame(extra)
    print(f"\níndice: {saida}  linhas={len(idx)}  (esperado 128 = 8 classes x 8 velocidades x 2 cargas)")

    print("\n## sha256 dos arquivos baixados vs API")
    ok = tot = 0
    for p in arquivos + [dados / "1_Time" / "Time_10_seconds_420000_samples.csv"]:
        chave = {"1_Unloaded_Condition": "mendeley_files_mat_unloaded_v2",
                 "2_Loaded_Condition": "mendeley_files_mat_loaded_v2",
                 "1_Time": "mendeley_files_time_v2"}[p.parent.name]
        tot += 1
        ok += hashlib.sha256(p.read_bytes()).hexdigest() == sha_api[(chave, p.name)]
    print(f"{ok}/{tot} conferem")
    tam = sum(p.stat().st_size for p in arquivos)
    print(f"tamanho dos 128 .mat: {tam / 1e6:.1f} MB")

    print("\n## formato medido")
    print(f"amostras por arquivo: {sorted(Counter(ex.n).items())}; colunas: {sorted(Counter(ex.ncol).items())}")
    print(f"duração por arquivo (s): {sorted(Counter(idx.duracao_s).items())}")
    print(f"passo de quantização do microfone (mediana entre níveis vizinhos): "
          f"min {ex.passo.min():.6f} V, max {ex.passo.max():.6f} V")
    print(f"níveis distintos do microfone por arquivo: min {ex.niveis.min()} ({ex.stem[ex.niveis.idxmin()]}), "
          f"mediana {int(ex.niveis.median())}, max {ex.niveis.max()} ({ex.stem[ex.niveis.idxmax()]})")
    print(f"faixa absoluta do microfone em toda a base: {ex.vmin.min():.4f} a {ex.vmax.max():.4f} V")
    print(f"offset DC do microfone: {ex.dc.min():.5f} a {ex.dc.max():.5f} V")
    print(f"amostras no valor extremo do arquivo (máx+mín), máx por arquivo: {ex.n_ext.max()}")

    print("\n## duração total por classe (s)")
    print(md(idx.groupby("classe").duracao_s.agg(["count", "sum"])))

    print("\n## matriz classe x unidade (n arquivos)")
    print(md(pd.crosstab(idx.classe, idx.unidade)))

    print("\n## matriz classe x condição (n arquivos)")
    print(md(pd.crosstab(idx.classe, idx.condicao)))

    print("\n## duplicatas do canal acústico (hash idêntico)")
    dup = [v for v in hashes_mic.values() if len(v) > 1]
    print(dup if dup else "nenhuma")

    print("\n## nível acústico: RMS AC (mV) por classe x velocidade, média das duas cargas")
    ex["classe"] = ex.cod.map(CLASSES)
    ex["vel"] = ex.v.map(lambda k: VELOC[k][0] + f"_{VELOC[k][1]}-{VELOC[k][2]}" if VELOC[k][0] != "const" else f"{VELOC[k][1]}")
    print(md((ex.pivot_table(index="classe", columns="vel", values="rms", aggfunc="mean") * 1e3).round(1)))
    print("\n## pico |AC| (mV) por classe x velocidade, máximo das duas cargas")
    print(md((ex.pivot_table(index="classe", columns="vel", values="pico", aggfunc="max") * 1e3).round(0)))
    print("\n## RMS AC (mV) por classe x carga (todas as velocidades, mediana)")
    print(md((ex.pivot_table(index="classe", columns="c", values="rms", aggfunc="median") * 1e3).round(1)))
    print("\n## razão max/min de RMS entre classes, por velocidade x carga")
    r = ex.groupby(["vel", "c"]).rms.agg(lambda s: s.max() / s.min()).round(1)
    print(md(r.to_frame("max/min")))
    print("\n## menor RMS entre as 7 falhas / RMS do saudável, por velocidade x carga")
    rs = ex.groupby(["vel", "c"]).apply(
        lambda g: g[g.cod != "H_H"].rms.min() / g[g.cod == "H_H"].rms.iloc[0], include_groups=False)
    print(md(rs.round(2).to_frame("min_falha/saudavel")))
    print(f"faixa: {rs.min():.2f} a {rs.max():.2f}; condições em que o saudável é o mais "
          f"silencioso: {int(sum(ex.loc[g.idxmin(), 'cod'] == 'H_H' for _, g in ex.groupby(['vel', 'c']).rms))}/16")

    print("\n## temperatura média (°C) por classe x carga")
    print(md(ex.pivot_table(index="classe", columns="c", values="temp", aggfunc="mean").round(2)))

    print("\n## rotação: frequência do inversor (artigo) x pico 1x do acelerômetro 2 (velocidade constante)")
    k = ex[ex.tipo == "const"].copy()
    k["razao"] = (k.f1x / k.f_art0).round(4)
    print(md(k[["stem", "f_art0", "f1x", "razao"]], idx=False))
    print(f"razão 1x/f_inversor: mediana {k.razao.median():.4f}, min {k.razao.min():.4f}, max {k.razao.max():.4f}")
    print("\n## rampas: 1x no 1º e no último segundo (acelerômetro 2) x extremos do artigo")
    rr = ex[ex.tipo != "const"][["stem", "f_art0", "f1x_ini", "f_art1", "f1x_fim"]]
    print(md(rr, idx=False))


if __name__ == "__main__":
    main()
