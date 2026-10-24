"""Inventário da base MaFaulDa a partir dos arquivos extraídos.

Uso:
    .venv\\Scripts\\python.exe -I scripts\\inventario_mafaulda.py "<raiz>\\data\\mafaulda" [saida.csv]

Lê:
  <dados>/full/**/*.csv                 (1951 CSV, 250 000 x 8, sem cabeçalho)
  <dados>/_metadados/full_zip_listing.txt  (lista do full.zip: tamanhos e datas de cada membro)
Escreve:
  out/indices/mafaulda.csv                       (uma linha por arquivo; colunas padrão)
  <dados>/_metadados/inventario_detalhes.csv     (medidas por arquivo: microfone, tacômetro, 1x)
  <dados>/_cache/**.npy                          (microfone e tacômetro, float32, via loader)
e imprime em stdout as tabelas em Markdown usadas em docs/bases/mafaulda.md.
Nenhum número da ficha é digitado à mão.
"""
from __future__ import annotations

import hashlib
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from lib.load_mafaulda import SR, _cache_de, rotacao_tacometro  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

TIPO_ROL = {"cage_fault": "gaiola", "ball_fault": "esfera", "outer_race": "pista_externa"}
SOS_LP = ss.butter(8, 20_000, fs=SR, output="sos")


def rotulos(rel: Path) -> dict:
    """Classe, unidade, condição a partir do caminho relativo a full/."""
    p = rel.parts
    if p[0] == "normal":
        return dict(classe="normal", unidade="bancada_mafaulda", condicao="sem_falha", sev="")
    if p[0] == "imbalance":
        return dict(classe="desbalanceamento", unidade=f"bancada_mafaulda+massa_{p[1]}",
                    condicao=f"massa_{p[1]}", sev=p[1])
    if p[0] == "horizontal-misalignment":
        return dict(classe="desalinhamento_horizontal", unidade="bancada_mafaulda",
                    condicao=f"desloc_{p[1]}", sev=p[1])
    if p[0] == "vertical-misalignment":
        return dict(classe="desalinhamento_vertical", unidade="bancada_mafaulda",
                    condicao=f"desloc_{p[1]}", sev=p[1])
    if p[0] in ("underhang", "overhang"):
        t = TIPO_ROL[p[1]]
        return dict(classe=f"rolamento_{p[0]}_{t}", unidade=f"rolamento_defeituoso_{t}",
                    condicao=f"massa_{p[2]}", sev=p[2])
    raise ValueError(rel)


def espectro_pico(x: np.ndarray, f0: float, lo: float, hi: float, nfft: int) -> float:
    x = x - x.mean()
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft))
    f = np.fft.rfftfreq(nfft, 1 / SR)
    m = (f >= lo * f0) & (f <= hi * f0)
    return float(f[m][np.argmax(X[m])])


def medir(args) -> dict:
    caminho, rel = args
    caminho = Path(caminho)
    with open(caminho, "r") as fh:
        ncol = len(fh.readline().split(","))
    m = pd.read_csv(caminho, header=None, engine="c", usecols=[0, 2, 4, 7],
                    dtype=np.float64).to_numpy()
    taco, acc_u_rad, acc_o_ax, mic = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    # cache (microfone, tacômetro) float32, o mesmo formato que o loader grava
    c = _cache_de(caminho)
    if c is not None and not c.exists():
        c.parent.mkdir(parents=True, exist_ok=True)
        tmp = c.with_suffix(".tmp.npy")
        np.save(tmp, np.stack([mic, taco], axis=1).astype(np.float32))
        tmp.replace(c)
    r = rotacao_tacometro(taco)
    # conferência de pulsos por volta: pico do acelerômetro em [0,3; 1,3]·f_taco
    f1x = espectro_pico(acc_u_rad, r["f_med"], 0.3, 1.3, 1 << 20) if np.isfinite(r["f_med"]) else np.nan
    # espectro do tacômetro: frequência fundamental (maior pico abaixo de 70 Hz e acima de 5 Hz)
    ft = espectro_pico(taco, 1.0, 5.0, 70.0, 1 << 20)
    # duty cycle e níveis do tacômetro
    lo, hi = np.percentile(taco, [5, 95])
    duty = float((taco > (lo + hi) / 2).mean())
    dc = float(mic.mean())
    ac = mic - dc
    lp = ss.sosfiltfilt(SOS_LP, ac)
    f, P = ss.welch(ac, SR, nperseg=8192)
    tot = P.sum()
    aud = (f >= 20) & (f < 20000)
    flat = float(np.exp(np.mean(np.log(P[aud] + 1e-30))) / P[aud].mean())
    mt = (f > 21500) & (f < 23500)
    tom = P[mt]
    # teste da taxa: zumbido de rede (60 Hz, harmônicos 2 e 3) no microfone e no tacômetro.
    # Saliência (dB acima da mediana local ±3 Hz) em 60·k Hz supondo SR = 51 200 (a do loader)
    # e na posição em que 60·k Hz cairia se a taxa real fosse 50 000 (60·k·51 200/50 000).
    nf = 1 << 20
    fq = np.fft.rfftfreq(nf, 1 / SR)
    sal = {}
    for nome, x in (("mic", ac), ("taco", taco - taco.mean())):
        X = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nf))
        for rot, fator in (("51k2", 1.0), ("50k", SR / 50_000)):
            v = []
            for h in (1, 2, 3):
                alvo = 60 * h * fator
                mk = (fq > alvo - 0.15) & (fq < alvo + 0.15)
                viz = (fq > alvo - 3) & (fq < alvo + 3) & ~mk
                v.append(20 * np.log10(X[mk].max() / np.median(X[viz])))
            sal[f"rede_{nome}_{rot}_db"] = float(max(v))
    u = np.unique(mic)
    # impulsos espúrios: |AC| acima de 20 x desvio robusto (MAD)
    mad = 1.4826 * np.median(np.abs(ac - np.median(ac)))
    esp = np.abs(ac) > 20 * mad
    # RMS robusto: mediana do RMS (<20 kHz) em janelas de 0,1 s
    w = SR // 10
    rms_jan = np.sqrt((lp[: len(lp) // w * w].reshape(-1, w) ** 2).mean(axis=1))
    return dict(
        rel=rel, ncol=ncol, n=len(mic), mtime=caminho.stat().st_mtime, bytes=caminho.stat().st_size,
        sha1_mic=hashlib.sha1(mic.tobytes()).hexdigest(),
        mic_dc=dc, mic_rms=float(ac.std()), mic_rms_lp20k=float(lp.std()),
        mic_pico=float(np.abs(ac).max()), mic_min=float(mic.min()), mic_max=float(mic.max()),
        mic_n_ext=int((mic == mic.max()).sum() + (mic == mic.min()).sum()),
        mic_niveis=len(u), mic_passo=float(np.min(np.diff(u))),
        mic_mad=float(mad), mic_n_espurio=int(esp.sum()), mic_rms_rob=float(np.median(rms_jan)),
        mic_rms_sem_esp=float(np.sqrt(np.mean(ac[~esp] ** 2))),
        **sal,
        f_taco_n_descartadas=r.get("n_descartadas", 0),
        mic_frac_tom22k=float(tom.sum() / tot), mic_f_tom=float(f[mt][np.argmax(tom)]),
        mic_frac_lt200=float(P[f < 200].sum() / tot), mic_frac_2k20k=float(P[(f >= 2000) & (f < 20000)].sum() / tot),
        mic_flat=flat,
        taco_min=float(taco.min()), taco_max=float(taco.max()), taco_duty=duty,
        f_taco=r["f_med"], f_taco_media=r["f_media"], f_taco_p5=r["f_p5"], f_taco_p95=r["f_p95"],
        n_bordas=r["n_bordas"], f_taco_fft=ft, f1x_acc=f1x,
    )


def md(df: pd.DataFrame, idx: bool = True) -> str:
    d = df.reset_index() if idx else df
    cab = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    linhas = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([cab, sep, *linhas])


def main() -> None:
    dados = Path(sys.argv[1]).resolve()
    saida = Path(sys.argv[2]) if len(sys.argv) > 2 else AQUI.parent / "out" / "indices" / "mafaulda.csv"
    raiz = dados / "full"
    arquivos = sorted(raiz.rglob("*.csv"))
    print(f"CSV encontrados: {len(arquivos)}")

    # lista do zip (tamanho e data/hora de cada membro)
    lista = {}
    for ln in (dados / "_metadados" / "full_zip_listing.txt").read_text(encoding="utf-8").splitlines():
        mm = re.match(r"\s*(\d+)\s+(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d)\s+(\S.*\.csv)$", ln)
        if mm:
            lista[mm.group(4)] = (int(mm.group(1)), f"{mm.group(2)} {mm.group(3)}")
    print(f"membros .csv na lista do zip: {len(lista)}")

    tarefas = [(str(p), p.relative_to(raiz).as_posix()) for p in arquivos]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(medir, tarefas, chunksize=4))
    ex_ = pd.DataFrame(res)

    linhas = []
    for r in res:
        rel = Path(r["rel"])
        lab = rotulos(rel)
        vel = float(rel.stem)
        obs = []
        if lab["classe"].startswith("rolamento"):
            obs.append("um rolamento defeituoso por tipo, o mesmo nas duas posicoes (Marins et al. 2018); "
                       "massa de desbalanceamento adicionada")
        if lab["classe"] == "desbalanceamento":
            obs.append("unidade = bancada + massa acoplada ao rotor")
        if r["mic_flat"] > 0.5 or r["mic_rms_rob"] > 0.3:
            obs.append(f"MICROFONE DEFEITUOSO: espectro plano (flatness {r['mic_flat']:.2f}), "
                       f"RMS robusto {r['mic_rms_rob'] * 1e3:.0f} mV, {r['mic_n_espurio']} impulsos")
        elif r["mic_n_espurio"] > 0:
            obs.append(f"microfone com {r['mic_n_espurio']} impulsos espurios")
        disp = (r["f_taco_p95"] - r["f_taco_p5"]) / r["f_taco"]
        if disp > 0.05:
            rot = r["f_taco_fft"]
            obs.append(f"tacometro ruidoso (dispersao {disp:.2f}); rotacao pelo pico do espectro do tacometro")
        else:
            rot = r["f_taco"]
        linhas.append(dict(
            base="mafaulda", arquivo=f"full/{r['rel']}", classe=lab["classe"], unidade=lab["unidade"],
            condicao=lab["condicao"], velocidade=vel, carga="nenhuma",
            gravacao_id=f"mafaulda/{rel.with_suffix('').as_posix()}", sr=SR, canais=1,
            duracao_s=round(r["n"] / SR, 6), f_alim_hz="", rot_esperada_hz=round(rot, 4),
            obs="; ".join(obs)))
    idx = pd.DataFrame(linhas)
    saida.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(saida, index=False, encoding="utf-8")

    ex_ = ex_.merge(idx[["arquivo", "classe", "unidade", "condicao", "velocidade"]].assign(
        rel=idx.arquivo.str[5:]), on="rel")
    ex_ = ex_.merge(idx[["arquivo", "rot_esperada_hz"]].assign(rel=idx.arquivo.str[5:]).drop(columns="arquivo"), on="rel")
    ex_["taco_ruidoso"] = (ex_.f_taco_p95 - ex_.f_taco_p5) / ex_.f_taco > 0.05
    ex_["ratio_nome_rot"] = ex_.velocidade / ex_.rot_esperada_hz
    ex_["ratio_nome"] = ex_.velocidade / ex_.f_taco
    ex_["ratio_fft"] = ex_.f_taco_fft / ex_.f_taco
    ex_["ratio_1x"] = ex_.f1x_acc / ex_.f_taco
    ex_["data_zip"] = ex_.rel.map(lambda k: lista.get(k, (None, None))[1])
    ex_["bytes_zip"] = ex_.rel.map(lambda k: lista.get(k, (None, None))[0])
    ex_["mtime_iso"] = ex_.mtime.map(lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"))
    det = dados / "_metadados" / "inventario_detalhes.csv"
    ex_.to_csv(det, index=False, encoding="utf-8")
    print(f"índice: {saida}  linhas={len(idx)}; detalhes: {det}")

    print("\n## integridade vs lista do zip")
    print(f"arquivos sem entrada na lista: {int(ex_.bytes_zip.isna().sum())}; "
          f"tamanho diferente: {int((ex_.bytes_zip != ex_.bytes).sum())}; total bytes {ex_.bytes.sum()}")

    print("\n## formato medido")
    print(f"colunas por arquivo: {sorted(Counter(ex_.ncol).items())}; amostras: {sorted(Counter(ex_.n).items())}")
    print(f"duração (s): {sorted(Counter(idx.duracao_s).items())}")

    print("\n## arquivos e duração por classe")
    print(md(idx.groupby("classe").duracao_s.agg(["count", "sum"])))
    print(f"total: {len(idx)} arquivos, {idx.duracao_s.sum():.0f} s")

    print("\n## matriz classe x unidade")
    print(md(pd.crosstab(idx.classe, idx.unidade)))

    print("\n## matriz classe x condição (severidade)")
    print(md(pd.crosstab(idx.classe, idx.condicao)))

    print("\n## velocidades nominais (nome do arquivo, Hz)")
    v = sorted(idx.velocidade.unique())
    print(f"{len(v)} valores distintos, de {v[0]} a {v[-1]}; múltiplos de 0,2048 Hz? "
          f"{all(abs(x / 0.2048 - round(x / 0.2048)) < 1e-6 for x in v)}")
    print("arquivos por (classe, condição): min/max de velocidades distintas:",
          idx.groupby(["classe", "condicao"]).velocidade.nunique().agg(["min", "max"]).to_dict())
    rep = idx.groupby(["classe", "condicao", "velocidade"]).size()
    print(f"(classe, condição, velocidade) com mais de 1 arquivo: {int((rep > 1).sum())}")

    print(f"\n## taxa de amostragem: zumbido de rede (máx. saliência em 60/120/180 Hz, dB), SR do loader = {SR}")
    for nome in ("mic", "taco"):
        a_, b_ = ex_[f"rede_{nome}_51k2_db"], ex_[f"rede_{nome}_50k_db"]
        print(f"{nome}: em 60·k Hz supondo 51,2 kHz: mediana {a_.median():.1f} dB (p5 {a_.quantile(.05):.1f}); "
              f"supondo 50 kHz: mediana {b_.median():.1f} dB (p5 {b_.quantile(.05):.1f}); "
              f"arquivos com 51,2k > 50k: {int((a_ > b_).sum())}/{len(ex_)}")
    print("por classe (mediana, microfone, 51,2k | 50k):")
    print(md(ex_.groupby("classe")[["rede_mic_51k2_db", "rede_mic_50k_db", "rede_taco_51k2_db", "rede_taco_50k_db"]]
             .median().round(1)))

    print("\n## verdade de rotação: tacômetro")
    print(f"bordas descartadas pelo período refratário: arquivos com >0: {int((ex_.f_taco_n_descartadas > 0).sum())}, "
          f"max {ex_.f_taco_n_descartadas.max()}")
    print(f"tacômetro ruidoso ((p95-p5)/mediana > 0,05): {int(ex_.taco_ruidoso.sum())} arquivos; por pasta:")
    print(md(ex_[ex_.taco_ruidoso].assign(p=ex_.rel.str.rsplit("/", n=1).str[0]).groupby("p").size().to_frame("n")))
    rr = ex_.ratio_nome_rot
    d = ex_.velocidade - ex_.rot_esperada_hz
    print(f"nome / rot_esperada (índice): mediana {rr.median():.4f}, p5 {rr.quantile(.05):.4f}, p95 {rr.quantile(.95):.4f}, "
          f"min {rr.min():.4f}, max {rr.max():.4f}")
    print(f"nome - rot_esperada (Hz): mediana {d.median():.3f}, p5 {d.quantile(.05):.3f}, p95 {d.quantile(.95):.3f}, "
          f"|d| <= 0,2048 Hz (1 bin do nome): {(d.abs() <= 0.2048).mean():.3f}; <= 0,5 Hz: {(d.abs() <= 0.5).mean():.3f}")
    print(f"idem só tacômetro limpo: mediana {d[~ex_.taco_ruidoso].median():.3f}, "
          f"|d|<=0,2048: {(d[~ex_.taco_ruidoso].abs() <= 0.2048).mean():.3f}, max |d| {d[~ex_.taco_ruidoso].abs().max():.3f}")
    print(f"idem tacômetro ruidoso: mediana {d[ex_.taco_ruidoso].median():.3f}, max |d| {d[ex_.taco_ruidoso].abs().max():.3f}")
    a2, b2 = np.polyfit(ex_.velocidade[~ex_.taco_ruidoso], ex_.rot_esperada_hz[~ex_.taco_ruidoso], 1)
    print(f"ajuste (tacômetro limpo) rot = {a2:.5f}·nome + {b2:.4f} Hz")
    r50 = ex_.velocidade / (ex_.f_taco * 50_000 / SR)
    print(f"nome / f_taco calculada com 50 kHz: mediana {r50.median():.4f} (p5 {r50.quantile(.05):.4f}, p95 {r50.quantile(.95):.4f})")
    print(f"bordas por arquivo: min {ex_.n_bordas.min()}, mediana {ex_.n_bordas.median()}, max {ex_.n_bordas.max()}")
    print(f"duty cycle do tacômetro: mediana {ex_.taco_duty.median():.3f} (p5 {ex_.taco_duty.quantile(.05):.3f}, "
          f"p95 {ex_.taco_duty.quantile(.95):.3f}); faixa de tensão: min {ex_.taco_min.min():.2f}, max {ex_.taco_max.max():.2f} V")
    print(f"f_taco (bordas, mediana) de {ex_.f_taco.min():.3f} a {ex_.f_taco.max():.3f} Hz")
    print(f"dispersão dentro do arquivo (p95-p5)/mediana: mediana {((ex_.f_taco_p95 - ex_.f_taco_p5) / ex_.f_taco).median():.4f}, "
          f"max {((ex_.f_taco_p95 - ex_.f_taco_p5) / ex_.f_taco).max():.4f}")
    print(f"FFT do tacômetro / bordas: mediana {ex_.ratio_fft.median():.4f}, min {ex_.ratio_fft.min():.4f}, max {ex_.ratio_fft.max():.4f}")
    print(f"pulsos por volta: pico do acelerômetro (col 3, underhang radial) em [0,3;1,3]·f_taco / f_taco: "
          f"mediana {ex_.ratio_1x.median():.4f}; fração em [0,98;1,02]: {((ex_.ratio_1x - 1).abs() < 0.02).mean():.3f}; "
          f"fração perto de 0,5: {((ex_.ratio_1x - 0.5).abs() < 0.02).mean():.3f}")
    print(f"nome / f_taco: mediana {ex_.ratio_nome.median():.4f}, p5 {ex_.ratio_nome.quantile(.05):.4f}, "
          f"p95 {ex_.ratio_nome.quantile(.95):.4f}, min {ex_.ratio_nome.min():.4f}, max {ex_.ratio_nome.max():.4f}")
    a, b = np.polyfit(ex_.velocidade, ex_.f_taco, 1)
    res_ = ex_.f_taco - (a * ex_.velocidade + b)
    print(f"ajuste f_taco = {a:.4f}·nome + {b:.4f} Hz; resíduo RMS {res_.std():.3f} Hz, max |res| {res_.abs().max():.3f} Hz")
    print("\nnome/f_taco por classe (mediana, min, max):")
    print(md(ex_.groupby("classe").ratio_nome.agg(["median", "min", "max"]).round(4)))
    print("\ndiferença nome - f_taco (Hz) por faixa de nome:")
    ex_["faixa_nome"] = pd.cut(ex_.velocidade, [0, 20, 30, 40, 50, 70])
    print(md(ex_.assign(d=ex_.velocidade - ex_.f_taco).groupby("faixa_nome", observed=True).d
             .agg(["count", "median", "min", "max"]).round(3)))
    out = ex_[(ex_.ratio_nome - ex_.ratio_nome.median()).abs() > 0.05]
    print(f"\narquivos com nome/f_taco a mais de 0,05 da mediana: {len(out)}")
    if len(out):
        print(md(out[["rel", "velocidade", "f_taco", "f_taco_fft", "n_bordas", "taco_duty"]].round(3).head(30), idx=False))
    # dentro de cada (classe, condição), a ordem por nome bate com a ordem por f_taco?
    inv = 0
    for _, g in ex_.groupby(["classe", "condicao"]):
        g = g.sort_values("velocidade")
        inv += int((np.diff(g.f_taco.to_numpy()) < 0).sum())
    print(f"inversões de ordem (f_taco cai quando o nome sobe) dentro de classe x condição: {inv}")

    print("\n## microfone: formato e nível")
    print(f"passo de quantização: valores {sorted(Counter(ex_.mic_passo.round(7)).items())[:6]}")
    print(f"níveis distintos: min {ex_.mic_niveis.min()}, mediana {int(ex_.mic_niveis.median())}, max {ex_.mic_niveis.max()}")
    print(f"faixa absoluta: {ex_.mic_min.min():.4f} a {ex_.mic_max.max():.4f} V; DC {ex_.mic_dc.min():.4f} a {ex_.mic_dc.max():.4f} V")
    print(f"amostras no valor extremo (máx+mín) por arquivo: mediana {ex_.mic_n_ext.median()}, max {ex_.mic_n_ext.max()}; "
          f"arquivos com >=10: {int((ex_.mic_n_ext >= 10).sum())}")
    vmax = ex_.mic_max.max(); vmin = ex_.mic_min.min()
    print(f"arquivos que tocam o máximo global ({vmax:.4f}): {int((ex_.mic_max == vmax).sum())}; "
          f"o mínimo global ({vmin:.4f}): {int((ex_.mic_min == vmin).sum())}")
    print(f"tom em ~22 kHz: frequência {sorted(Counter(ex_.mic_f_tom.round(0)).items())[:8]}; "
          f"fração da potência: mediana {ex_.mic_frac_tom22k.median():.3f}, min {ex_.mic_frac_tom22k.min():.3f}, max {ex_.mic_frac_tom22k.max():.3f}")
    print("\nRMS AC total e RMS <20 kHz (mV), fração do tom 22 kHz, flatness 20 Hz-20 kHz, por classe (medianas):")
    t = ex_.groupby("classe").agg(n=("rel", "size"), rms_mV=("mic_rms", "median"), rms_lp20k_mV=("mic_rms_lp20k", "median"),
                                  rms_lp_min=("mic_rms_lp20k", "min"), rms_lp_max=("mic_rms_lp20k", "max"),
                                  pico_mV=("mic_pico", "max"), frac_tom=("mic_frac_tom22k", "median"),
                                  flat=("mic_flat", "median"), n_flat_gt05=("mic_flat", lambda s: int((s > 0.5).sum())),
                                  n_arq_espurio=("mic_n_espurio", lambda s: int((s > 0).sum())),
                                  rms_rob_mV=("mic_rms_rob", "median"), rms_sem_esp_mV=("mic_rms_sem_esp", "median"))
    for c in ["rms_mV", "rms_lp20k_mV", "rms_lp_min", "rms_lp_max", "pico_mV", "rms_rob_mV", "rms_sem_esp_mV"]:
        t[c] = (t[c] * 1e3).round(1)
    print(md(t.round(3)))
    print(f"\nimpulsos espúrios (|AC| > 20·MAD): arquivos com pelo menos 1: {int((ex_.mic_n_espurio > 0).sum())}; "
          f"amostras por arquivo: mediana {ex_.mic_n_espurio.median()}, max {ex_.mic_n_espurio.max()}")
    print("arquivos com impulsos por classe x condição:")
    print(md(ex_.pivot_table(index="classe", columns="condicao", values="mic_n_espurio", fill_value=-1,
                             aggfunc=lambda s: int((s > 0).sum())).astype(int).replace(-1, "")))
    print("\nRMS robusto (mediana de janelas de 0,1 s, <20 kHz, mV) por classe x condição (mediana):")
    print(md((ex_.pivot_table(index="classe", columns="condicao", values="mic_rms_rob", aggfunc="median") * 1e3).round(1)))
    print("\nRMS <20 kHz (mV) por classe x condição (mediana):")
    print(md((ex_.pivot_table(index="classe", columns="condicao", values="mic_rms_lp20k", aggfunc="median") * 1e3).round(1)))
    print("\nflatness mediana por classe x condição:")
    print(md(ex_.pivot_table(index="classe", columns="condicao", values="mic_flat", aggfunc="median").round(2)))
    print("\nRMS robusto (mV) por classe x faixa de rotação nominal (mediana):")
    print(md((ex_.pivot_table(index="classe", columns="faixa_nome", values="mic_rms_rob", aggfunc="median",
                              observed=True) * 1e3).round(1)))
    fl = ex_[ex_.mic_flat > 0.5]
    print(f"\narquivos com flatness > 0,5 (espectro quase branco): {len(fl)}")
    if len(fl):
        print(md(fl.groupby(["classe", "condicao"]).size().to_frame("n")))

    print("\n## duplicatas do microfone (hash idêntico)")
    dup = [g.rel.tolist() for _, g in ex_.groupby("sha1_mic") if len(g) > 1]
    print(dup if dup else "nenhuma")

    print("\n## datas (lista do zip = mtime dos arquivos) por pasta")
    ex_["pasta"] = ex_.rel.map(lambda s: "/".join(s.split("/")[:-1]))
    print(md(ex_.groupby("pasta").agg(n=("rel", "size"), primeiro=("mtime_iso", "min"), ultimo=("mtime_iso", "max"))))
    print("\nordem de gravação (mtime) x ordem de velocidade dentro de cada pasta (Spearman):")
    sp = ex_.groupby("pasta").apply(lambda g: g.mtime.rank().corr(g.velocidade.rank()), include_groups=False)
    print(f"mediana {sp.median():.3f}, min {sp.min():.3f}, max {sp.max():.3f}")
    print("\nsequência das pastas por mtime mínimo:")
    print(md(ex_.groupby("pasta").mtime_iso.min().sort_values().to_frame("inicio")))


if __name__ == "__main__":
    main()
