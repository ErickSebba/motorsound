"""Figuras do artigo (PDF + PNG, coluna IEEE de 3,5 in, 8 pt, sem título interno).

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_artigo_figuras.py

Agente E (Etapa 7). Rodar DEPOIS de gerar_artigo_numeros.py (lê paper/numeros.json).

- arquitetura          diagrama do sistema; rótulos numéricos vêm de numeros.json
- diag_balacc_en_<ts>  versão em inglês de diag_balacc (mesmo CSV do diagnóstico)
- diag_confusao_p11_en_<ts>, diag_espectro_en_<ts>   idem (o espectro relê janelas do
                       train/ do IDMT, nunca o test/)
- equivalencia_erro_etapas_en   razão erro/tolerância por etapa (resumo por build)
- esp32_latencia_<alvo>_<ts>    SÓ se results/metrics/esp32_lat_*.json e bancada.json existirem;
                       caso contrário nenhuma figura é criada (sem figura falsa).

Escreve também paper/figuras.tex com uma macro de caminho por figura (\\FigArq, …) e o
\\newif\\iffigesp que o main.tex usa para incluir a figura do ESP32 só quando ela existe.
"""
from __future__ import annotations

import json
import sys
import warnings

import gerar_artigo_comum as C

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "font.family": "serif"})
# mesma paleta de diagnostico_saidas.py (skill dataviz, slots 1-4)
CORES = {"b00_rmspico_norm": "#eda100", "b00_rmspico_bruto": "#eb6834",
         "p04_classicas": "#1baf7a", "p11_decifrae_harmsum": "#2a78d6"}
HACH = {"b00_rmspico_norm": "////", "b00_rmspico_bruto": "", "p04_classicas": "....",
        "p11_decifrae_harmsum": ""}
ROT = {"b00_rmspico_norm": "RMS/peak (RMS-norm.)", "b00_rmspico_bruto": "RMS/peak (raw)",
       "p04_classicas": "Classical (p04)", "p11_decifrae_harmsum": "Decifrae (p11)"}
CASOS = [("idmt", "aleatorio", "IDMT\nrandom*"), ("idmt", "blocos", "IDMT\nblocks"),
         ("a2212", "aleatorio", "A2212\nrandom*"), ("a2212", "arquivo", "A2212\nfile"),
         ("a2212", "motor", "A2212\nmotor-out"), ("a2212", "motor_sem_M4", "A2212\nw/o M4")]
SAIDAS = {}   # nome lógico -> (pdf, [fontes])


def _salvar(fig, nome, fontes):
    for ext in ("pdf", "png"):
        d = C.FIGS / ext
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{nome}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    SAIDAS[nome] = (C.FIGS / "pdf" / f"{nome}.pdf", [C.rel(f) for f in fontes])
    return nome


def N():
    return C.ler_json(C.PAPER / "numeros.json")["numeros"]


# ------------------------------------------------------------------ arquitetura
def fig_arquitetura(n):
    def v(k):
        x = n[k]["tex"].replace("{,}", ",").replace("$", "")
        return x.replace("\\times10^{", "e").rstrip("}") if "times" in x else x

    fig, ax = plt.subplots(figsize=(3.5, 3.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 112)
    ax.axis("off")
    azul, cinza, laranja = "#2a78d6", "#52514e", "#eb6834"

    def caixa(x, y, w, h, txt, cor=azul, fc="#eef4fc", ls="-", fs=6.6, peso="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=cor, lw=0.7, ls=ls))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color="#0b0b0b", weight=peso, linespacing=1.15)

    def seta(x0, y0, x1, y1, cor=cinza):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=cor, shrinkA=0, shrinkB=0,
                                    mutation_scale=7))

    # domínio do MCU
    nh1 = int(n["VParSr"]["valor"] * n["VParSegBldc"]["valor"])
    nh3 = int(n["VParSr"]["valor"] * n["VParSegIdmt"]["valor"])
    ax.add_patch(FancyBboxPatch((1, 1), 98, 92, boxstyle="round,pad=0.3,rounding_size=2",
                                fc="none", ec=laranja, lw=0.9, ls="--"))
    ax.text(50, 91.2, "runs on the ESP32: portable C, float32, no malloc (same source as PC build)",
            fontsize=5.0, color=laranja, ha="center", va="top", style="italic")
    # entrada
    caixa(4, 100, 61, 9, f"Audio {v('VParSrKhz')} kHz, window {v('VParSegIdmt')} s (IDMT) / "
          f"{v('VParSegBldc')} s (A2212)\nnow: sent over UART (processor-in-the-loop)",
          cor=cinza, fc="#f4f3f0", fs=5.3)
    caixa(70, 100, 27, 9, "I2S microphone\n(future work)", cor=cinza, fc="white", ls=":", fs=5.3)
    seta(34.5, 100, 34.5, 86.5)
    # cadeia principal (coluna esquerda)
    xL, wL = 4, 61
    ys = [79.0, 68.0, 57.0, 46.0, 35.0, 24.0, 13.0]
    hs = 7.5
    txt = [f"RMS normalisation (target {v('VParRmsTarget')})",
           f"Per-frame STFT, Hann {v('VParNfft')}, hop {v('VParHop')} → |X|",
           f"Harmonic sum, {v('VParNharm')} harmonics, {v('VParJ')} states\n"
           f"({v('VParFmin')}–{v('VParFmax')} Hz, {v('VParStepCents')}-cent grid)",
           "Rotation Viterbi (silence state, octave jump)\n"
           "→ $f_{rot}(t)$, interpretable output",
           f"{v('VParNFeat')} features (rotation, harmonic energy,\n"
           "envelope modulation, spectral shape)",
           f"z-score → cosine to class prototypes\n→ softmax (T = {v('VParTempCos')})",
           "Online HMM forward filter, per window\n→ operating state"]
    for y, t in zip(ys, txt):
        caixa(xL, y, wL, hs, t, fs=5.3)
    for a, b in zip(ys[:-1], ys[1:]):
        seta(xL + wL / 2, a, xL + wL / 2, b + hs)
    # ramo do envelope (coluna direita)
    caixa(70, 55, 27, 18, f"Envelope branch:\n|Hilbert(y)|\nButterworth {v('VParButterOrder')}, "
          f"{v('VParButterFc')} Hz\n↓{v('VParDecim')}, Welch", fs=5.1)
    seta(65, 82.75, 70, 66)
    seta(70, 60, 65, 38.75)
    # anotações de memória e FFT
    caixa(70, 33, 27, 16, f"Mixed-radix FFT\n(portable C): STFT,\nHilbert of N = {nh1}\n"
          f"or {nh3}, Welch", cor=laranja, fc="#fdf0e9", fs=5.1)
    caixa(70, 5, 27, 23, f"PSRAM arena\n(analytical estimate:\n{v('VEstWsFloatThreeMb')} MB @ "
          f"{v('VParSegIdmt')} s,\n{v('VEstWsFloatOneMb')} MB @ {v('VParSegBldc')} s)\n"
          "saliency, Viterbi\nback-pointers, Hilbert", cor=laranja, fc="#fdf0e9", fs=5.1)
    return _salvar(fig, "arquitetura", [C.PAPER / "numeros.json"])


# ------------------------------------------------------------------ diagnóstico em inglês
def _diag_fontes():
    fdiag = C.mais_recente(C.RES / "csv", "diagnostico_2*.csv")
    ts = fdiag.stem.replace("diagnostico_", "")
    return fdiag, ts


def fig_balacc_en():
    fdiag, ts = _diag_fontes()
    M = pd.read_csv(fdiag)
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    vs = list(ROT)
    w = 0.8 / len(vs)
    x = np.arange(len(CASOS))
    for j, v in enumerate(vs):
        val, lo, hi = [], [], []
        for b, p, _ in CASOS:
            r = M[(M.variante == v) & (M.base == b) & (M.protocolo == p) & (M.classe == "_macro")
                  & (M.metrica == "acuracia_balanceada")].iloc[0]
            val.append(r.valor); lo.append(r.valor - r.ic95_lo); hi.append(r.ic95_hi - r.valor)
        xx = x + (j - (len(vs) - 1) / 2) * w
        ax.bar(xx, val, w * 0.92, color=CORES[v], hatch=HACH[v], edgecolor="white",
               linewidth=0.4, label=ROT[v], zorder=2)
        ax.errorbar(xx, val, yerr=[lo, hi], fmt="none", ecolor="#0b0b0b", elinewidth=0.6,
                    capsize=1.2, zorder=3)
    ax.axhline(1 / 3, color="#52514e", lw=0.6, ls=":", zorder=1)
    ax.set_xticks(x, [c[2] for c in CASOS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", color="#d9d8d4", lw=0.4, zorder=0)
    ax.legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1.4, columnspacing=0.8)
    return _salvar(fig, f"diag_balacc_en_{ts}", [fdiag])


def fig_confusao_en():
    fdiag, ts = _diag_fontes()
    fj = C.RES / "metrics" / f"diagnostico_{ts}.json"
    m = C.ler_json(fj)["metricas"]["p11_decifrae_harmsum"]
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for ax, (b, p, tit) in zip(axs, [("idmt", "blocos", "IDMT, blocks"),
                                     ("a2212", "arquivo", "A2212, by file")]):
        cm = np.array(m[b][p]["matriz_confusao"]["valores"], dtype=float)
        cl = m[b][p]["matriz_confusao"]["linhas_verdade_colunas_pred"]
        cn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(cl)):
            for k in range(len(cl)):
                ax.text(k, i, f"{int(cm[i, k])}", ha="center", va="center", fontsize=7,
                        color="white" if cn[i, k] > 0.6 else "#0b0b0b")
        ab = {"good": "good", "broken": "brok.", "heavyload": "heavy", "Healthy": "Heal.",
              "Propeller": "Prop.", "Bearing": "Bear."}
        ax.set_xticks(range(len(cl)), [ab[c] for c in cl])
        ax.set_yticks(range(len(cl)), [ab[c] for c in cl])
        ax.set_xlabel(f"Predicted ({tit})", fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("True")
    fig.tight_layout(w_pad=1.0)
    return _salvar(fig, f"diag_confusao_p11_en_{ts}", [fj])


def fig_espectro_en():
    """Mesmas janelas de diagnostico_saidas.fig_espectro (train/ do IDMT, nunca o test/)."""
    fdiag, ts = _diag_fontes()
    from motorsound import io, runner
    from motorsound.config import AVAL_IDMT
    from motorsound.rotacao import rastrear, stft_mag
    from pipelines import _decifrae as dd
    from pipelines import p11_decifrae_harmsum as p11
    pre = p11.PRE
    segs = runner.segmentos_de("idmt", AVAL_IDMT)
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 2.9), gridspec_kw={"height_ratios": [1.5, 1]})
    fontes = []
    for classe, cor in (("good", "#2a78d6"), ("broken", "#eb6834")):
        ss_ = [s for s in segs if s.classe == classe]
        s = ss_[len(ss_) // 2]
        assert "/test" not in s.rec_id
        fontes.append(s.rec_id)
        y = runner._preparar(io.audio_segmento(s, pre.sr), AVAL_IDMT)
        f_rot, _, _ = rastrear(y, pre.sr, pre.fmin, pre.fmax, pre.passo_cents,
                               np.arange(1, dd.N_HARM + 1), pre.n_fft, pre.hop,
                               usar_pyin=pre.usar_pyin, usar_viterbi=pre.usar_viterbi,
                               max_salto=pre.max_salto)
        S, f = stft_mag(y, pre.sr, pre.n_fft, pre.hop)
        Pdb = 10 * np.log10((S ** 2).mean(axis=1) + 1e-12)
        fm = float(np.nanmedian(f_rot))
        mk = f <= 650
        axs[0].plot(f[mk], Pdb[mk] - Pdb[mk].max(), color=cor, lw=0.6,
                    label=f"{classe} ($f_{{rot}}$={fm:.2f} Hz)")
        for k in range(1, dd.N_HARM + 1):
            if k * fm <= 650:
                axs[0].axvline(k * fm, color=cor, lw=0.3, alpha=0.35,
                               ls="-" if classe == "good" else "--")
        tq = (np.arange(len(f_rot)) * pre.hop + pre.n_fft / 2) / pre.sr
        axs[1].plot(tq, f_rot, color=cor, lw=1.0, marker="o", ms=2)
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Power (dB rel.)")
    axs[0].set_ylim(-80, 3)
    axs[0].set_xlim(0, 650)
    axs[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                  handlelength=1.2, columnspacing=0.8)
    axs[1].set_xlabel("Time within window (s)")
    axs[1].set_ylabel("$f_{rot}$ (Hz)")
    axs[1].set_ylim(pre.fmin, pre.fmax)
    for a in axs:
        a.grid(color="#e6e5e1", lw=0.4)
    fig.tight_layout(h_pad=0.6)
    nome = _salvar(fig, f"diag_espectro_en_{ts}", [fdiag])
    SAIDAS[nome][1].extend(f"data/{r} (janela do meio, train/)" for r in fontes)
    return nome


def fig_equivalencia_en():
    fres = C.mais_recente(C.RES / "csv", "equivalencia_resumo_por_build_*.csv")
    S = pd.read_csv(fres)
    rot = {"double": "C double (PC)", "float": "C float32 (PC)",
           "float-sosd": "C float32 + SOS in double (PC)"}
    cor = {"double": "#2a78d6", "float": "#eb6834", "float-sosd": "#1baf7a"}
    etapas = [e for e in S[S.build == "double"].etapa if e not in
              ("vit_path", "decisao", "argmax", "hmm_viterbi", "hmm_online", "absorb")]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    x = np.arange(len(etapas))
    w = 0.27
    for j, b in enumerate(rot):
        s = S[S.build == b].set_index("etapa")
        r = np.array([max(float(s.loc[e, "razao_tol_max"]), 1e-6) for e in etapas])
        ax.bar(x + (j - 1) * w, r, w * 0.95, color=cor[b], label=rot[b], zorder=2)
        for i, e in enumerate(etapas):
            if s.loc[e, "fail"] > 0:
                ax.text(x[i] + (j - 1) * w, r[i] * 2, "F", ha="center", fontsize=5.5,
                        color=cor[b])
            elif s.loc[e, "pass_empate"] > 0:
                ax.text(x[i] + (j - 1) * w, r[i] * 2, "T", ha="center", fontsize=5.5,
                        color=cor[b])
    ax.axhline(1, color="#0b0b0b", lw=0.6, ls="--", zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("max error / tolerance")
    ax.set_xticks(x, etapas, rotation=70, ha="right",
                  fontsize=5.8)
    ax.grid(axis="y", color="#e6e5e1", lw=0.4, zorder=0)
    ax.legend(frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return _salvar(fig, "equivalencia_erro_etapas_en", [fres])


# ------------------------------------------------------------------ ESP32 (condicional)
ETAPAS_CHIP = [("prep", "preparar", "RMS norm."), ("rastreio", "rastrear_quadros", "STFT + harm. sum"),
               ("viterbi", "viterbi", "Viterbi"), ("envelope", "envelope_mod", "envelope"),
               ("features", "features", "features"), ("cosseno", "cosseno", "cosine")]


def fig_latencia_esp32(resumo_chip, resumo_pc, nome, fontes):
    """Barras empilhadas por etapa (mediana), ESP32 medido × C no PC medido, por base."""
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.7))
    paleta = ["#9c9a94", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7a52c7"]
    for ax, (base, tit) in zip(axs, (("idmt", "IDMT, 3 s"), ("a2212", "A2212, 1 s"))):
        linhas = [("ESP32", resumo_chip[base], 0), ("PC", resumo_pc[f"c_float_pc/{base}"], 1)]
        for rot, r, i in linhas:
            acc = 0.0
            for k, (ec, ep, lab) in enumerate(ETAPAS_CHIP):
                chave = f"{ec}_ms" if rot == "ESP32" else f"{ep}_ms"
                val = r[chave]["mediana"]
                ax.barh(i, val, left=acc, color=paleta[k], height=0.6,
                        label=lab if (base == "idmt" and rot == "ESP32") else None)
                acc += val
        ax.set_yticks([0, 1], ["ESP32", "PC"])
        ax.set_xscale("log")
        ax.set_xlabel(f"median ms ({tit})", fontsize=7)
    fig.legend(ncol=3, frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    return _salvar(fig, nome, fontes)


def fig_esp32_se_houver():
    fb = C.RES / "manifests" / "bancada.json"
    flat = C.mais_recente(C.RES / "metrics", "esp32_lat_*.json")
    fpc = C.mais_recente(C.RES / "metrics", "bench_pc_*.json")
    if not (fb.exists() and flat is not None):
        print("ESP32: sem bancada.json/esp32_lat_*.json -> nenhuma figura de latência criada")
        return None
    chip = C.ler_json(flat)["resumo"]
    pc = C.ler_json(fpc)["resumo"]
    return fig_latencia_esp32(chip, pc, flat.stem, [fb, flat, fpc])


def escrever_figuras_tex(nomes, chip):
    def cam(n):
        return f"../figures/pdf/{n}.pdf"
    L = ["% Gerado por embarcados/src/python/gerar_artigo_figuras.py. NÃO EDITAR À MÃO.",
         "% Caminhos relativos a embarcados/paper/.",
         r"\newif\iffigesp",
         (r"\figesptrue" if chip else r"\figespfalse") + "  % true só se a figura do ESP32 existir"]
    for macro, n in nomes.items():
        L.append(f"\\newcommand{{\\{macro}}}{{{cam(n)}}}")
    if not chip:
        L.append(r"\newcommand{\FigEspLat}{}")
    (C.PAPER / "figuras.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (C.PAPER / "figuras.json").write_text(json.dumps(
        {k: {"pdf": C.rel(v[0]), "fontes": v[1]} for k, v in SAIDAS.items()}, indent=1,
        ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    n = N()
    nomes = {"FigArq": fig_arquitetura(n), "FigBalacc": fig_balacc_en(),
             "FigConf": fig_confusao_en(), "FigEquiv": fig_equivalencia_en()}
    if "--sem-espectro" not in (argv or sys.argv[1:]):
        nomes["FigSpec"] = fig_espectro_en()
    else:
        fdiag, ts = _diag_fontes()
        nomes["FigSpec"] = f"diag_espectro_en_{ts}"
    chip = fig_esp32_se_houver()
    if chip:
        nomes["FigEspLat"] = chip
    escrever_figuras_tex(nomes, bool(chip))
    print("figuras:", ", ".join(nomes.values()))


if __name__ == "__main__":
    main()
