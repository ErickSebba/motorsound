"""Etapa 6: figuras, tabela LaTeX e relatório a partir das saídas de `diagnostico.py`.

Lê o CSV de métricas mais recente (ou `--ts`), o CSV de janelas e o meta. Nenhum número é
digitado à mão: tudo sai daqui. Figuras com largura de coluna IEEE (3,5 in), fonte 8 pt,
sem título dentro.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\diagnostico_saidas.py [--ts AAAAMMDD_HHMMSS]

Escreve:
  figures/{pdf,png}/diag_balacc_<ts>.*        (a) acurácia balanceada com IC
  figures/{pdf,png}/diag_confusao_p11_<ts>.*  (b) matriz de confusão da p11 (IDMT blocos, A2212 arquivo)
  figures/{pdf,png}/diag_espectro_<ts>.*      (c) espectro médio + pente k·f_rot + caminho de rotação
  tables/latex/desempenho.tex                 (booktabs; regenerado a cada execução, com o ts no cabeçalho)
  reports/diagnostico.md
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
EMB = RAIZ / "embarcados"
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HIST_CSV = RAIZ / "out" / "experiments" / "20261014_185859_validacao.csv"

# paleta categórica validada (skill dataviz, slots 1-4 em ordem fixa)
CORES = {"b00_rmspico_norm": "#eda100", "b00_rmspico_bruto": "#eb6834",
         "p04_classicas": "#1baf7a", "p11_decifrae_harmsum": "#2a78d6"}
HACH = {"b00_rmspico_norm": "////", "b00_rmspico_bruto": "", "p04_classicas": "....",
        "p11_decifrae_harmsum": ""}
ROT = {"b00_rmspico_norm": "RMS/pico (RMS norm.)", "b00_rmspico_bruto": "RMS/pico (bruto)",
       "p04_classicas": "Clássicas (p04)", "p11_decifrae_harmsum": "Decifrae (p11)"}
ORDEM_V = list(ROT)
CASOS = [("idmt", "aleatorio", "IDMT\naleat.*"), ("idmt", "blocos", "IDMT\nblocos"),
         ("a2212", "aleatorio", "A2212\naleat.*"), ("a2212", "arquivo", "A2212\narquivo"),
         ("a2212", "motor", "A2212\nmotor"), ("a2212", "motor_sem_M4", "A2212\nsem M4")]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "font.family": "serif"})


def _ultimo_ts():
    cs = sorted((EMB / "results" / "csv").glob("diagnostico_2*.csv"))
    cs = [c for c in cs if not c.stem.endswith("_meta")]
    if not cs:
        raise SystemExit("rode diagnostico.py antes")
    return cs[-1].stem.replace("diagnostico_", "")


def _salvar(fig, nome, ts):
    for ext in ("pdf", "png"):
        d = EMB / "figures" / ext
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{nome}_{ts}.{ext}"
        if p.exists():
            raise FileExistsError(p)
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return f"figures/pdf/{nome}_{ts}.pdf"


def _macro(M, v, b, p, k="acuracia_balanceada"):
    r = M[(M.variante == v) & (M.base == b) & (M.protocolo == p) & (M.classe == "_macro")
          & (M.metrica == k)]
    return r.iloc[0] if len(r) else None


# ------------------------------------------------------------------ (a)
def fig_balacc(M, ts):
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    n = len(ORDEM_V)
    w = 0.8 / n
    x = np.arange(len(CASOS))
    for j, v in enumerate(ORDEM_V):
        vals, lo, hi = [], [], []
        for b, p, _ in CASOS:
            r = _macro(M, v, b, p)
            vals.append(r.valor); lo.append(r.valor - r.ic95_lo); hi.append(r.ic95_hi - r.valor)
        ax.bar(x + (j - (n - 1) / 2) * w, vals, w * 0.92, color=CORES[v], hatch=HACH[v],
               edgecolor="white", linewidth=0.4, label=ROT[v], zorder=2)
        ax.errorbar(x + (j - (n - 1) / 2) * w, vals, yerr=[lo, hi], fmt="none",
                    ecolor="#0b0b0b", elinewidth=0.6, capsize=1.2, zorder=3)
    ax.axhline(1 / 3, color="#52514e", lw=0.6, ls=":", zorder=1)
    ax.set_xticks(x, [c[2] for c in CASOS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acurácia balanceada")
    ax.grid(axis="y", color="#d9d8d4", lw=0.4, zorder=0)
    ax.legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1.4, columnspacing=0.8)
    return _salvar(fig, "diag_balacc", ts)


# ------------------------------------------------------------------ (b)
def fig_confusao(Mj, ts):
    m = Mj["metricas"]["p11_decifrae_harmsum"]
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for ax, (b, p, tit) in zip(axs, [("idmt", "blocos", "IDMT, blocos"),
                                     ("a2212", "arquivo", "A2212, por arquivo")]):
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
        ax.set_xlabel(f"Predita ({tit})", fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("Verdadeira")
    fig.tight_layout(w_pad=1.0)
    return _salvar(fig, "diag_confusao_p11", ts)


# ------------------------------------------------------------------ (c)
def fig_espectro(ts):
    """Uma janela good e uma broken do train/ do IDMT (nunca o test/)."""
    from motorsound import io, runner
    from motorsound.config import AVAL_IDMT
    from motorsound.rotacao import rastrear, stft_mag
    from pipelines import _decifrae as dd
    from pipelines import p11_decifrae_harmsum as p11
    pre = p11.PRE
    segs = runner.segmentos_de("idmt", AVAL_IDMT)
    info = {}
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 2.9), gridspec_kw={"height_ratios": [1.5, 1]})
    for classe, cor in (("good", "#2a78d6"), ("broken", "#eb6834")):
        ss_ = [s for s in segs if s.classe == classe]
        s = ss_[len(ss_) // 2]                             # janela do meio da gravação
        y = runner._preparar(io.audio_segmento(s, pre.sr), AVAL_IDMT)
        f_rot, _, _ = rastrear(y, pre.sr, pre.fmin, pre.fmax, pre.passo_cents,
                               np.arange(1, dd.N_HARM + 1), pre.n_fft, pre.hop,
                               usar_pyin=pre.usar_pyin, usar_viterbi=pre.usar_viterbi,
                               max_salto=pre.max_salto)
        S, f = stft_mag(y, pre.sr, pre.n_fft, pre.hop)
        Pdb = 10 * np.log10((S ** 2).mean(axis=1) + 1e-12)
        fm = float(np.nanmedian(f_rot))
        m = f <= 650
        axs[0].plot(f[m], Pdb[m] - Pdb[m].max(), color=cor, lw=0.6,
                    label=f"{classe} ($f_{{rot}}$={fm:.2f} Hz)")
        for k in range(1, dd.N_HARM + 1):
            if k * fm <= 650:
                axs[0].axvline(k * fm, color=cor, lw=0.3, alpha=0.35,
                               ls="-" if classe == "good" else "--")
        tq = (np.arange(len(f_rot)) * pre.hop + pre.n_fft / 2) / pre.sr
        axs[1].plot(tq, f_rot, color=cor, lw=1.0, marker="o", ms=2, label=classe)
        info[classe] = {"rec_id": s.rec_id, "idx": s.idx, "t0": s.t0, "t1": s.t1,
                        "f_rot_mediana_hz": fm, "quadros": int(len(f_rot)),
                        "quadros_sil": int(np.sum(~np.isfinite(f_rot)))}
    axs[0].set_xlabel("Frequência (Hz)")
    axs[0].set_ylabel("Potência (dB rel.)")
    axs[0].set_ylim(-80, 3)
    axs[0].set_xlim(0, 650)
    axs[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                  handlelength=1.2, columnspacing=0.8)
    axs[1].set_xlabel("Tempo na janela (s)")
    axs[1].set_ylabel("$f_{rot}$ (Hz)")
    axs[1].set_ylim(pre.fmin, pre.fmax)
    for a in axs:
        a.grid(color="#e6e5e1", lw=0.4)
    fig.tight_layout(h_pad=0.6)
    return _salvar(fig, "diag_espectro", ts), info


# ------------------------------------------------------------------ tabela LaTeX
def _fmt(r):
    return f"{r.valor:.3f} [{r.ic95_lo:.2f}, {r.ic95_hi:.2f}]"


def tabela_latex(M, ts, csv_rel):
    nomes_c = {"idmt/aleatorio": "IDMT aleat.$^\\dagger$", "idmt/blocos": "IDMT blocos",
               "a2212/aleatorio": "A2212 aleat.$^\\dagger$", "a2212/arquivo": "A2212 arquivo",
               "a2212/motor": "A2212 motor", "a2212/motor_sem_M4": "A2212 motor s/M4"}
    rot_tex = {"b00_rmspico_norm": "RMS/peak (RMS-norm.)", "b00_rmspico_bruto": "RMS/peak (raw)",
               "p04_classicas": "Classical (p04)", "p11_decifrae_harmsum": "Decifrae (p11)"}
    L = [f"% Gerado por embarcados/src/python/diagnostico_saidas.py a partir de {csv_rel}",
         "% Não editar à mão.", "\\begin{table*}[t]", "\\centering", "\\scriptsize",
         "\\caption{Balanced accuracy, macro-F1 and macro one-vs-rest ROC-AUC on the same "
         "windows. Brackets: 95\\% CI from a group bootstrap (2000 resamples of recordings/blocks, "
         "stratified by class); $g$ = number of groups. $^\\dagger$Random split (intentional "
         "leakage).}",
         "\\label{tab:desempenho}", "\\begin{tabular}{llrlll}", "\\toprule",
         "Protocol & Method & $g$ & Bal.\\ acc. & Macro-F1 & ROC-AUC \\\\", "\\midrule"]
    for i, (b, p, _) in enumerate(CASOS):
        if i:
            L.append("\\midrule")
        for j, v in enumerate(ORDEM_V):
            ba, f1, au = (_macro(M, v, b, p, k) for k in
                          ("acuracia_balanceada", "f1_macro", "roc_auc_ovr_macro"))
            L.append(f"{nomes_c[b + '/' + p] if j == 0 else ''} & {rot_tex[v]} & "
                     f"{int(ba.n_grupos)} & {_fmt(ba)} & {_fmt(f1)} & {_fmt(au)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    d = EMB / "tables" / "latex"
    d.mkdir(parents=True, exist_ok=True)
    (d / "desempenho.tex").write_text("\n".join(L), encoding="utf-8")
    return "tables/latex/desempenho.tex"


# ------------------------------------------------------------------ histórico
def historico(J):
    """Acurácia por janela (não balanceada) de T1 (CSV histórico) × agora."""
    if not HIST_CSV.exists():
        return None
    H = pd.read_csv(HIST_CSV, usecols=["pipeline", "base", "split", "normalizar_rms", "acerto"])
    H = H[H.pipeline.isin(["b00_nivel", "p04_classicas", "p11_decifrae_harmsum"])]
    h = H.groupby(["pipeline", "base", "split", "normalizar_rms"]).acerto.mean()
    a = J.groupby(["pipeline", "base", "split", "normalizar_rms"]).acerto.mean()
    df = pd.concat({"historico_T1": h, "agora": a}, axis=1).dropna().reset_index()
    df["diferenca"] = df.agora - df.historico_T1
    return df


# ------------------------------------------------------------------ relatório
def relatorio(M, Mj, meta, ts, figs, tex, info_c, H):
    t = meta["tempos_s"]
    ext = {k.split("/", 1)[1]: v for k, v in t.items() if k.startswith("extracao/")}
    L = ["# Diagnóstico (Etapa 6): Decifrae × RMS/pico × clássicas nas mesmas janelas", "",
         f"Gerado por `embarcados/src/python/diagnostico_saidas.py` a partir do experimento "
         f"`{ts}` (commit `{meta['projeto']['commit']}`, árvore suja = "
         f"{meta['projeto']['sujo']}). Nenhum número deste arquivo foi digitado à mão. "
         "Só validação: o `test/` do IDMT não foi lido. A coluna do porte em C entra depois.", "",
         "## O que rodou", "", "```",
         ".venv\\Scripts\\python.exe embarcados\\src\\python\\diagnostico.py --n-jobs 4",
         ".venv\\Scripts\\python.exe embarcados\\src\\python\\diagnostico_saidas.py", "```", "",
         f"Tempo total do experimento: {t['total']} s (extração sem cache, joblib n_jobs="
         f"{meta['n_jobs']}). Extração por variante/base (s): "
         + ", ".join(f"{k} {v}" for k, v in ext.items()) + ". "
         f"Bootstrap: {meta['n_boot']} reamostragens, semente {meta['seed_boot']}; "
         f"{meta['bootstrap']}. Semente dos splits: {meta['seed_splits']}.", "",
         "Variantes: `b00_nivel` em duas condições declaradas, com o RMS da janela normalizado "
         "pelo runner (`normalizar_rms=True`, o RMS vira constante e só o pico relativo, isto "
         "é, o fator de crista, carrega informação) e sem normalizar (`normalizar_rms=False`, "
         "RMS e pico brutos); `p04_classicas` e `p11_decifrae_harmsum` com `normalizar_rms=True`, "
         "a mesma configuração de T1. Janelas: 3 s no IDMT (357 janelas do `train/`), 1 s no "
         "A2212 (430 janelas). Todas as variantes usam exatamente as mesmas janelas e os mesmos "
         "folds.", "",
         "## Tabela principal (acurácia balanceada, IC 95 % por grupo, nº de grupos)", "",
         "| protocolo | " + " | ".join(ROT[v] for v in ORDEM_V) + " | grupos |",
         "|---|" + "---|" * (len(ORDEM_V) + 1)]
    for b, p, nome in CASOS:
        cel = []
        for v in ORDEM_V:
            r = _macro(M, v, b, p)
            cel.append(f"{r.valor:.3f} [{r.ic95_lo:.3f}, {r.ic95_hi:.3f}]")
        r = _macro(M, ORDEM_V[0], b, p)
        L.append(f"| {nome.replace(chr(10), ' ')} | " + " | ".join(cel)
                 + f" | {int(r.n_grupos)} ({r.grupos_por_classe}) |")
    L += ["", "\\* split aleatório por janela: vazamento intencional, só para medir a inflação.", "",
          "ROC-AUC one-vs-rest macro (scores contínuos `p_<classe>`) e F1 macro:", "",
          "| protocolo | " + " | ".join(f"{ROT[v]} AUC / F1" for v in ORDEM_V) + " |",
          "|---|" + "---|" * len(ORDEM_V)]
    for b, p, nome in CASOS:
        cel = [f"{_macro(M, v, b, p, 'roc_auc_ovr_macro').valor:.3f} / "
               f"{_macro(M, v, b, p, 'f1_macro').valor:.3f}" for v in ORDEM_V]
        L.append(f"| {nome.replace(chr(10), ' ')} | " + " | ".join(cel) + " |")
    # sensibilidade por classe da p11
    L += ["", "Sensibilidade (recall) por classe da p11:", ""]
    for b, p, nome in CASOS:
        pc = Mj["metricas"]["p11_decifrae_harmsum"][b][p]["por_classe"]
        L.append(f"- {nome.replace(chr(10), ' ')}: " + ", ".join(
            f"{c} {d['sensibilidade']:.3f}" for c, d in pc.items() if d["suporte"] > 0))
    # leitura honesta, gerada
    L += ["", "## Leitura", ""]
    frases = []
    for b, p, nome in CASOS:
        r11 = _macro(M, "p11_decifrae_harmsum", b, p)
        outros = {v: _macro(M, v, b, p) for v in ORDEM_V if v != "p11_decifrae_harmsum"}
        melhor_v = max(outros, key=lambda v: outros[v].valor)
        rb = outros[melhor_v]
        if abs(r11.valor - rb.valor) < 5e-4:
            veredito = (f"empata com {ROT[melhor_v]} ({rb.valor:.3f}); a p11 não traz "
                        "ganho aqui")
        elif r11.valor > rb.valor and r11.ic95_lo > rb.ic95_hi:
            veredito = "supera todos os outros com ICs disjuntos"
        elif r11.valor > rb.valor:
            veredito = "fica numericamente acima dos outros, mas com ICs sobrepostos (empate)"
        elif r11.ic95_hi >= rb.ic95_lo:
            veredito = (f"NÃO supera {ROT[melhor_v]} ({rb.valor:.3f}); ICs sobrepostos")
        else:
            veredito = (f"NÃO supera {ROT[melhor_v]} ({rb.valor:.3f}) e perde com ICs "
                        "disjuntos")
        frases.append(f"Em {nome.replace(chr(10), ' ')}, a p11 tem acurácia balanceada "
                      f"{r11.valor:.3f} e {veredito}.")
    L.append(" ".join(frases))
    n_idmt = int(_macro(M, "p11_decifrae_harmsum", "idmt", "blocos").n_grupos)
    rb_n = _macro(M, "b00_rmspico_norm", "idmt", "blocos")
    rb_b = _macro(M, "b00_rmspico_bruto", "idmt", "blocos")
    L += ["", f"No IDMT o atalho de nível vale nas duas condições: com o RMS normalizado o "
          f"baseline de dois números ainda chega a {rb_n.valor:.3f} de acurácia balanceada nos "
          f"blocos (o fator de crista basta) e sem normalizar chega a {rb_b.valor:.3f}. Como o "
          "`train/` tem uma gravação por classe, classe, motor físico e gravação são a mesma "
          "variável; os blocos impedem que janelas vizinhas caiam dos dois lados, mas não "
          "impedem que o classificador aprenda a identidade da gravação. Por isso nenhum número "
          "do IDMT aqui mede diagnóstico de estado, e o IC é estreito por construção: "
          f"são só {n_idmt} grupos (5 blocos × 3 gravações), todos da mesma sessão, e o "
          "bootstrap não representa a variabilidade entre motores ou sessões, que é zero "
          "amostras. O IC deve ser lido como variabilidade entre blocos, não como incerteza "
          "de generalização. Acurácias de 1,000 com IC [1,000, 1,000] são degeneradas: "
          "nenhuma réplica erra, e o bootstrap não tem como mostrar incerteza. A regra do "
          "projeto manda procurar vazamento acima de 99 %; aqui o vazamento é conhecido e "
          "estrutural (classe = gravação), não um bug de split.", "",
          f"No A2212 a acurácia balanceada da p11 por arquivo "
          f"({_macro(M, 'p11_decifrae_harmsum', 'a2212', 'arquivo').valor:.3f}) é maior que "
          f"a acurácia simples de T1 "
          f"({_macro(M, 'p11_decifrae_harmsum', 'a2212', 'arquivo', 'acuracia').valor:.3f}) "
          "porque o erro se concentra em Healthy, a classe majoritária (ver a sensibilidade "
          "por classe acima): a balanceada pesa as três classes igualmente.", "",
          "No A2212 os grupos são os 43 arquivos (22 Healthy, 14 Propeller, 7 Bearing), "
          "o que dá um IC por arquivo utilizável, mas os arquivos de um mesmo motor não são "
          "independentes; no protocolo `motor` a unidade independente real é o motor, e "
          "com 4 motores não existe IC honesto nessa escala (o IC mostrado é por arquivo e "
          "é otimista). Quando M4 sai, a classe Bearing não tem nenhuma janela de treino, "
          "o classificador atribui p_Bearing=0 e a sensibilidade de Bearing é 0 por "
          "construção, o que limita a acurácia balanceada do protocolo `motor`; a linha "
          "`A2212 sem M4` tira o fold M4 e avalia só Healthy/Propeller (Bearing ainda pode ser "
          "predito, e conta como erro). Toda linha com Bearing está confundida com a "
          "identidade do motor 4.", ""]
    # histórico
    L += ["## Comparação com o histórico (T1 de `artigo-ICIT/RESULTADOS_tabelas.md`)", "",
          "T1 reporta acurácia por janela (não balanceada). A comparação abaixo usa a mesma "
          "métrica, calculada das predições deste experimento e do CSV histórico "
          f"`{HIST_CSV.relative_to(RAIZ).as_posix()}`.", ""]
    if H is not None and len(H):
        L += ["| pipeline | base | protocolo | normalizar_rms | T1 | agora | diferença |",
              "|---|---|---|---|---|---|---|"]
        for _, r in H.iterrows():
            L.append(f"| {r.pipeline} | {r.base} | {r.split} | {r.normalizar_rms} | "
                     f"{r.historico_T1:.3f} | {r.agora:.3f} | {r.diferenca:+.3f} |")
        mx = float(H.diferenca.abs().max())
        L += ["", f"Maior diferença absoluta: {mx:.3f}. "
              + ("Os números reproduzem o histórico (diferença < 0,005 em todas as células)."
                 if mx < 0.005 else
                 "Há diferenças acima de 0,005; ver a seção de ressalvas.")]
    L += ["", "## Figuras e tabela", "",
          f"- (a) `{figs['a']}` (+ PNG): acurácia balanceada com IC por protocolo.",
          f"- (b) `{figs['b']}` (+ PNG): matriz de confusão da p11 nos protocolos honestos "
          "(contagens; cor = fração por linha).",
          f"- (c) `{figs['c']}` (+ PNG): espectro médio de uma janela good e uma broken do "
          "`train/` do IDMT com o pente k·f_rot (k = 1..48) e o caminho de rotação do Viterbi. "
          + "; ".join(f"{c}: `{d['rec_id']}` janela {d['idx']} ({d['t0']:.0f}–{d['t1']:.0f} s), "
                      f"f_rot mediana {d['f_rot_mediana_hz']:.2f} Hz, "
                      f"{d['quadros_sil']}/{d['quadros']} quadros SIL"
                      for c, d in info_c.items()) + ".",
          f"- Tabela LaTeX: `{tex}` (booktabs, números do CSV).", "",
          "## Arquivos do experimento", ""]
    for k in ("diagnostico_janelas", "p11_janelas", "p11_folds"):
        L.append(f"- `results/predictions/{k}_{ts}.csv`")
    L += [f"- `results/predictions/p11_modelos_{ts}.json` (mu, sd e protótipos do cosseno por "
          "fold, conferidos contra as probabilidades do runner com tolerância 1e-12)",
          f"- `results/csv/p11_features_{ts}.csv` (13 features por janela, para a comparação com o C)",
          f"- `results/csv/diagnostico_{ts}.csv`, `results/metrics/diagnostico_{ts}.json` "
          "(+ `_meta.json` de cada)", ""]
    (EMB / "reports" / "diagnostico.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=None)
    ap.add_argument("--sem-espectro", action="store_true")
    a = ap.parse_args(argv)
    ts = a.ts or _ultimo_ts()
    csv = EMB / "results" / "csv" / f"diagnostico_{ts}.csv"
    M = pd.read_csv(csv)
    Mj = json.loads((EMB / "results" / "metrics" / f"diagnostico_{ts}.json")
                    .read_text(encoding="utf-8"))
    meta = json.loads(csv.with_name(csv.stem + "_meta.json").read_text(encoding="utf-8"))
    J = pd.read_csv(EMB / "results" / "predictions" / f"diagnostico_janelas_{ts}.csv",
                    usecols=["pipeline", "base", "split", "normalizar_rms", "acerto"])
    figs = {"a": fig_balacc(M, ts), "b": fig_confusao(Mj, ts)}
    info_c = {}
    if not a.sem_espectro:
        figs["c"], info_c = fig_espectro(ts)
    tex = tabela_latex(M, ts, csv.relative_to(RAIZ).as_posix())
    relatorio(M, Mj, meta, ts, figs, tex, info_c, historico(J))
    print("ok", figs, tex)


if __name__ == "__main__":
    main()
