"""Tabelas LaTeX do artigo (booktabs), geradas a partir de paper/numeros.json.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_artigo_tabelas.py

Agente E (Etapa 7). Rodar DEPOIS de gerar_artigo_numeros.py. Nenhum número é digitado aqui:
todo valor sai de numeros.json (que por sua vez sai dos CSV/JSON de results/ e do código da
referência). Texto descritivo (nomes de protocolo, o que cada um permite concluir) é redação.

Saídas em tables/latex/:
  bases_protocolos.tex      bases e o que cada protocolo permite concluir
  parametros_features.tex   parâmetros da p11 (lidos do código) e as 13 features
  recursos.tex              recursos por categoria: estimativa | PC | build | ESP32 (p11_c_lowmem)
  desempenho_compacto.tex   acurácia balanceada, 4 métodos × 6 protocolos (versão de 1 coluna)
  equivalencia_compacta.tex veredito por grupo de etapas (versão de 1 coluna)
`desempenho.tex` e `equivalencia.tex` NÃO são gerados aqui (diagnostico_saidas.py e
resumo_equivalencia.py); o main.tex os reaproveita.
"""
from __future__ import annotations

import pandas as pd

import gerar_artigo_comum as C

CAB = ("% Gerado por embarcados/src/python/gerar_artigo_tabelas.py a partir de "
       "embarcados/paper/numeros.json.\n% NÃO EDITAR À MÃO.\n")


def T(n, k):
    return n[k]["tex"]


def bases_protocolos(n):
    L = [CAB, r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}",
         (r"\caption{Datasets and evaluation protocols. IDMT: \texttt{train/} only, "
          f"{T(n,'VDataIdmtRecordings')} recordings of {T(n,'VDataIdmtMotors')} motors, "
          f"{T(n,'VDataIdmtWindows')} windows of {T(n,'VParSegIdmt')}\\,s. A2212: "
          f"{T(n,'VDataBldcRecordings')} files of {T(n,'VDataBldcMotors')} motors "
          f"({T(n,'VDataBldcFilesHealthy')} Healthy, {T(n,'VDataBldcFilesPropeller')} Propeller, "
          f"{T(n,'VDataBldcFilesBearing')} Bearing), {T(n,'VDataBldcWindows')} windows of "
          f"{T(n,'VParSegBldc')}\\,s. $g$: groups used by the bootstrap. "
          r"$^\dagger$Intentional leakage, reported only to measure inflation.}"),
         r"\label{tab:protocols}",
         r"\begin{tabular}{@{}p{1.55cm}p{1.35cm}r p{3.75cm}@{}}", r"\toprule",
         r"Protocol & Split unit & $g$ & What it can / cannot support \\", r"\midrule"]
    rows = [
        ("IDMT random$^\\dagger$", "window", "VDataIdmtRandGroups",
         "Upper bound only: neighbouring windows of one recording on both sides."),
        ("IDMT blocks", "contiguous block", "VDataIdmtBlkGroups",
         "One recording per class: class = motor = recording. Measures recording identity, "
         "not operating state."),
        ("A2212 random$^\\dagger$", "window", "VDataBldcRandGroups",
         "Upper bound only: windows of one file on both sides."),
        ("A2212 file", "file", "VDataBldcFileGroups",
         "State of a \\emph{known} motor (same motors in training and test). Bearing exists "
         "only on motor M4."),
        ("A2212 motor-out", "motor", "VDataBldcMotorGroups",
         "Unseen motor. With M4 held out, Bearing has no training data (recall "
         f"{T(n, 'VDiagRecDecBldcMotorBearing')} by construction). Only "
         f"{T(n, 'VDataBldcMotors')} motors: no motor-level CI."),
        ("A2212 motor-out w/o M4", "motor (M1--M3)", "VDataBldcNoMfourGroups",
         "Healthy vs.\\ Propeller on an unseen motor; "
         f"{n['VDataBldcMotors']['valor'] - 1} folds."),
    ]
    for a, b, g, c in rows:
        L.append(f"{a} & {b} & {T(n, g)} & {c} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def parametros_features(n):
    t = lambda k: T(n, k)  # noqa: E731
    par = [
        ("Sampling rate; window", f"{t('VParSrKhz')}\\,kHz; {t('VParSegIdmt')}\\,s (IDMT), "
                                  f"{t('VParSegBldc')}\\,s (A2212); RMS $\\rightarrow$ {t('VParRmsTarget')}"),
        ("STFT", f"symmetric Hann, $N$={t('VParNfft')}, hop {t('VParHop')}, no centring; "
                 f"{t('VParFramesIdmt')} / {t('VParFramesBldc')} frames"),
        ("Rotation grid", f"{t('VParFmin')}--{t('VParFmax')}\\,Hz, {t('VParStepCents')}-cent step, "
                          f"$J$={t('VParJ')} states + silence"),
        ("Harmonic sum", f"$K$={t('VParNharm')} harmonics, unit weights, linear interpolation"),
        ("Emission cost", f"$-\\log({t('VParWEnergy')}\\,\\tilde e + {t('VParEmisEps').strip('$')})$; "
                          "silence from frame energy"),
        ("Viterbi", f"step {t('VParWStep')}/semitone, jumps $\\le${t('VParMaxJump')} states, "
                    f"octave $\\pm${t('VParOctStates')} states (cost {t('VParWOct')}), "
                    f"silence {t('VParWSil')}"),
        ("Envelope", f"$|\\mathcal{{H}}(y)|$, Butterworth {t('VParButterOrder')}, "
                     f"{t('VParButterFc')}\\,Hz (causal \\texttt{{sosfilt}}), $\\downarrow${t('VParDecim')}, "
                     f"Welch; peak in {t('VParModLo')}--{t('VParModHi')}\\,Hz"),
        ("Classifier", f"z-score, cosine to class means, softmax $T$={t('VParTempCos')}"),
        ("Online HMM", f"change penalty swept ({t('VParNChangePen')} values, {t('VParCpMin')}--"
                       f"{t('VParCpMax')}), $\\log p_N$={t('VParNLogprob')}"),
    ]
    g = [(t(f"VParGroup{x}Lo"), t(f"VParGroup{x}Hi")) for x in ("One", "Two", "Three", "Four")]
    feats = [
        "median $f_{rot}$", "IQR of $f_{rot}$", "silent-frame fraction",
        "mean normalised saliency on the path",
        f"harmonic ratio ($k f_{{rot}}\\pm1$ bin / {t('VParBandLo')}--{t('VParBandHi')}\\,Hz)",
        *[f"energy of orders {a}--{b} (dB)" for a, b in g],
        "envelope-modulation peak frequency", "its prominence (dB)",
        "spectral centroid", "spectral flatness"]
    L = [CAB, r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         (r"\caption{Parameters of the p11 reference, read from the Python code "
          r"(\texttt{PreDecifrae}, \texttt{rotacao}, \texttt{\_envelope\_mod}), and the "
          f"{t('VParNFeat')} features in the order of the reference.}}"),
         r"\label{tab:params}", r"\begin{tabular}{@{}p{1.55cm}p{6.6cm}@{}}", r"\toprule"]
    for a, b in par:
        L.append(f"{a} & {b} \\\\")
    L.append(r"\midrule")
    L.append(r"Features & " + "; ".join(f"({i}) {f}" for i, f in enumerate(feats)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def recursos(n):
    t = lambda k: T(n, k)  # noqa: E731
    na = "--"
    W = f"{t('VParSegIdmt')}\\,s / {t('VParSegBldc')}\\,s"
    ref, lm = r"\texttt{p11\_c\_ref}", r"\texttt{" + t("VChipVariant") + "}"
    rows = [
        ("Variant", f"{ref} arena", f"{ref}", f"{ref}; {lm}", f"{lm}"),
        (f"Latency, C float32, median [p95] (ms), {W}", na,
         f"{t('VPcCIdmtMed')} [{t('VPcCIdmtPninetyfive')}] / {t('VPcCBldcMed')} [{t('VPcCBldcPninetyfive')}]",
         na, f"{t('VChipLatIdmtMed')} / {t('VChipLatBldcMed')} [{t('VChipLatBldcPninetyfive')}]"),
        (f"Latency, Python reference, median (ms), {W}", na,
         f"{t('VPcPyIdmtMed')} / {t('VPcPyBldcMed')}", na, na),
        (f"Real-time factor (C, median), {W}", na, f"{t('VPcCIdmtRtf')} / {t('VPcCBldcRtf')}", na,
         f"{t('VChipRtfIdmt')} / {t('VChipRtfBldc')}"),
        (f"Share of STFT + harmonic sum (\\%), {t('VParSegBldc')}\\,s", na,
         f"{t('VPcCBldcTrackPct')}", na, f"{t('VChipBldcTrackPct')}"),
        (f"Working memory (KiB), {W}", f"{t('VEstWsFloatThreeKib')} / {t('VEstWsFloatOneKib')}",
         na, na, f"{t('VChipWsIdmtKib')} / {t('VChipWsBldcKib')} (pool peak)"),
        ("Internal RAM handed to the pool (KiB)", na, na, na, f"{t('VChipPoolKib')}"),
        (f"Min.\\ free internal heap (KiB), {t('VParSegBldc')}\\,s", na, na, na,
         f"{t('VChipHeapIntBldcKib')}"),
        (f"Stack headroom (B), {t('VParSegBldc')}\\,s", na, na, na, f"{t('VChipStackBldc')}"),
        (f"App binary (KiB): ESP32, -S3 ({ref}); ESP32 ({lm})", na, na,
         f"{t('VBuildEspBinKb')}, {t('VBuildEspsBinKb')}; {t('VBuildLmBinKb')}", na),
        (f"Static DRAM, IRAM used / region (KiB), ESP32, {lm}", na, na,
         f"{t('VBuildLmDramKb')}/{t('VBuildLmDramTotalKb')}, {t('VBuildLmIramKb')}/{t('VBuildLmIramTotalKb')}",
         na),
    ]
    L = [CAB, r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         (r"\caption{Resources, separated by evidence category. (1) analytical estimate "
          r"(\texttt{p11\_host ws}, same layout function as the arena of the reference variant); "
          r"(2) measured on the PC "
          f"({t('VPcOs')}, {t('VPcCompiler')}; {t('VPcNIdmt')} timings per dataset cycling "
          f"over {t('VPcDistinctIdmt')} distinct windows after {t('VPcWarmup')} warm-up runs; the C "
          r"executable runs as a new process per window, so warm-up does not warm it, and its time "
          r"excludes the workspace initialisation, which the Python timing cannot exclude); "
          r"(3) firmware build, \texttt{idf.py size} (ESP-IDF "
          f"{t('VBuildIdf')}), static only; (4) measured on the {t('VChipModel')} "
          f"({t('VChipCpuMhz')}\\,MHz, {t('VChipPsramMb')}\\,MB PSRAM; {t('VChipLatBldcN')} timed "
          f"windows cycling over {t('VChipLatDistinct')} distinct ones after {t('VChipLatWarmup')} "
          r"warm-up runs). Only column (4) is a measurement on the microcontroller, and it uses "
          f"the named low-memory variant {lm}; the {t('VParSegIdmt')}-s IDMT window did not fit.}}"),
         r"\label{tab:resources}", r"\begin{tabular}{@{}lllll@{}}", r"\toprule",
         r"Quantity & (1) Estimate & (2) PC & (3) Build & (4) ESP32 \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def desempenho_compacto(n):
    t = lambda k: T(n, k)  # noqa: E731
    prots = [("IdmtRand", "IDMT random$^\\dagger$"), ("IdmtBlk", "IDMT blocks"),
             ("BldcRand", "A2212 random$^\\dagger$"), ("BldcFile", "A2212 file"),
             ("BldcMotor", "A2212 motor-out"), ("BldcNoMfour", "A2212 w/o M4")]
    L = [CAB, r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         (r"\caption{Balanced accuracy on the same windows (validation only); $g$ = groups. "
          f"{t('VDataCiLevel')}\\% CIs from a group bootstrap ({t('VDataNBoot')} resamples) are "
          r"shown in Fig.~\ref{fig:balacc}. $^\dagger$Intentional leakage. IDMT: class = motor = "
          r"recording. A2212: Bearing exists only on motor M4 (confounded with motor identity); "
          r"with M4 held out, Bearing recall is zero by construction.}"),
         r"\label{tab:desempenho-compacto}", r"\begin{tabular}{@{}lrcccc@{}}", r"\toprule",
         r"Protocol & $g$ & RMS/pk (norm.) & RMS/pk (raw) & Classical & p11 \\", r"\midrule"]
    for k, nome in prots:
        g = t(f"VData{k}Groups")
        vals = [t(f"VDiagBal{v}{k}") for v in ("RmsNorm", "RmsRaw", "Cls", "Dec")]
        L.append(f"{nome} & {g} & " + " & ".join(vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def equivalencia_compacta(n):
    fres = C.mais_recente(C.RES / "csv", "equivalencia_resumo_por_build_*.csv")
    S = pd.read_csv(fres)
    grupos = [("Front end", ["y_prep", "stft_S", "grade", "sal", "e_norm", "custo"]),
              ("Rotation Viterbi", ["vit_path", "vit_score", "f_rot"]),
              ("Spectrum, envelope", ["P_mean", "env_abs", "env_dec", "welch_f", "welch_P", "env_mod"]),
              (f"{n['VParNFeat']['tex']} features", ["features"]), ("z-score", ["z"]),
              ("Cosine, softmax, decision", ["cos", "probs", "decisao"]),
              ("Sequences (HMM)", ["hmm_log_em", "argmax", "hmm_viterbi", "hmm_online_alpha",
                                   "hmm_online", "absorb"])]
    assert sum(len(g) for _, g in grupos) == len(S[S.build == "double"])
    builds = [("double", "C double"), ("float", "C float32"), ("float-sosd", "float32+SOS64")]
    L = [CAB.replace("embarcados/paper/numeros.json",
                     C.rel(fres) + " (veredito por etapa)"),
         r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Stage group (\#stages) & " + " & ".join(b for _, b in builds) + r" \\", r"\midrule"]
    for nome, et in grupos:
        cel = []
        for b, _ in builds:
            # leitura conservadora (auditoria F, M1): PASS_EMPATE conta como falha
            s = S[(S.build == b) & (S.etapa.isin(et))]
            ruim = s.fail + s.pass_empate
            ok = int((ruim == 0).sum())
            c = f"{ok}/{len(s)}"
            if ruim.sum():
                c += f" (FAIL in {int(ruim.max())}/{int(s[ruim > 0].n.iloc[0])})"
            cel.append(c)
        L.append(f"{nome} ({len(et)}) & " + " & ".join(cel) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n", fres


def main():
    n = C.ler_json(C.PAPER / "numeros.json")["numeros"]
    C.TABS.mkdir(parents=True, exist_ok=True)
    saidas = {"bases_protocolos.tex": bases_protocolos(n),
              "parametros_features.tex": parametros_features(n),
              "recursos.tex": recursos(n),
              "desempenho_compacto.tex": desempenho_compacto(n)}
    eq, fres = equivalencia_compacta(n)
    saidas["equivalencia_compacta.tex"] = eq
    for nome, txt in saidas.items():
        (C.TABS / nome).write_text(txt, encoding="utf-8")
    print("tabelas:", ", ".join(saidas))


if __name__ == "__main__":
    main()
