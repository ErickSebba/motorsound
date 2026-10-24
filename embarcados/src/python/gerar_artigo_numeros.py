"""Gera paper/numeros.tex e paper/numeros.json: uma macro por número citado no artigo.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_artigo_numeros.py

Agente E (Etapa 7). Nenhum número é digitado aqui: cada valor é lido de um CSV/JSON de
`embarcados/results/`, do código Python da referência (import), da saída de
`p11_host ws <n>` (estimativa analítica) ou do `size.txt` do `idf.py size` (build do firmware).

Categorias (prefixo da macro):
  \\VPar…   parâmetro lido do código da referência
  \\VData…  contagens das bases/protocolos (predições do experimento)
  \\VDiag…  classificação (validação, mesmas janelas)          \\VRot… rotação estimada
  \\VEq…    equivalência Python × C no PC
  \\VEst…   1 · estimativa analítica          \\VPc…    2 · medição no PC
  \\VBuild… 3 · build do firmware (idf.py size)
  \\VChip…  4 · medição no ESP32 físico — \\textbf{[pending]} se a bancada não rodou.
"""
from __future__ import annotations

import inspect
import json
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import gerar_artigo_comum as C


class Registro:
    def __init__(self):
        self.n = {}

    def add(self, chave, valor, tex, fonte, coluna, cat, nota=""):
        C.nome_macro(chave)
        if chave in self.n:
            raise KeyError(f"macro duplicada: {chave}")
        pend = valor is None
        self.n[chave] = {
            "valor": valor if not isinstance(valor, (np.floating, np.integer)) else valor.item(),
            "tex": C.PENDENTE_TEX if pend else tex,
            "pt": C.tex_para_pt(None if pend else tex),
            "status": "pending" if pend else "ok",
            "fonte": [C.rel(f) for f in (fonte if isinstance(fonte, (list, tuple)) else [fonte])],
            "coluna": coluna, "categoria": cat, "nota": nota}

    def set_tex(self, chave, tex):
        self.n[chave]["tex"] = tex
        self.n[chave]["pt"] = C.tex_para_pt(tex)

    def num(self, chave, valor, fonte, coluna, cat, casas=3, sci=False, nota=""):
        tex = None if valor is None else C.fmt(valor, casas, sci)
        if sci and tex is not None:
            tex = r"\ensuremath{" + tex + "}"
        self.add(chave, valor, tex, fonte, coluna, cat, nota)

    def txt(self, chave, valor, fonte, coluna, cat, nota=""):
        tex = None if valor is None else str(valor).replace("_", r"\_")
        self.add(chave, valor, tex, fonte, coluna, cat, nota)


# ------------------------------------------------------------------ 1. código da referência
def parametros(R):
    from motorsound import rotacao
    from motorsound.config import AVAL_A2212, AVAL_IDMT, PostConfig
    from pipelines import _decifrae as dd
    from pipelines import _classicas as cl
    from pipelines import p11_decifrae_harmsum as p11

    f_dd = RAIZ_PY / "pipelines" / "_decifrae.py"
    f_rot = RAIZ_PY / "motorsound" / "rotacao.py"
    f_cfg = RAIZ_PY / "motorsound" / "config.py"
    pre, post = p11.PRE, p11.POST
    R.num("VParSr", pre.sr, f_dd, "PreDecifrae.sr", "codigo", 0)
    R.num("VParSrKhz", pre.sr / 1000, f_dd, "PreDecifrae.sr/1000", "codigo", 0)
    R.num("VParNfft", pre.n_fft, f_dd, "PreDecifrae.n_fft", "codigo", 0)
    R.num("VParHop", pre.hop, f_dd, "PreDecifrae.hop", "codigo", 0)
    R.num("VParFmin", pre.fmin, f_dd, "PreDecifrae.fmin", "codigo", 0)
    R.num("VParFmax", pre.fmax, f_dd, "PreDecifrae.fmax", "codigo", 0)
    R.num("VParStepCents", pre.passo_cents, f_dd, "PreDecifrae.passo_cents", "codigo", 0)
    J = len(rotacao.grade(pre.fmin, pre.fmax, pre.passo_cents))
    R.num("VParJ", J, f_rot, "len(rotacao.grade(fmin,fmax,passo))", "codigo", 0)
    R.num("VParNbins", pre.n_fft // 2 + 1, f_rot, "n_fft//2+1 (rfft)", "codigo", 0)
    R.num("VParDf", pre.sr / pre.n_fft, f_rot, "sr/n_fft", "codigo", 2)
    R.num("VParNharm", dd.N_HARM, f_dd, "N_HARM", "codigo", 0)
    g = dd.GRUPOS
    for i, (a, b) in enumerate(g):
        nome = ["One", "Two", "Three", "Four"][i]
        R.num(f"VParGroup{nome}Lo", a, f_dd, f"GRUPOS[{i}][0]", "codigo", 0)
        R.num(f"VParGroup{nome}Hi", b, f_dd, f"GRUPOS[{i}][1]", "codigo", 0)
    R.num("VParMaxJump", pre.max_salto, f_dd, "PreDecifrae.max_salto", "codigo", 0)
    R.num("VParMaxJumpCents", pre.max_salto * pre.passo_cents, f_dd,
          "max_salto*passo_cents", "codigo", 0)
    src_vit = inspect.getsource(rotacao.viterbi_rotacao)
    m = re.search(r"k_oct = int\(round\((\d+) / passo_cents\)\)", src_vit)
    R.num("VParOctStates", int(round(int(m.group(1)) / pre.passo_cents)), f_rot,
          "viterbi_rotacao: k_oct = round(1200/passo)", "codigo", 0)
    R.num("VParWStep", rotacao.VIT_W_STEP, f_rot, "VIT_W_STEP", "codigo", 2)
    R.num("VParWOct", rotacao.VIT_W_OCTAVE, f_rot, "VIT_W_OCTAVE", "codigo", 1)
    R.num("VParWSil", rotacao.VIT_W_SIL, f_rot, "VIT_W_SIL", "codigo", 1)
    src_em = inspect.getsource(rotacao.emissao_baixo)
    m = re.search(r"(\d\.\d+)\s*\*\s*e_norm", src_em)
    R.num("VParWEnergy", float(m.group(1)), f_rot, "emissao_baixo: peso de e_norm", "codigo", 2)
    m = re.search(r"\+\s*(1e-\d+)\)", src_em)
    R.num("VParEmisEps", float(m.group(1)), f_rot, "emissao_baixo: eps do -log", "codigo", 0,
          sci=False, nota="1e-3")
    R.set_tex("VParEmisEps", r"\ensuremath{10^{" + str(int(np.log10(float(m.group(1))))) + "}}")
    R.num("VParTempCos", post.temp_cos, f_dd, "PostDecifrae.temp_cos", "codigo", 1)
    src_env = inspect.getsource(dd._envelope_mod)
    m = re.search(r"butter\((\d+),\s*(\d+)", src_env)
    R.num("VParButterOrder", int(m.group(1)), f_dd, "_envelope_mod: butter(ordem, fc)", "codigo", 0)
    R.num("VParButterFc", int(m.group(2)), f_dd, "_envelope_mod: butter(ordem, fc)", "codigo", 0)
    m = re.search(r"\(f >= (\d+)\) & \(f <= (\d+)\)", src_env)
    R.num("VParModLo", int(m.group(1)), f_dd, "_envelope_mod: banda", "codigo", 0)
    R.num("VParModHi", int(m.group(2)), f_dd, "_envelope_mod: banda", "codigo", 0)
    m = re.search(r"sr // (\d+)", src_env)
    R.num("VParEnvRateHz", int(m.group(1)), f_dd, "_envelope_mod: [:: sr//1000]", "codigo", 0)
    R.num("VParDecim", pre.sr // int(m.group(1)), f_dd, "sr // 1000", "codigo", 0)
    src_ex = inspect.getsource(dd.extrair)
    m = re.search(r"\(f >= (\d+)\) & \(f <= (\d+)\)", src_ex)
    R.num("VParBandLo", int(m.group(1)), f_dd, "extrair: banda das features", "codigo", 0)
    R.num("VParBandHi", int(m.group(2)), f_dd, "extrair: banda das features", "codigo", 0)
    R.num("VParSegIdmt", AVAL_IDMT.seg_s, f_cfg, "AVAL_IDMT.seg_s", "codigo", 0)
    R.num("VParSegBldc", AVAL_A2212.seg_s, f_cfg, "AVAL_A2212.seg_s", "codigo", 0)
    R.num("VParRmsTarget", AVAL_IDMT.rms_alvo, f_cfg, "AvaliacaoConfig.rms_alvo", "codigo", 1)
    R.num("VParFramesIdmt", 1 + (int(AVAL_IDMT.seg_s * pre.sr) - pre.n_fft) // pre.hop, f_rot,
          "1+(N-n_fft)//hop (stft_mag)", "codigo", 0)
    R.num("VParFramesBldc", 1 + (int(AVAL_A2212.seg_s * pre.sr) - pre.n_fft) // pre.hop, f_rot,
          "1+(N-n_fft)//hop (stft_mag)", "codigo", 0)
    R.num("VParNClassicFeat", cl.N_FEATURES, RAIZ_PY / "pipelines" / "_classicas.py",
          "N_FEATURES", "codigo", 0)
    # parâmetros de sequência: os que o ouro e o seq_pc gravaram (lidos do código pelo agente B)
    pj = C.EMB / "gold" / "seqs" / "idmt_seq00" / "params.json"
    p = C.ler_json(pj)
    R.num("VParNChangePen", len(p["change_penalties"]), pj, "change_penalties", "codigo", 0)
    R.num("VParCpMin", min(p["change_penalties"]), pj, "change_penalties", "codigo", 1)
    R.num("VParCpMax", max(p["change_penalties"]), pj, "change_penalties", "codigo", 0)
    R.num("VParNLogprob", p["n_logprob"], pj, "n_logprob", "codigo", 0)
    R.num("VParHmmTemp", p["temp"], pj, "temp", "codigo", 0)
    _ = PostConfig  # documentação: PostConfig.n_logprob difere do usado nas sequências


# ------------------------------------------------------------------ 2. dados e protocolos
def dados(R):
    fpred = C.mais_recente(C.RES / "predictions", "diagnostico_janelas_*.csv")
    fdiag = C.mais_recente(C.RES / "csv", "diagnostico_2*.csv")
    P = pd.read_csv(fpred, usecols=["variante", "base", "split", "rec_id", "idx", "classe",
                                    "motor"])
    P = P[(P.variante == "p11_decifrae_harmsum")]
    for base, nome in (("idmt", "Idmt"), ("a2212", "Bldc")):
        q = P[P.base == base].drop_duplicates(["rec_id", "idx"])
        R.num(f"VData{nome}Windows", len(q), fpred, "len(unique(rec_id,idx)) da p11", "dados", 0)
        R.num(f"VData{nome}Recordings", q.rec_id.nunique(), fpred, "nunique(rec_id)", "dados", 0)
        R.num(f"VData{nome}Motors", q.motor.nunique(), fpred, "nunique(motor)", "dados", 0)
        R.num(f"VData{nome}Classes", q.classe.nunique(), fpred, "nunique(classe)", "dados", 0)
    q = P[P.base == "a2212"].drop_duplicates("rec_id")
    for cl in ("Healthy", "Propeller", "Bearing"):
        R.num(f"VDataBldcFiles{cl}", int((q.classe == cl).sum()), fpred,
              "nunique(rec_id) por classe", "dados", 0)
    D = pd.read_csv(fdiag)
    for (base, prot), nome in PROTS.items():
        g = D[(D.base == base) & (D.protocolo == prot)].n_grupos.iloc[0]
        R.num(f"VData{nome}Groups", int(g), fdiag, "n_grupos", "dados", 0)
    meta = C.ler_json(fdiag.with_name(fdiag.stem + "_meta.json"))
    R.num("VDataNBoot", meta["n_boot"], fdiag.with_name(fdiag.stem + "_meta.json"), "n_boot",
          "dados", 0)
    nivel = int(re.search(r"ic(\d+)_lo", " ".join(D.columns)).group(1))
    R.num("VDataCiLevel", nivel, fdiag, "nível do IC no nome da coluna ic<nível>_lo", "dados", 0)
    R.num("VDataSeedBoot", meta["seed_boot"], fdiag.with_name(fdiag.stem + "_meta.json"),
          "seed_boot", "dados", 0)
    R.set_tex("VDataSeedBoot", str(meta["seed_boot"]))
    return fdiag


PROTS = {("idmt", "aleatorio"): "IdmtRand", ("idmt", "blocos"): "IdmtBlk",
         ("a2212", "aleatorio"): "BldcRand", ("a2212", "arquivo"): "BldcFile",
         ("a2212", "motor"): "BldcMotor", ("a2212", "motor_sem_M4"): "BldcNoMfour"}
VARS = {"b00_rmspico_norm": "RmsNorm", "b00_rmspico_bruto": "RmsRaw",
        "p04_classicas": "Cls", "p11_decifrae_harmsum": "Dec"}


# ------------------------------------------------------------------ 3. diagnóstico
def diagnostico(R, fdiag):
    D = pd.read_csv(fdiag)
    for (base, prot), pn in PROTS.items():
        for v, vn in VARS.items():
            for met, mn in (("acuracia_balanceada", "Bal"), ("roc_auc_ovr_macro", "Auc"),
                            ("f1_macro", "Fone")):
                r = D[(D.variante == v) & (D.base == base) & (D.protocolo == prot)
                      & (D.metrica == met) & (D.classe == "_macro")]
                if r.empty:
                    continue
                r = r.iloc[0]
                col = f"valor[{v},{base},{prot},{met},_macro]"
                R.num(f"VDiag{mn}{vn}{pn}", r.valor, fdiag, col, "diagnostico", 3)
                if mn == "Bal":
                    R.num(f"VDiag{mn}{vn}{pn}Lo", r.ic95_lo, fdiag, col.replace("valor", "ic95_lo"),
                          "diagnostico", 3)
                    R.num(f"VDiag{mn}{vn}{pn}Hi", r.ic95_hi, fdiag, col.replace("valor", "ic95_hi"),
                          "diagnostico", 3)
    # sensibilidade por classe da p11 (A2212)
    for prot, pn in (("arquivo", "BldcFile"), ("motor", "BldcMotor")):
        for cl in ("Healthy", "Propeller", "Bearing"):
            r = D[(D.variante == "p11_decifrae_harmsum") & (D.base == "a2212")
                  & (D.protocolo == prot) & (D.metrica == "sensibilidade") & (D.classe == cl)]
            R.num(f"VDiagRecDec{pn}{cl}", r.valor.iloc[0], fdiag,
                  f"valor[p11,a2212,{prot},sensibilidade,{cl}]", "diagnostico", 3)
    # diferenças derivadas (inflação do split aleatório e queda com motor fora)
    b = R.n
    for vn in VARS.values():
        R.num(f"VDiagInfl{vn}Bldc", b[f"VDiagBal{vn}BldcRand"]["valor"]
              - b[f"VDiagBal{vn}BldcFile"]["valor"], fdiag,
              "Bal[aleatorio]-Bal[arquivo] (a2212)", "diagnostico", 3)
        R.num(f"VDiagDrop{vn}Bldc", b[f"VDiagBal{vn}BldcFile"]["valor"]
              - b[f"VDiagBal{vn}BldcMotor"]["valor"], fdiag,
              "Bal[arquivo]-Bal[motor] (a2212)", "diagnostico", 3)


# ------------------------------------------------------------------ 4. rotação (interpretabilidade)
def rotacao_estimada(R):
    from pipelines import p11_decifrae_harmsum as p11
    ff = C.mais_recente(C.RES / "csv", "p11_features_*.csv")
    fpred = C.mais_recente(C.RES / "predictions", "diagnostico_janelas_*.csv")
    F = pd.read_csv(ff)
    R.num("VParNFeat", len([c for c in F.columns if re.fullmatch(r"f\d+", c)]), ff,
          "nº de colunas f0..f12", "codigo", 0)
    for cl, nome in (("good", "Good"), ("broken", "Broken"), ("heavyload", "Heavy")):
        x = F[(F.base == "idmt") & (F.classe == cl)].f0
        R.num(f"VRotIdmt{nome}Med", float(np.median(x)), ff, f"mediana de f0 (idmt, {cl})",
              "dados", 2)
        R.num(f"VRotIdmt{nome}Iqr", float(np.subtract(*np.percentile(x, [75, 25]))), ff,
              f"IQR de f0 (idmt, {cl})", "dados", 2)
    xg = F[(F.base == "idmt") & (F.classe == "good")].f0
    LIM_OIT, LIM_TETO = 1.5, 0.99   # limiares desta análise (registrados como macros)
    R.num("VRotOctThr", LIM_OIT, ff, "limiar: f0 > LIM_OIT·mediana da classe", "dados", 1)
    R.num("VRotCeilPctThr", 100 * (1 - LIM_TETO), ff, "limiar: f0 >= LIM_TETO·fmax", "dados", 0)
    R.num("VRotIdmtGoodOctPct", 100 * float(np.mean(xg > LIM_OIT * np.median(xg))), ff,
          "% janelas good com f0 > LIM_OIT·mediana da classe (salto de oitava)", "dados", 1)
    xb = F[F.base == "a2212"].f0
    lim = LIM_TETO * p11.PRE.fmax
    R.num("VRotBldcCeilPct", 100 * float(np.mean(xb >= lim)), ff,
          "% janelas A2212 com f0 >= 0,99·fmax", "dados", 1)
    P = pd.read_csv(fpred, usecols=["base", "rec_id", "motor", "velocidade"]).drop_duplicates()
    A = F[F.base == "a2212"].merge(P[P.base == "a2212"], on=["base", "rec_id"])
    G = A.groupby(["rec_id", "motor", "velocidade"]).f0.median().reset_index()
    rho = spearmanr(G.velocidade, G.f0).statistic
    R.num("VRotBldcSpearman", float(rho), [ff, fpred],
          "Spearman(velocidade nominal do arquivo, mediana de f0 do arquivo), 43 arquivos",
          "dados", 2, nota="arquivos do mesmo motor não são independentes")
    for m, nome in (("M1", "One"), ("M2", "Two"), ("M3", "Three"), ("M4", "Four")):
        g = G[G.motor == m]
        R.num(f"VRotBldcSpearmanM{nome}", float(spearmanr(g.velocidade, g.f0).statistic),
              [ff, fpred], f"Spearman por motor {m} ({len(g)} arquivos)", "dados", 2)
        R.num(f"VRotBldcFilesM{nome}", len(g), [ff, fpred], f"nº de arquivos do {m}", "dados", 0)


# ------------------------------------------------------------------ 5. equivalência no PC
def equivalencia(R):
    fres = C.mais_recente(C.RES / "csv", "equivalencia_resumo_por_build_*.csv")
    S = pd.read_csv(fres)
    meta = C.ler_json(fres.with_name(fres.stem + "_meta.json"))["fontes"]
    nomes = {"double": "Double", "float": "Float", "float-sosd": "Sosd"}
    f_det = {b: C.RES / "csv" / meta[b]["csv"] for b in nomes}
    etapas_seq = ["hmm_log_em", "argmax", "hmm_viterbi", "hmm_online_alpha", "hmm_online", "absorb"]
    s0 = S[S.build == "double"]
    R.num("VEqStagesTotal", len(s0), fres, "nº de etapas por build", "equivalencia", 0)
    R.num("VEqStagesCase", int((~s0.etapa.isin(etapas_seq)).sum()), fres,
          "etapas de janela", "equivalencia", 0)
    R.num("VEqStagesSeq", int(s0.etapa.isin(etapas_seq).sum()), fres, "etapas de sequência",
          "equivalencia", 0)
    R.num("VEqCases", int(s0[s0.etapa == "y_prep"].n.iloc[0]), fres, "n[y_prep]", "equivalencia", 0)
    R.num("VEqSeqCombos", int(s0[s0.etapa == "absorb"].n.iloc[0]), fres, "n[absorb]",
          "equivalencia", 0)
    for b, nm in nomes.items():
        s = S[S.build == b]
        R.num(f"VEq{nm}StagesPass", int((s.veredito == "PASS").sum()), fres,
              f"nº de etapas com veredito PASS ({b})", "equivalencia", 0)
        ok = s[(s.fail == 0) & (s.pass_empate == 0)]
        R.num(f"VEq{nm}MaxRatio", float(ok.razao_tol_max.max()), fres,
              f"máx razao_tol_max nas etapas sem empate/FAIL ({b})", "equivalencia", 2)
        d = s[s.etapa == "decisao"].iloc[0]
        R.num(f"VEq{nm}DecisionPass", int(d["pass"] + d.pass_empate), fres,
              f"pass+pass_empate[decisao] ({b})", "equivalencia", 0)
    z = S[(S.build == "float") & (S.etapa == "z")].iloc[0]
    R.num("VEqFloatZFail", int(z.fail), fres, "fail[float,z]", "equivalencia", 0)
    R.num("VEqFloatZPass", int(z["pass"]), fres, "pass[float,z]", "equivalencia", 0)
    fs = S[(S.build == "float") & (S.veredito != "PASS")].etapa.tolist()
    R.txt("VEqFloatFailStages", ", ".join(fs), fres, "etapas sem PASS (float)", "equivalencia")
    emp = S[(S.build == "float") & (S.pass_empate > 0)]
    R.num("VEqFloatTieStages", len(emp), fres, "nº de etapas com PASS_EMPATE (float)",
          "equivalencia", 0)
    # detalhe do empate (comparador do agente D)
    Dt = pd.read_csv(f_det["float"])
    e = Dt[(Dt.etapa == "env_mod") & (Dt.status == "PASS_EMPATE")]
    det = e.detalhe.iloc[0]
    m = re.search(r"C bin (\d+), ref bin (\d+); P_ref\[ref\]-P_ref\[C\] = ([0-9.e+-]+) "
                  r"\(2·tol = ([0-9.e+-]+)\)", det)
    R.num("VEqFloatTieCases", e.id.nunique(), f_det["float"], "ids com PASS_EMPATE em env_mod",
          "equivalencia", 0)
    R.txt("VEqFloatTieCaseIds", ", ".join(sorted(e.id.unique())), f_det["float"], "id",
          "equivalencia")
    R.num("VEqFloatTieGap", float(m.group(3)), f_det["float"], "detalhe[env_mod]: P_ref[ref]-P_ref[C]",
          "equivalencia", 1, sci=True)
    R.num("VEqFloatTieTol", float(m.group(4)), f_det["float"], "detalhe[env_mod]: 2·tol",
          "equivalencia", 1, sci=True)
    fm = Dt[(Dt.etapa == "features[9]:f_mod") & (Dt.status == "PASS_EMPATE")].err_max.max()
    R.num("VEqFloatFmodErr", float(fm), f_det["float"], "err_max[features[9]:f_mod] (empate)",
          "equivalencia", 1)
    zz = Dt[(Dt.etapa == "z") & (Dt.status == "FAIL")]
    R.num("VEqFloatZErrMax", float(zz.err_max.max()), f_det["float"], "err_max[z] (FAIL)",
          "equivalencia", 2)
    R.num("VEqFloatZFailIsEdge", int((zz.tipo_caso == "borda").all()), f_det["float"],
          "todos os FAIL de z são casos de borda (1=sim)", "equivalencia", 0)
    # env_dec: erro relativo ao máximo, mediana nos casos (float × variante sosd)
    for b, nm in (("float", "Float"), ("float-sosd", "Sosd"), ("double", "Double")):
        Db = pd.read_csv(f_det[b], usecols=["tipo_item", "etapa", "err_rel_max"])
        x = Db[(Db.tipo_item == "caso") & (Db.etapa == "env_dec")].err_rel_max
        R.num(f"VEqEnvDec{nm}Med", float(x.median()), f_det[b], "mediana err_rel_max[env_dec]",
              "equivalencia", 1, sci=True)
    # congelamento das tolerâncias
    ftol = C.EMB / "configs" / "embedded" / "tolerancias.json"
    R.txt("VEqTolSha", C.sha256(ftol)[:12], ftol, "sha256 (12 primeiros)", "equivalencia")
    R.txt("VEqTolShaRec", meta["float"]["sha256_tolerancias"][:12], fres,
          "sha256_tolerancias gravado no _meta da comparação", "equivalencia")
    gl = subprocess.run(["git", "-C", str(C.RAIZ), "log", "-1", "--format=%h|%ci", "--",
                         "embarcados/configs/embedded/tolerancias.json"],
                        capture_output=True, text=True).stdout.strip().split("|")
    R.txt("VEqTolCommit", gl[0], ftol, "git log -1 do tolerancias.json", "equivalencia")
    R.txt("VEqTolCommitTime", gl[1][:16], ftol, "data do commit do tolerancias.json", "equivalencia")
    R.txt("VEqRunTime", meta["float"]["data"].replace("T", " ")[:16], fres,
          "data da comparação formal (float)", "equivalencia")
    # ouro
    fg = C.EMB / "gold" / "manifest.json"
    g = C.ler_json(fg)
    from collections import Counter
    c = Counter(x["tipo"] for x in g["casos"])
    R.num("VEqGoldClean", c["limpo"], fg, "casos tipo limpo", "equivalencia", 0)
    R.num("VEqGoldNoisy", c["ruidoso"], fg, "casos tipo ruidoso", "equivalencia", 0)
    R.num("VEqGoldEdge", c["borda"], fg, "casos tipo borda", "equivalencia", 0)
    R.num("VEqGoldSeqs", g["n_seqs"], fg, "n_seqs", "equivalencia", 0)
    R.num("VEqGoldPerts", len(g["perturbacoes"]), fg, "nº de perturbações", "equivalencia", 0)
    R.num("VEqMinDurValues", len(C.ler_json(C.EMB / "gold" / "seqs" / "idmt_seq00" /
                                            "params.json")["min_durs_s"]),
          C.EMB / "gold" / "seqs" / "idmt_seq00" / "params.json", "len(min_durs_s)",
          "equivalencia", 0)
    # emulações de calibração (sem o C)
    for b, nm in (("emul_double", "Double"), ("emul_float", "Float")):
        f = C.mais_recente(C.RES / "csv", f"equivalencia_{b}_*.csv")
        mt = C.ler_json(f.with_name(f.stem + "_meta.json"))
        R.num(f"VEqEmul{nm}Rows", mt["n_linhas"], f.with_name(f.stem + "_meta.json"), "n_linhas",
              "equivalencia", 0)
        R.num(f"VEqEmul{nm}Pass", mt["contagem_status"].get("PASS", 0),
              f.with_name(f.stem + "_meta.json"), "contagem_status.PASS", "equivalencia", 0)
    return f_det


# ------------------------------------------------------------------ 6. medição no PC
def pc(R):
    fj = C.mais_recente(C.RES / "metrics", "bench_pc_*.json")
    fc = C.mais_recente(C.RES / "csv", "bench_pc_*.csv")
    J = C.ler_json(fj)
    r = J["resumo"]
    for base, nb in (("idmt", "Idmt"), ("a2212", "Bldc")):
        c, p = r[f"c_float_pc/{base}"], r[f"python_ref/{base}"]
        for est, ne in (("mediana", "Med"), ("p95", "Pninetyfive"), ("max", "Max")):
            R.num(f"VPcC{nb}{ne}", c["total_ms"][est], fj, f"resumo.c_float_pc/{base}.total_ms.{est}",
                  "pc", 2)
            R.num(f"VPcPy{nb}{ne}", p["total_ms"][est], fj, f"resumo.python_ref/{base}.total_ms.{est}",
                  "pc", 1)
        R.num(f"VPcC{nb}Rtf", c["fator_tempo_real_mediana"], fj, "fator_tempo_real_mediana", "pc", 4)
        R.num(f"VPcPy{nb}Rtf", p["fator_tempo_real_mediana"], fj, "fator_tempo_real_mediana", "pc", 3)
        R.num(f"VPcSpeedup{nb}", p["total_ms"]["mediana"] / c["total_ms"]["mediana"], fj,
              "mediana Python / mediana C", "pc", 1)
        for et, ne in (("rastrear_quadros", "Track"), ("viterbi", "Vit"), ("envelope_mod", "Env"),
                       ("features", "Feat"), ("cosseno", "Cos")):
            R.num(f"VPcC{nb}{ne}Med", c[f"{et}_ms"]["mediana"], fj, f"{et}_ms.mediana", "pc",
                  3 if ne in ("Feat", "Cos") else 2)
            R.num(f"VPcC{nb}{ne}Pct", 100 * c[f"{et}_ms"]["mediana"] / c["total_ms"]["mediana"], fj,
                  f"100·{et}_ms.mediana/total_ms.mediana", "pc", 0)
        R.num(f"VPcN{nb}", c["total_ms"]["n"], fj, "total_ms.n", "pc", 0)
    m = J["meta"]
    R.num("VPcWarmup", m["argv"]["aquecimento"], fj, "meta.argv.aquecimento", "pc", 0)
    B = pd.read_csv(fc)
    for base, nb in (("idmt", "Idmt"), ("a2212", "Bldc")):
        R.num(f"VPcDistinct{nb}", int(B[B.base == base].n_janelas_distintas.iloc[0]), fc,
              "n_janelas_distintas", "pc", 0)
    R.txt("VPcCompiler", m["compilador"].split(" (")[0].replace("/", r"\slash{}"), fj,
          "meta.compilador", "pc")
    R.num("VPcNcpu", m["maquina"]["n_cpu"], fj, "meta.maquina.n_cpu", "pc", 0)
    R.txt("VPcOs", m["maquina"]["plataforma"].split("-")[0] + " " +
          m["maquina"]["plataforma"].split("-")[1], fj, "meta.maquina.plataforma", "pc")
    # fluxo contínuo no PC
    fs = C.mais_recente(C.RES / "csv", "seq_pc_*.csv")
    ms = C.ler_json(fs.with_name(fs.stem + "_meta.json"))
    S = pd.read_csv(fs)
    R.num("VPcSeqWindows", len(S), fs, "nº de linhas (janelas)", "pc", 0)
    R.num("VPcSeqSwitches", len(ms["deteccao_online_c"]), fs.with_name(fs.stem + "_meta.json"),
          "len(deteccao_online_c)", "pc", 0)
    R.num("VPcSeqDelayC", max(d["atraso_janelas"] for d in ms["deteccao_online_c"]),
          fs.with_name(fs.stem + "_meta.json"), "max atraso_janelas (C)", "pc", 0)
    R.num("VPcSeqDelayPy", max(d["atraso_janelas"] for d in ms["deteccao_online_python"]),
          fs.with_name(fs.stem + "_meta.json"), "max atraso_janelas (Python)", "pc", 0)
    R.num("VPcSeqSameDecisions", int(ms["decisoes_iguais_python_c"]),
          fs.with_name(fs.stem + "_meta.json"), "decisoes_iguais_python_c (1=sim)", "pc", 0)
    R.num("VPcSeqMaxProbDiff", ms["max_abs_dif_probs"], fs.with_name(fs.stem + "_meta.json"),
          "max_abs_dif_probs", "pc", 1, sci=True)
    R.num("VPcSeqCp", ms["cp"], fs.with_name(fs.stem + "_meta.json"), "cp", "pc", 0)
    R.num("VPcSeqPerClass", int(len(S) / S.verdade.nunique()), fs, "janelas por classe", "pc", 0)


# ------------------------------------------------------------------ 7. estimativa analítica
def estimativa(R):
    out = C.PAPER / "fontes"
    out.mkdir(parents=True, exist_ok=True)
    regs = {}
    for b in ("float", "double"):
        exe = C.EMB / f"build-{b}" / "p11_host.exe"
        for n in (16000, 48000):
            s = subprocess.run([str(exe), "ws", str(n)], capture_output=True, text=True,
                               check=True).stdout.strip()
            d = json.loads(s)
            d["comando"] = f"embarcados/build-{b}/p11_host.exe ws {n}"
            d["sha256_executavel"] = C.sha256(exe)
            f = out / f"ws_{b}_{n}.json"
            f.write_text(json.dumps(d, indent=1), encoding="utf-8")
            regs[(b, n)] = (d, f)
    for b, nb in (("float", "Float"), ("double", "Double")):
        for n, nn in ((16000, "One"), (48000, "Three")):
            d, f = regs[(b, n)]
            R.num(f"VEstWs{nb}{nn}Mb", d["total_com_tabela"] / 1e6, f, "total_com_tabela/1e6",
                  "estimativa", 2)
            R.num(f"VEstWs{nb}{nn}NoTabMb", d["total_sem_tabela"] / 1e6, f, "total_sem_tabela/1e6",
                  "estimativa", 2)
            R.num(f"VEstWs{nb}{nn}PhaseBPct", 100 * d["faseB_envelope"] /
                  (d["persistente"] + max(d["faseA_rastreio"], d["faseB_envelope"])), f,
                  "faseB/(persistente+max(faseA,faseB))", "estimativa", 0)
    d, f = regs[("float", 16000)]
    R.num("VEstHarmTableMb", (d["total_com_tabela"] - d["total_sem_tabela"]) / 1e6, f,
          "total_com_tabela - total_sem_tabela", "estimativa", 2)
    return regs


# ------------------------------------------------------------------ 8. build do firmware
def _size(alvo):
    f = C.ESPRESSIF / f"build_p11_{alvo}" / "size.txt"
    if not f.exists():
        return None, f
    t = f.read_text(encoding="utf-8", errors="replace")
    d = {}
    m = re.search(r"p11_bancada\.bin binary size 0x([0-9a-f]+) bytes\. Smallest app partition is "
                  r"0x([0-9a-f]+)", t)
    d["bin"], d["part"] = int(m.group(1), 16), int(m.group(2), 16)
    for linha in t.splitlines():
        c = [x.strip() for x in linha.split("│") if x.strip() != ""]
        if len(c) >= 2 and re.fullmatch(r"\d+", c[1]):
            nome = c[0]
            nums = [x for x in c[1:]]
            d[nome] = nums
    return d, f


def build(R):
    pd_ = C.ESPRESSIF / "build_p11_esp32s3" / "project_description.json"
    if pd_.exists():
        j = C.ler_json(pd_)
        R.txt("VBuildIdf", j["git_revision"], pd_, "git_revision", "build")
        m = re.search(r"esp-(\d+\.\d+\.\d+)", j["c_compiler"])
        R.txt("VBuildGcc", m.group(1), pd_, "c_compiler (versão do toolchain esp-)", "build")
    for alvo, na in (("esp32", "Esp"), ("esp32s3", "Esps")):
        d, f = _size(alvo)
        if d is None:
            for k in ("BinKb", "FlashCodeKb", "FlashDataKb", "DramKb", "DramTotalKb", "IramKb"):
                R.num(f"VBuild{na}{k}", None, f, "size.txt ausente", "build")
            continue
        R.num(f"VBuild{na}BinKb", d["bin"] / 1024, f, "binary size (hex)/1024", "build", 0)
        R.num(f"VBuild{na}BinPct", 100 * d["bin"] / d["part"], f, "bin/partição app", "build", 0)
        R.num(f"VBuild{na}FlashCodeKb", int(d["Flash Code"][0]) / 1024, f,
              "Flash Code Used/1024", "build", 0)
        R.num(f"VBuild{na}FlashDataKb", int(d["Flash Data"][0]) / 1024, f,
              "Flash Data Used/1024", "build", 0)
        dram = "DIRAM" if "DIRAM" in d else "DRAM"
        R.num(f"VBuild{na}DramKb", int(d[dram][0]) / 1024, f, f"{dram} Used/1024", "build", 0)
        R.num(f"VBuild{na}DramTotalKb", int(d[dram][-1]) / 1024, f, f"{dram} Total/1024", "build", 0)
        R.num(f"VBuild{na}IramKb", int(d["IRAM"][0]) / 1024, f, "IRAM Used/1024", "build", 0)
        R.txt(f"VBuild{na}DramName", dram, f, "nome da região de dados interna", "build")
        tot = int(d[dram][-1])
        R.num(f"VBuild{na}WsRatio", R.n["VEstWsFloatThreeMb"]["valor"] * 1e6 / tot, [f],
              f"estimativa ws float 3 s / {dram} total", "estimativa", 1,
              nota="razão entre estimativa analítica (cat. 1) e capacidade do build (cat. 3)")
        R.num(f"VBuild{na}WsRatioOne", R.n["VEstWsFloatOneMb"]["valor"] * 1e6 / tot, [f],
              f"estimativa ws float 1 s / {dram} total", "estimativa", 1)


# ------------------------------------------------------------------ 9. medição no ESP32
def esp32(R):
    fb = C.RES / "manifests" / "bancada.json"
    flat = C.mais_recente(C.RES / "metrics", "esp32_lat_*.json")
    feq = C.mais_recente(C.RES / "csv", "equivalencia_esp32_*_resumo.csv", excluir=("_meta",))
    fsq = C.mais_recente(C.RES / "csv", "esp32_seq_*.csv")
    b = C.ler_json(fb) if fb.exists() else None
    tem_lat = b is not None and flat is not None
    R.txt("VChipModel", b["chip_modelo"] if b else None, fb, "chip_modelo", "esp32")
    R.num("VChipCpuMhz", b["cpu_mhz"] if b else None, fb, "cpu_mhz", "esp32", 0)
    R.num("VChipPsramMb", b["psram_mb"] if b else None, fb, "psram_mb", "esp32", 0)
    L = C.ler_json(flat)["resumo"] if tem_lat else {}
    for base, nb in (("idmt", "Idmt"), ("a2212", "Bldc")):
        x = L.get(base)
        for est, ne in (("mediana", "Med"), ("p95", "Pninetyfive"), ("max", "Max")):
            R.num(f"VChipLat{nb}{ne}", x["total_ms"][est] if x else None, flat or fb,
                  f"resumo.{base}.total_ms.{est}", "esp32", 1)
        R.num(f"VChipLat{nb}N", x["total_ms"]["n"] if x else None, flat or fb,
              f"resumo.{base}.total_ms.n", "esp32", 0)
        R.num(f"VChipRtf{nb}", x["fator_tempo_real_mediana"] if x else None, flat or fb,
              "fator_tempo_real_mediana", "esp32", 3)
        R.num(f"VChipRtfMax{nb}", x["fator_tempo_real_max"] if x else None, flat or fb,
              "fator_tempo_real_max", "esp32", 3)
        R.num(f"VChipHeapInt{nb}Kb", x["heap_min_livre_interna"] / 1024 if x else None, flat or fb,
              "heap_min_livre_interna/1024", "esp32", 0)
        R.num(f"VChipStack{nb}", x["pilha_folga_bytes_min"] if x else None, flat or fb,
              "pilha_folga_bytes_min", "esp32", 0)
        R.num(f"VChipWsPsram{nb}", x["ws_na_psram"] if x else None, flat or fb, "ws_na_psram",
              "esp32", 0)
        R.num(f"VChipWs{nb}Mb", x["ws_bytes"] / 1e6 if x else None, flat or fb, "ws_bytes/1e6",
              "esp32", 2)
    if feq is not None and b is not None:
        E = pd.read_csv(feq)
        R.num("VChipEqStagesPass", int((E.veredito == "PASS").sum()), feq, "veredito==PASS", "esp32", 0)
        R.num("VChipEqStages", len(E), feq, "nº de etapas comparadas no chip", "esp32", 0)
        d = E[E.etapa == "decisao"]
        R.num("VChipEqDecisionPass", int(d["pass"].iloc[0] + d.pass_empate.iloc[0]) if len(d) else None,
              feq, "pass+pass_empate[decisao]", "esp32", 0)
    else:
        for k in ("VChipEqStagesPass", "VChipEqStages", "VChipEqDecisionPass"):
            R.num(k, None, fb, "equivalência no chip não rodou", "esp32", 0)
    if fsq is not None and b is not None:
        ms = C.ler_json(fsq.with_name(fsq.stem + "_meta.json"))
        R.num("VChipSeqDelay", max((d["atraso_janelas"] or 0) for d in ms["deteccao"]),
              fsq.with_name(fsq.stem + "_meta.json"), "max atraso_janelas", "esp32", 0)
    else:
        R.num("VChipSeqDelay", None, fb, "fluxo contínuo no chip não rodou", "esp32", 0)
    return tem_lat


# ------------------------------------------------------------------ saída
def escrever(R, chip_medido):
    linhas = ["% Gerado por embarcados/src/python/gerar_artigo_numeros.py. NÃO EDITAR À MÃO.",
              f"% commit {COMMIT['commit']} (árvore suja em embarcados/: {COMMIT['sujo']})",
              "% Uma macro por número. Comentário: [categoria] fonte :: coluna.",
              "% Prefixos: VPar código | VData/VDiag/VRot dados e validação | VEq equivalência (PC)",
              "%           VEst 1·estimativa analítica | VPc 2·medição no PC |"
              " VBuild 3·build do firmware | VChip 4·medição no ESP32",
              r"\newif\ifchipmedido",
              (r"\chipmedidotrue" if chip_medido else r"\chipmedidofalse")
              + "  % true só se results/manifests/bancada.json e esp32_lat_* existirem", ""]
    for k, v in R.n.items():
        linhas.append(f"\\newcommand{{\\{k}}}{{{v['tex']}}}% [{v['categoria']}] "
                      f"{' + '.join(v['fonte'])} :: {v['coluna']}")
    (C.PAPER / "numeros.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    fontes = sorted({f for v in R.n.values() for f in v["fonte"]})
    out = {"gerado_por": "embarcados/src/python/gerar_artigo_numeros.py", "projeto": COMMIT,
           "chip_medido": chip_medido, "categorias": C.CAT,
           "fontes": {f: C.sha256(C.RAIZ / f if not f.startswith("C:") else f) for f in fontes},
           "numeros": R.n}
    (C.PAPER / "numeros.json").write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                                     default=float), encoding="utf-8")


RAIZ_PY = C.RAIZ
COMMIT = C.commit()


def main():
    C.PAPER.mkdir(parents=True, exist_ok=True)
    R = Registro()
    parametros(R)
    fdiag = dados(R)
    diagnostico(R, fdiag)
    rotacao_estimada(R)
    equivalencia(R)
    pc(R)
    estimativa(R)
    build(R)
    chip = esp32(R)
    escrever(R, chip)
    pend = [k for k, v in R.n.items() if v["status"] == "pending"]
    print(f"{len(R.n)} macros; {len(pend)} pending: {', '.join(pend)}")


if __name__ == "__main__":
    main()
