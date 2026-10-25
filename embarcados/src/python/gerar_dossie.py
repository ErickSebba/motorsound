"""Dossiê de evidências para o manuscrito (pedido de dados de 25/10/2026).

Gera `embarcados/reports/DOSSIE_MANUSCRITO.md` lendo só CSV/JSON/relatórios já existentes:
nenhum número é digitado à mão. Cada valor sai com o arquivo de origem. O que não foi medido
aparece como "não medido"; o que o pedido supõe e não corresponde ao projeto é dito.

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_dossie.py

Não lê o conjunto de teste do IDMT (contagens do teste vêm de docs/DATASETS.md, que as leu
pelo cabeçalho em 14/10). Os relatórios de tamanho do firmware são copiados de
C:\\Espressif\\build_p11_<alvo>\\size.txt para embarcados/results/firmware/ (com carimbo).
"""
from __future__ import annotations

import glob
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import bancada_comum as bc

EMB = bc.EMB
RAIZ = bc.RAIZ
OUT = EMB / "reports" / "DOSSIE_MANUSCRITO.md"


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return "`" + p.relative_to(RAIZ).as_posix() + "`"
    except ValueError:
        return "`" + str(p) + "`"


def ultimo(padrao: str) -> Path:
    fs = sorted(f for f in glob.glob(str(padrao)) if not f.endswith("_meta.json"))
    if not fs:
        raise FileNotFoundError(padrao)
    return Path(fs[-1])


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):d}"
    return f"{x:.{d}f}"


def md_tab(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lin = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lin.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lin)


# ---------------------------------------------------------------- 1. bases
def sec_bases(L):
    L.append("## 1. Bases de dados e protocolo experimental\n")
    L.append("**ToyCar / DCASE 2022 Task 2: não usado.** Nenhum experimento deste projeto usa ToyCar "
             "ou qualquer dado DCASE. As únicas bases são IDMT-ISA-ELECTRIC-ENGINE e BLDC A2212 "
             f"(regra em {rel(RAIZ / 'CLAUDE.md')} e {rel(EMB / 'CLAUDE.md')}; inventário em "
             f"{rel(RAIZ / 'docs' / 'DATASETS.md')}). Se o manuscrito cita ToyCar, isso precisa sair.\n")
    # contagens do lado de treino/A2212 por listagem de arquivos (sem abrir test/)
    idmt = RAIZ / "data" / "idmt" / "train_cut"
    linhas = []
    for c in sorted(p.name for p in idmt.iterdir() if p.is_dir()):
        linhas.append({"base": "IDMT", "subconjunto": "train_cut (recortes de 3 s de train/<classe>/pure.wav)",
                       "classe": c, "arquivos": len(list((idmt / c).glob("*.wav"))),
                       "gravações de origem": 1})
    a = RAIZ / "data" / "a2212" / "BLDC_sound_data"
    for c in sorted(p.name for p in a.iterdir() if p.is_dir()):
        mot = sorted(p.name for p in (a / c).iterdir() if p.is_dir())
        n = sum(len(list((a / c / m).glob("*.wav"))) for m in mot)
        linhas.append({"base": "A2212", "subconjunto": "todos (sem split oficial)", "classe": c,
                       "arquivos": n, "gravações de origem": f"{n} ({', '.join(mot)})"})
    L.append("Contagem por listagem de arquivos feita por este script (o `test/` do IDMT não foi "
             "listado nem lido):\n")
    L.append(md_tab(pd.DataFrame(linhas)) + "\n")
    L.append("Versões, formatos, sha256 dos zips, contagens do `test/`/`test_cut/` do IDMT (lidas só "
             "do cabeçalho em 14/10) e as particularidades (uma gravação por classe no `train/` do IDMT; "
             "classe = unidade física no IDMT; Bearing só no motor M4 do A2212; um arquivo A2212 a "
             "44,1 kHz; saturação em Propeller/M1) estão em "
             f"{rel(RAIZ / 'docs' / 'DATASETS.md')} §1–2. **Arquivos excluídos:** nenhum arquivo "
             "inteiro; a sobra final < 3 s de cada `train/<classe>/pure.wav` não vira recorte "
             "(decisão dos autores da base).\n")
    v = json.load(open(RAIZ / "out" / "experiments" / "20261014_185859_validacao_meta.json", encoding="utf-8"))
    L.append("**Divisão.** Desenvolvimento só em validação cruzada (o teste fica trancado por código "
             f"em {rel(RAIZ / 'motorsound' / 'io.py')}). O teste oficial do IDMT foi lido **uma única "
             "vez**, em 25/10/2026, com autorização do Erick e depois do congelamento commitado "
             "das configurações (§2, \"Teste oficial do IDMT\"). Os demais números são de validação:\n")
    L.append("- IDMT: blocos temporais. Cada `train/<classe>/pure.wav` em segmentos de 3 s agrupados "
             "em 5 blocos contíguos; deixa-um-bloco-de-fora, com o primeiro segmento de cada bloco "
             "descartado como guarda. Grupo = (gravação, bloco). Fonte: "
             f"{rel(RAIZ / 'docs' / 'DATASETS.md')} §3, {rel(RAIZ / 'motorsound' / 'splits.py')}.")
    L.append("- A2212: `StratifiedGroupKFold` de 5 dobras agrupado por arquivo, segmentos de 1 s; e "
             "deixa-um-motor-de-fora (protocolo `motor`), que é o teste com **unidades não vistas**.")
    L.append("- Protocolo `aleatorio` (vazamento intencional, janelas da mesma gravação nos dois lados) "
             "só como auditoria da inflação.")
    L.append(f"- Seed dos splits: `{v.get('seed_splits')}` (meta {rel(RAIZ / 'out' / 'experiments' / '20261014_185859_validacao_meta.json')}). "
             "Todas as 20 pipelines rodaram nos mesmos cenários e partições: "
             f"`{v.get('cenarios')}`.\n")
    L.append("**Supervisionado, não one-class.** As pipelines são classificadores supervisionados "
             "com rótulo de estado (IDMT good/broken/heavyload; A2212 Healthy/Propeller/Bearing) no "
             "treino de cada dobra. As duas exceções são p30_anomalia e p30b_ae, treinadas só no "
             "normal (rótulos usados só na avaliação). Normalização, protótipos e pesos são ajustados "
             "só no treino de cada dobra. Se o manuscrito fala em \"anomaly detection\" one-class "
             "para a pipeline selecionada, isso não corresponde ao que foi feito.\n")


# ---------------------------------------------------------------- 2. 20 pipelines
def sec_pipelines(L):
    L.append("## 2. Comparação das 20 pipelines\n")
    fv = [RAIZ / "out" / "experiments" / "20261014_185859_validacao.csv",
          RAIZ / "out" / "experiments" / "20261014_203110_validacao_cnn.csv"]
    v = pd.concat([pd.read_csv(f) for f in fv])
    v = v[~v.vazamento_intencional | (v.split == "aleatorio")]
    cen = [("idmt", "blocos", True), ("idmt", "aleatorio", True), ("a2212", "arquivo", True),
           ("a2212", "motor", True), ("a2212", "aleatorio", True)]
    linhas = []
    for p, g in v.groupby("pipeline", sort=True):
        r = {"pipeline": p, "tipo": g.tipo.iloc[0]}
        for b, s, nz in cen:
            h = g[(g.base == b) & (g.split == s) & (g.normalizar_rms == nz)]
            r[f"{b} {s}"] = fmt(h.acerto.mean()) if len(h) else "—"
            r[f"n {b} {s}"] = len(h)
        linhas.append(r)
    L.append("**Lista.** As 20 configurações, a origem (artigo de referência ou Decifrae), o que "
             "muda em cada uma e as suposições estão em "
             f"{rel(RAIZ / 'docs' / 'PIPELINES.md')} (uma linha por pipeline) e no cabeçalho de "
             f"cada arquivo em {rel(RAIZ / 'pipelines')}. Nenhuma ficou como \"reprodução\": todas "
             "dependem de parâmetros que os artigos omitem.\n")
    L.append("**Métrica:** acurácia por janela (fração de janelas com classe prevista = rótulo), "
             "agregada sobre as dobras de validação; `n` = número de janelas avaliadas. "
             "Plataforma: PC (Python). Fontes: " + ", ".join(rel(f) for f in fv) +
             f"; tabela histórica equivalente em {rel(RAIZ / 'artigo-ICIT' / 'RESULTADOS_tabelas.md')} T1 "
             f"(gerada por {rel(RAIZ / 'experimentos' / 'tabelas_resultados.py')}). "
             "Validação, não teste.\n")
    L.append(md_tab(pd.DataFrame(linhas)) + "\n")
    m = json.load(open(RAIZ / "out" / "experiments" / "20261014_185859_validacao_meta.json", encoding="utf-8"))
    L.append("**Custo:** só **estimativa analítica** (parâmetros, RAM, flash, MFLOP/s) no campo "
             f"`custo_mcu` de {rel(RAIZ / 'out' / 'experiments' / '20261014_185859_validacao_meta.json')} "
             "e do meta da CNN; tabela em RESULTADOS_tabelas.md T9. **Nenhuma das 20 teve custo "
             "medido** (nem no PC por pipeline, nem no ESP32); só a p11 teve medição, ver §5.\n")
    L.append("**Seleção da p11.** A p11_decifrae_harmsum **não** foi escolhida por um critério "
             "formal de compromisso desempenho × custo. Ela é a versão embarcável da pipeline "
             "Decifrae (o tema do trabalho), definida antes da comparação. A comparação mostra que "
             "ela **perde** para CNNs e features clássicas no A2212 (relatório de decisão "
             f"{rel(RAIZ / 'artigo-ICIT' / 'DECISAO.md')}). O manuscrito não pode dizer que ela foi "
             "selecionada como o melhor compromisso.\n")
    d = pd.read_csv(EMB / "results" / "csv" / "diagnostico_20261023_222831.csv")
    d = d[(d.classe == "_macro") & (d.metrica == "acuracia_balanceada")]
    t = d.pivot_table(index="variante", columns=["base", "protocolo"], values="valor").round(3)
    t.columns = [f"{b} {p}" for b, p in t.columns]
    t = t.reset_index()
    L.append("**Diagnóstico refeito para o artigo do ESP32** (acurácia balanceada macro, IC95 por "
             "bootstrap de gravações, 4 variantes; fonte "
             f"{rel(EMB / 'results' / 'csv' / 'diagnostico_20261023_222831.csv')}, script "
             f"{rel(EMB / 'src' / 'python' / 'diagnostico.py')}; os ICs estão no CSV):\n")
    L.append(md_tab(t) + "\n")
    ft = sorted(p for p in (EMB / "results" / "csv").glob("teste_idmt_2*.csv")
                if "_curvas" not in p.name and "_meta" not in p.name)[-1]
    tt = pd.read_csv(ft)
    m = tt[(tt.classe == "_macro") & (tt.metrica == "acuracia_balanceada")]
    tab = m.pivot_table(index="condicao", columns="variante", values="valor").round(3).reset_index()
    tot = m[m.condicao == "_todas"][["variante", "valor", "ic95_lo", "ic95_hi", "n_janelas", "n_gravacoes"]].round(3)
    auc = tt[(tt.classe == "_macro") & (tt.metrica == "roc_auc_ovr_macro") & (tt.condicao == "_todas")][["variante", "valor"]].round(3)
    cong = sorted((EMB / "configs" / "experiments").glob("congelado_teste_idmt_*.json"))[-1]
    L.append("**Teste oficial do IDMT** (lido uma vez; congelamento em "
             f"{rel(cong)}, commitado antes do resultado; script "
             f"{rel(EMB / 'src' / 'python' / 'teste_idmt.py')}; resultado {rel(ft)}; janelas "
             "em `embarcados/results/predictions/teste_idmt_janelas_*.csv`; curvas ROC/PR em "
             "`results/csv/teste_idmt_curvas_*.csv`). Treino em todo o `train/` (pure), teste em todas "
             "as janelas de 3 s do `test/`. Acurácia balanceada agregada (IC95 por bootstrap de gravações):\n")
    L.append(md_tab(tot) + "\n")
    L.append("AUC macro um-contra-o-resto:\n")
    L.append(md_tab(auc) + "\n")
    L.append("Por condição (acurácia balanceada; cada condição tem 3 gravações, uma por classe):\n")
    L.append(md_tab(tab) + "\n")
    L.append("Classe = motor também no teste; o resultado mede transferência entre sessões e ruído "
             "dos mesmos três motores, não reconhecimento de estado.\n")
    L.append("Baselines realmente executadas: b00_nivel (RMS + pico, normalizado e bruto) e "
             "b01_forma_espectral. Ablações executadas: p11a (grade 10 cents), p12 (sem Viterbi de "
             "rotação), p20b (só rotação), HMM × argmax (\"p13\", em "
             f"{rel(RAIZ / 'experimentos' / 'sequencias.py')}).\n")


# ---------------------------------------------------------------- 3. p11
def sec_p11(L):
    L.append("## 3. Configuração exata da pipeline selecionada (p11_decifrae_harmsum)\n")
    L.append("**Não há CQT.** A p11 usa STFT (n_fft 8192, hop 1024, Hann) e uma soma harmônica numa "
             "grade logarítmica de frequência; não existe CQT em nenhuma etapa. Ordem real (contrato "
             f"em {rel(EMB / 'reports' / 'CONTRATO_INTERFACES.md')}, mapeamento Python → C em "
             f"{rel(EMB / 'reports' / 'c_port.md')} §2):\n")
    with open(EMB / "src" / "c" / "p11_tipos.h", encoding="utf-8") as f:
        defs = [l.split()[1:3] for l in f if l.startswith("#define P11_") and len(l.split()) >= 3]
    par = {k: v for k, v in defs}
    pega = lambda k: par.get(k, "?")
    L.append(f"1. Pré-processamento: mono, reamostragem para {pega('P11_SR')} Hz (só o arquivo "
             f"A2212 a 44,1 kHz precisa), normalização RMS para {pega('P11_RMS_ALVO')}; janelas de 3 s "
             "(IDMT) ou 1 s (A2212).")
    L.append(f"2. STFT: n_fft {pega('P11_N_FFT')}, hop {pega('P11_HOP')}, Hann simétrico, |X|.")
    L.append(f"3. Soma harmônica de {pega('P11_N_HARM')} harmônicos numa grade de {pega('P11_FMIN')}–"
             f"{pega('P11_FMAX')} Hz com passo de {pega('P11_PASSO_CENTS')} cents ({pega('P11_J')} pontos); "
             "emissão −log(0,45·e_norm + 1e-3) mais estado de silêncio.")
    L.append(f"4. Viterbi de rotação: salto máximo {pega('P11_MAX_SALTO')} pontos, custo de passo "
             f"{pega('P11_VIT_W_STEP')}, salto de oitava ±{pega('P11_K_OCT')} pontos com custo "
             f"{pega('P11_VIT_W_OCT')}, custo de silêncio {pega('P11_VIT_W_SIL')} → rotação f_rot.")
    L.append(f"5. Envelope de modulação: Hilbert sem padding, Butterworth ordem 4 a 200 Hz em SOS "
             f"(`sosfilt`, causal), decimação ×{pega('P11_DECIM')} para {pega('P11_FS_ENV')} Hz, Welch.")
    L.append(f"6. {pega('P11_N_FEAT')} features (rotação, saliência, razões harmônicas, ganhos por "
             f"grupo de harmônicos, f_mod e proeminência, centroide, planura); tabela em "
             f"{rel(EMB / 'tables' / 'latex' / 'parametros_features.tex')}.")
    L.append("7. Decisão por cosseno: z-score com média/desvio do treino, cosseno com o protótipo "
             "(média dos z do treino) de cada classe, softmax com temperatura T = 0,1.")
    L.append("8. HMM de estados (opcional, por sequência): Viterbi em log com `change_penalty` fixo "
             "(não há matriz de transição treinada), filtro online e absorção de trechos curtos; "
             "parâmetros cp = 2, n_logprob = −50, temp = 1 nos experimentos de fluxo "
             f"({rel(EMB / 'src' / 'python' / 'seq_pc.py')}).\n")
    L.append("Constantes do filtro geradas a partir do scipy da referência em "
             f"{rel(EMB / 'src' / 'c' / 'p11_constantes.h')}. Código de referência: "
             f"{rel(RAIZ / 'pipelines' / 'p11_decifrae_harmsum.py')}, {rel(RAIZ / 'pipelines' / '_decifrae.py')}, "
             f"{rel(RAIZ / 'motorsound' / 'rotacao.py')}, {rel(RAIZ / 'motorsound' / 'classificador.py')}, "
             f"{rel(RAIZ / 'motorsound' / 'hmm.py')}.\n")
    b = json.load(open(EMB / "results" / "csv" / "bench_pc_20261024-091911_meta.json", encoding="utf-8"))
    L.append(f"Bibliotecas: Python {b['maquina']['python']}, numpy {b['maquina']['numpy']} "
             f"(meta {rel(EMB / 'results' / 'csv' / 'bench_pc_20261024-091911_meta.json')}); scipy na "
             f"versão registrada no cabeçalho de {rel(EMB / 'src' / 'c' / 'p11_constantes.h')}. "
             "Precisão: referência em float32/float64 como no numpy; C em dois builds, `float` "
             "(o do MCU) e `double` (só PC).\n")


# ---------------------------------------------------------------- 4. Python × C
def sec_equiv(L):
    L.append("## 4. Consistência entre Python e C\n")
    L.append("**Objetivo: equivalência dentro de tolerância, por etapa**, não equivalência exata. "
             "As tolerâncias foram congeladas em `configs/embedded/tolerancias.json` antes da "
             "comparação formal, mas **depois** de o desenvolvedor do C ter depurado o porte contra "
             "os mesmos vetores de ouro (cronologia na auditoria "
             f"{rel(EMB / 'reports' / 'independent_scientific_audit.md')}, M2).\n")
    L.append("Entradas idênticas: o Python de referência e o C recebem o mesmo `audio.f32` de cada "
             f"caso do ouro (gerador {rel(EMB / 'src' / 'python' / 'gerar_ouro.py')}, relatório "
             f"{rel(EMB / 'reports' / 'gold_vectors.md')}). Comparador: "
             f"{rel(EMB / 'tests' / 'equivalence' / 'comparar.py')}.\n")
    L.append(f"Resultados por etapa, com maior erro e razão erro/tolerância: "
             f"{rel(EMB / 'reports' / 'c_equivalence_resultados_pc.md')} (gerado de CSV), CSVs:\n")
    for nome in ("double", "float", "float-sosd"):
        f = ultimo(EMB / "results" / "csv" / f"equivalencia_{nome}_2*_resumo.csv")
        r = pd.read_csv(f)
        L.append(f"- `{nome}`: {rel(f)} — itens PASS {int(r.n_pass.sum())}, PASS_EMPATE "
                 f"{int(r.n_pass_empate.sum())}, FAIL {int(r.n_fail.sum())}, em {len(r)} linhas de etapa.")
    f = ultimo(EMB / "results" / "csv" / "equivalencia_float-lowmem_2*_resumo.csv")
    r = pd.read_csv(f)
    L.append(f"- variante `p11_c_lowmem` (float, só janelas de 1 s): {rel(f)} — PASS "
             f"{int(r.n_pass.sum())}, FAIL {int(r.n_fail.sum())}.")
    f = ultimo(EMB / "results" / "csv" / "equivalencia_esp32_*_resumo.csv")
    r = pd.read_csv(f)
    dec = r[r.etapa == "decisao"].iloc[0]
    L.append(f"- **no ESP32 físico**, `p11_c_lowmem`: {rel(f)} — FAIL {int(r.n_fail.sum())}; decisão "
             f"{int(dec.n_pass)}/{int(dec.n_pass + dec.n_fail)} casos; razão erro/tolerância máx. "
             f"{r.razao_tol_max.max():.3g}. Só as saídas finais são comparadas (o firmware não devolve as "
             "etapas intermediárias, que aparecem como ausentes).")
    f = ultimo(EMB / "results" / "csv" / "aud_holdout_2*_resumo.csv")
    L.append(f"- teste fora da amostra (24 janelas fora do ouro, PC): {rel(f)}.\n")
    L.append("Divergências conhecidas: nos 2 casos sintéticos de borda DC + ruído do build float, "
             "o pico do espectro de modulação está no piso de arredondamento float32 nas duas "
             "implementações; f_mod difere e o z-score falha; **a decisão de classe coincide**. "
             "A auditoria (M1) pede para contar isso como FAIL de borda, não como empate. "
             "Concordância de decisão: etapa `decisao` no relatório acima (casos do ouro).\n")
    L.append("Fluxo contínuo (Python × C no PC, filtro online do HMM):")
    for f in sorted(glob.glob(str(EMB / "results" / "csv" / "seq_pc_*_meta.json"))):
        m = json.load(open(f, encoding="utf-8"))
        L.append(f"- {rel(f)}: base {m.get('base', 'idmt')}, decisões iguais Python × C = "
                 f"{m['decisoes_iguais_python_c']}, max |Δp| = {m['max_abs_dif_probs']:.3g}, "
                 f"detecções {m['deteccao_online_c']}.")
    L.append("")


# ---------------------------------------------------------------- 5. ESP32
def copia_size():
    dst = EMB / "results" / "firmware"
    dst.mkdir(parents=True, exist_ok=True)
    out = {}
    for alvo in ("esp32", "esp32_lowmem"):
        src = Path(r"C:\Espressif") / f"build_p11_{alvo}" / "size.txt"
        if src.exists():
            ts = datetime.fromtimestamp(src.stat().st_mtime).strftime("%Y%m%d-%H%M%S")   # hora local
            d = dst / f"size_{alvo}_{ts}.txt"
            if not d.exists():
                shutil.copy2(src, d)
            out[alvo] = d
    return out


def sec_esp32(L):
    L.append("## 5. Execução no ESP32 físico\n")
    b = json.load(open(EMB / "results" / "manifests" / "bancada.json", encoding="utf-8"))
    ci = b["esp_chip_info"]
    L.append(f"Manifesto: {rel(EMB / 'results' / 'manifests' / 'bancada.json')} (lido do chip por "
             "`esp_chip_info` pelo firmware; sessões anteriores em `results/manifests/bancada_*.json`).\n")
    L.append(f"- Chip: **{b['chip_modelo']}** clássico (revisão {ci['revisao']}, {ci['cores']} núcleos), "
             f"CPU {b['cpu_mhz']} MHz, flash {b['flash_mb']} MB, PSRAM {b['psram_mb']} MB. "
             "**Não é ESP32-S3.** O esptool identificou o chip como ESP32-D0WD-V3 durante a gravação "
             "(saída de console, não registrada em arquivo).")
    L.append("- Placa portadora: placa de desenvolvimento com conversor USB-serial CH9102 e botões "
             "BOOT/EN (ou seja, **é uma placa estilo DevKit**). Fabricante e modelo exatos da placa: "
             "**não registrados**.")
    L.append(f"- SDK: ESP-IDF {b['idf']}; compilador do IDF (xtensa-esp32-elf-gcc da toolchain "
             f"do IDF {b['idf']}); flags do núcleo `{b['flags_nucleo']}`; variante `{b['variante']}`; "
             f"compilado em {ci['compilado']}. Projeto e configuração: "
             f"{rel(EMB / 'firmware')} (`sdkconfig.defaults`, `main/CMakeLists.txt`). Wi-Fi e BT "
             "desligados.")
    L.append("- Execução: processor-in-the-loop. O áudio chega pela serial (ou pela flash); "
             "o tempo é medido com `esp_timer_get_time()` só em torno do processamento, "
             "sem aquisição.\n")
    m0 = pd.read_csv(EMB / "results" / "csv" / "esp32_memoria_esp32_20261024-182937.csv")
    L.append(f"**Memória, medida no chip.** Variante de referência `p11_c_ref` "
             f"({rel(EMB / 'results' / 'csv' / 'esp32_memoria_esp32_20261024-182937.csv')}): "
             "o workspace não cabe —")
    for _, r in m0.iterrows():
        L.append(f"  - {r.base}, janela de {r.n_amostras} amostras: precisa {fmt(r.get('ws_bytes'))} B "
                 f"contíguos, maior bloco livre {fmt(r.get('maior_bloco_livre_interna'))} B, PSRAM "
                 f"{fmt(r.get('maior_bloco_livre_psram'))} B.")
    L.append(f"\nVariante `p11_c_lowmem` (pool de blocos, só janelas de 1 s): pool disponível no "
             f"chip {fmt(ci.get('pool_8bit'))} B de DRAM + {fmt(ci.get('pool_32bit'))} B de IRAM, "
             f"blocos {ci.get('pool')}; heap livre após o pool {fmt(ci.get('heap_livre_8bit'))} B. "
             "Tentativas registradas (as primeiras falharam na alocação ou no acesso à IRAM, até "
             "as correções do firmware):")
    for f in sorted(glob.glob(str(EMB / "results" / "csv" / "esp32_memoria_esp32_*.csv"))):
        if f.endswith("_meta.json"):
            continue
        r = pd.read_csv(f)
        res = "; ".join(f"{x.base}: coube={x.coube}" + (f", ws={int(x.ws_bytes)} B, heap mín. "
                        f"{int(x.heap_min_livre_interna)} B" if x.coube == 1 and "heap_min_livre_interna" in r
                        else "") for x in r.itertuples())
        L.append(f"  - {rel(f)}: {res}")
    sz = copia_size()
    L.append("\n**Tamanho do firmware (build, `idf.py size`, não é medição em execução):** " +
             ", ".join(f"{k}: {rel(v)}" for k, v in sz.items()) + ".\n")
    fl = ultimo(EMB / "results" / "csv" / "esp32_lat_esp32_*.csv")
    lt = pd.read_csv(fl)
    lt = lt[~lt.aquecimento.astype(bool)]
    est = lt.total_ms.agg(["count", "min", "mean", "median", lambda s: s.quantile(0.95), "max"])
    est.index = ["n", "min", "media", "mediana", "p95", "max"]
    L.append(f"**Tempo de processamento, medido no ESP32** ({rel(fl)}; variante `p11_c_lowmem`, janelas "
             f"de 1 s do A2212, aquecimento excluído; `esp_timer_get_time()` só em torno do "
             f"processamento, sem aquisição nem transmissão), ms por janela:\n")
    L.append(md_tab(est.round(1).to_frame("total_ms").T.reset_index(drop=True)) + "\n")
    eta = [c for c in lt.columns if c.endswith("_ms") and c != "total_ms"]
    L.append("Mediana por etapa (ms): " + ", ".join(f"{c[:-3]} {lt[c].median():.1f}" for c in eta) + ".")
    L.append(f"Fator de tempo real (mediana do total / 1000 ms): {lt.total_ms.median() / 1000:.2f} — "
             "**não é tempo real** nesta placa. A janela de 3 s do IDMT não cabe.\n")
    fs = ultimo(EMB / "results" / "csv" / "esp32_seq_esp32_*.csv")
    ms = json.load(open(fs.with_name(fs.stem + "_meta.json"), encoding="utf-8"))
    sq = pd.read_csv(fs)
    pcf = sorted((EMB / "results" / "csv").glob("seq_pc_*.csv"))
    pcm = [p for p in pcf if json.load(open(p.with_name(p.stem + "_meta.json"), encoding="utf-8")).get("base") == ms.get("base")]
    iguais = (sq.decisao_online.values == pd.read_csv(pcm[-1]).online_python.values).sum() if pcm else None
    L.append(f"**Fluxo contínuo, medido no ESP32** ({rel(fs)}): {len(sq)} janelas lidas da flash, "
             f"detecções {ms['deteccao']}; decisões iguais às do Python ({rel(pcm[-1]) if pcm else '—'}): "
             f"{iguais}/{len(sq)}.\n")
    L.append("**Energia: não medida** (não há sensor de corrente). O manuscrito não pode afirmar "
             "tempo real nem baixo consumo no ESP32.\n")
    bp = pd.read_csv(EMB / "results" / "csv" / "bench_pc_20261024-091911.csv")
    bp = bp[~bp.aquecimento]
    t = bp.groupby(["impl", "base"]).total_ms.agg(
        n="count", min="min", media="mean", mediana="median",
        p95=lambda s: s.quantile(0.95), max="max").round(3).reset_index()
    L.append("Para comparação, **medição no PC** (não ESP32) por janela, ms, sem aquecimento "
             f"({rel(EMB / 'results' / 'csv' / 'bench_pc_20261024-091911.csv')}; máquina e compilador no "
             "meta; o C roda num processo novo por janela e `total_sem_init_ms` exclui a "
             "inicialização — ver auditoria m5):\n")
    L.append(md_tab(t) + "\n")


# ---------------------------------------------------------------- 6. figuras
def sec_figuras(L):
    L.append("## 6. Tabelas, gráficos e arquivos de comprovação\n")
    L.append("Já existem (dados + script de origem no manifesto "
             f"{rel(EMB / 'results' / 'manifests' / 'manifesto.json')}, gerados por "
             f"{rel(EMB / 'src' / 'python' / 'gerar_artigo_tudo.py')}):\n")
    for f in sorted((EMB / "figures" / "png").glob("*.png")):
        L.append(f"- {rel(f)}")
    for f in sorted((EMB / "tables" / "latex").glob("*.tex")):
        L.append(f"- {rel(f)}")
    L.append(f"- {rel(RAIZ / 'out' / 'figuras' / 'pareto_custo.png')} (desempenho × custo **estimado**, "
             "fase exploratória)\n")
    L.append("Situação dos itens pedidos:\n")
    L.append("- Tabela comparativa das 20 pipelines: **pronta** (§2 e RESULTADOS_tabelas.md T1), em validação.")
    L.append("- Desempenho × custo: **pronta só com custo estimado** (pareto_custo.png); com custo medido é inviável (§5).")
    L.append("- Matriz de confusão: **pronta em validação** para a p11 (diag_confusao_p11_*.png; contagens `cm_pred_*` no CSV de diagnóstico).")
    L.append("- ROC/PR **no teste oficial do IDMT**: **pronto** (pontos em `results/csv/teste_idmt_curvas_*.csv`, "
             "ROC em `figures/png/teste_idmt_roc_en.png`; PR só em CSV). O A2212 não tem conjunto de teste "
             "separado, então lá só há AUC de validação (`roc_auc_ovr_macro` no CSV de diagnóstico).")
    L.append("- Matriz de confusão **no teste**: **pronta** (`cm_pred_*` em `results/csv/teste_idmt_2*.csv`).")
    L.append("- Python × C com linha de identidade e distribuição dos erros: **parcial**; existe "
             "equivalencia_erro_etapas.png (erro por etapa contra a tolerância). Um gráfico de "
             "dispersão Python × C não foi feito, mas os dados existem (`gold/` + `results/c_out/`, não versionados por tamanho).")
    L.append("- Distribuição dos tempos no ESP32: **dados prontos** (§5, CSV por janela); a figura do "
             "artigo mostra a mediana por etapa (`figures/png/esp32_lat_ram_*`), não a distribuição "
             "(a variação entre janelas é < 1 ms).\n")
    L.append("Código: Python de referência em `pipelines/`, `motorsound/`; C em `embarcados/src/c/`; "
             "firmware em `embarcados/firmware/`; testes em `embarcados/tests/`. Manifesto do ouro "
             "(partições e casos): `embarcados/gold/manifest.json` (não versionado; gerado por "
             "`gerar_ouro.py`). Registros de compilação: `results/firmware/size_*.txt`; testes unitários: "
             "`tests/unit/test_p11.py`.\n")


def main():
    L = ["# Dossiê de evidências para o manuscrito",
         "",
         "> Gerado por `embarcados/src/python/gerar_dossie.py` em " + bc.carimbo() +
         f" (commit {bc.commit()['commit'][:7]}). Não editar à mão. Todo valor sai de um arquivo "
         "citado; \"não medido\" e \"não disponível\" são literais.",
         ""]
    sec_bases(L)
    sec_pipelines(L)
    sec_p11(L)
    sec_equiv(L)
    sec_esp32(L)
    sec_figuras(L)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
