# `embarcados/paper/` — artigo ICIT 2027 (Etapa 7)

Tudo aqui é gerado por script a partir de `embarcados/results/`, do código Python da referência
e das saídas de build. Nenhum número é digitado à mão em `.md`, `.tex` ou figura. O único arquivo
redigido à mão é `main.tex`, e ele só cita números por macro (`\V…`, de `numeros.tex`).

## Como regenerar (diretório de trabalho: `Artigo3(MotorSound)`)

```
.venv\Scripts\python.exe embarcados\src\python\gerar_artigo_tudo.py            # tudo, na ordem
.venv\Scripts\python.exe embarcados\src\python\gerar_artigo_tudo.py --sem-espectro  # sem reler áudio
```

Ordem interna (cada script também roda sozinho):

1. `gerar_artigo_numeros.py` → `numeros.tex`, `numeros.json`, `fontes/ws_*.json` (roda
   `build-float|double\p11_host.exe ws 16000|48000`; lê `C:\Espressif\build_p11_*\size.txt`).
2. `gerar_artigo_figuras.py` → `figures/{pdf,png}/arquitetura`, `diag_*_en_<ts>`,
   `equivalencia_erro_etapas_en`, e `esp32_lat_<alvo>_<ts>` **só se** a bancada tiver medido;
   escreve `figuras.tex` (caminhos) e `figuras.json`. O espectro relê janelas do `train/` do IDMT.
3. `gerar_artigo_tabelas.py` → `tables/latex/{bases_protocolos,parametros_features,recursos,
   desempenho_compacto,equivalencia_compacta}.tex`.
4. `gerar_artigo_textos.py` → `METODOLOGIA.md`, `RESULTADOS.md` (modelos com `{VNome}` preenchidos
   pelo campo `pt` de `numeros.json`).
5. `gerar_artigo_manifesto.py` → `results/manifests/manifesto.json` (tabela/figura/macro → script,
   fontes com sha256, commit).
6. `gerar_artigo_checar.py` → `checagem_latex.txt` (inputs, refs, cites, macros, figuras,
   ambientes, dígitos soltos na prosa; nas duas versões).

Compilação (MiKTeX em `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`, fora do PATH), dentro de
`embarcados/paper/`:

```
pdflatex -jobname=main "\input{main}"            ; bibtex main      ; pdflatex ×2   → regular
pdflatex -jobname=main_wip "\def\usewip{1}\input{main}" ; bibtex main_wip ; pdflatex ×2 → WiP
```

Estado em 24/10 (após a auditoria F): regular = 6 páginas, WiP = 4 páginas, sem referência indefinida e sem erro.

Pré-requisitos já existentes (não são deste agente): `diagnostico.py` + `diagnostico_saidas.py`
(gera `desempenho.tex`, `diag_*` em PT), `comparar.py` + `resumo_equivalencia.py`
(`equivalencia.tex`), `bench_pc.py`, `seq_pc.py`, builds `build-float`/`build-double`, `gold/`.

## Quando o ESP32 chegar

Rodar `src/python/pil_esp32.py info`, `lat`, `equiv` (+ `tests/equivalence/comparar.py --build
esp32_<alvo> --tol-build float`, que gera `equivalencia_esp32_<alvo>_*_resumo.csv`) e `seq`; depois
`gerar_artigo_tudo.py`. O `\ifchipmedido` vira verdadeiro sozinho, os `[pending]` são trocados pelos
valores medidos, a figura de latência é criada e o texto do `main.tex` muda para o ramo "medido".
Revisar à mão as frases do ramo `\ifchipmedido` depois da primeira medição real.

## O que fica pendente sem o chip (macros `\VChip…` = `\textbf{[pending]}`)

Modelo do chip, frequência e PSRAM (`bancada.json`); latência mediana/p95/máx por base e fator de
tempo real; heap interno mínimo, folga de pilha e arena alocada; equivalência no chip (etapas e
decisão); atraso de detecção no fluxo contínuo no chip; a figura de latência por etapa. A lista
exata das que o `main.tex` usa está no fim de `checagem_latex.txt`.

## Observações

- As figuras `diag_*_<ts>` originais (de `diagnostico_saidas.py`) têm rótulos em português; o
  artigo usa as versões `_en` geradas aqui a partir dos mesmos CSV/JSON.
- `desempenho.tex` (tabela completa com F1/AUC e ICs), a matriz de confusão e o espectro continuam gerados,
  mas saíram do texto por espaço; o artigo usa `desempenho_compacto.tex` + Fig. de acurácia com IC.
- Referências: só as com ficha em `docs/literatura/`; `references.bib` anota a ficha e o tipo de
  leitura de cada uma (`vanaraj2026` e `xi2015` foram lidas só no resumo e são citadas só pelo
  que o resumo afirma).
