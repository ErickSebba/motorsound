# embarcados/ — porte em C da p11_decifrae_harmsum, equivalência e artigo ICIT 2027

Pasta de trabalho da branch `embarcado-icit`. Regras em `CLAUDE.md`; pedido original em
`PROMPT_TESTES_EMBARCADO.md`. Este README não repete números: cada valor citado no artigo sai de
script e está em `paper/numeros.json` (com a fonte), e os relatórios abaixo são gerados ou
escritos a partir dos CSVs de `results/`.

## Mapeamento com a estrutura do colaborador

O porte original estava em `Embarcado/decifrae_core.{c,h}` na raiz (commit `e00f2a9`; na árvore
de trabalho foi movido, sem commit, para `artigo-ICIT/Proposta_de_Artigo/Embarcado/`). Ele está
copiado sem alteração em `src/c/colaborador_original/`. O núcleo novo (`src/c/p11_*`) reaproveita
a ideia das features e do cosseno, com as 16 divergências corrigidas
(`reports/repository_audit.md`).

## Como reproduzir (diretório de trabalho: `Artigo3(MotorSound)`)

| passo | comando | saída |
|---|---|---|
| vetores de ouro (≈2 min) | `.venv\Scripts\python.exe embarcados\src\python\gerar_ouro.py` | `gold/` (gitignored) |
| build do C no PC | `& "embarcados\src\c\build_host.bat"` (PowerShell) | `build-float`, `build-double` |
| variante p11_c_sosd | `& "embarcados\src\c\build_variante_sosd.bat"` | `build-float-sosd` |
| testes unitários | `.venv\Scripts\python.exe -m pytest embarcados\tests\unit embarcados\tests\equivalence\test_comparador.py` | |
| equivalência no PC | `.venv\Scripts\python.exe embarcados\tests\equivalence\comparar.py --build double --gerar` (idem `float`; `--build float-sosd --tol-build float`) | `results/csv/equivalencia_*` |
| resumo da equivalência | `.venv\Scripts\python.exe embarcados\tests\equivalence\resumo_equivalencia.py` | `reports/c_equivalence_resultados_pc.md`, `tables/latex/equivalencia.tex` |
| diagnóstico (etapa 6) | `.venv\Scripts\python.exe embarcados\src\python\diagnostico.py --n-jobs 4` e `diagnostico_saidas.py` | `results/{csv,metrics,predictions}/diagnostico_*` |
| bancada no PC | `cd embarcados\src\python; ..\..\..\.venv\Scripts\python.exe bench_pc.py --n 200 --aquecimento 5` | `results/csv/bench_pc_*` |
| fluxo contínuo no PC | `... pil_esp32.py seq-imagem --por-classe 8` e `... seq_pc.py --imagem <bin> --cp 2` | `results/csv/seq_pc_*` |
| firmware | `embarcados\firmware\build_fw.bat <esp32 \| esp32s3 \| esp32s3oct> [espdsp]` | `C:\Espressif\build_p11_<alvo>` |
| ESP32 físico | `firmware\gravar_fw.bat <alvo> COMx`, `pil_esp32.py info/equiv/lat`, `firmware\gravar_seq.bat COMx <bin>`, `pil_esp32.py seq` | `results/csv/esp32_*`, `results/manifests/bancada.json` |
| artigo | `.venv\Scripts\python.exe embarcados\src\python\gerar_artigo_tudo.py` (ver `paper/README.md`) | `paper/main.pdf`, `paper/main_wip.pdf` |

Ferramentas: MSVC 2022 Build Tools + CMake (PC), ESP-IDF v5.5.6 em `C:\Espressif` (ambiente por
`firmware/idf_env.bat`), MiKTeX 25.12 (usuário). Ver `reports/BLOQUEIOS.md`.

## Resumo

**Bases usadas e limites.** IDMT-ISA-ELECTRIC-ENGINE (só `train/`; o teste oficial nunca foi
aberto) e BLDC A2212. No IDMT cada classe é um motor e uma gravação, então todo protocolo mede
reconhecimento de unidade conhecida; no A2212 a classe Bearing existe só no motor 4. Nenhuma
base nova foi baixada. Detalhes: `reports/diagnostico.md`.

**O que foi rodado.** Auditoria do porte; vetores de ouro instrumentando a referência sem
editá-la (164 casos, 20 sequências); porte completo em C (float32 e double); tolerâncias
congeladas e commitadas antes da comparação formal (`447a7c2`); comparação etapa por etapa no
PC; diagnóstico p11 × RMS/pico × features clássicas em três protocolos com bootstrap por grupo;
bancada no PC (Python × C); firmware ESP-IDF para esp32/esp32s3; artigo; auditoria independente.

**Status do porte e equivalência.** Ver `reports/c_port.md` e
`reports/c_equivalence_resultados_pc.md`. Build double passa em todas as etapas. Build float32
falha nos dois casos sintéticos de borda DC+ruído (argmax do Welch no piso numérico; contados
como FAIL após a auditoria), com a decisão de classe igual em todos os casos; a variante
`p11_c_sosd` passa em tudo. **Não** dizer "equivalente à p11" sem essas ressalvas.

**Medições no ESP32.** Pendentes até o chip ser conectado (`reports/BLOQUEIOS.md`, B1). Hoje há só
estimativa analítica do workspace (`p11_host ws`), medição no PC e build do firmware
(`idf.py size`), que são categorias diferentes e não podem ser chamadas de "medido no ESP32".
A estimativa indica que a p11 de referência exige PSRAM.

**Figuras e tabelas.** `figures/{pdf,png}/`, `tables/latex/`, ligadas a experimento e commit em
`results/manifests/manifesto.json`.

**Diagnóstico.** A p11 não supera as features clássicas em nenhum protocolo
(`reports/diagnostico.md`, `tables/latex/desempenho_compacto.tex`): a contribuição do artigo é a
viabilidade embarcada com equivalência verificada, não acurácia.

**Bloqueios restantes.** B1 (chip). Ver `reports/BLOQUEIOS.md`.

**Veredito do auditor.** `reports/independent_scientific_audit.md` (antes do chip): A COND,
B PASS, C COND (só PC), D FAIL (sem medição no chip), E COND; as correções de texto pedidas
foram aplicadas em `5c70782`.

**Recomendação.** Pendente do chip. Pela regra do prompt: regular (6 pp) se C e D passarem com
equivalência e medições no chip; WiP (4 pp) se a equivalência passou só no PC ou as medições
estão incompletas — que é a situação atual.
