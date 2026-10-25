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
| variante p11_c_lowmem no PC | `.venv\Scripts\python.exe embarcados\tests\equivalence\gerar_lowmem.py --build float --pool "<layout do chip>"` | `results/csv/equivalencia_float-lowmem_*` |
| firmware | `embarcados\firmware\build_fw.bat <esp32 \| esp32s3 \| esp32s3oct> [lowmem \| espdsp]` | `C:\Espressif\build_p11_<alvo>[_<variante>]` |
| gravar (placa sem reset de GPIO0) | 1ª vez: `firmware\gravar_tudo.bat COM3 esp32_lowmem <seq.bin> no_reset` (BOOT+EN antes); depois: `firmware\gravar_boot_espera.py COM3 C:\Espressif\build_p11_esp32_lowmem 600` segurando BOOT | flash |
| ESP32 físico | `pil_esp32.py info`, `mem`, `equiv` (+ `comparar.py --build esp32_esp32_p11_c_lowmem --tol-build float --c-out results\c_out\esp32_esp32_p11_c_lowmem`), `lat --bases a2212 --n 200 --aquecimento 5`, `seq --imagem <bin> --cp 2` | `results/csv/esp32_*`, `results/csv/equivalencia_esp32_*`, `results/manifests/bancada.json` |
| números da auditoria | `aud_holdout.py`, `aud_rotacao_motor.py`, `aud_motor_igual.py`, `gerar_artigo_numeros_aud.py` (em `embarcados\src\python`) | `results/csv/aud_*`, `paper/numeros_aud.tex` |
| teste oficial do IDMT (uma vez) | `teste_idmt.py congelar` → commit → `teste_idmt.py avaliar --congelado <json>`; `gerar_artigo_teste.py` | `configs/experiments/congelado_*`, `results/csv/teste_idmt_*`, `paper/numeros_teste.tex` |
| artigo | `.venv\Scripts\python.exe embarcados\src\python\gerar_artigo_tudo.py` (ver `paper/README.md`) | `paper/main.pdf`, `paper/main_wip.pdf` |
| dossiê para o manuscrito | `.venv\Scripts\python.exe embarcados\src\python\gerar_dossie.py` | `reports/DOSSIE_MANUSCRITO.md` |

Ferramentas: MSVC 2022 Build Tools + CMake (PC), ESP-IDF v5.5.6 em `C:\Espressif` (ambiente por
`firmware/idf_env.bat`), MiKTeX 25.12 (usuário). Ver `reports/BLOQUEIOS.md`.

## Resumo

**Bases usadas e limites.** IDMT-ISA-ELECTRIC-ENGINE (desenvolvimento só no `train/`; o teste
oficial foi lido uma vez em 25/10, com autorização do Erick e configurações congeladas antes) e
BLDC A2212. No IDMT cada classe é um motor e uma gravação, então todo protocolo mede
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

**Medições no ESP32.** Feitas num ESP32 clássico sem PSRAM (`results/manifests/bancada.json`).
A referência `p11_c_ref` não cabe; a variante nomeada `p11_c_lowmem` cabe só em janelas de 1 s
(A2212), passa na equivalência no chip com as tolerâncias congeladas (só saídas finais), e **não
é tempo real** (fator em `paper/numeros.json`, `VChipRtfBldc`); a janela de 3 s do IDMT não cabe.
Energia não medida. Números e fontes: `reports/DOSSIE_MANUSCRITO.md` §5.

**Teste oficial do IDMT.** A validação quase perfeita não se sustenta no teste
(`results/csv/teste_idmt_*`, `paper/RESULTADOS.md`); classe = motor também lá.

**Figuras e tabelas.** `figures/{pdf,png}/`, `tables/latex/`, ligadas a experimento e commit em
`results/manifests/manifesto.json`.

**Diagnóstico.** A p11 não supera as features clássicas em nenhum protocolo
(`reports/diagnostico.md`, `tables/latex/desempenho_compacto.tex`): a contribuição do artigo é a
viabilidade embarcada com equivalência verificada, não acurácia.

**Bloqueios restantes.** Nenhum que impeça a submissão. Limites da placa (sem PSRAM) e da
conexão USB registrados em `reports/BLOQUEIOS.md` (B5, B6).

**Veredito do auditor.** Antes do chip: `reports/independent_scientific_audit.md` (A COND, B PASS,
C COND, D FAIL, E COND). Depois do chip e do teste: `reports/independent_scientific_audit_final.md`.

**Decisão.** Em 25/10 o Erick escolheu o artigo regular (6 pp), com a afirmação de
embarcabilidade restrita ao que foi medido (1 s, não tempo real). `paper/main.pdf` (6 pp) e
`paper/main_wip.pdf` (4 pp) compilam.
