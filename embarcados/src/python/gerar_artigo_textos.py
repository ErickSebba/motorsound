"""Gera paper/METODOLOGIA.md e paper/RESULTADOS.md (português, estilo artigo).

    .venv\\Scripts\\python.exe embarcados\\src\\python\\gerar_artigo_textos.py

Agente E (Etapa 7). Os textos abaixo são modelos com marcadores {VNome}; cada marcador é
trocado pelo valor formatado em português que está em paper/numeros.json (campo "pt").
Nenhum número é digitado aqui. Marcador sem macro correspondente = erro (o script para).
Os números do ESP32 físico aparecem como **[pendente]** enquanto a bancada não rodar.
"""
from __future__ import annotations

import re

import gerar_artigo_comum as C

METODOLOGIA = """# Metodologia (artigo ICIT 2027, `embarcados/`)

> Gerado por `embarcados/src/python/gerar_artigo_textos.py` a partir de `paper/numeros.json`
> (commit {COMMIT}). Não editar à mão. Os valores entre parênteses vêm das macros de
> `paper/numeros.tex`, que trazem arquivo e coluna de origem. O histórico da fase exploratória
> está em `artigo-ICIT/METODOLOGIA.md`, `RESULTADOS.md` e `DECISAO.md` (somente leitura).

## Pipeline e analogia

A pipeline avaliada é a `p11_decifrae_harmsum`, herdada do transcritor de baixo e do
classificador de acordes do projeto Decifrae. O motor é tratado como um "instrumento de uma corda
só": a frequência de rotação faz o papel da fundamental, rastreada por uma soma de harmônicos e
decodificada por um Viterbi com estado de silêncio e custo de salto de oitava. A analogia motiva a
representação e não afirma que o som do motor seja uma nota musical nem que o desempenho em música
se transfira para máquinas.

Cada janela de análise ({VParSegIdmt} s no IDMT, {VParSegBldc} s no A2212, ambas a {VParSrKhz} kHz)
é normalizada em RMS (alvo {VParRmsTarget}) e processada sozinha. A STFT usa janela de Hann
simétrica de {VParNfft} amostras e salto de {VParHop}, sem centralização ({VParFramesIdmt} quadros
em {VParSegIdmt} s e {VParFramesBldc} em {VParSegBldc} s). Para cada uma das {VParJ} frequências
candidatas, numa grade logarítmica de {VParFmin} a {VParFmax} Hz com passo de {VParStepCents} cents,
a saliência soma {VParNharm} harmônicos com interpolação linear entre bins e é normalizada por
quadro. O custo de emissão é −log({VParWEnergy}·ẽ + {VParEmisEps}), e o estado de silêncio recebe um
custo que depende da energia do quadro relativa ao quadro mais forte da janela. O Viterbi admite
saltos de até {VParMaxJump} estados ({VParMaxJumpCents} cents) com custo de {VParWStep} por
semitom, saltos de oitava de ±{VParOctStates} estados com custo {VParWOct} e entrada ou saída do
silêncio com custo {VParWSil}. Todos esses valores foram lidos do código Python da referência por
import (`PreDecifrae`, `motorsound.rotacao`, `_envelope_mod`), não de comentário nem do C.

Do caminho de rotação e do espectro médio saem {VParNFeat} features, na ordem da referência:
mediana e IQR de f_rot, fração de quadros em silêncio, saliência média no caminho, razão harmônica,
energia relativa de quatro grupos de ordens harmônicas, frequência e proeminência do pico de
modulação do envelope (Hilbert, Butterworth de ordem {VParButterOrder} em {VParButterFc} Hz com
`sosfilt` causal, dizimação por {VParDecim}, Welch, pico entre {VParModLo} e {VParModHi} Hz),
centróide e planura espectral. As features são padronizadas com média e desvio do treino e
comparadas por cosseno a um protótipo por classe; um softmax com temperatura {VParTempCos} dá as
probabilidades. Nas sequências, um HMM com penalidade de troca fixa e um estado extra "N" roda como
filtro forward causal, uma janela por vez ({VParNChangePen} penalidades varridas, de {VParCpMin} a
{VParCpMax}; log-probabilidade do estado N igual a {VParNLogprob}).

## Porte em C

O porte segue o Python linha a linha e compila sem alteração no PC e no ESP-IDF. Ele nunca guarda
o espectrograma inteiro: cada quadro é janelado, transformado por uma FFT real, acumulado no
espectro médio e reduzido à sua linha de saliência. O Viterbi precisa da janela inteira, porque o
custo do silêncio usa a energia máxima da janela; por isso a saliência normalizada e os
ponteiros de retorno ficam guardados. O envelope de Hilbert precisa de uma FFT do tamanho da
janela, que não é potência de dois, e o porte inclui uma FFT de raiz mista. Toda a memória é uma
arena fornecida pelo chamador, cujo tamanho sai da mesma função de layout que distribui os
ponteiros. O tipo real é escolhido na compilação (float32 para o microcontrolador, double para a
verificação), e as decisões que mudam inteiros (grade, medianas, arredondamento das posições
harmônicas) são feitas em double nos dois builds, reproduzindo o arredondamento meio-para-par e o
desempate pelo primeiro índice da referência. Toda mudança de aritmética é uma variante nomeada.
A avaliada aqui, `p11_c_sosd`, mantém em double o estado do filtro do envelope dentro do build
float32, como o scipy faz; ela é mais precisa, não uma aproximação. A placa disponível não tem
PSRAM, e para ela existe a variante nomeada `{VChipVariant}`: as mesmas funções do núcleo, mas a
memória vem de um pool de {VChipPoolBlocks} blocos de RAM interna (DRAM e IRAM, esta só para
vetores de float lidos por palavra inteira, como as tabelas de giro da FFT), alocados por fase com
um plano encontrado por busca com retrocesso; a tabela de interpolação harmônica sai e as posições
dos harmônicos são recalculadas em double a cada quadro; o Hilbert é feito por uma FFT de quatro
passos, que é a mesma DFT em outra ordem de operações e por isso muda o erro numérico. Ela é
comparada com a referência pelas mesmas tolerâncias do float32. Existe no firmware a opção
`p11_c_espdsp` (FFTs de potência de dois pela ESP-DSP), que nunca foi comparada com a referência
nem medida no chip e por isso fica fora do artigo.

## Bases, protocolos e métricas

Usamos só o `train/` do IDMT ({VDataIdmtRecordings} gravações de {VDataIdmtMotors} motores,
{VDataIdmtWindows} janelas; o teste oficial não foi aberto) e o A2212 inteiro
({VDataBldcRecordings} arquivos de {VDataBldcMotors} motores, {VDataBldcFilesHealthy} Healthy,
{VDataBldcFilesPropeller} Propeller e {VDataBldcFilesBearing} Bearing; {VDataBldcWindows} janelas).
Todos os métodos usam exatamente as mesmas janelas e os mesmos folds. No IDMT comparamos o split
aleatório por janela (vazamento intencional, {VDataIdmtRandGroups} grupos) e o split por blocos
contíguos ({VDataIdmtBlkGroups} grupos), que ainda não separa classe de motor porque há uma gravação
por classe. No A2212 comparamos o split aleatório, o split agrupado por arquivo
({VDataBldcFileGroups} grupos), o motor fora e o motor fora sem o fold do M4
({VDataBldcNoMfourGroups} grupos), porque o rolamento só existe no motor M4. A tabela
`tables/latex/bases_protocolos.tex` resume o que cada protocolo permite concluir.

Os baselines são o de nível (RMS e pico com regressão logística, com e sem normalização de RMS) e o
de features clássicas p04 ({VParNClassicFeat} estatísticas de ZCR, RMS, centróide e MFCC com voto
suave), inspirado no resumo de Vanaraj et al. (ficha `docs/literatura/vanaraj2026.md`), que só foi lido no resumo. As métricas são
acurácia balanceada, F1 macro, ROC-AUC macro um-contra-todos com os scores contínuos e
sensibilidade por classe. Os intervalos de {VDataCiLevel} % vêm de um bootstrap que reamostra
grupos (blocos ou arquivos), estratificado por classe, com {VDataNBoot} reamostragens (semente
{VDataSeedBoot}). No protocolo de motor fora a unidade independente é o motor, e com
{VDataBldcMotors} motores não há intervalo honesto nessa escala; o intervalo por arquivo é otimista.

## Protocolo de equivalência

A referência instrumentada gerou vetores de ouro de todas as etapas em {VEqCases} janelas:
{VEqGoldClean} limpas da validação das duas bases, {VEqGoldNoisy} ruidosas ({VEqGoldPerts}
perturbações: ruído branco em três SNRs e os distúrbios sintéticos de afundamento e cintilação) e
{VEqGoldEdge} casos de borda (silêncio, clipping, rotação sintética fora da faixa e nível constante
com ruído mínimo). As etapas do HMM foram conferidas em {VEqGoldSeqs} sequências de probabilidade,
{VParNChangePen} penalidades e {VEqMinDurValues} durações mínimas ({VEqSeqCombos} configurações). As
tolerâncias de cada etapa vêm de uma análise de arredondamento de primeira ordem e de uma emulação
numérica independente de cada build; foram congeladas antes da comparação formal, no commit
`{VEqTolCommit}` ({VEqTolCommitTime}, sha256 `{VEqTolSha}…`; comparação em {VEqRunTime}), e o hash vai em cada linha do
resultado. As duas emulações passaram em todas as linhas ({VEqEmulDoublePass} e {VEqEmulFloatPass}
de {VEqEmulFloatRows}) antes de o C ser comparado. Etapas contínuas são comparadas pelo erro
máximo relativo à escala da etapa; saídas discretas (caminho de rotação, decisão, estados do HMM)
têm de coincidir exatamente, salvo empate demonstrado com os valores da própria referência. Uma
falha numa etapa marca as seguintes como "a jusante". Cronologia: o porte em C foi depurado contra
os mesmos vetores de ouro pelo seu desenvolvedor antes do congelamento (commit do porte anterior ao
das tolerâncias). Por isso falamos em tolerâncias "congeladas antes da comparação formal", não em
pré-registro. A auditoria independente (`reports/independent_scientific_audit.md`) conferiu também
janelas fora do ouro, que passaram com as tolerâncias congeladas; o número dessas janelas não está
em CSV do repositório e não é citado. As tolerâncias dos casos de borda foram calibradas no erro da
emulação nos mesmos sinais, evidência mais fraca, e algumas são largas no float32 (proeminência de
modulação e grupos de ordens). A grade de {VParStepCents} cents e a faixa de {VParFmin} a
{VParFmax} Hz do rastreador foram escolhidas olhando o `train/` do IDMT.

## Categorias de custo

Separamos quatro categorias, também nos nomes das macros. A estimativa analítica (`VEst…`) é o
tamanho da arena devolvido pela função de layout do C (`p11_host ws <n>`). A medição no PC
(`VPc…`) faz {VPcNIdmt} medições por base depois de {VPcWarmup} de aquecimento, em ciclo sobre
{VPcDistinctIdmt} janelas distintas ({VPcCompiler}, {VPcOs}, {VPcNcpu} CPUs lógicas). O executável C
roda num processo novo por janela (o aquecimento não aquece o C) e o tempo é tomado dentro dele, em
torno das chamadas do núcleo, sem a inicialização do workspace; a referência Python é cronometrada
nas mesmas janelas e não tem como excluir a própria inicialização. O build do firmware (`VBuild…`) é o `idf.py size` para ESP32 e ESP32-S3 (ESP-IDF
{VBuildIdf}, GCC {VBuildGcc}), só estático; para a variante `{VChipVariant}` o build é o do
ESP32. A medição no ESP32 (`VChip…`) foi feita num {VChipModel} (revisão {VChipRev}, {VChipCpuMhz}
MHz, {VChipFlashMb} MB de flash, {VChipPsramMb} MB de PSRAM, ESP-IDF {VChipIdf}), com rádios nunca
iniciados. As janelas chegam pela serial e o tempo é tomado com o temporizador do chip só em torno
de cada chamada do núcleo, junto com as marcas de heap e de pilha; a latência usa {VChipLatBldcN}
janelas de {VParSegBldc} s em ciclo sobre {VChipLatDistinct} janelas distintas, depois de
{VChipLatWarmup} de aquecimento, que ficam fora do resumo. O fluxo contínuo lê as janelas da flash
do próprio chip. Só essa última categoria pode ser chamada de "medido no ESP32". Energia não é
medida e não é estimada por datasheet.
"""

RESULTADOS = """# Resultados (artigo ICIT 2027, `embarcados/`)

> Gerado por `embarcados/src/python/gerar_artigo_textos.py` a partir de `paper/numeros.json`
> (commit {COMMIT}). Não editar à mão. Validação apenas; o teste oficial do IDMT não foi lido.
> Os números do ESP32 vêm da bancada física, com a variante `{VChipVariant}`.

## Equivalência Python × C (medição no PC)

Em double, as {VEqStagesTotal} verificações de etapa passam em todas as {VEqCases} janelas e nas
{VEqSeqCombos} configurações de sequência ({VEqDoubleStagesPass} de {VEqStagesTotal}); a maior razão
erro/tolerância é {VEqDoubleMaxRatio}. Caminho de rotação, decisão e todas as saídas do HMM
coincidem exatamente. Em float32, que é o build do microcontrolador, as {VEqFloatRealAllPass} de
{VEqRealCases} janelas de áudio real (limpas e ruidosas) passam em todas as etapas. As falhas
restantes estão em {VEqFloatBadCases} janelas, ambas o caso de borda sintético de nível constante
com ruído (uma por modelo: {VEqFloatTieCaseIds}). Ali o envelope é quase constante e o espectro de
modulação fica no piso de arredondamento do float32 nas duas implementações, de modo que a posição
do pico é indeterminada nessa precisão: o pico da referência vale {VEqDcRefPeakRatio} vezes a
potência da referência no bin escolhido pelo C, então não é empate. A frequência de modulação muda
{VEqFloatFmodErr} Hz. Sob a regra congelada o comparador marcou PASS_EMPATE em `env_mod` e
`features` e FAIL em `z`; adotamos a leitura conservadora e contamos as três etapas
({VEqFloatStrictFailStages}) como FAIL nesses casos, o que deixa {VEqFloatStagesPassStrict} de
{VEqStagesTotal} etapas passando em todas as janelas (erro máximo de z {VEqFloatZErrMax}). As
tolerâncias e o comparador não foram alterados. Cossenos, probabilidades e decisão passam em
todas as janelas ({VEqFloatDecisionPass} de {VEqCases}). Com a variante nomeada `p11_c_sosd`, as
{VEqSosdStagesPass} verificações passam; ela reduz o erro do envelope dizimado (máximo relativo ao pico, mediana entre
janelas) de {VEqEnvDecFloatMed} para {VEqEnvDecSosdMed}, porque os polos do passa-baixa ficam perto
do círculo unitário e o arredondamento em float32 se acumula no estado do filtro. A afirmação que os
dados sustentam é concordância etapa a etapa dentro de tolerâncias congeladas antes da comparação
formal, com a exceção acima; não é equivalência incondicional. A auditoria independente (agente F)
reproduziu a comparação a partir de um build limpo, e as correções dela estão aplicadas aqui.

A variante `{VChipVariant}` foi comparada com as mesmas tolerâncias do float32, primeiro no PC e
depois no chip. No PC, as {VEqLmCases} janelas de {VParSegBldc} s do ouro passam nas {VEqLmStages}
etapas comparadas ({VEqLmStagesPass} sem falha; maior razão erro/tolerância {VEqLmMaxRatio};
decisão igual em {VEqLmDecisionPass} de {VEqLmCases}). No ESP32, as mesmas {VChipEqCases} janelas
({VChipEqCasesReal} de áudio real e {VChipEqCasesEdge} casos de borda) passam nas {VChipEqStages}
saídas que o firmware devolve pela serial (caminho e escore do Viterbi, envelope de modulação,
features, z, cossenos, probabilidades e decisão), com {VChipEqFail} falhas, decisão igual em
{VChipEqDecisionPass} de {VChipEqCases} e maior razão erro/tolerância {VChipEqMaxRatio}. As
{VChipEqStagesAbsent} etapas intermediárias não voltam do chip e ficam ausentes, e as janelas de
{VParSegIdmt} s do IDMT não couberam na placa, então não há comparação no chip para elas. A
auditoria independente não cobriu os resultados do chip.

## Custo

Medido no PC, o C float32 leva {VPcCIdmtMed} ms por janela de {VParSegIdmt} s (p95
{VPcCIdmtPninetyfive} ms, máximo {VPcCIdmtMax} ms) e {VPcCBldcMed} ms por janela de
{VParSegBldc} s, contra {VPcPyIdmtMed} ms e {VPcPyBldcMed} ms da referência Python; a razão
({VPcSpeedupIdmt} vezes) inclui o custo do interpretador e não é uma comparação de algoritmos. No PC,
STFT por quadro com soma harmônica fica com {VPcCIdmtTrackPct} % da mediana, o Viterbi de rotação
com {VPcCIdmtVitPct} % e o ramo do envelope com {VPcCIdmtEnvPct} %; features e cosseno são
desprezíveis. O fator de tempo real no PC ({VPcCIdmtRtf}) não diz nada sobre o microcontrolador.

A estimativa analítica da arena em float32 é {VEstWsFloatThreeMb} MB para {VParSegIdmt} s e
{VEstWsFloatOneMb} MB para {VParSegBldc} s ({VEstWsFloatThreeNoTabMb} e {VEstWsFloatOneNoTabMb} MB
sem a tabela de interpolação harmônica, que ocupa {VEstHarmTableMb} MB); os buffers do Hilbert
dominam (fase do envelope = {VEstWsFloatThreePhaseBPct} % do total em {VParSegIdmt} s). Em double
seriam {VEstWsDoubleThreeMb} e {VEstWsDoubleOneMb} MB. Numa unidade só, a arena
({VEstWsFloatThreeKib} e {VEstWsFloatOneKib} KiB) é pelo menos {VBuildEspsWsRatio} e
{VBuildEspsWsRatioOne} vezes a região DIRAM interna inteira que o build reporta para o ESP32-S3
({VBuildEspsDramTotalKb} KiB, dos quais só {VBuildEspsDramFreeKib} KiB sobram depois da alocação
estática; a IRAM do S3 está {VBuildEspsIramPct} % usada), e {VBuildEspWsRatio} e
{VBuildEspWsRatioOne} vezes a DRAM do ESP32 ({VBuildEspDramTotalKb} KiB): a arena contígua da
referência (`p11_c_ref`) exige PSRAM. No build do firmware da referência, a imagem ocupa
{VBuildEspBinKb} KiB (ESP32, {VBuildEspBinPct} % da partição) e {VBuildEspsBinKb} KiB (ESP32-S3,
{VBuildEspsBinPct} %); o uso estático de RAM interna é {VBuildEspDramKb} KiB de DRAM mais
{VBuildEspIramKb} KiB de IRAM no ESP32 e {VBuildEspsDramKb} KiB de DIRAM no ESP32-S3. O firmware da
variante `{VChipVariant}` para o ESP32 ocupa {VBuildLmBinKb} KiB ({VBuildLmBinPct} % da partição;
{VBuildLmFlashCodeKb} KiB de código e {VBuildLmFlashDataKb} KiB de dados constantes na flash) e usa,
de forma estática, {VBuildLmDramKb} de {VBuildLmDramTotalKb} KiB de DRAM e {VBuildLmIramKb} de
{VBuildLmIramTotalKb} KiB de IRAM. A memória de trabalho é alocada em tempo de execução e não entra
nesses números.

Medido no ESP32 ({VChipModel}, {VChipCpuMhz} MHz, {VChipPsramMb} MB de PSRAM), a referência não
cabe: a arena estimada é pelo menos {VChipRefOverPoolOne} vezes ({VParSegBldc} s) e
{VChipRefOverPoolThree} vezes ({VParSegIdmt} s) a RAM interna que o firmware consegue entregar ao
pool ({VChipPoolKib} KiB, medida). Com a variante `{VChipVariant}`, a janela de {VParSegBldc} s do
A2212 cabe: o pico do pool é {VChipWsBldcB} bytes ({VChipWsBldcKib} KiB, {VChipWsPoolPct} % do
pool), nada vai para PSRAM, o heap interno mínimo livre durante o processamento é
{VChipHeapIntBldcB} bytes e a folga mínima de pilha é {VChipStackBldc} bytes. A janela de
{VParSegIdmt} s do IDMT ({VChipIdmtSamples} amostras) não cabe nesta placa: o plano de alocação não
encontra lugar com {VChipIdmtFreeAtFailKib} KiB livres, e por isso latência, memória e
equivalência do IDMT no chip ficam como "{VChipLatIdmtMed}", sem número. A latência mediana por
janela de {VParSegBldc} s é {VChipLatBldcMed} ms (p95 {VChipLatBldcPninetyfive} ms, máximo
{VChipLatBldcMax} ms, {VChipLatBldcN} janelas), ou seja, um fator de tempo real de
{VChipRtfBldc}: o chip leva {VChipRtfBldc} vezes a duração do áudio e não opera em tempo real nesta
placa. A STFT por quadro com soma harmônica leva {VChipBldcTrackMed} ms ({VChipBldcTrackPct} % da
mediana), o ramo do envelope {VChipBldcEnvMed} ms ({VChipBldcEnvPct} %), o Viterbi de rotação
{VChipBldcVitMed} ms ({VChipBldcVitPct} %), as features {VChipBldcFeatMed} ms e a normalização
{VChipBldcPrepMed} ms; cosseno e saliência no caminho são desprezíveis. No PC, a etapa de rastreio
da referência (com tabela) fica com {VPcCBldcTrackPct} % do tempo do C. Atribuímos o peso dela no
chip ao fato de a variante recalcular as posições harmônicas em double sem tabela, e o ESP32 só tem double
por emulação em software; o tempo da variante no PC não foi medido. Os
caminhos para fechar a diferença são uma variante que faça essas contas em float (que teria de ser
verificada como variante nomeada) ou uma placa com PSRAM, onde a tabela da referência cabe.
Energia não foi medida.

## Classificação nos três protocolos

No IDMT todos os métodos ficam perto do teto nos dois protocolos. O baseline de nível com RMS
normalizado, que só vê o fator de crista, chega a {VDiagBalRmsNormIdmtBlk} de acurácia balanceada nos
blocos ({VDiagBalRmsNormIdmtBlkLo}–{VDiagBalRmsNormIdmtBlkHi}), e a p11 e as clássicas chegam a
{VDiagBalDecIdmtBlk} e {VDiagBalClsIdmtBlk}. Como cada classe é uma gravação de um motor, esses
números medem a identidade da gravação e não ordenam os métodos.

No A2212 com split por arquivo, as clássicas chegam a {VDiagBalClsBldcFile}
({VDiagBalClsBldcFileLo}–{VDiagBalClsBldcFileHi}), a p11 a {VDiagBalDecBldcFile}
({VDiagBalDecBldcFileLo}–{VDiagBalDecBldcFileHi}) e o nível bruto a {VDiagBalRmsRawBldcFile}. A
p11 erra sobretudo em Healthy (sensibilidade {VDiagRecDecBldcFileHealthy}). O split aleatório
infla a acurácia balanceada em {VDiagInflDecBldc} na p11 e {VDiagInflClsBldc} nas clássicas. Com o
motor fora, todos caem: as clássicas para {VDiagBalClsBldcMotor} e a p11 para
{VDiagBalDecBldcMotor}, com sensibilidade de Bearing {VDiagRecDecBldcMotorBearing} por construção.
Sem o fold do M4 (Healthy × Propeller em motor não visto), as clássicas ficam em
{VDiagBalClsBldcNoMfour}, o nível com RMS normalizado em {VDiagBalRmsNormBldcNoMfour} e a p11 em
{VDiagBalDecBldcNoMfour}. Comparando igual por igual (Healthy e Propeller dos motores M1–M3 no
split por arquivo: clássicas {VDiagBalClsBldcFileHP}, p11 {VDiagBalDecBldcFileHP}, nível com RMS
normalizado {VDiagBalRmsNormBldcFileHP}), tirar o motor custa {VDiagDropHPClsBldc} às clássicas e
{VDiagDropHPDecBldc} à p11; a queda grande do protocolo de motor fora completo vem sobretudo da
classe Bearing, que perde o seu único motor de treino. A p11 não supera as features clássicas em
nenhum protocolo (empata no teto do IDMT) e, com motor fora, a estimativa pontual dela fica abaixo
também do baseline de nível, com intervalos sobrepostos. ROC-AUC macro da p11 por arquivo:
{VDiagAucDecBldcFile}; das clássicas: {VDiagAucClsBldcFile}. No protocolo de motor fora a AUC junta
scores de modelos de folds diferentes e, no fold do M4, p_Bearing = 0 é imposto; por isso não a
usamos para ordenar métodos. Em toda linha com Bearing, a classe está confundida com o motor M4.

## Rotação estimada (interpretabilidade)

No IDMT a rotação mediana rastreada é {VRotIdmtGoodMed}, {VRotIdmtBrokenMed} e {VRotIdmtHeavyMed}
Hz para good, broken e heavyload, com IQR de {VRotIdmtGoodIqr}, {VRotIdmtBrokenIqr} e
{VRotIdmtHeavyIqr} Hz; {VRotIdmtGoodOctPct} % das janelas good ficam acima de {VRotOctThr} vezes a
mediana da classe (todas entre {VRotOctRatioMin} e {VRotOctRatioMax} vezes), um erro de oitava.
Como classe = motor no IDMT, essas diferenças não podem ser atribuídas ao estado. No A2212, juntando
os {VDataBldcRecordings} arquivos, a correlação de postos entre a mediana por arquivo e a velocidade
que dá nome aos arquivos é {VRotBldcSpearman}, mas dentro de cada motor ela é positiva (M1
{VRotBldcSpearmanMOne}, M2 {VRotBldcSpearmanMTwo}, M3 {VRotBldcSpearmanMThree}, M4
{VRotBldcSpearmanMFour}; {VRotBldcFilesMOne}, {VRotBldcFilesMTwo}, {VRotBldcFilesMThree} e
{VRotBldcFilesMFour} arquivos): o valor global perto de zero vem de diferenças de nível entre
motores (efeito de Simpson). O nível absoluto não é comparável entre motores e {VRotBldcCeilPct} %
das janelas ficam a menos de {VRotCeilPctThr} % do limite superior da faixa. A faixa e a grade foram
escolhidas no IDMT; no A2212 a saída do rastreador não deve ser lida como a velocidade do eixo. Arquivos do mesmo motor não são
independentes, então a correlação é descritiva.

## Decisão em fluxo

Numa sequência de {VPcSeqWindows} janelas de {VParSegBldc} s de validação do A2212
({VPcSeqPerClass} por classe, Healthy → Propeller → Bearing), o HMM online com penalidade
{VPcSeqCp} detectou as {VPcSeqSwitches} trocas com atraso de {VPcSeqDelayC} janelas no C e
{VPcSeqDelayPy} no Python, com decisões iguais (diferença máxima de probabilidade
{VPcSeqMaxProbDiff}), medido no PC. No ESP32, a mesma sequência gravada na flash do chip e
processada janela a janela pela variante `{VChipVariant}` detectou as {VChipSeqSwitches} trocas com
atraso de {VChipSeqDelay} janelas, e a decisão online do chip coincide com a do Python em
{VChipSeqSame} de {VChipSeqWindows} janelas; o passo do HMM leva {VChipSeqHmmUs} µs (mediana). As
janelas são as limpas do ouro, quantizadas em 16 bits na imagem (o PC processou o mesmo áudio
desquantizado), e Bearing é o motor M4; isso confere a implementação, não o detector.

## O que as bases permitem concluir

O IDMT sem o seu teste não sustenta nenhuma afirmação sobre discriminar estado de operação: uma
gravação por classe torna classe, motor e sessão a mesma variável, e o baseline de dois números é
tão bom quanto qualquer modelo. O A2212 com split por arquivo sustenta afirmações sobre o estado de
motores já vistos no treino, e ali as features clássicas são melhores que a p11. Generalizar para
um motor novo é a pergunta relevante para um monitor implantado, e com {VDataBldcMotors} motores,
um defeito de rolamento num só deles e faixas de velocidade diferentes por classe, os dados só
mostram que todo método piora. Nenhum experimento trata de predição de falha futura e nenhum usa
motores industriais. Nos nossos dados o split aleatório inflou pouco a acurácia do A2212
({VDiagInflClsBldc} a {VDiagInflDecBldc}), e tirar o motor, igual por igual em Healthy × Propeller,
custou um valor parecido ({VDiagDropHPClsBldc} às clássicas, {VDiagDropHPDecBldc} à p11); a queda
de {VDiagDropClsBldc} e {VDiagDropDecBldc} do protocolo de motor fora completo é dominada pela
classe Bearing, que só existe no M4.
"""


def preencher(modelo, n, commit):
    def sub(m):
        k = m.group(1)
        if k == "COMMIT":
            return commit
        if k not in n:
            raise KeyError(f"marcador sem macro: {k}")
        return n[k]["pt"].replace("\\_", "_")
    return re.sub(r"\{(V[A-Za-z]+|COMMIT)\}", sub, modelo)


def main():
    j = C.ler_json(C.PAPER / "numeros.json")
    n, commit = j["numeros"], j["projeto"]["commit"][:7]
    for nome, modelo in (("METODOLOGIA.md", METODOLOGIA), ("RESULTADOS.md", RESULTADOS)):
        (C.PAPER / nome).write_text(preencher(modelo, n, commit), encoding="utf-8")
    print("textos: METODOLOGIA.md, RESULTADOS.md")


if __name__ == "__main__":
    main()
