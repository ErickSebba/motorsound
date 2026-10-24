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
desempate pelo primeiro índice da referência. A única aproximação usada é a variante nomeada
`p11_c_sosd`, que mantém em double o estado do filtro do envelope dentro do build float32.

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
numérica independente de cada build; foram commitadas no commit `{VEqTolCommit}` ({VEqTolCommitTime},
sha256 `{VEqTolSha}…`) antes da comparação formal ({VEqRunTime}), e o hash vai em cada linha do
resultado. As duas emulações passaram em todas as linhas ({VEqEmulDoublePass} e {VEqEmulFloatPass}
de {VEqEmulFloatRows}) antes de o C ser comparado. Etapas contínuas são comparadas pelo erro
máximo relativo à escala da etapa; saídas discretas (caminho de rotação, decisão, estados do HMM)
têm de coincidir exatamente, salvo empate demonstrado com os valores da própria referência. Uma
falha numa etapa marca as seguintes como "a jusante". Uma ressalva de cronologia: o porte em C foi
depurado contra os vetores de ouro pelo agente C antes do congelamento (commit do porte anterior
ao das tolerâncias); o "pré-registro" vale para a comparação formal do agente D.

## Categorias de custo

Separamos quatro categorias, também nos nomes das macros. A estimativa analítica (`VEst…`) é o
tamanho da arena devolvido pela função de layout do C (`p11_host ws <n>`). A medição no PC
(`VPc…`) cronometra {VPcNIdmt} janelas por base depois de {VPcWarmup} de aquecimento, em ciclo sobre
{VPcDistinctIdmt} janelas distintas, dentro do executável e em torno das chamadas do núcleo
({VPcCompiler}, {VPcOs}, {VPcNcpu} CPUs lógicas); a referência Python é cronometrada nas mesmas
janelas. O build do firmware (`VBuild…`) é o `idf.py size` para ESP32 e ESP32-S3 (ESP-IDF
{VBuildIdf}, GCC {VBuildGcc}), só estático. A medição no ESP32 (`VChip…`) manda as janelas pela
serial e mede com o temporizador do chip só em torno do processamento, junto com as marcas de heap
e de pilha. Só essa última categoria pode ser chamada de "medido no ESP32", e ela ainda está
pendente. Energia não é medida e não é estimada por datasheet.
"""

RESULTADOS = """# Resultados (artigo ICIT 2027, `embarcados/`)

> Gerado por `embarcados/src/python/gerar_artigo_textos.py` a partir de `paper/numeros.json`
> (commit {COMMIT}). Não editar à mão. Validação apenas; o teste oficial do IDMT não foi lido.
> Valores **[pendente]** dependem da bancada do ESP32 físico.

## Equivalência Python × C (medição no PC)

Em double, as {VEqStagesTotal} verificações de etapa passam em todas as {VEqCases} janelas e nas
{VEqSeqCombos} configurações de sequência ({VEqDoubleStagesPass} de {VEqStagesTotal}); a maior razão
erro/tolerância é {VEqDoubleMaxRatio}. Caminho de rotação, decisão e todas as saídas do HMM
coincidem exatamente. Em float32, que é o build do microcontrolador, passam {VEqFloatStagesPass} de
{VEqStagesTotal} verificações. A exceção é o z-score, que falha em {VEqFloatZFail} de {VEqCases}
janelas, ambas o caso de borda sintético de nível constante com ruído (uma por modelo:
{VEqFloatTieCaseIds}). Ali o envelope é quase constante e o espectro de Welch tem dois picos cujas
potências na referência diferem de {VEqFloatTieGap} (tolerância de empate {VEqFloatTieTol}); o C
escolhe o outro pico, a feature de frequência de modulação muda {VEqFloatFmodErr} Hz e o z-score,
cuja tolerância é propagada das features e não herda a exceção de empate, falha (erro máximo
{VEqFloatZErrMax}). Cossenos, probabilidades e decisão passam em todas as janelas
({VEqFloatDecisionPass} de {VEqCases}). Com a variante nomeada `p11_c_sosd`, as {VEqSosdStagesPass}
verificações passam; ela reduz o erro do envelope dizimado (máximo relativo ao pico, mediana entre
janelas) de {VEqEnvDecFloatMed} para {VEqEnvDecSosdMed}, porque os polos do passa-baixa ficam perto
do círculo unitário e o arredondamento em float32 se acumula no estado do filtro. A afirmação que os
dados sustentam é concordância etapa a etapa dentro das tolerâncias pré-registradas, com a exceção
acima; não é equivalência incondicional. A mesma comparação no chip está pendente
({VChipEqStagesPass} de {VChipEqStages} etapas; decisão {VChipEqDecisionPass}), e a auditoria
independente (agente F) também.

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
seriam {VEstWsDoubleThreeMb} e {VEstWsDoubleOneMb} MB. Isso é {VBuildEspsWsRatio} e
{VBuildEspsWsRatioOne} vezes a região DIRAM interna que o build reporta para o ESP32-S3
({VBuildEspsDramTotalKb} KiB), e {VBuildEspWsRatio} e {VBuildEspWsRatioOne} vezes a DRAM do ESP32
({VBuildEspDramTotalKb} KiB): a referência exige PSRAM, e qualquer redução (Hilbert por blocos,
tabelas menores) muda a aritmética e teria de ser verificada como variante nomeada. No build do
firmware, a imagem ocupa {VBuildEspBinKb} KiB (ESP32, {VBuildEspBinPct} % da partição) e
{VBuildEspsBinKb} KiB (ESP32-S3, {VBuildEspsBinPct} %); o uso estático de RAM interna é
{VBuildEspDramKb} KiB de DRAM mais {VBuildEspIramKb} KiB de IRAM no ESP32 e {VBuildEspsDramKb} KiB
de DIRAM no ESP32-S3. A arena é alocada em tempo de execução e não entra nesses números.

Medido no ESP32: modelo {VChipModel}, {VChipCpuMhz} MHz, PSRAM {VChipPsramMb} MB; latência mediana
{VChipLatIdmtMed} ms em {VParSegIdmt} s (p95 {VChipLatIdmtPninetyfive}, máximo {VChipLatIdmtMax})
e {VChipLatBldcMed} ms em {VParSegBldc} s; fator de tempo real {VChipRtfIdmt} e {VChipRtfBldc};
heap interno mínimo livre {VChipHeapIntIdmtKb} KiB; folga de pilha {VChipStackIdmt} bytes. Nenhuma
estimativa foi posta no lugar desses valores.

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
{VDiagBalDecBldcNoMfour}. A p11 não supera as features clássicas em nenhum protocolo e, com motor
fora, fica abaixo também do baseline de nível. ROC-AUC macro da p11 por arquivo:
{VDiagAucDecBldcFile}; das clássicas: {VDiagAucClsBldcFile}.

## Rotação estimada (interpretabilidade)

No IDMT a rotação mediana rastreada é {VRotIdmtGoodMed}, {VRotIdmtBrokenMed} e {VRotIdmtHeavyMed}
Hz para good, broken e heavyload, com IQR de {VRotIdmtGoodIqr}, {VRotIdmtBrokenIqr} e
{VRotIdmtHeavyIqr} Hz; {VRotIdmtGoodOctPct} % das janelas good ficam acima de {VRotOctThr} vezes a
mediana da classe, um erro do tipo oitava. Como classe = motor no IDMT, essas diferenças não podem
ser atribuídas ao estado. No A2212 a rotação rastreada não acompanha a velocidade que dá nome aos
arquivos: a correlação de postos entre a mediana por arquivo e esse rótulo é {VRotBldcSpearman} nos
{VDataBldcRecordings} arquivos (por motor: M1 {VRotBldcSpearmanMOne}, M2 {VRotBldcSpearmanMTwo}, M3
{VRotBldcSpearmanMThree}, M4 {VRotBldcSpearmanMFour}; {VRotBldcFilesMOne}, {VRotBldcFilesMTwo},
{VRotBldcFilesMThree} e {VRotBldcFilesMFour} arquivos), e {VRotBldcCeilPct} % das janelas ficam a
menos de {VRotCeilPctThr} % do limite superior da faixa. A faixa foi escolhida no IDMT; no A2212 a
saída do rastreador não deve ser lida como a velocidade do eixo. Arquivos do mesmo motor não são
independentes, então a correlação é descritiva.

## Decisão em fluxo

Numa sequência de {VPcSeqWindows} janelas de validação do IDMT ({VPcSeqPerClass} por classe, good →
broken → heavyload), o HMM online com penalidade {VPcSeqCp} detectou as {VPcSeqSwitches} trocas com
atraso de {VPcSeqDelayC} janelas no C e {VPcSeqDelayPy} no Python, com decisões iguais (diferença
máxima de probabilidade {VPcSeqMaxProbDiff}), medido no PC. Dada a saturação do IDMT, isso confere a
implementação, não o detector. No chip: atraso máximo {VChipSeqDelay} janelas.

## O que as bases permitem concluir

O IDMT sem o seu teste não sustenta nenhuma afirmação sobre discriminar estado de operação: uma
gravação por classe torna classe, motor e sessão a mesma variável, e o baseline de dois números é
tão bom quanto qualquer modelo. O A2212 com split por arquivo sustenta afirmações sobre o estado de
motores já vistos no treino, e ali as features clássicas são melhores que a p11. Generalizar para
um motor novo é a pergunta relevante para um monitor implantado, e com {VDataBldcMotors} motores,
um defeito de rolamento num só deles e faixas de velocidade diferentes por classe, os dados só
mostram que todo método piora. Nenhum experimento trata de predição de falha futura e nenhum usa
motores industriais. Nos nossos dados o split aleatório inflou pouco a acurácia do A2212
({VDiagInflClsBldc} a {VDiagInflDecBldc}), enquanto tirar o motor custou {VDiagDropClsBldc} às
clássicas e {VDiagDropDecBldc} à p11.
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
