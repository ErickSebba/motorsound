# MotorSound — estado de motores elétricos pelo som

Pesquisa que leva as técnicas de transcrição musical do [Decifrae](https://decifrae.com.br) (pYIN,
soma harmônica, Viterbi) para outro domínio: **identificar o estado de operação de motores
elétricos a partir do áudio**, com o objetivo final de embarcar o classificador num ESP32 como
monitor preventivo de baixo custo. É o terceiro artigo do Decifrae, escrito para o IEEE ICIT 2027.

*In English: research project that applies music-transcription techniques (pYIN, harmonic sum,
Viterbi) to classify the operating state of electric motors from their sound. It shows that the
usual random split inflates accuracy on public datasets by up to 18 points, that a rotation-speed
feature is the most noise-robust one tested, and that the feature extractor runs on an ESP32,
though not in real time.*

![Acurácia por SNR com ruído branco e rosa no IDMT: a feature de rotação (p20b_so_frot) é a mais robusta](out/figuras/snr_idmt.png)

## O que o projeto faz

- **25 pipelines comparadas no mesmo protocolo:** linhas de base simples (RMS e pico, forma
  espectral), reimplementações de trabalhos da literatura (CNNs com log-mel e MFCC, ETNeXt,
  features clássicas) e as pipelines Decifrae (rastreador de rotação por soma harmônica, pYIN e
  Viterbi, decodificação de estados com HMM).
- **Auditoria de vazamento:** cada pipeline roda com o split aleatório usado na literatura e com
  splits por gravação e por motor, para medir quanto da acurácia vem do split.
- **Robustez a ruído:** treino limpo e validação com ruído branco, rosa, percussivo, reverberação
  e desafinação, em várias SNRs.
- **Custo embarcado:** estimativa de MFLOP por segundo de áudio e de memória de cada pipeline.

## Resultados principais

Medidos só na validação das bases IDMT-ISA-ELECTRIC-ENGINE e BLDC A2212:

- **As bases públicas medem menos do que parece.** No IDMT cada classe é um motor gravado numa
  sessão só, e dois números (RMS e pico) bastam para 99 % na validação. No A2212 o split aleatório
  infla a acurácia em até 18 pontos, e deixar um motor de fora derruba os métodos para 0,33–0,78.
  Os trabalhos que reportam 96–100 % nessas bases usam o split que infla.
- **O rastreador de rotação vindo do transcritor de baixo funciona:** acerta a rotação sintética
  com erro abaixo de 0,1 % e é a feature mais robusta ao ruído da comparação, a única acima de
  0,9 com ruído branco a 0 dB.
- **A pipeline Decifrae completa não supera a literatura:** fica cerca de 20 pontos atrás das CNNs
  e das features clássicas no A2212. A contribuição é a feature robusta e barata, não a pipeline
  inteira.

### No ESP32

O extrator de features da pipeline Decifrae (`p11_decifrae_harmsum`) foi portado para C e medido
num ESP32 clássico (240 MHz, 4 MB de flash, sem PSRAM), na branch
[`embarcado-icit`](../../tree/embarcado-icit):

- **Equivalência com o Python:** no chip, 79 de 79 casos com a mesma decisão de classe. No PC, o
  build em double passa em todas as etapas; o build em float32 falha em dois casos sintéticos de
  borda (DC + ruído, com o pico do espectro no piso numérico), sem mudar a decisão de classe.
- **Memória:** só cabe com janelas de 1 s (base A2212): 296 KiB de área de trabalho, 96 % do que
  sobra de RAM. A janela de 3 s do IDMT não cabe.
- **Tempo:** 2.647 ms (mediana de 200 execuções) para processar 1 s de áudio, ou seja, **2,65×
  mais lento que o tempo real**. 89 % do tempo é o rastreador de rotação.

O porte mostra que a pipeline é embarcável em memória, mas precisa de otimização (ou de um chip
com PSRAM e mais RAM) para rodar em tempo real.

![Custo estimado por segundo de áudio × acurácia com ruído rosa a 10 dB](out/figuras/pareto_custo.png)

## Como rodar

Testado com Python 3.14. As bases não fazem parte do repositório: baixe a IDMT-ISA-ELECTRIC-ENGINE
e a base BLDC A2212 e coloque-as em `data/idmt/` e `data/a2212/BLDC_sound_data/`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python

# testes (os de split precisam das bases em data/)
.venv/Scripts/python -m pytest tests

# matriz de validação e auditoria de vazamento
.venv/Scripts/python experimentos/validacao.py --pipelines b00_nivel b01_forma_espectral p11_decifrae_harmsum

# robustez a ruído
.venv/Scripts/python experimentos/robustez.py --pipelines p11_decifrae_harmsum b01_forma_espectral --base idmt

# tabelas e figuras a partir dos CSVs de out/experiments/
.venv/Scripts/python experimentos/tabelas_resultados.py
.venv/Scripts/python experimentos/analise.py
```

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `motorsound/` | núcleo: leitura das bases, splits sem vazamento, rotação, HMM, classificador, métricas, cache |
| `pipelines/` | as 25 pipelines (`b0x` linhas de base, `p0x` literatura, `p1x` Decifrae, `p2x` rotação e envelope, `p3x` anomalia) |
| `ruidos/` | bateria de ruídos (branco, rosa, Lee & Shen) e sinais sintéticos |
| `experimentos/` | validação, robustez, sequências, estatística, tacômetro, tabelas |
| `artigo-2027/` | scripts da auditoria de mais seis bases públicas (KAIST, UOEMD, MIMII, ToyCar, IMVACD, MaFaulDa) para o artigo completo |
| `Embarcado/` | primeiro porte em C do núcleo de features, feito pelo [DmZ-git](https://github.com/DmZ-git); o porte completo e medido está na branch `embarcado-icit` |
| `out/` | resultados dos experimentos (CSVs comprimidos) e figuras |

O porte completo para o ESP32 (C em float e double, testes de equivalência com o Python, firmware
ESP-IDF e medições de latência e memória na placa) está na branch
[`embarcado-icit`](../../tree/embarcado-icit), em `embarcados/`. O inventário das bases e as
decisões de split estão em [`docs/DATASETS.md`](docs/DATASETS.md); a lista das pipelines, em
[`docs/PIPELINES.md`](docs/PIPELINES.md).

## O que aprendi

- A parte mais difícil foi o começo: achar boa bibliografia e entender que o ruído de um motor
  pode ser tratado como um instrumento de uma corda, o que abriu caminho para usar as técnicas
  de transcrição musical.
- Se fizesse de novo, compararia outros tipos de análise de dados desde o início e procuraria
  antes onde encontrar mais dados. Assim faria mais testes antes do protótipo no ESP32 e
  economizaria tempo.

## Autores

[ErickSebba](https://github.com/ErickSebba) e [DmZ-git](https://github.com/DmZ-git) (porte em C).
