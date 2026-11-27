# Inventário das bases

Escrito em 14/10/2026 a partir da leitura dos arquivos, não da documentação. Onde a
documentação diverge do que foi medido, as duas versões estão registradas. Todos os números
deste arquivo saíram de `soundfile.info` (cabeçalho) ou de leitura das amostras **apenas do
lado de treino**; nenhuma amostra de `test/` ou `test_cut/` foi lida.

## 1. IDMT-ISA-ELECTRIC-ENGINE

Origem: `data/raw/IDMT-ISA-ELECTRIC-ENGINE.zip` (1 455 MB, sha256
`9f070747cf92fff80bee0e3fe78ea3a42d9599f190bdd707abc4615de7c2910b`), descompactado em
`data/idmt/`. Havia **um** zip na pasta, não dois; a questão da duplicata não se aplicou.
O zip lista 2 409 arquivos e a pasta descompactada tem os mesmos 2 409.

### 1.1 Árvore

```
data/idmt/
├── IDMT_ISA_ELECTRIC_ENGINE_discription.pdf
├── train/      engine1_good/pure.wav
│               engine2_broken/pure.wav
│               engine3_heavyload/pure.wav
├── train_cut/  engine{1_good,2_broken,3_heavyload}/pure_<k>.wav          k = 0, 1, 2, ...
├── test/       engine{...}/<cond>.wav
│               cond ∈ {atmo_low, atmo_medium, atmo_high, talking_1..4, whitenoise_low, stresstest}
└── test_cut/   engine{...}/<cond>_<k>.wav
```

### 1.2 O que diz o PDF de descrição

Versão 1.0, 2017, gravado em março de 2017 no Fraunhofer IDMT. Motor 2ACT Brushless DC
42BLF01, 4000 RPM, 24 VDC. **"three similar units of an electrical engine"**; os estados
good, heavy load e broken "were provoked by a change of supply voltage and loading weight".
"In each file, only one of the engines is active". Condições: pure, talking, white_noise,
atmo (low, medium, high) e stress test ("slightly changed input gains ... and people knocking
on the casing"). Duração total declarada 42,32 min; 774 good, 789 broken, 815 heavyload,
2 378 wav; 44,1 kHz, **32 bit**, mono. Licença CC BY-SA 4.0. O PDF não descreve a
convenção de nomes nem diz que a tensão é chaveada a cada 18 ms (essa informação precisa ser confirmada no artigo EUSIPCO de Grollmisch et al., 2019).

### 1.3 O que foi medido

| Split | Classe | Arquivos | Formato | Duração total | Por arquivo |
|---|---|---|---|---|---|
| train | engine1_good | 1 (`pure`) | 44,1 kHz, PCM_24, mono | 315,8 s | — |
| train | engine2_broken | 1 (`pure`) | idem | 372,9 s | — |
| train | engine3_heavyload | 1 (`pure`) | idem | 385,8 s | — |
| train_cut | engine1_good | 105 | 44,1 kHz, PCM_16 | 315 s | 3,000 s |
| train_cut | engine2_broken | 124 | idem | 372 s | 3,000 s |
| train_cut | engine3_heavyload | 128 | idem | 384 s | 3,000 s |
| test | cada classe | 9 (uma por condição) | PCM_24; `stresstest` em FLOAT | 2 015 / 2 003 / 2 073 s | 123–421 s |
| test_cut | engine1_good | 669 | PCM_16 | 2 007 s | 3,000 s |
| test_cut | engine2_broken | 665 | PCM_16 | 1 995 s | 3,000 s |
| test_cut | engine3_heavyload | 687 | PCM_16 | 2 061 s | 3,000 s |

Recortes do `test_cut/` por condição (good / broken / heavyload): atmo_high 70/70/70,
atmo_low 72/68/70, atmo_medium 70/74/70, stresstest 41/41/41, talking_1 70/72/86,
talking_2 72/71/70, talking_3 70/70/70, talking_4 70/70/70, whitenoise_low 134/129/140.

Duração das gravações de `test/` (s), good / broken / heavyload: atmo_high 210,5/210,6/211,9;
atmo_low 216,5/205,7/210,6; atmo_medium 211,9/224,0/212,4; stresstest 123,2/123,9/123,5;
talking_1 212,0/216,1/259,8; talking_2 216,3/213,4/211,0; talking_3 210,7/210,7/210,7;
talking_4 210,7/210,8/212,1; whitenoise_low 403,4/387,9/420,6.

Divergências em relação ao PDF: a resolução real é 24 bit inteiro (32 bit float só no
`stresstest`), e os recortes são 16 bit. O total de recortes (357 + 2 021 = 2 378) bate com
o PDF; a soma por classe também (105 + 669 = 774 etc.).

### 1.4 Gravação de origem de cada recorte

Verificado amostra a amostra no treino: `train_cut/<classe>/pure_<k>.wav` é exatamente o
trecho `[3k, 3k+3)` s de `train/<classe>/pure.wav` (erro máximo < 1e-3, a quantização para
16 bit), para todo k de 0 até o último, sem lacunas e sem sobreposição. A sobra final (< 3 s)
foi descartada pelos autores. Para o teste, por nome, `test_cut/<classe>/<cond>_<k>.wav` sai
de `test/<classe>/<cond>.wav`; isso **não** foi verificado por conteúdo para não ler o teste.

Identificador de gravação adotado em `motorsound/splits.py`: `idmt/<split>/<classe>/<cond>`.

### 1.5 Consequências para o protocolo (achados, não suposições)

1. **O `train/` tem uma única gravação por classe, toda na condição `pure`.** Não existe
   validação "de dentro do `train/`, agrupada por gravação" que contenha as três classes:
   tirar uma gravação tira uma classe inteira. A regra de agrupar por gravação não é satisfazível
   como está escrita; ver a decisão registrada em §3.
2. **Classe e unidade física coincidem.** O PDF fala em três unidades do motor e as pastas são
   `engine1_good`, `engine2_broken`, `engine3_heavyload`. Tudo indica que good foi sempre o
   motor 1, broken o 2 e heavyload o 3, inclusive no teste. Um classificador pode estar
   reconhecendo a unidade e não o estado. É a mesma confusão do rolamento no A2212, mas aqui
   atinge as três classes. Toda tabela do IDMT deve declarar isso.
3. **O nível de sinal separa uma classe sozinho.** No treino, o RMS é 0,088 (good),
   0,057 (broken) e 0,260 (heavyload); o pico é 0,33 / 0,30 / 1,00 (heavyload tem 4·10⁻⁶ das
   amostras com |y| > 0,99). Pipelines sem normalização de nível ganham um atalho. O runner
   registra o desempenho com e sem normalização RMS.
4. **Treino e teste são sessões diferentes.** Treino é só `pure`; nenhuma condição do teste
   aparece no treino. "Generalização entre condições" no IDMT oficial é treino em pure,
   teste em todas as outras — exatamente o que o split oficial já faz.

## 2. BLDC A2212 (Estacio et al., Data in Brief 50, 2023)

Origem: `data/raw/a2212.zip` (14,5 MB, sha256
`8162fee39bfbd4e45bb703d032c5e6013f876a712369471dba233ba83bd1c82b`, renomeado de
`Brushless DC Motor sound dataset for PdM.zip`), descompactado em
`data/a2212/BLDC_sound_data/<Classe>/<Motor>/<velocidade>.wav`. Não há arquivo de
documentação dentro do zip.

| Classe | Motor | Velocidades (nome do arquivo) | Arquivos |
|---|---|---|---|
| Healthy | M1 | 1400 1450 1550 1650 1750 1850 1950 2000 | 8 |
| Healthy | M2 | 1400 1450 1550 1650 1750 1850 1950 2000 | 8 |
| Healthy | M3 | 1450 1550 1650 1750 1850 1950 | 6 |
| Propeller | M1 | 1400 1450 1550 1650 1750 1850 | 6 |
| Propeller | M2 | 1400 1450 1550 1650 1750 1850 1950 2000 | 8 |
| Bearing | M4 | 1700 1750 1800 1850 1900 1950 2000 | 7 |

Total 43 arquivos, 10,1–10,5 s cada, **estéreo**, PCM_16, **16 kHz exceto
`Healthy/M1/1400.wav`, que está a 44,1 kHz**. O loader converte para mono pela média dos
canais e reamostra para 16 kHz. Os níveis são baixos (RMS 0,001–0,01) exceto
`Propeller/M1` a partir de 1550, que satura (pico 1,0 em 1750 e 1850; RMS 0,325 em 1850).
O nome do arquivo é a velocidade comandada; a unidade (RPM ou valor de PWM) precisa ser
confirmada na ficha `estacio2023`.

Consequências: o rolamento só existe no motor 4 e o motor 4 só tem rolamento, então
"Bearing" é indistinguível de "M4" com estes dados. A hélice só existe em M1 e M2, que também
têm Healthy, então Healthy × Propeller é a única comparação dentro do mesmo motor. A
saturação de `Propeller/M1` é outro atalho de nível.

## 3. Decisão sobre a validação do IDMT

Decidido em 14/10/2026: **blocos temporais**. Cada `train/<classe>/pure.wav` é
cortado em segmentos de 3 s (os mesmos do `train_cut/`) e os segmentos são agrupados em 5
blocos contíguos, alinhados por posição relativa entre as classes. A validação cruzada deixa
um bloco de fora; o primeiro segmento de cada bloco a partir do segundo é descartado como
guarda, de modo que nenhum segmento de treino é vizinho imediato de um de validação. O grupo
é (gravação, bloco), não gravação, e `tests/test_splits.py` falha se um grupo cair nos dois
lados. Limitações declaradas: a validação mede só `pure`, dentro da mesma sessão de gravação
e com classe = unidade física; é otimista por construção. As condições de ruído do IDMT só
aparecem no teste oficial, lido uma vez depois do congelamento; na validação, as condições
são simuladas pela bancada de ruídos (Etapa 5). O `test/` fica trancado por código
(`motorsound/io.py`, variável `MOTORSOUND_ABRIR_TESTE`).

A2212: `StratifiedGroupKFold` de 5 dobras por arquivo, segmentos de 1 s. Em paralelo,
deixa-um-motor-de-fora, que expõe a confusão rolamento = M4.
