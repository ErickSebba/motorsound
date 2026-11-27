# Porte C de p11_decifrae_harmsum (23–24/10/2026)

Este relatório descreve o núcleo C portável `p11` em `embarcados/src/c/`. Ele **não declara
equivalência** com a referência Python: a verificação é feita à parte, com tolerâncias
congeladas antes (`c_equivalence.md`). Os números de comparação abaixo saem de scripts de depuração
(`tests/unit/depurar_gold.py`, `tests/unit/depurar_seqs.py`) e dos testes unitários
(`tests/unit/test_p11.py`); servem para localizar erro, não como prova.

## 1. Arquivos

| arquivo | conteúdo |
|---|---|
| `src/c/p11_tipos.h` | `p11_real` (float ou double por `-DP11_REAL`), `p11_cpx`, constantes de `PreDecifrae`/`VIT_*` |
| `src/c/p11_fft.h`, `p11_fft.c` | FFT complexa de raiz mista (2, 3, 4, 5, genérica até primo 64), FFT real via meia FFT complexa, stub ESP-DSP |
| `src/c/p11.h` | API pública e a struct de workspace `p11_ws` |
| `src/c/p11_core.c` | workspace (arena), `_preparar`, STFT por quadro, soma harmônica, custo, Viterbi de rotação, envelope (Hilbert, sosfilt, Welch), 13 features |
| `src/c/p11_decisao.c` | cosseno + softmax, HMM (log-emissões, Viterbi, filtro online passo a passo), absorb_short |
| `src/c/p11_soma.h` | soma em pares com a blocagem do numpy (`pairwise_sum`) |
| `src/c/p11_constantes.h` | GERADO por `src/python/gerar_constantes.py`: SOS de `butter(4, 200, fs=16000)` em repr float64 e float32 |
| `src/c/p11_host.c` | CLI `p11_host` (caso, seq, bench, ws); único arquivo com malloc/stdio |
| `src/c/build_host.bat`, `build_variante_sosd.bat` | builds MSVC + CMake (NMake) |
| `CMakeLists.txt` | biblioteca `p11`, `p11_host`, `p11_testes` |
| `tests/unit/p11_testes.c`, `test_p11.py` | testes unitários por peça |
| `tests/unit/depurar_gold.py`, `depurar_seqs.py` | depuração contra `gold/` (não é a verificação formal) |

O porte do colaborador (`src/c/colaborador_original/`) não foi editado. Nenhum trecho dele foi
copiado: ele não tinha STFT, rastreador, Viterbi de rotação nem envelope, e as features e o
cosseno dele divergem do Python em pontos que importam (percentil em float, `searchsorted`
binário sobre `f_rot` em float, limiar extra no denominador do cosseno). O C novo foi escrito
a partir da leitura linha a linha do Python.

## 2. Mapeamento Python → C

| Python (somente leitura) | C |
|---|---|
| `motorsound/runner.py:70` `_preparar` | `p11_core.c:195` `p11_preparar` |
| `motorsound/rotacao.py:30` `grade` | `p11_core.c:145` `calc_grade` (chamada em `p11_ws_init`, `p11_core.c:160`) |
| `motorsound/rotacao.py:40` `stft_mag` | `p11_core.c:208` `p11_rastrear_quadros` (laço por quadro) + `p11_fft.c:296` `p11_rfft`; janela em `p11_ws_init` |
| `motorsound/rotacao.py:51` `saliencia_harmonica` | `p11_core.c:208` (mesmo laço); tabela `(i0, a)` em `p11_ws_init` |
| `motorsound/rotacao.py:70` `emissao_baixo` | `p11_core.c:273` `custo_j`, `:275` `custo_sil`, `:285` `p11_custo_linha` |
| `motorsound/rotacao.py:102` `viterbi_rotacao` | `p11_core.c:299` `p11_viterbi` (inclui o score do contrato §4) |
| `motorsound/rotacao.py:162` `rastrear` (f_rot) | `p11_core.c:392` `p11_f_rot` |
| `pipelines/_decifrae.py:50` `_envelope_mod` | `p11_core.c:524` `p11_envelope_mod`, `:450` `p11_sosfilt_dec`, `:475` `p11_welch` |
| `pipelines/_decifrae.py:63` `extrair` | `p11_core.c:598` `p11_features`, `:402` `p11_saliencia_caminho`, `:585` `percentil_linear` |
| `motorsound/classificador.py:29,34` `emissao`, `predict_proba`; `_decifrae.py:104` `prever` | `p11_decisao.c:18` `p11_cosseno` |
| `motorsound/hmm.py:26` `_log_emissoes` | `p11_decisao.c:46` `p11_hmm_log_em` |
| `motorsound/hmm.py:44` `decode` | `p11_decisao.c:76` `p11_hmm_decode` |
| `motorsound/hmm.py:70` `filtro_online` | `p11_decisao.c:134,142` `p11_hmm_online_init/_passo`; `np.logaddexp` em `:106` |
| `motorsound/hmm.py:96,108,129,134` `_merge`, `absorb_short`, `rotulos_para_segmentos`, `segmentos_para_rotulos` | `p11_decisao.c:163` `merge`, `:179` `p11_absorb_short` |

## 3. Decisões

**Porte por quadro.** O caminho de MCU nunca guarda S (F×T). Para cada quadro: janela
(Hann simétrico de `np.hanning`, calculado como o numpy, `0.5+0.5cos(πn/(M−1))` com
`n = 1−M, …, M−1`, convertido para float32), FFT real de 8192, |X| por `hypot` (o `np.abs`
complexo), acumulação de P_mean em t e soma harmônica do quadro. P_mean soma
sequencialmente em t e divide por T no fim, porque é assim que o numpy reduz o eixo não
contíguo de S (conferido numericamente). O CLI guarda S e sal só em modo dump.

**A janela inteira para o Viterbi.** O custo SIL usa `e_q.max()` sobre todos os quadros, então
o Viterbi só pode começar depois do último quadro. O C guarda `e_norm` (T×J) e `e_q` (T) e
calcula o custo na hora (`custo_j`, `custo_sil`), sem matriz de custo. `psi` é `int16`
(T×(J+1); J = 1201 cabe). O laço segue o Python literalmente: saltos de −50 a +50 em ordem,
depois +600 e −600, depois SIL, sempre com `<` estrito; SIL com `argmin` primeiro índice e
`<=`; `argmin` final primeiro índice. O custo de passo é calculado como
`((0,55·|o|)·2)/100` em double, a mesma ordem do numpy. O score (§4) é somado na mesma ordem
do `delta`, por isso `vit_score == vit_min_delta` bit a bit nos dois builds.

**Tabela de interpolação.** `pos = (h·f_j)/df` em double; a tabela guarda `i0` (`uint16`) e
`a` (`p11_real`) por (h, j): 48×1201×6 = 345 888 bytes no build float. Com `usar_tabela = 0`
a mesma conta é feita na hora (resultado idêntico, não é variante). Na grade usada,
`pos < F−1` vale sempre (máx. 48·60/1,953125 = 1474,6). No ESP32 a tabela pode ir para a
flash como `const`; isso ainda não foi feito.

**Índice direto em vez de `searchsorted`.** Na feature 3 o Python faz
`searchsorted(grade, f_rot)` com `side='left'`. Como `f_rot[t] = grade[path[t]]` é o próprio
elemento de uma grade estritamente crescente, o resultado é `path[t]`. O C usa `path[t]`.
Isso evita a comparação de floats que o colaborador fazia.

**Decisões de índice em double nos dois builds.** A grade, a mediana e o IQR de f_rot (a partir
de `grade[path]`), `pos = rint(k·f_med/df)` na feature 4 (`rint` = meio-para-par, como
`np.round`), as frequências do Welch e os tempos do absorb (`i·passo_s`, `rint` como o
`round()` do Python) ficam em double. São poucas operações e mudam inteiros.

**Somas.** `p11_soma.h` reproduz a soma em pares do numpy (n < 8 sequencial, n ≤ 128 com
8 acumuladores, acima disso divide). É usada onde o Python faz `sum`/`mean` sobre vetor
contíguo. `_preparar` usa a mesma soma em float32 sobre `y*y`, sem buffer, e o escalar
`0,1/r` é convertido para float32 antes do produto (regra de escalar fraco do numpy 2.5).
No build float todas as acumulações longas são em float: soma harmônica de 48 termos, P_mean
em T quadros, somas de banda (~2000 bins), centróide (4097 bins), média do envelope e Welch.

**Hilbert e Welch.** Hilbert como o scipy: FFT de n pontos sem padding, ×2 em 1..n/2−1, zero
acima de n/2, IFFT com fator 1/n e `hypot`. n = 16 000 ou 48 000 não é potência de 2, daí a
FFT de raiz mista. sosfilt em forma direta II transposta, estado zero, dizimação `[::16]`.
O Welch foi conferido no scipy 1.18.1 instalado (`_spectral_py.py`: `csd` → `ShortTimeFFT`):
janela `hann` periódica, `noverlap = nps//2`, `(n−noverlap)//passo` segmentos, detrend
constante por segmento, escala `1/(fs·Σw²)`, ×2 exceto DC e Nyquist, média entre segmentos.
`nperseg` ímpar devolve erro (não ocorre: n = 16 000 dá 1000, n = 48 000 dá 1024).

**Cosseno.** Sem o limiar no denominador do colaborador. Com a feature 2 tendo
`sd = 1e-9` no modelo IDMT, um quadro SIL dá z ≈ 1e9 e z² ≈ 1e18, que cabe em float
(máx. 3,4e38). Nos casos de silêncio do ouro o build float não gerou inf nem NaN
(`depurar_gold.py float`: nenhuma divergência de finitude). Não foi preciso calcular a norma
em double.

**HMM.** `logaddexp` igual ao `npy_logaddexp` (caso `x == y` soma `ln 2`), redução sequencial
da esquerda como `np.logaddexp.reduce`. O filtro online é passo a passo
(`p11_hmm_online_passo`), com estado de `S` reais; é o que roda no MCU. O `psi` do HMM é
`uint8` (S ≤ 255).

**Sem malloc no núcleo.** O chamador fornece uma arena; `p11_ws_bytes(n, usar_tabela)` devolve
o tamanho exato pela mesma função de layout que distribui os ponteiros. A arena tem uma parte
persistente e uma união: fase A (rastreio: quadro, e_norm, psi, delta) e fase B (envelope:
tabelas e buffers do Hilbert, Welch) ocupam a mesma memória. Por isso a feature 3 é calculada
(`p11_saliencia_caminho`) antes de `p11_envelope_mod`.

## 4. Variante `float-sosd` (P11_SOS_DOUBLE)

`P11_SOS_DOUBLE` é uma **opção de precisão nomeada, desligada por padrão** (`-DP11_SOS_DOUBLE=ON`
no CMake; script `src/c/build_variante_sosd.bat`, saída em `build-float-sosd/`). Ela não muda a
conta: só faz o estado e os coeficientes do sosfilt do envelope serem double no build float,
como o scipy, que promove a entrada para float64. Existe porque a depuração contra o ouro
mostrou que o sosfilt em float é a maior fonte de erro do build float. Os polos do Butterworth
de 200 Hz a 16 kHz ficam perto de 1 (a1 = −1,936, a2 = 0,942), então o erro de arredondamento
do estado se acumula. Na saída `env_dec` o erro relativo ao máximo foi de 1,2e-5 na mediana dos
164 casos (4,6e-5 no pior) com float puro, e caiu para 1,2e-7 (2,8e-7 no pior) com `sosd`. No caso
`borda_dc_ruido3s_midmt` (DC mais ruído, envelope quase constante) o erro de `env_mod`
foi 0,73 em float puro (a proeminência em dB muda muito) e 2,8e-4 com `sosd`. No ESP32 o double é
emulado em software, então o custo de tempo desta opção precisa ser medido no chip. A decisão de
usá-la ou não no firmware fica para quando o D tiver as tolerâncias e houver medição.

## 5. Estimativa analítica de RAM do workspace (ESTIMATIVA)

Saída real de `p11_host ws <n>`, que usa a mesma função de layout da arena (alinhamento de 16
bytes incluído). É **estimativa analítica**, não medição em ESP32. Não inclui pilha, código,
modelo (C×13×3 reais) nem o buffer de áudio de entrada.

| build | janela | T | persistente | fase A | fase B | total com tabela | total sem tabela |
|---|---|---|---|---|---|---|---|
| float | 1 s (16 000) | 8 | 534 368 | 186 824 | 424 096 | 958 480 | 612 592 |
| float | 3 s (48 000) | 39 | 663 008 | 410 280 | 1 200 960 | 1 863 984 | 1 518 096 |
| double | 1 s | 8 | 943 712 | 354 352 | 844 096 | 1 787 824 | 1 211 344 |
| double | 3 s | 39 | 1 200 608 | 726 736 | 2 397 728 | 3 598 352 | 3 021 872 |

Composição no build float (bytes): tabela harmônica 345 888; tabelas da rfft de 8192 65 536;
y_prep 4n (64 000 / 192 000); P_mean 16 388; quadro (buffers da FFT e |X|) 114 700;
e_norm 4·T·J (38 432 / 187 356); psi 2·T·(J+1) (19 232 / 93 756); Hilbert 24n (tabela de
giro n complexos mais dois buffers de n complexos: 384 000 / 1 152 000). A fase B (Hilbert)
domina. Mesmo sem a tabela harmônica, 1 s em float precisa de ~600 kB, acima da DRAM interna
de um ESP32 clássico (~320 kB utilizáveis); isso exige PSRAM ou uma reorganização do Hilbert.
Qualquer reorganização dessas (FFT no lugar, tabela de giro de quarto de onda, Hilbert por blocos)
mexe no arredondamento ou no algoritmo e teria de entrar como variante nomeada.

## 6. Build e saídas reais

```
& "embarcados\src\c\build_host.bat"
```

O `.bat` chama `vcvars64.bat` (MSVC 14.44.35207), roda `cmake -G "NMake Makefiles"
-DCMAKE_BUILD_TYPE=Release -DP11_REAL=float|double` e `cmake --build` em `build-float` e
`build-double`. Opções: `/O2 /fp:precise /W3 /utf-8`. Saída: os três alvos (`p11`, `p11_host`,
`p11_testes`) compilam nos dois builds sem aviso e o script termina em `BUILD OK`.
Observação: com `cmd /c "<caminho>"` o cmd quebra no parêntese de `Artigo3(MotorSound)`; pelo
PowerShell, use o operador `&` como acima. Fora do MSVC, o CMake usa `-O2 -ffp-contract=off`.
O firmware deve usar `-ffp-contract=off` também, porque o GCC funde em FMA por padrão.

Uso do CLI:

```
p11_host caso  <dir_caso> <modelo.bin> <dir_saida>     # todas as etapas do contrato + etapas.json
p11_host seq   <probs.f64> <T> <C> <cp> <temp> <n_logprob> <min_dur> <passo_s> <dir_saida>
p11_host bench <dir_caso> <modelo.bin> <n_rep>          # medição no PC
p11_host ws    <n_amostras>                             # estimativa analítica
```

`caso` grava ainda `vit_min_delta`. `seq` grava `hmm_log_em`, `hmm_viterbi`, `hmm_online`,
`hmm_online_alpha` (T, C+1), `argmax` e `absorb` com forma (3, T) (eixo 0 = argmax, viterbi,
online, a mesma ordem de `gold/seqs/*/absorb.i32`); chama-se uma vez por (cp, min_dur).

**Medição no PC** (`p11_host bench`, 30 repetições, mediana em ms, Windows, QueryPerformanceCounter;
não é ESP32):

| build | caso | rastrear_quadros | viterbi | envelope_mod | features | cosseno | total |
|---|---|---|---|---|---|---|---|
| float | idmt_limpo_good_00 (3 s) | 5,60 | 5,24 | 2,64 | 0,014 | 0,0003 | 13,66 |
| float | a2212_branco0_Bearing_00 (1 s) | 1,00 | 1,11 | 0,63 | 0,012 | 0,0003 | 2,86 |
| double | idmt_limpo_good_00 (3 s) | 4,88 | 4,89 | 2,90 | 0,017 | 0,0005 | 13,09 |

## 7. Testes unitários

`.venv\Scripts\python.exe -m pytest embarcados\tests\unit\test_p11.py -q` → **33 passed**
(float e double). O que eles cobrem: FFT complexa e inversa contra `np.fft` (N = 1000, 8192,
16 000, 48 000 e 1001 = 7·11·13 pela borboleta genérica); rfft (1000, 1024, 8192); sosfilt
contra `scipy.signal.sosfilt` (dizimação 1 e 16); Welch contra `scipy.signal.welch` (L = 1000,
1500, 2048, 3000; `f` idêntico bit a bit); Viterbi de rotação contra `rotacao.viterbi_rotacao` em
12 entradas aleatórias com saltos de oitava e quadros SIL (caminho idêntico e
`min_delta == score` no double); HMM (`_log_emissoes`, `decode`, `filtro_online`) e absorb contra
`hmm.py` em 16 sequências aleatórias com temp ∈ {1; 0,5}, n_logprob ∈ {−50; −2,5} e
passo ∈ {1; 3; 0,1} (caminhos idênticos no double); absorb com passo 0,1 (meio-para-par).
Tolerâncias destes testes estão em `TOL` no arquivo (não são as do D).

Depuração contra o ouro (164 casos, 20 seqs), `depurar_gold.py` e `depurar_seqs.py`: nos builds
float e double, `vit_path` e a classe prevista coincidem com o ouro nos 164 casos. No double,
`y_prep`, `grade`, `f_rot` e `welch_f` são idênticos ao ouro, e as demais etapas ficam em ~1e-7
relativo (mediana), que é a ordem da precisão float32 da referência (STFT, Hilbert e P em float32
no numpy/scipy). O pior caso é `borda_dc_ruido3s_midmt`: lá a própria referência tem ruído de
float32 na ordem de 1e-7 do bin DC, que vaza para `sal`. Nas sequências, as 240 combinações
(cp × min_dur) coincidem com o ouro; no double, `hmm_log_em` e `alpha` são bit a bit iguais; no
float, a diferença máxima é 9e-7 e 4,6e-6.

## 8. O que falta para o ESP-IDF

1. Instalar ESP-IDF v5.x e ESP-DSP (na época, ainda não instalados nesta máquina) e criar `firmware/` com um componente que
   compile `p11_fft.c`, `p11_core.c` e `p11_decisao.c` com `-DP11_REAL=float -ffp-contract=off`.
2. O caminho `#if defined(ESP_PLATFORM) && defined(P11_USAR_ESP_DSP)` em `p11_fft.c` (rfft de
   8192 e Welch de 1024 por `dsps_fft2r_fc32` + `dsps_bit_rev_fc32`) **não foi compilado**: não há
   IDF nem ESP32 nesta máquina. Ele exige `dsps_fft2r_init_fc32` na inicialização e buffers
   alinhados, e a diferença numérica contra o C portável terá de ser medida pelo D. Os N que não
   são potência de 2 (Hilbert 16 000/48 000, Welch 1000) continuam no C portável.
3. A RAM (§5) não cabe em DRAM interna sem PSRAM. A tabela harmônica pode ir para flash
   (`const`); para o Hilbert, qualquer redução de memória é variante nomeada.
4. Decidir o uso de `P11_SOS_DOUBLE` depois de medir o custo do double emulado no chip.
5. Medições no ESP32 físico (B1): tempo por etapa com `esp_timer_get_time()`, `idf.py size`
   e heap livre.
