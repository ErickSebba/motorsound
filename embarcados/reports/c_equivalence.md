# Equivalência referência Python × porte C (p11_decifrae_harmsum)

Verificação, 23–24/10/2026. Este relatório é independente do porte: não usa os testes do C como
prova, não leu `results/c_out/` e não rodou o C contra o ouro. As tolerâncias foram fixadas
antes de qualquer saída do C existir.

## 1. Congelamento

| arquivo | sha256 |
|---|---|
| `embarcados/configs/embedded/tolerancias.json` | `d74dae800cebfe4d670c33ba2c0a72639ad1e04e95bcbd9a904be13c294c1b5d` |

A partir deste hash as tolerâncias não mudam. O comparador grava o sha256 do json em cada linha
do CSV e no `_meta.json`, então uma comparação feita com outro arquivo é visível. Se for preciso
mudar algum número depois de ver a saída do C, isso vira uma versão nova (`versao: 2`) com o
motivo registrado aqui, e os resultados das duas versões são reportados lado a lado.

## 2. Tolerâncias e justificativa

### 2.1 Princípio

Cada tolerância é o maior entre (a) uma estimativa analítica de primeira ordem do erro de
arredondamento e (b) um fator de segurança vezes o erro máximo medido por uma emulação
independente do build. As unidades de arredondamento são u = 2⁻⁵³ ≈ 1,11e-16 (double) e
u = 2⁻²⁴ ≈ 5,96e-8 (float32; eps32 = 2u ≈ 1,19e-7). A emulação
(`tests/equivalence/emular_build.py`, que usa as funções de `estudo_tolerancias.py`) refaz
toda a cadeia do contrato em numpy e aceita dois modos. No modo `emul_double` tudo é float64.
No modo `emul_float` tudo é float32, com somas **sequenciais** (o laço ingênuo de um C) e uma
FFT radix-2 com twiddles float (o modelo de `dsps_fft2r_fc32`). É o pior caso plausível do C
float. O fator é 10 no double e 5 no float, porque a emulação float já é pessimista. A
calibração rodou nos 164 casos e 20 sequências do ouro, e as duas emulações passam com
razão erro/tolerância ≤ 0,33 em toda etapa, exceto as features no float: razão ≤ 0,57 antes do
último ajuste, que subiu a `rtol` do centróide de 2e-5 para 5e-5. O estudo sobre sinais de
treino independentes do ouro está em `results/csv/estudo_tolerancias_20261023-223624.csv`.

Há um ponto que muda a leitura dos dois builds: **a própria referência roda parte da cadeia em
float32.** No numpy 2.5.4 instalado, `np.fft.rfft` de entrada float32 devolve complex64, e
`scipy.signal.hilbert` (scipy 1.18.1) preserva complex64 para entrada float32. Isso foi
conferido no `.venv`. Também ficam em float32 o `y` de `_preparar`, `S`, `P_mean = (S**2).mean`
(acumulador float32) e a planura (`log` e `mean` de `P[banda]`). Já `sosfilt` com SOS float64
devolve float64, e o Welch também. Por isso o build double **não** coincide com a referência a
menos de eps64: ele herda o erro float32 da referência, da ordem de 1e-7 relativo ao máximo nas
etapas espectrais. As tolerâncias do double são, então, tolerâncias de "erro de float32 da
referência", da ordem de eps32·log₂N, e não de eps64. Ainda assim ficam 50 a 400 vezes abaixo
do efeito dos erros de convenção medidos (§3). **As convenções são verificadas no build double;
o build float só limita a perda de precisão.** No float, um erro de convenção do tamanho de
1e-4 (Hann periódico) fica abaixo da tolerância, mas aparece no double.

### 2.2 Tabela (normal = casos limpos e ruidosos; bordas em §2.8)

| etapa | regra | double | float | fundamento |
|---|---|---|---|---|
| y_prep | comb | rtol 2e-6, atol 1e-12 | rtol 2e-4, atol 1e-9 | a ref arredonda y·(0,1/r) em float32 (≤1 ulp) e r vem de soma float32; no C float a soma ingênua de N = 48 000 quadrados deu erro de escala de 4,2e-5 (pior caso γ_N ≈ N·u = 2,9e-3) |
| stft_S | relmax | 2e-6 | 2,5e-4 | FFT: erro ~ eps·log₂N relativo ao máximo (eps32·13 = 1,5e-6); no float soma-se a escala de y_prep, que passa inteira para S |
| grade | comb | rtol 1e-13 | rtol 1e-6 | pow/exp2: poucos ulp, mais o erro do expoente j·2/1200 (float: 2 ulp·(1 + ln2·x) ≈ 4,4e-7) |
| sal, e_norm, custo | cond_S | ρ = 1,5e-5 | ρ = 1,5e-3 | §2.3 |
| vit_path, vit_score | caminho / soma | exato, ou empate demonstrado | idem | §2.4 |
| f_rot | comb | rtol 1e-13 | rtol 1e-6 | = grade[caminho] quando o caminho coincide; posições NaN exatas |
| P_mean | relmax | 4e-6 | 5e-4 | 2× o erro relativo de S (quadrado), mais a média de 39 quadros |
| env_abs | relmax | 6e-6 | 3e-4 | hilbert por FFT de N = 48 000 (não potência de 2): eps32·log₂N ≈ 1,9e-6; no float, mais a escala de y_prep |
| env_dec | relmax | 1e-5 | 4e-4 | passa-baixa com norma L1 da resposta ao impulso ≈ 1,3: herda o erro de env_abs |
| welch_f | comb | rtol 1e-14 | rtol 1,2e-7 | k·1000/1024 = k·125/128 é exato em binário |
| welch_P | relmax | 2e-5 | 1e-3 | periodograma ∝ \|FFT\|²: 2× o erro de env_dec |
| env_mod | f: argmax/empate; prom: atol | 5e-4 dB | 1,5e-2 dB | Δprom ≈ 4,343·(ΔPpk/Ppk + Δmed/med) |
| features | comb por feature | §2.6 | §2.6 | |
| z, cos, probs | propagadas | §2.7 | §2.7 | |
| decisao | argmax | exato, ou empate demonstrado | idem | §2.7 |
| hmm_log_em | comb | atol 1e-13, rtol 1e-14 | atol 2e-6, rtol 2,4e-7 | log(p + 1e-12): conversão f64→f32 (Δlog ≤ u) e log com \|log\| ≤ 27,6 (u·27,6 ≈ 1,6e-6) |
| hmm_online_alpha | comb | atol 1e-11 | atol 2e-4 | filtro forward com renormalização: erro por passo ~1e-5 (float), contraído pela mistura; o pior caso linear em T ≤ 114 dá 9e-4 |
| argmax, hmm_viterbi, hmm_online, absorb | discretos | exato, ou empate demonstrado | idem | §2.5 |

### 2.3 Cadeia da rotação: tolerância condicionada pela STFT

O erro absoluto da FFT é proporcional ao **máximo** do espectro, não a cada bin. A saliência
`sal[t, j] = Σ_k S(k·f_j)` soma bins que podem ser muito menores que max(S). No caso
`dc_ruido3s`, o pico de DC fica fora de todos os harmônicos, e uma tolerância relativa ao
máximo de `sal` falharia sem que nada estivesse errado. A regra `cond_S` propaga o erro em
primeira ordem a partir de um único parâmetro ρ, que é o erro de `sal` relativo a max(S).
Nos estados, o erro admitido é atol + ρ·maxS em `sal`, atol + 2ρ·maxS/max_j sal[t,j] em
`e_norm` (numerador e denominador da normalização) e atol + rtol·|c| + 2ρ·exp(c)·q_t em
`custo`. O fator exp(c) = 1/(0,45·e + 1e-3) é a derivada de −log, o que limita o condicionamento
a ≤ 1000. A troca de 0,45 por 1 deixa uma folga de 2,2×. Na coluna SIL, q = maxS/max(sal).
No double, ρ = 1,5e-5 ≈ √48 × a tolerância de S (48 erros de bins pouco correlacionados). No
float, ρ = 1,5e-3. Esse valor cobre a escala de y_prep e o erro de posição h·f/df calculado em
float (até 1474 bins × 2u ≈ 1,8e-4 bin por harmônico; no pior caso somado, 4,3e-3). Na
emulação float o máximo medido foi 2,3e-4. O atol do custo float (2,4e-7) cobre −log(1,001)
em float no silêncio. Sem ele, os 4 casos de silêncio falhavam por 9,6e-8 numa tolerância de
1e-9.

### 2.4 Viterbi de rotação

O caminho tem de ser **idêntico**. Uma divergência só é aceita (PASS_EMPATE, documentada
caso a caso na coluna `detalhe`) se for demonstrado um empate real. Sejam S_ref(p) e S_C(p) o
custo do contrato §4 de um caminho p, avaliado em float64 com o custo da referência e com o
custo gravado pelo C. Se o C acha o ótimo para os próprios custos, então
S_C(p_C) ≤ S_C(p_ref) + acc. Daí sai a margem exata, calculada com os dados:
gap = S_ref(p_C) − S_ref(p_ref) ≤ E(p_ref) − E(p_C) + acc, com E(p) = S_C(p) − S_ref(p) e
acc = 2·T·u·(|S_ref| + 1). O termo acc vem de T = 39 quadros de acumulação; no float ele vale
≈ 1e-4 para |S| ≈ 35. A exceção exige ainda que o `custo` do C passe na tolerância e que o
caminho do C seja ótimo para o custo do próprio C; se não for, o Viterbi do C está errado. O
score é comparado com |ΔS| ≤ Σ_t tol_custo(t, p_t) + acc. Se não houver custo do C, a margem
cai para o limite da tolerância, Σ tol_custo nos dois caminhos + acc. A emulação float teve 0
divergências de caminho no ouro. Uma emulação anterior, sobre um DC sintético diferente, teve
2–3 quadros com gap de 1,5e-4 a 2,9e-4: são esses empates que a regra separa de erro.

### 2.5 Sequências

`argmax` é exato. A exceção vale só se P[t, ref] − P[t, C] ≤ 4u·P, o que cobre a conversão de
probs para float. `hmm_viterbi` segue a mesma regra de empate demonstrado da §2.4, em log-score
e com a log-emissão do C. `hmm_online` é exato, ou empate quando alpha_ref[t, ref] −
alpha_ref[t, C] ≤ tol_alpha nos dois estados. `absorb` é exato. `absorb_short` é determinístico
dado o rótulo de entrada, e as durações i·passo_s são exatas em double porque passo_s ∈ {1, 3}.
Uma divergência de absorb com a entrada (argmax/viterbi/online) também divergente é
classificada como a jusante.

### 2.6 Features (13, escalas diferentes)

Os valores ficam em `features.atol/rtol` do json. f_med = mediana de valores da grade e iqr =
diferença de percentis lineares desses valores. Com o caminho igual, o erro é o da grade
(double rtol 1e-13; float rtol 1e-6 e iqr atol 1,2e-4 = 2·1e-6·60 Hz). frac_sil = 1 − k/T tem
erro absoluto ≤ u (atol 3e-16 / 1,2e-7). sal_caminho é a média de e_norm no caminho, com o erro
de e_norm (1e-5 / 5e-5). A razão harmônica é uma fração de somas de P_mean (1e-5 / 5e-5). Os
grupos 5–8 são 10·log10 de razões pequenas. O cancelamento em dB amplifica o erro relativo dos
bins harmônicos por 4,343, o que dá 5e-5 dB / 2e-4 dB. f_mod é a frequência do argmax de welch_P
em 2–100 Hz: exata ou empate (§2.2). prom vale 5e-4 dB / 1,5e-2 dB. O centróide tem rtol 4e-6
(double) e 5e-5 (float: a soma ingênua de 4097 termos tem erro típico √n·u ≈ 3,8e-6 e pior caso
n·u = 2,4e-4). A planura tem atol 3e-6/2e-5 e rtol 1e-5/1e-4: o log de bins pequenos tem erro
relativo grande, e a média geométrica o carrega.

### 2.7 Classificador: tolerâncias propagadas (sd = 1e-9 na feature 2 do modelo IDMT)

z = (x − μ)/σ. Uma tolerância absoluta fixa em z não serve, porque σ₂ = 1e-9 no modelo IDMT
(todo o treino teve frac_sil = 0). Basta um quadro SIL para z₂ ≈ 2,6e7. Por isso a tolerância
de z é **propagada** da tolerância da feature:
tol_z,k = (tol_x,k + 4u·(|x_k| + |μ_k|))/σ_k. O segundo termo cobre o cancelamento quando
x ≈ μ. Com frac_sil exato (k/T, erro ≤ u), z₂ fica dentro da tolerância mesmo com σ = 1e-9,
porque o erro relativo de z₂ é o de x₂. Se frac_sil diverge, isso aparece primeiro em
`vit_path` ou `features`, e z é a jusante. Depois, cos = M·z/‖z‖ com ‖M_i‖ = 1, então
|Δcos| ≤ 2‖Δz‖/‖z‖, o que dá tol_cos = min(2, 2‖tol_z‖/‖z_ref‖) + atol. O softmax com T = 0,1
amplifica esse erro por 10: |Δp_i| ≤ 2·p_i·tol_cos/T, logo tol_p_i = atol + min(1, 20·p_i·tol_cos).
A decisão (argmax) é exata. A exceção vale só se, na referência,
p_top1 − p_C ≤ tol_p[top1] + tol_p[C].

### 2.8 Casos de borda

Silêncio, DC + ruído σ = 1e-5, clip, f₀ = 8 Hz e f₀ = 90 Hz são mal condicionados por
construção. No DC, as flutuações do envelope ficam em ~2e-5 da média, e as features de
envelope e as de energia na banda caem no piso de ruído float32 **da própria referência**. As
regras `cond_S` funcionam neles sem ajuste. Só welch_P (float), prom e as features 3–8, 11 e 12
recebem tolerâncias próprias (`builds.<b>.borda`), iguais a 3× o máximo medido nos casos de
borda do ouro. Exemplos: prom 3e-2 dB (double) e 3,5 dB (float); grupos 2,5e-3 dB / 0,15 dB.
Essas tolerâncias foram calibradas nos mesmos sinais, então valem como evidência mais fraca, e o
relatório final as apresenta separadas das dos casos normais.

### 2.9 Primeira etapa que diverge

A primeira etapa que diverge é a primeira com FAIL na ordem do contrato. Nos casos a ordem é
y_prep, stft_S, grade, sal, e_norm, custo, vit_path, vit_score, f_rot, P_mean, env_abs, env_dec,
welch_f, welch_P, env_mod, features, z, cos, probs, decisao. Nas sequências é hmm_log_em,
argmax, hmm_viterbi, hmm_online_alpha, hmm_online, absorb. Todo FAIL posterior no mesmo item vira
`FAIL_JUSANTE`, com a causa na coluna `primeira_divergencia`. A referência **não** é
reexecutada com o caminho do C. Uma divergência de vit_path torna as features 0–4 desse caso
discrepâncias a jusante. PASS_EMPATE não conta como divergência.

## 3. Convenções que o C precisa reproduzir (fontes de falsa divergência)

O efeito de cada convenção errada foi medido em 3 janelas do IDMT train_cut, em erro relativo
ao máximo. Todas ficam acima da tolerância double:

- STFT com Hann **simétrico** (`np.hanning(8192)`, divisor N−1): o periódico muda S em 1,2e-4 a
  1,3e-4. Sem centralização, `n = 1 + (len − 8192)//1024` quadros e zero-padding só se
  len < 8192.
- Welch com Hann **periódico** (padrão do scipy), `nperseg = min(len, 1024)` (A2212:
  len = 1000, um segmento), noverlap = nperseg/2, detrend **constante** por segmento, densidade,
  unilateral e média. Usar Hann simétrico muda welch_P em 3,7e-4 a 8,3e-4. A subtração de
  `env.mean()` antes do Welch é redundante com o detrend (efeito medido de 1e-16), mas deve ser
  mantida.
- `rfft` sem normalização, módulo de |X|. Uma normalização por 1/N é erro de 8192×.
- `sosfilt` causal com estado inicial zero, nunca `sosfiltfilt` (efeito de 0,57 a 0,76).
  Coeficientes de `butter(4, 200, fs=16000, output='sos')` em float64 (o `p11_constantes.h` do
  porte C). Decimação `[::16]` a partir do índice 0.
- `hilbert` **sem padding**, FFT de N = 48 000 ou 16 000, que não são potências de 2: o C
  precisa de FFT de raiz mista ou Bluestein. Padding para 65 536 muda env_abs em 0,10 a 0,23.
  O espectro analítico usa h[0] = h[N/2] = 1 e h[1..N/2−1] = 2.
- `np.round` arredonda o meio para o par em `pos = round(k·f_med/df)`. `round()` do C arredonda o
  meio para longe de zero, e a diferença aparece em .5 exato (raro, mas possível).
- `nanpercentile` linear (método padrão). Mediana de contagem par = média dos dois centrais.
  `searchsorted` à esquerda (a feature 3 busca o índice da grade).
- argmax/argmin no primeiro índice. Viterbi: comparação estrita `<` na ordem de varredura
  (offsets −50…+50, depois +k_oct, −k_oct, depois SIL). Na coluna SIL, `prev[J] <= prev[jb] + w_sil`
  (empate favorece ficar em SIL). Backtrace a partir de `argmin(delta)`.
- Épsilons: +1e-12 (r de _preparar e máximo de sal), +1e-3 (custo), +1e-20 e +1e-10 (features),
  +1e-12 e +1e-20 em prom, +1e-9 (normas do cosseno; σ já inclui o +1e-9). 1e-20 é normal em
  float32 (o mínimo normal é 1,18e-38).
- Silêncio: e_norm = 0/(0 + 1e-12) = 0. Todos os estados ficam com custo −log(1e-3), o que gera
  empates exatos resolvidos pelo primeiro índice. Com f_rot todo NaN, f_med = iqr = sal = 0.
  Em prom, argmax de zeros = primeiro bin da banda.
- Precisão mista da referência (§2.1): o C double fica *mais* preciso que a referência nessas
  etapas, e isso é esperado.

## 4. Comparador

`tests/equivalence/comparar.py`. Com a raiz `Artigo3(MotorSound)` como diretório de trabalho:

```
.venv\Scripts\python.exe embarcados\tests\equivalence\comparar.py --build double [--gerar]
.venv\Scripts\python.exe embarcados\tests\equivalence\comparar.py --build float  [--gerar]
```

O comparador lê `gold/` e `results/c_out/<build>/` (ou `--c-out`). Com `--gerar`, chama
`p11_host caso <dir_caso> <modelo.bin> <dir_saida>` e
`p11_host seq <probs.f64> T C cp temp n_logprob min_dur passo_s <dir_saida>` para o que faltar.
As sequências vão para `c_out/<build>/seqs/<id>/cp<cp>_md<md>/`, uma pasta por (change_penalty,
min_dur). O executável é procurado em `embarcados/build-<build>/`. A saída é
`results/csv/equivalencia_<build>_<ts>.csv`, uma linha por (item, etapa) e por feature, com erro
máximo, médio, RMS, erro relativo ao máximo, tolerância, razão, concordância, status e primeira
divergência. Saem também `_resumo.csv` (por etapa) e `_meta.json` (commit, sha256 das
tolerâncias e do manifesto, build, data, contagens). O código de saída é 0 só se não houver FAIL
nem AUSENTE.

Autotestes (`tests/equivalence/test_comparador.py`, sem o C): ver §5.

## 5. Resultados dos autotestes do comparador

`test_comparador.py` passa em 4 de 4 testes (`pytest`, 4,6 s), sobre um subconjunto do ouro
copiado para um diretório temporário com 6 casos (inclusive silêncio e DC) e 2 sequências × 6
change_penalty × 2 min_dur.

(a) O ouro comparado consigo mesmo dá tudo PASS e nenhuma primeira divergência, nos conjuntos
double e float.

(b) As perturbações caem onde deveriam. Um elemento do custo deslocado 10× acima da sua
tolerância dá FAIL em `custo`, que vira a primeira divergência, com as etapas anteriores em PASS
e os outros casos intactos. Um quadro do caminho deslocado 25 estados, sem empate, dá FAIL em
`vit_path`, com `f_rot` como FAIL_JUSANTE. frac_sil acrescida de 1/39 dá FAIL em `features` e
`features[2]`, com z a jusante, mesmo com σ = 1e-9. O pico de welch_P com 2× a tolerância float
dá FAIL em `welch_P`. Um rótulo trocado no Viterbi do HMM dá FAIL só naquele change_penalty.

(c) A STFT com Hann periódico dá FAIL em `stft_S` no double e PASS no float, o comportamento
documentado em §2.1.

(d) Um empate real (diferença de 2e-9) na decisão dá PASS_EMPATE. A mesma troca com 0,9 contra
0,1 dá FAIL em `probs`, com `decisao` a jusante.

Também como controle de calibração, sem o C, as emulações completas sobre os 164 casos e as 20
sequências deram PASS em todas as 6 852 linhas, nos dois builds:
`results/csv/equivalencia_emul_double_20261024-090737.csv` e
`results/csv/equivalencia_emul_float_20261024-090743.csv`, com o sha256 das tolerâncias
`d74dae80…` no `_meta.json`.
