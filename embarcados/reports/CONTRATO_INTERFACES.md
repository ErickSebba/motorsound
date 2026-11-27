# Contrato de interfaces: referência Python × porte C (p11_decifrae_harmsum)

Escrito em 23/10/2026, antes do porte começar. Define os formatos que o gerador dos vetores de
ouro (wrapper Python), o porte C e o verificador de equivalência usam, para que cada parte seja
feita sem olhar as saídas das outras.

## 1. Escopo do núcleo C

O núcleo C recebe **um segmento já em 16 kHz, float32, antes da normalização de RMS**
(o que `runner._extrair_gravacao` passa para `_preparar`). A leitura do arquivo e a
reamostragem (`io.carregar`, `librosa.resample(res_type="soxr_hq")` de 44,1 kHz para 16 kHz no
IDMT) ficam **fora** do núcleo: no MCU o microfone amostraria direto em 16 kHz. Isso é declarado
como escopo, não como equivalência.

Etapas, na ordem, com o identificador usado em nomes de arquivo:

| id | etapa | referência Python | forma |
|---|---|---|---|
| `y_prep` | normalização de RMS (alvo 0,1) | `runner._preparar` | (N,) |
| `stft_S` | módulo da STFT, Hann simétrico `np.hanning`, sem centralização | `rotacao.stft_mag` | (F=4097, T) |
| `grade` | grade de rotação, 15–60 Hz, 2 cents | `rotacao.grade` | (J=1201,) |
| `sal` | soma harmônica, 48 harmônicos, interpolação linear | `rotacao.saliencia_harmonica` | (T, J) |
| `e_norm` | saliência normalizada por quadro | idem | (T, J) |
| `custo` | custo de emissão (sem pYIN), último estado = SIL | `rotacao.emissao_baixo` | (T, J+1) |
| `vit_path` | caminho do Viterbi de rotação (−1 = SIL) | `rotacao.viterbi_rotacao` | (T,) int32 |
| `vit_score` | custo total do caminho decodificado (ver §4) | recalculado | (1,) |
| `f_rot` | rotação por quadro (NaN = SIL) | `rotacao.rastrear` | (T,) |
| `P_mean` | espectro de potência médio `(S**2).mean(axis=1)` | `_decifrae.extrair` | (F,) |
| `env_abs` | `abs(hilbert(y))` | `_decifrae._envelope_mod` | (N,) |
| `env_dec` | passa-baixa Butterworth 4ª ordem 200 Hz (`sosfilt`), dizimado por 16, **depois** de subtrair a média (`env - env.mean()`, `_decifrae.py:54`) | idem | (N/16,) |
| `welch_f`, `welch_P` | Welch a 1 kHz, `nperseg=min(len,1024)` | idem | (B,) |
| `env_mod` | (frequência do pico, proeminência em dB) | idem | (2,) |
| `features` | as 13 features, na ordem de `_decifrae.py` | `_decifrae.extrair` | (13,) |
| `z` | z-score `(x − mu)/sd` | `CossenoPrototipos.emissao` | (13,) |
| `cos` | cossenos contra os protótipos | idem | (C,) |
| `probs` | softmax com T = 0,1 | `CossenoPrototipos.predict_proba` / `prever` | (C,) |

Etapas de sequência (sobre matrizes de probabilidade por janela, (T, C)):

| id | etapa | referência |
|---|---|---|
| `hmm_log_em` | log-emissões com estado N | `hmm._log_emissoes` |
| `hmm_viterbi` | decodificação offline | `hmm.decode` |
| `hmm_online` | filtro forward causal | `hmm.filtro_online` |
| `absorb` | `absorb_short` sobre os rótulos | `hmm.absorb_short` + conversões |

Parâmetros das sequências: os que `experimentos/sequencias.py` e `sequencias.avaliar` usam
(change_penalty varrido; `n_logprob=-50.0`, `temp=1.0` em `avaliar`, que diferem do
`PostConfig`). O gerador de ouro confirma no código e grava no manifesto.

## 2. Arquivos

Todo vetor é binário cru, little-endian, sem cabeçalho, com descritor JSON ao lado.

```
embarcados/gold/                       (gitignored; regerável por src/python/gerar_ouro.py)
  manifest.json                        casos: id, base, rec_id, idx, classe, fold, perturbação,
                                       n_amostras, sr, modelo usado, hash sha256 do áudio
  modelos/<nome>.bin + <nome>.json     artefatos do modelo (§3)
  casos/<id>/audio.f32                 entrada do C (segmento em 16 kHz antes de _preparar)
  casos/<id>/<etapa>.f64 | .i32        saídas da referência (float64; discretos em int32)
  casos/<id>/etapas.json               {etapa: {"arquivo", "dtype", "shape"}}
  seqs/<id>/probs.f64, params.json, hmm_viterbi.i32, hmm_online.i32, absorb.i32, ...
embarcados/results/c_out/<build>/casos/<id>/<etapa>.f32|.f64|.i32 + etapas.json
```

`shape` segue a ordem C (linha maior). `stft_S` é gravado como (F, T), igual ao Python.
NaN é gravado como NaN IEEE. Estados do HMM: 0..C−1 classes na ordem do modelo, C = estado N.

## 3. Artefatos do modelo

`modelos/<base>_<fold>.bin`: `int32 n_classes, int32 n_feat(=13), float64 temp,
float64 mu[13], float64 sd[13], float64 M[n_classes][13]` (M já normalizado como em `fit`).
O `.json` traz as mesmas coisas legíveis, mais as classes, o fold, os índices de treino
e o hash do `.bin`. `sd` já inclui o `+1e-9` de `fit`.

Constantes de projeto que o C não recalcula (coeficientes SOS de `butter(4, 200, fs=16000)`)
saem de um script Python para `src/c/p11_constantes.h`, com cabeçalho de procedência.

## 4. Score do Viterbi

`viterbi_rotacao` não devolve o score. O wrapper e o C reportam o custo do caminho
decodificado pela mesma regra: Σ custo[t, path_t] + Σ custo de transição (ficar 0; passo de
|Δ| ≤ max_salto estados = w_step·|Δ|·passo/100; ±k_oct = w_oct; entrar/sair de SIL = w_sil).
O C também reporta `min(delta)` final, que deve ser igual a esse custo.

## 5. Builds do C

O mesmo fonte compila com `P11_REAL=float` (o que roda no MCU) e `P11_REAL=double` (só no
PC, para separar erro de algoritmo de erro de precisão). Desempates (argmin/argmax: primeiro
índice; ordem de varredura dos saltos do Viterbi) seguem o Python literalmente.
