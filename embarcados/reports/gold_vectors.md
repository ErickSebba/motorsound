# Vetores de ouro da p11_decifrae_harmsum

Gerado em 23/10/2026. Todos os números abaixo foram copiados de `embarcados/gold/manifest.json`
da execução descrita (commit do projeto `c9a205d`, árvore suja; Python 3 do `.venv`, numpy 2.5.4,
scipy 1.18.1). A fonte de verdade é o manifesto regenerado, não este texto.

## Como regerar

Com o diretório de trabalho na raiz `Artigo3(MotorSound)`:

```
.venv\Scripts\python.exe embarcados\src\python\gerar_ouro.py
```

O script apaga e recria `embarcados/gold/` (gitignored; nunca contém dado original), leva cerca
de 2 min (115 s na última execução, extração sequencial `n_jobs=1`) e sai com código 1 se algum
assert falhar. Duas execuções seguidas produziram os 3 462 arquivos binários de `casos/`, `seqs/`
e `modelos/` idênticos byte a byte (sha256). Tamanho: 351 MB em `casos/`, 1,2 MB em `seqs/`.

Arquivos de código: `embarcados/src/python/ouro_p11.py` (wrapper instrumentado: `etapas_p11`,
`score_caminho`, `_viterbi_com_delta`, `filtro_online_alpha`) e
`embarcados/src/python/gerar_ouro.py` (gerador). A referência (`pipelines/`, `motorsound/`,
`ruidos/`) só é importada. A extração usa `runner.extrair(..., usar_cache=False)`; o cache da raiz
só é lido (`cache.carregar`) para conferência. Só o `train/` do IDMT e o A2212 são lidos.

## Asserts (todos passaram, nos 164 casos)

Por caso, gravados em `manifest.json → casos[i].checagens`:

- `f_rot==rastrear`, `e_norm==rastrear`, `grade==rastrear`: as etapas chamadas uma a uma
  (`stft_mag → saliencia_harmonica → emissao_baixo → viterbi_rotacao`) reproduzem
  `rotacao.rastrear` exatamente (NaN incluído);
- `vit_path==copia` e `vit_score==min(delta)_exato`: o score pela regra do contrato §4,
  acumulado na ordem do Viterbi (`(delta + transição) + emissão`), é **bit a bit igual** ao
  `min(delta)` final de uma cópia literal de `viterbi_rotacao` que devolve `delta` (164/164);
- `env_mod==_envelope_mod`: as linhas reexecutadas do envelope dão a mesma tupla;
- `features==_d.extrair`: as 13 features recalculadas a partir dos intermediários são iguais a
  `_decifrae.extrair` (`np.array_equal`);
- `features==runner`: para casos limpos, iguais à linha de `runner.extrair` (sem cache) do
  segmento; para ruidosos, iguais a `runner._extrair_gravacao(..., perturbacao, usar_cache=False)`
  (prova que o áudio ruidoso gravado é o mesmo do runner, com a mesma semente);
- `cos==clf.emissao`, `softmax==clf.predict_proba`, `probs==_d.prever`: exatos.

Também conferido: features recalculadas sem cache são idênticas ao cache existente em todas as
gravações (IDMT 3/3, A2212 43/43); as probabilidades de validação dos dois folds batem com as
colunas `p_*` dos CSVs antigos `out/experiments/20261014_185859_validacao.csv` (IDMT, linhas com
`normalizar_rms=true`) e `20261014_193054_robustez_a2212.csv` (A2212, limpo) com
`max|ΔP| = 1,1e-16`. Nas sequências, `filtro_online_alpha` reproduz `hmm.filtro_online` e
`argmax(alpha)` dá as mesmas decisões, para todo change_penalty e toda sequência.

## Modelos

| arquivo | fold | n treino | sha256 do `.bin` |
|---|---|---|---|
| `gold/modelos/idmt_bloco0.bin` | `idmt_bloco0` (split `blocos`) | 273 | `e6d1e3456d99ea1c17ba5ea9130c326f8e687e015255fb894bee1a3459afc008` |
| `gold/modelos/a2212_arq0.bin` | `a2212_arq0` (split `arquivo`) | 340 | `0102f1561324174687e5af4074d4fb6b71604802f1298cf3b8cd7b8521f65601` |

É exatamente `P11.treinar(X[f.treino], y[f.treino], POST, n_classes)` do runner. Layout do `.bin`
como no contrato §3; o `.json` traz mu, sd, M, classes, índices de treino (posições na lista
`runner.segmentos_de`), pares (rec_id, idx) e o hash. Acurácia da validação do fold
(diagnóstico): IDMT 1,0 (72 janelas; classe = motor, esperado), A2212 0,589 (90 janelas).

## Casos (`gold/casos/<id>/`)

Cada caso tem `audio.f32` (segmento em 16 kHz antes de `_preparar`, já com ruído), todas as etapas
do contrato §1 (`.f64`; `vit_path.i32`) e `etapas.json` (arquivo, dtype, shape, `dtype_ref` = dtype
original no Python: `y_prep`, `stft_S`, `P_mean` e `env_abs` são float32 na referência e foram
gravados em float64 sem perda). O manifesto traz id, tipo, base, rec_id, idx, t0/t1, classe,
fold, perturbação e semente, n_amostras, sr, modelo, sha256 do áudio e a predição.

- **Limpos (60)**: `<base>_limpo_<classe>_<k>`, 10 por classe por base, só da validação do fold.
  Escolha: round-robin pelas gravações da classe em ordem sorteada (`SEED_SELECAO = 20261023`),
  espaçamento uniforme dentro de cada gravação. No IDMT a validação de `idmt_bloco0` tem uma
  gravação por classe (o `train/` tem três gravações), então as 10 janelas vêm do bloco 0 de uma
  só gravação. No A2212, `Bearing` tem um único arquivo na validação (M4/1950, exatamente 10
  janelas: todas usadas); `Propeller` vem de 3 arquivos e `Healthy` de 5.
- **Ruidosos (90)**: `<base>_<pert>_<classe>_<k>`, as 3 primeiras janelas limpas de cada classe
  × 5 perturbações: `branco20`, `branco5`, `branco0` (`{"tipo":"branco","snr_db":…}`, formato de
  `experimentos/robustez.py:26`), `leeshen_sag` e `leeshen_flicker`
  (`{"tipo":"lee_shen","disturbio":…}`). Semente = a do runner
  (`crc32(rec_id#idx) & 0xFFFFFF + pert.get("seed", 0)`, `motorsound/runner.py:94-95`).
- **Bordas (14)**: `borda_<sinal>_m<base>`, cada sinal rodado com os dois modelos (com o `aval`
  da base do modelo, que só muda `seg_s`, irrelevante aqui): `silencio3s`, `silencio1s` (zeros),
  `clip_idmt` e `clip_a2212` (primeira janela limpa da 1ª classe ×20 e `clip` em ±1 antes de
  `_preparar`), `rot90hz3s` e `rot8hz3s` (48 harmônicos de amplitude 1/k, fases sorteadas),
  `dc_ruido3s` (0,5 + ruído σ = 1e-5). Semente `SEED_BORDAS = 20261024`.

## Sequências (`gold/seqs/<base>_seqNN/`)

Probabilidades da referência em todas as janelas de validação do fold (limpo), 10 sequências por
base com `sequencias.montar(seed=0)`; T de 47 a 114 janelas (IDMT) e de 29 a 57 (A2212). Arquivos:
`probs.f64 (T,C)`, `verdade.i32`, `argmax.i32`, `hmm_log_em.f64 (T,C+1)`, `hmm_viterbi.i32` e
`hmm_online.i32` com forma `(n_cp, T)`, `hmm_online_alpha.f64 (n_cp,T,C+1)`,
`hmm_online_alpha_final.f64 (n_cp,C+1)` (log, normalizado) e `absorb.i32` com forma
`(3 decodificações [argmax, viterbi, online], n_cp, n_min_dur, T)`. Estados: 0..C−1 classes na
ordem do modelo, C = N. `params.json` traz os eixos, a matriz `log_T` de cada change_penalty e a
origem de cada parâmetro:

| parâmetro | valor | de onde |
|---|---|---|
| change_penalty | 0,5; 1; 2; 4; 8; 16 | `experimentos/sequencias.py:25` (`PENALIDADES`, varrido na linha 54) |
| temp | 1,0 | default de `sequencias.avaliar`, `motorsound/sequencias.py:50` (a chamada em `experimentos/sequencias.py:55` não o passa) |
| n_logprob | −50,0 | idem, `motorsound/sequencias.py:50` (difere de `PostConfig.n_logprob = −2,5`) |
| min_dur | 0 nos experimentos | `motorsound/sequencias.py:51`; nenhum experimento passa outro valor |
| passo_s | 3,0 (IDMT), 1,0 (A2212) | `experimentos/sequencias.py:49` (`seg_s` do aval) |
| dmin, dmax | (5, 20) IDMT, (3, 10) A2212 | `experimentos/sequencias.py:50` |
| seed | 0 | `experimentos/sequencias.py:36` |
| n_seq | 10 (o experimento usa 30) | `experimentos/sequencias.py:35` |

Os valores de `temp`, `n_logprob` e `PENALIDADES` são lidos do código em tempo de execução
(`inspect.signature` e import do módulo), não copiados.

## Problemas e incertezas

1. **Distúrbios Lee & Shen nunca foram usados na validação.** Nem `experimentos/robustez.py` nem
   `ruidos.CONDICOES_VAL` têm `lee_shen`; ele só aparece na bancada sintética. O formato
   `{"tipo":"lee_shen","disturbio":"sag"|"flicker"}` é o que `ruidos/noise.py::aplicar` aceita, com
   parâmetros sorteados pela semente (faixas de `lee_shen.sortear_parametros`); a escolha de sag e
   flicker é deste gerador. Branco 20, 5 e 0 dB estão todos na `CURVA_SNR` de
   `experimentos/robustez.py:26`.
2. **`min_dur > 0` não existe nos experimentos.** O `absorb.i32` usa min_dur = 2 e 5 passos
   (6 e 15 s no IDMT, 2 e 5 s no A2212), escolhidos só para exercitar `absorb_short`.
3. **Sequências de um fold só.** `experimentos/sequencias.py` monta os trechos com a validação de
   todos os folds do split; aqui só a do fold do modelo (pedido da tarefa), então as sequências não
   são as mesmas dos CSVs de sequências antigos, embora a semente seja.
4. **z explosivo no IDMT.** No modelo `idmt_bloco0`, a feature 2 (fração de quadros SIL) é 0 em
   todo o treino, então `sd = 1e-9` e qualquer quadro SIL dá `z ≈ 1e9`. Nos silêncios isso domina o
   vetor, os cossenos ficam ~1e-7 e as probabilidades saem exatamente uniformes (1/3). O C em float32
   precisa reproduzir isso (1e9 é representável, mas a normalização de `z` e os cossenos ficam
   sensíveis ao arredondamento); vale tolerância específica para esses casos.
5. **Bordas de rotação fora da faixa não ficam fora da faixa.** `rot90hz3s` sai com `f_rot` mediana
   45 Hz (sub-harmônico) e `rot8hz3s` com 16 Hz (2º harmônico): o rastreador sempre acha algo em
   15–60 Hz. É comportamento da referência, não erro do ouro.
6. Nos silêncios o `vit_score` é negativo (custo de SIL = −log(1 + 1e-3)); correto pela regra.
