# Equivalência Python × C no PC: resultados

Gerado por `tests/equivalence/resumo_equivalencia.py` a partir dos CSVs abaixo; não editar à mão. Tolerâncias congeladas no commit `447a7c2` (`configs/embedded/tolerancias.json`), antes de qualquer saída do C existir. Categoria: **medição no PC** (MSVC 19.44, x64, `/O2 /fp:precise`). Nada aqui é medição no ESP32.

- C double (PC): `results/csv/equivalencia_double_20261024-091106.csv` (sha256 `0462b21cd040d898…`, tolerâncias `d74dae800cebfe4d…`, conjunto de tolerâncias `double`, commit `447a7c220731bfd1393b046895a6b50c24301c21`)
- C float32 (PC): `results/csv/equivalencia_float_20261024-091122.csv` (sha256 `38e87620d895ce26…`, tolerâncias `d74dae800cebfe4d…`, conjunto de tolerâncias `float`, commit `447a7c220731bfd1393b046895a6b50c24301c21`)
- variante p11_c_sosd (PC): `results/csv/equivalencia_float-sosd_20261024-091222.csv` (sha256 `fb79aad99d6e607a…`, tolerâncias `d74dae800cebfe4d…`, conjunto de tolerâncias `float`, commit `447a7c220731bfd1393b046895a6b50c24301c21`)

## Veredito por etapa

| etapa | C double (PC) | C float32 (PC) | variante p11_c_sosd (PC) |
|---|---|---|---|
| `y_prep` | PASS (164/164; máx 0× tol) | PASS (164/164; máx 0× tol) | PASS (164/164; máx 0× tol) |
| `stft_S` | PASS (164/164; máx 0.072× tol) | PASS (164/164; máx 0.0012× tol) | PASS (164/164; máx 0.0012× tol) |
| `grade` | PASS (164/164; máx 0× tol) | PASS (164/164; máx 0.057× tol) | PASS (164/164; máx 0.057× tol) |
| `sal` | PASS (164/164; máx 0.021× tol) | PASS (164/164; máx 0.0032× tol) | PASS (164/164; máx 0.0032× tol) |
| `e_norm` | PASS (164/164; máx 0.01× tol) | PASS (164/164; máx 0.002× tol) | PASS (164/164; máx 0.002× tol) |
| `custo` | PASS (164/164; máx 0.0087× tol) | PASS (164/164; máx 0.19× tol) | PASS (164/164; máx 0.19× tol) |
| `vit_path` | PASS (164/164; máx nan× tol) | PASS (164/164; máx nan× tol) | PASS (164/164; máx nan× tol) |
| `vit_score` | PASS (164/164; máx 0.001× tol) | PASS (164/164; máx 0.13× tol) | PASS (164/164; máx 0.13× tol) |
| `f_rot` | PASS (164/164; máx 0× tol) | PASS (164/164; máx 0.05× tol) | PASS (164/164; máx 0.05× tol) |
| `P_mean` | PASS (164/164; máx 0.059× tol) | PASS (164/164; máx 0.00063× tol) | PASS (164/164; máx 0.00063× tol) |
| `env_abs` | PASS (164/164; máx 0.099× tol) | PASS (164/164; máx 0.0032× tol) | PASS (164/164; máx 0.0032× tol) |
| `env_dec` | PASS (164/164; máx 0.02× tol) | PASS (164/164; máx 0.11× tol) | PASS (164/164; máx 0.0007× tol) |
| `welch_f` | PASS (164/164; máx 0× tol) | PASS (164/164; máx 0× tol) | PASS (164/164; máx 0× tol) |
| `welch_P` | PASS (164/164; máx 0.042× tol) | PASS (164/164; máx 0.33× tol) | PASS (164/164; máx 0.0009× tol) |
| `env_mod` | PASS (164/164; máx 0.22× tol) | PASS (162/164, 2 empate; máx 0.24× tol) | PASS (164/164; máx 0.0041× tol) |
| `features` | PASS (164/164; máx 0.55× tol) | PASS (162/164, 2 empate; máx 6.1e+06× tol) | PASS (164/164; máx 0.23× tol) |
| `z` | PASS (164/164; máx 0.55× tol) | FAIL (162/164; máx 1.6e+06× tol) | PASS (164/164; máx 0.22× tol) |
| `cos` | PASS (164/164; máx 0.078× tol) | PASS (164/164; máx 0.34× tol) | PASS (164/164; máx 0.001× tol) |
| `probs` | PASS (164/164; máx 0.067× tol) | PASS (164/164; máx 0.2× tol) | PASS (164/164; máx 0.0025× tol) |
| `decisao` | PASS (164/164; máx nan× tol) | PASS (164/164; máx nan× tol) | PASS (164/164; máx nan× tol) |
| `hmm_log_em` | PASS (240/240; máx 0× tol) | PASS (240/240; máx 0.15× tol) | PASS (240/240; máx 0.15× tol) |
| `argmax` | PASS (240/240; máx nan× tol) | PASS (240/240; máx nan× tol) | PASS (240/240; máx nan× tol) |
| `hmm_viterbi` | PASS (240/240; máx nan× tol) | PASS (240/240; máx nan× tol) | PASS (240/240; máx nan× tol) |
| `hmm_online_alpha` | PASS (240/240; máx 0× tol) | PASS (240/240; máx 0.023× tol) | PASS (240/240; máx 0.023× tol) |
| `hmm_online` | PASS (240/240; máx nan× tol) | PASS (240/240; máx nan× tol) | PASS (240/240; máx nan× tol) |
| `absorb` | PASS (240/240; máx nan× tol) | PASS (240/240; máx nan× tol) | PASS (240/240; máx nan× tol) |

`máx k× tol` é a maior razão erro/tolerância entre os itens da etapa (PASS exige ≤ 1, salvo empate demonstrado).

## Todas as discrepâncias (status diferente de PASS)

- **C float32 (PC)**, `borda_dc_ruido3s_midmt` (borda, modelo idmt_bloco0), etapa `env_mod`: PASS_EMPATE; erro máx 0.2051, razão 0.05859× tol. argmax de welch_P: C bin 90, ref bin 52; P_ref[ref]-P_ref[C] = 9.414e-16 (2·tol = 6.335e-11)
- **C float32 (PC)**, `borda_dc_ruido3s_midmt` (borda, modelo idmt_bloco0), etapa `features`: PASS_EMPATE; erro máx 37.11, razão 6.089e+06× tol. features[9]:f_mod
- **C float32 (PC)**, `borda_dc_ruido3s_midmt` (borda, modelo idmt_bloco0), etapa `features[9]:f_mod`: PASS_EMPATE; erro máx 37.11, razão 6.089e+06× tol. argmax de welch_P: C bin 90, ref bin 52; P_ref[ref]-P_ref[C] = 9.414e-16 (2·tol = 6.335e-11)
- **C float32 (PC)**, `borda_dc_ruido3s_midmt` (borda, modelo idmt_bloco0), etapa `z`: FAIL; erro máx 1.607, razão 1.389e+06× tol. 
- **C float32 (PC)**, `borda_dc_ruido3s_ma2212` (borda, modelo a2212_arq0), etapa `env_mod`: PASS_EMPATE; erro máx 0.2051, razão 0.05859× tol. argmax de welch_P: C bin 90, ref bin 52; P_ref[ref]-P_ref[C] = 9.414e-16 (2·tol = 6.335e-11)
- **C float32 (PC)**, `borda_dc_ruido3s_ma2212` (borda, modelo a2212_arq0), etapa `features`: PASS_EMPATE; erro máx 37.11, razão 6.089e+06× tol. features[9]:f_mod
- **C float32 (PC)**, `borda_dc_ruido3s_ma2212` (borda, modelo a2212_arq0), etapa `features[9]:f_mod`: PASS_EMPATE; erro máx 37.11, razão 6.089e+06× tol. argmax de welch_P: C bin 90, ref bin 52; P_ref[ref]-P_ref[C] = 9.414e-16 (2·tol = 6.335e-11)
- **C float32 (PC)**, `borda_dc_ruido3s_ma2212` (borda, modelo a2212_arq0), etapa `z`: FAIL; erro máx 1.265, razão 1.573e+06× tol. 

## Nota da auditoria independente

Os vereditos acima aplicam a regra congelada em `447a7c2`, que aceita como PASS_EMPATE um argmax do Welch diferente quando P_ref[bin da referência] − P_ref[bin do C] ≤ 2·tol absoluta. A auditoria independente (M1) mostrou que nos casos de borda DC+ruído essa diferença absoluta é pequena só porque o espectro inteiro está no piso do float32: em termos relativos o pico da referência é várias vezes o bin escolhido pelo C, então não é empate real. As tolerâncias e o comparador não foram alterados depois disso; o artigo conta esses casos, de forma conservadora, como FAIL em `env_mod`, `features` e `z` (a razão exata é recalculada por `src/python/gerar_artigo_numeros.py`). A decisão de classe não muda em nenhum caso.
