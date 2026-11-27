# Diagnóstico (Etapa 6): Decifrae × RMS/pico × clássicas nas mesmas janelas

Gerado por `embarcados/src/python/diagnostico_saidas.py` a partir do experimento `20261023_222831` (commit `9081500`, árvore suja = True). Nenhum número deste arquivo foi digitado à mão. Só validação: o `test/` do IDMT não foi lido. A coluna do porte em C entra depois.

## O que rodou

```
.venv\Scripts\python.exe embarcados\src\python\diagnostico.py --n-jobs 4
.venv\Scripts\python.exe embarcados\src\python\diagnostico_saidas.py
```

Tempo total do experimento: 57.4 s (extração sem cache, joblib n_jobs=4). Extração por variante/base (s): b00_rmspico_norm/idmt 3.2, b00_rmspico_bruto/idmt 2.5, p04_classicas/idmt 1.1, p11_decifrae_harmsum/idmt 13.3, b00_rmspico_norm/a2212 0.1, b00_rmspico_bruto/a2212 0.0, p04_classicas/a2212 0.1, p11_decifrae_harmsum/a2212 3.0. Bootstrap: 2000 reamostragens, semente 20261023; reamostra grupos com reposição, estratificado pela classe do grupo; grupo = (gravação, bloco) no IDMT e arquivo no A2212; IC percentil 95 %. Semente dos splits: 20261014.

Variantes: `b00_nivel` em duas condições declaradas, com o RMS da janela normalizado pelo runner (`normalizar_rms=True`, o RMS vira constante e só o pico relativo, isto é, o fator de crista, carrega informação) e sem normalizar (`normalizar_rms=False`, RMS e pico brutos); `p04_classicas` e `p11_decifrae_harmsum` com `normalizar_rms=True`, a mesma configuração de T1. Janelas: 3 s no IDMT (357 janelas do `train/`), 1 s no A2212 (430 janelas). Todas as variantes usam exatamente as mesmas janelas e os mesmos folds.

## Tabela principal (acurácia balanceada, IC 95 % por grupo, nº de grupos)

| protocolo | RMS/pico (RMS norm.) | RMS/pico (bruto) | Clássicas (p04) | Decifrae (p11) | grupos |
|---|---|---|---|---|---|
| IDMT aleat.* | 0.992 [0.983, 1.000] | 0.997 [0.990, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 18 ({"broken": 6, "good": 6, "heavyload": 6}) |
| IDMT blocos | 0.991 [0.982, 0.997] | 0.993 [0.987, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 15 ({"broken": 5, "good": 5, "heavyload": 5}) |
| A2212 aleat.* | 0.564 [0.515, 0.607] | 0.625 [0.571, 0.660] | 0.998 [0.995, 1.000] | 0.849 [0.790, 0.905] | 43 ({"Bearing": 7, "Healthy": 22, "Propeller": 14}) |
| A2212 arquivo | 0.562 [0.512, 0.606] | 0.608 [0.544, 0.656] | 0.981 [0.950, 0.998] | 0.816 [0.747, 0.881] | 43 ({"Bearing": 7, "Healthy": 22, "Propeller": 14}) |
| A2212 motor | 0.542 [0.485, 0.596] | 0.503 [0.412, 0.580] | 0.613 [0.579, 0.641] | 0.456 [0.397, 0.519] | 43 ({"Bearing": 7, "Healthy": 22, "Propeller": 14}) |
| A2212 sem M4 | 0.814 [0.720, 0.887] | 0.754 [0.623, 0.873] | 0.920 [0.868, 0.962] | 0.685 [0.593, 0.774] | 36 ({"Healthy": 22, "Propeller": 14}) |

\* split aleatório por janela: vazamento intencional, só para medir a inflação.

ROC-AUC one-vs-rest macro (scores contínuos `p_<classe>`) e F1 macro:

| protocolo | RMS/pico (RMS norm.) AUC / F1 | RMS/pico (bruto) AUC / F1 | Clássicas (p04) AUC / F1 | Decifrae (p11) AUC / F1 |
|---|---|---|---|---|
| IDMT aleat.* | 0.998 / 0.992 | 1.000 / 0.997 | 1.000 / 1.000 | 1.000 / 1.000 |
| IDMT blocos | 0.998 / 0.991 | 1.000 / 0.994 | 1.000 / 1.000 | 1.000 / 1.000 |
| A2212 aleat.* | 0.784 / 0.533 | 0.786 / 0.588 | 1.000 / 0.998 | 0.955 / 0.789 |
| A2212 arquivo | 0.759 / 0.532 | 0.706 / 0.576 | 0.996 / 0.982 | 0.935 / 0.753 |
| A2212 motor | 0.489 / 0.509 | 0.508 / 0.509 | 0.618 / 0.563 | 0.484 / 0.418 |
| A2212 sem M4 | 0.871 / 0.824 | 0.912 / 0.823 | 0.984 / 0.923 | 0.822 / 0.685 |

Sensibilidade (recall) por classe da p11:

- IDMT aleat.*: good 1.000, broken 1.000, heavyload 1.000
- IDMT blocos: good 1.000, broken 1.000, heavyload 1.000
- A2212 aleat.*: Healthy 0.641, Propeller 0.921, Bearing 0.986
- A2212 arquivo: Healthy 0.577, Propeller 0.929, Bearing 0.943
- A2212 motor: Healthy 0.441, Propeller 0.929, Bearing 0.000
- A2212 sem M4: Healthy 0.441, Propeller 0.929

## Leitura

Em IDMT aleat.*, a p11 tem acurácia balanceada 1.000 e empata com Clássicas (p04) (1.000); a p11 não traz ganho aqui. Em IDMT blocos, a p11 tem acurácia balanceada 1.000 e empata com Clássicas (p04) (1.000); a p11 não traz ganho aqui. Em A2212 aleat.*, a p11 tem acurácia balanceada 0.849 e NÃO supera Clássicas (p04) (0.998) e perde com ICs disjuntos. Em A2212 arquivo, a p11 tem acurácia balanceada 0.816 e NÃO supera Clássicas (p04) (0.981) e perde com ICs disjuntos. Em A2212 motor, a p11 tem acurácia balanceada 0.456 e NÃO supera Clássicas (p04) (0.613) e perde com ICs disjuntos. Em A2212 sem M4, a p11 tem acurácia balanceada 0.685 e NÃO supera Clássicas (p04) (0.920) e perde com ICs disjuntos.

No IDMT o atalho de nível vale nas duas condições: com o RMS normalizado o baseline de dois números ainda chega a 0.991 de acurácia balanceada nos blocos (o fator de crista basta) e sem normalizar chega a 0.993. Como o `train/` tem uma gravação por classe, classe, motor físico e gravação são a mesma variável; os blocos impedem que janelas vizinhas caiam dos dois lados, mas não impedem que o classificador aprenda a identidade da gravação. Por isso nenhum número do IDMT aqui mede diagnóstico de estado, e o IC é estreito por construção: são só 15 grupos (5 blocos × 3 gravações), todos da mesma sessão, e o bootstrap não representa a variabilidade entre motores ou sessões, que é zero amostras. O IC deve ser lido como variabilidade entre blocos, não como incerteza de generalização. Acurácias de 1,000 com IC [1,000, 1,000] são degeneradas: nenhuma réplica erra, e o bootstrap não tem como mostrar incerteza. A regra do projeto manda procurar vazamento acima de 99 %; aqui o vazamento é conhecido e estrutural (classe = gravação), não um bug de split.

No A2212 a acurácia balanceada da p11 por arquivo (0.816) é maior que a acurácia simples de T1 (0.751) porque o erro se concentra em Healthy, a classe majoritária (ver a sensibilidade por classe acima): a balanceada pesa as três classes igualmente.

No A2212 os grupos são os 43 arquivos (22 Healthy, 14 Propeller, 7 Bearing), o que dá um IC por arquivo utilizável, mas os arquivos de um mesmo motor não são independentes; no protocolo `motor` a unidade independente real é o motor, e com 4 motores não existe IC honesto nessa escala (o IC mostrado é por arquivo e é otimista). Quando M4 sai, a classe Bearing não tem nenhuma janela de treino, o classificador atribui p_Bearing=0 e a sensibilidade de Bearing é 0 por construção, o que limita a acurácia balanceada do protocolo `motor`; a linha `A2212 sem M4` tira o fold M4 e avalia só Healthy/Propeller (Bearing ainda pode ser predito, e conta como erro). Toda linha com Bearing está confundida com a identidade do motor 4.

## Comparação com o histórico (tabela T1 dos resultados do ICIT)

T1 reporta acurácia por janela (não balanceada). A comparação abaixo usa a mesma métrica, calculada das predições deste experimento e do CSV histórico `out/experiments/20261014_185859_validacao.csv`.

| pipeline | base | protocolo | normalizar_rms | T1 | agora | diferença |
|---|---|---|---|---|---|---|
| b00_nivel | a2212 | aleatorio | True | 0.730 | 0.730 | +0.000 |
| b00_nivel | a2212 | arquivo | False | 0.774 | 0.774 | +0.000 |
| b00_nivel | a2212 | arquivo | True | 0.728 | 0.728 | +0.000 |
| b00_nivel | a2212 | motor | True | 0.693 | 0.693 | +0.000 |
| b00_nivel | idmt | aleatorio | True | 0.992 | 0.992 | +0.000 |
| b00_nivel | idmt | blocos | False | 0.994 | 0.994 | +0.000 |
| b00_nivel | idmt | blocos | True | 0.991 | 0.991 | +0.000 |
| p04_classicas | a2212 | aleatorio | True | 0.998 | 0.998 | +0.000 |
| p04_classicas | a2212 | arquivo | True | 0.979 | 0.979 | +0.000 |
| p04_classicas | a2212 | motor | True | 0.777 | 0.777 | +0.000 |
| p04_classicas | idmt | aleatorio | True | 1.000 | 1.000 | +0.000 |
| p04_classicas | idmt | blocos | True | 1.000 | 1.000 | +0.000 |
| p11_decifrae_harmsum | a2212 | aleatorio | True | 0.788 | 0.788 | +0.000 |
| p11_decifrae_harmsum | a2212 | arquivo | True | 0.751 | 0.751 | +0.000 |
| p11_decifrae_harmsum | a2212 | motor | True | 0.528 | 0.528 | +0.000 |
| p11_decifrae_harmsum | idmt | aleatorio | True | 1.000 | 1.000 | +0.000 |
| p11_decifrae_harmsum | idmt | blocos | True | 1.000 | 1.000 | +0.000 |

Maior diferença absoluta: 0.000. Os números reproduzem o histórico (diferença < 0,005 em todas as células).

## Figuras e tabela

- (a) `figures/pdf/diag_balacc_20261023_222831.pdf` (+ PNG): acurácia balanceada com IC por protocolo.
- (b) `figures/pdf/diag_confusao_p11_20261023_222831.pdf` (+ PNG): matriz de confusão da p11 nos protocolos honestos (contagens; cor = fração por linha).
- (c) `figures/pdf/diag_espectro_20261023_222831.pdf` (+ PNG): espectro médio de uma janela good e uma broken do `train/` do IDMT com o pente k·f_rot (k = 1..48) e o caminho de rotação do Viterbi. good: `idmt/train/good/pure` janela 52 (156–159 s), f_rot mediana 25.73 Hz, 0/39 quadros SIL; broken: `idmt/train/broken/pure` janela 62 (186–189 s), f_rot mediana 27.99 Hz, 0/39 quadros SIL.
- Tabela LaTeX: `tables/latex/desempenho.tex` (booktabs, números do CSV).

## Arquivos do experimento

- `results/predictions/diagnostico_janelas_20261023_222831.csv`
- `results/predictions/p11_janelas_20261023_222831.csv`
- `results/predictions/p11_folds_20261023_222831.csv`
- `results/predictions/p11_modelos_20261023_222831.json` (mu, sd e protótipos do cosseno por fold, conferidos contra as probabilidades do runner com tolerância 1e-12)
- `results/csv/p11_features_20261023_222831.csv` (13 features por janela, para a comparação com o C)
- `results/csv/diagnostico_20261023_222831.csv`, `results/metrics/diagnostico_20261023_222831.json` (+ `_meta.json` de cada)
