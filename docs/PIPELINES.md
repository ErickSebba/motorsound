# Pipelines

Uma linha por pipeline. "Verificação rápida" é uma checagem preliminar (só imprime;
não é número oficial): IDMT blocos limpo / IDMT blocos com rosa 10 dB / A2212 por arquivo /
A2212 aleatório. Os números oficiais saem de `experimentos/validacao.py` e
`experimentos/robustez.py` e ficam em `out/experiments/`. Suposições completas no
cabeçalho de cada arquivo. Custo = estimativa analítica (nada medido no ESP32 ainda).

| id | tipo | origem | estado | verificação rápida | custo MCU (estimado) | notas |
|---|---|---|---|---|---|---|
| b00_nivel | baseline | — | validada | 0,991 / 0,620 / 0,728 / 0,730 | 9 params | RMS + pico; com RMS normalizado o fator de crista sozinho separa o IDMT |
| b01_forma_espectral | baseline | — | validada | 1,000 / 0,641 / 0,935 / 0,998 | ~1,6 MFLOP/s | B1 do plano: 32 bandas + forma espectral + LR |
| p00_logmel_cnn | inspirada | strantzalis2022 | validada | 1,000 / 0,641 / 0,965 / 0,998 | 7 703 params (=artigo), ~3,6 MFLOP/s; ESP32 est. ~14 ms/inf. | arquitetura idêntica; otimizador e normalização supostos; 1 440 passos = nº de atualizações do artigo |
| p01_mfcc_cnn1d | inspirada | strantzalis2022 | validada | 1,000 / 0,667 / 0,874 / 0,891 | 3 553 params, ~2,4 MFLOP/s | MFCC estilo Edge Impulse com padrões supostos |
| p02_grollmisch | inspirada | grollmisch2019 | validada | 1,000 / 1,000 / 0,947 / 1,000 | 542 k params, ~0,53 MB int8, ~28,6 MFLOP/s | 60 épocas em vez de 1000 (custo); média por janela em vez de voto por arquivo |
| p03_etnext | inspirada | etnext2026 | validada | 1,000 / 0,890 / 0,905 / 0,981 | ~0,8 MFLOP/s, ~28 kB flash | NCA própria (fscnca não existe no sklearn); seleção DENTRO do fold (no artigo, no mesmo CV do número) |
| p03b_etnext_knn | inspirada | etnext2026 | validada | 1,000 / 0,829 / 0,770 / 0,953 | ~270 kB flash (guarda o treino) | Fine kNN = 1-NN; Δ aleatório−arquivo = 18 pp |
| p04_classicas | inspirada | vanaraj2026 | validada | 1,000 / 0,641 / 0,979 / 0,998 | ~292 kB flash, ~2,1 MFLOP/s | só resumo lido; voto SVM/RF/kNN/LR suposto |
| p04b_msaf | inspirada | glowacz2018 | resultado negativo | 0,875 / 0,475 / 0,344 / 0,374 | ~0,6 MFLOP/s | bins selecionados caem em linhas que dependem da velocidade; protótipo médio falha no A2212 |
| p04c_classicas_lr | inspirada | vanaraj2026 | validada | 0,997 / 0,672 / 0,974 / 0,995 | 163 params, ~2,0 MFLOP/s | p04 só com LR; praticamente empata com a p04 |
| p05_melcnn | inspirada | melcnn2025 | validada | 1,000 / 0,693 / 0,977 / 1,000 | 2,45 M params, 9,4 MB f32 (não cabe) | fim da rede suposto (texto truncado); 22,05 kHz suposto |
| p10_decifrae_pyin_viterbi | decifrae | — | validada | 1,000 / 0,907 / — / — (grade 10 c) | pYIN não embarcável | emissão do baixo V3 com pYIN |
| p11_decifrae_harmsum | decifrae | — | validada | 0,997 / 0,980 / 0,770 / — (grade 10 c) | ver custo_mcu | versão embarcável: só soma harmônica; grade 2 c desde 14/10 |
| p11a_grade10 | decifrae | — | registro | = p11 antes da correção | | grade de 10 cents (trava em 4/3·f no `good`) |
| p12_decifrae_sem_viterbi | decifrae | — | validada | 0,991 / — / 0,774 / — (grade 10 c) | | argmax por quadro (isola o Viterbi de rotação) |
| p13 | decifrae | — | realizada como decodificação | — | | "p11 sem HMM" = decodificação `argmax` em `experimentos/sequencias.py`; o HMM de estados é aplicado a todas as pipelines |
| p20_hfs | inspirada | wheat2025 | validada | 1,000 / 0,548 / 0,542 / 0,614 | ~11,9 MFLOP/s | f_o do rastreador; carga não é regressor; PCA(2) é o gargalo; variantes de POST exploradas olhando a validação ficam só como registro, vale a configuração padrão |
| p20b_so_frot | inspirada | wheat2025 (controle) | validada | 0,922 / 0,890 / 0,449 / 0,498 | = rastreador | só a rotação: separa o IDMT e resiste ao ruído; não serve no A2212 (faixa 15–60 Hz) |
| p21_envelope | inspirada | decifrae | validada | 1,000 / 0,881 / 0,940 / 0,993 | ~3,1 MFLOP/s, ~24 kB RAM | espectro de modulação do envelope, todo causal |
| p30_anomalia | inspirada | harada2023 | validada | acc limitada a 2/3 por desenho; AUC A2212 0,955, M1+M2 0,944 | ~4,4 kB flash, 1,7 MFLOP/s | Mahalanobis Ledoit-Wolf treinada só no normal |
| p30b_ae | inspirada | purohit2019 | validada | AUC A2212 0,940, M1+M2 0,920 | ~50 kB int8, 4,8 MFLOP/s | AE 320-64-64-8; Bearing 0,989 confundido com M4 |

Nenhuma pipeline ficou como "reprodução": todas dependem de parâmetros que os artigos omitem.
