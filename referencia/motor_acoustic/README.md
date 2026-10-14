# Diagnóstico acústico de motores — ICIT 2027

Adaptação do pipeline do Decifrae para estado de operação de motores por som,
pensada para rodar num ESP32. Prazo de submissão: **26/10/2026 (hard)**,
artigo regular de 6 páginas.

## Arquitetura

O pipeline tem duas camadas, e cada uma reaproveita uma ideia do Decifrae.

A **camada contínua** rastreia a rotação. A saliência de cada candidato de f0
vem da soma da energia nos harmônicos, com penalidade nos inter-harmônicos para
matar o erro de sub-oitava. Em seguida, o Viterbi escolhe o caminho mais
plausível no tempo, com salto máximo limitado. Ela substitui o pYIN, que fica
como baseline.

A **camada discreta** decide o estado. Um vetor de 18 features físicas por
janela de 1 s (rotação, razões harmônicas, espectro de modulação do envelope,
forma espectral) alimenta uma regressão logística, que é o que vai para o MCU.
Por cima, um HMM com transição "pegajosa" consolida a decisão: Viterbi no
offline e filtro forward no online.

```
wav -> 16 kHz -> passa-altas + RMS -> STFT 4096/1024 -> soma harmônica -> Viterbi -> f0(t)
                                   \-> envelope 1 kHz -> FFT -> pico de modulação
                                   \-> centróide, planicidade, 4 bandas
              features (18) -> logreg -> P(estado) -> HMM -> alarme
```

## Arquivos

| Arquivo | Papel |
|---|---|
| `motoracoustic/config.py` | Todos os parâmetros; um config por base |
| `motoracoustic/io.py` | Leitura e indexação do IDMT e do A2212, com grupos para split sem vazamento |
| `motoracoustic/preprocess.py` | Passa-altas (1 biquad), normalização RMS, janelamento |
| `motoracoustic/ridge.py` | Soma harmônica + Viterbi + refino parabólico; ablação sem Viterbi |
| `motoracoustic/features.py` | As 18 features e os grupos para ablação |
| `motoracoustic/hmm.py` | Viterbi de estados (offline) e ForwardFilter (online) |
| `motoracoustic/baselines.py` | pYIN e log-mel |
| `motoracoustic/evaluate.py` | Acurácia, F1, por condição, atraso de detecção, trocas espúrias, erro de f0 |
| `motoracoustic/mcu_budget.py` | Estimativa de RAM, flash e carga de CPU no ESP32 |
| `scripts/explore_spectrum.py` | **Primeiro passo com a base**: onde estão os ridges |
| `scripts/run_benchmark.py` | Benchmark completo do artigo |
| `scripts/compare_trackers_synth.py` | Ridge vs argmax vs pYIN em sinal sintético |
| `tests/test_synthetic.py` | 7 testes, todos passando |

## Como rodar

```bash
pip install numpy scipy scikit-learn librosa matplotlib pytest
python -m pytest tests -q
python scripts/explore_spectrum.py idmt /caminho/IDMT-ISA-ELECTRIC-ENGINE
python scripts/explore_spectrum.py bldc /caminho/BLDC
# ajustar f0_min / f0_max em config.py conforme as figuras
python scripts/run_benchmark.py idmt /caminho/IDMT-ISA-ELECTRIC-ENGINE
python scripts/run_benchmark.py bldc /caminho/BLDC
python -m motoracoustic.mcu_budget
```

## Bases

- **IDMT-ISA-ELECTRIC-ENGINE** (Grollmisch et al., EUSIPCO 2019; Zenodo 7551261).
  As classes são condições de tensão e carga, não defeitos mecânicos.
  Usar o split train/test oficial.
- **BLDC A2212** (Estacio et al., Data in Brief 50, 2023; Mendeley j4yr5fmhv4).
  43 arquivos de ~10 s a 16 kHz, com classes healthy, propeller e bearing.
  Split agrupado por arquivo; nunca por segmento.

## Pendências conhecidas

1. `f0_min/f0_max` são chutes até rodar o `explore_spectrum.py` nas bases reais.
2. Os padrões de nome de arquivo em `io.py` (classe, condição de ruído, id do
   motor) foram escritos às cegas. Conferir com `summarize()` na primeira execução.
3. O sintético com ruído branco é fácil demais: os três rastreadores empatam.
   A diferença real vai aparecer nas condições talking/atmo/stresstest do IDMT.
4. O orçamento do MCU é analítico. Medir com `esp_timer_get_time()` no ESP32.
