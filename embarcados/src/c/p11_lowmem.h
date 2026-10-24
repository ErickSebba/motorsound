/* p11_lowmem.h -- variante nomeada p11_c_lowmem: a mesma p11 em pouca RAM.
 *
 * Por que existe: o ESP32-D0WD-V3 da bancada não tem PSRAM; o workspace contíguo da
 * referência (p11_ws_bytes) pede 958 480 B (1 s) e o maior bloco livre medido no chip é de
 * 110 592 B (results/csv/esp32_memoria_esp32_*.csv).
 *
 * O que muda em relação a p11_c_ref (nenhuma conta do pipeline muda):
 *   1. memória: em vez de uma arena única, vários blocos (lm_pool) com alocação por fases;
 *      vetores só de float podem ficar em memória de acesso só em 32 bits (IRAM do ESP32);
 *   2. sem a tabela de interpolação harmônica (usar_tabela = 0: o mesmo i0/a calculado na
 *      hora, já existente em p11_rastrear_quadros);
 *   3. |X| do quadro compactado no próprio buffer da rfft (qS e qX no mesmo lugar);
 *   4. Hilbert por FFT de quatro passos (N = N1*N2, N2 = 128) com os dados em linhas de N2
 *      complexos espalhadas pelos blocos, em vez de FFT de raiz mista direta sobre um vetor
 *      contíguo de N complexos. É a mesma DFT, em outra ordem de operações: o erro numérico
 *      muda, e é medido pelo comparador do Agente D contra as mesmas tolerâncias do float.
 * STFT, soma harmônica, Viterbi, saliência no caminho, sosfilt, Welch, features, cosseno e
 * HMM são as funções de p11.h/p11_core.c, chamadas sem alteração.
 */
#ifndef P11_LOWMEM_H
#define P11_LOWMEM_H

#include "p11.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LM_MAX_BLOCOS 24
#define LM_N2 128            /* colunas do Hilbert em quatro passos */
#define LM_MAX_LINHAS 512    /* N1 máximo (n <= 65536) */

typedef struct {
    unsigned char *p;
    size_t tam, usado, marca;
    int so32;                /* 1: só aceita vetores de float (acesso de 32 bits) */
} lm_bloco;

typedef struct {
    lm_bloco b[LM_MAX_BLOCOS];
    int nb;
    size_t pico;             /* maior soma de 'usado' observada (bytes) */
    size_t falhou_bytes;     /* tamanho do último pedido que não coube (0 se nenhum) */
} lm_pool;

void lm_pool_vazio(lm_pool *pl);
int lm_pool_add(lm_pool *pl, void *mem, size_t tam, int so32);
/* aceita32 = 1: o vetor só é lido/escrito como float, pode ir para bloco so32 */
void *lm_corta(lm_pool *pl, size_t bytes, int aceita32);
void lm_marca(lm_pool *pl);       /* guarda o uso atual de cada bloco */
void lm_volta(lm_pool *pl);       /* devolve tudo o que foi cortado depois da marca */
size_t lm_em_uso(const lm_pool *pl);

/* pontos de marcação de tempo (o firmware mede entre eles; NULL no PC) */
enum { LM_T_STFT = 0, LM_T_VITERBI, LM_T_SAL, LM_T_HILBERT, LM_T_SOS, LM_T_WELCH, LM_N_T };
typedef void (*lm_marcador)(void *ctx, int etapa);

typedef struct {
    p11_ws ws;               /* os mesmos campos que p11_core.c usa, apontando para o pool */
    lm_marcador marca;       /* chamado ao fim de cada etapa LM_T_* (opcional) */
    void *marca_ctx;
    lm_pool *pool;
    int n1, n2;
    p11_fft_plano pl_n1, pl_n2;
    p11_cpx *tw_r128;        /* W_N^r, r < N2 (double -> real) */
    p11_cpx *tw_q;           /* W_N^(N2*q), q < N1 */
    p11_cpx *lin[LM_MAX_LINHAS];
    p11_cpx *col_a, *col_b, *row_t;
    size_t bytes_persist, bytes_faseA, bytes_faseB, bytes_faseC;
} p11_lm;

/* aloca a parte persistente (y = buffer de entrada do áudio, grade, janela, P_mean, rfft)
 * e preenche as tabelas como p11_ws_init. y é escrito pelo chamador (áudio cru) e depois
 * normalizado no lugar por p11_preparar(y, n, y). Devolve 0 ou negativo. */
int p11_lm_init(p11_lm *lm, lm_pool *pool, int n);

/* STFT por quadro + soma harmônica + Viterbi + saliência no caminho (fase A).
 * Dumps opcionais (NULL no MCU): e_norm (T*J), custo (T*(J+1)). */
int p11_lm_rastrear(p11_lm *lm, p11_real *min_delta, p11_real *score,
                    p11_real *dump_e_norm, p11_real *dump_custo);

/* Hilbert em quatro passos + sosfilt/dizimação (fase B) e Welch + pico (fase C).
 * Dumps opcionais: env_abs (n), env_dec (n/16), welch_f/welch_P (nperseg/2+1). */
int p11_lm_envelope(p11_lm *lm, p11_real *dump_env_abs, p11_real *dump_env_dec,
                    p11_real *dump_welch_f, p11_real *dump_welch_P, p11_real out[2]);

/* features (usa um rascunho de F reais cortado do pool) */
int p11_lm_features(p11_lm *lm, const p11_real env_mod[2], p11_real feat[P11_N_FEAT]);

#ifdef __cplusplus
}
#endif
#endif /* P11_LOWMEM_H */
