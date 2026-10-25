/* main.c -- bancada de medição da p11 no ESP32 (processor-in-the-loop pela serial).
 *
 * Protocolo (linhas ASCII; respostas do firmware começam com "@@" para o host separar
 * do log do sistema):
 *   INFO\n                       -> @@INFO {json}   chip, frequência, flash, PSRAM, IDF, flags
 *   MODEL <nbytes>\n <bytes>     -> @@OK MODEL C=<C>  (formato dos .bin de gold/modelos, contrato §3)
 *   WIN <id> <n>\n <n float32>   -> @@RES {json}    tempos por etapa (us), saídas, memória
 *   SEQ <cp> <n_logprob> <temp>\n -> @@SEQW {json} por janela e @@SEQFIM; lê a partição "seq"
 *
 * O núcleo C é o mesmo do PC (embarcados/src/c), compilado com P11_REAL=float. O tempo é
 * medido com esp_timer_get_time() só em torno de cada chamada do núcleo. A sequência de
 * chamadas é a mesma de p11_host.c::cmd_caso, sem os dumps.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_psram.h"
#include "esp_system.h"
#include "esp_idf_version.h"
#include "esp_private/esp_clk.h"
#include "esp_rom_crc.h"
#include "esp_memory_utils.h"

#include "p11.h"
#ifdef P11_LOWMEM
#include "p11_lowmem.h"
#endif
#ifdef P11_USAR_ESP_DSP
#include "dsps_fft2r.h"
#endif

#define UART_P11 UART_NUM_0
#define PILHA_TAREFA (24 * 1024)
#define LINHA_MAX 128
#define SAIDA_MAX 4096

static p11_modelo g_mod;
static int g_tem_modelo = 0;

/* buffers reaproveitados entre janelas do mesmo tamanho */
static int g_n = 0;
static float *g_audio = NULL;
static void *g_mem = NULL;
static size_t g_mem_bytes = 0;
static int g_mem_psram = 0;

static char g_out[SAIDA_MAX];
static int g_rc_det = 0;   /* código interno da última falha (p11_lm_rastrear: -10 = pool) */

/* ------------------------------------------------------------------ serial */
static void escreve(const char *s, size_t n) { uart_write_bytes(UART_P11, s, n); }

static void escreve_str(const char *s) { escreve(s, strlen(s)); }

static void le_exato(void *dst, size_t n)
{
    uint8_t *p = (uint8_t *)dst;
    while (n) {
        int r = uart_read_bytes(UART_P11, p, n > 1024 ? 1024 : n, portMAX_DELAY);
        if (r > 0) { p += r; n -= (size_t)r; }
    }
}

/* Leitura com confirmação: depois de cada pedaço de CONF_BYTES o firmware responde "@@A" e o
 * host só manda o próximo depois disso. Sem isso, a 921600 baud bytes se perdiam (24/10). */
#define CONF_BYTES 1024
/* Cada pedaço passa por um buffer em DRAM e vai para dst em palavras de 32 bits: na variante
 * lowmem o áudio pode morar na IRAM, que não aceita acesso por byte (nem o uart_read_bytes nem
 * o CRC). Devolve o CRC32 (o mesmo de zlib.crc32) dos bytes recebidos. n é múltiplo de 4. */
static uint32_t g_conf[CONF_BYTES / 4];
static uint32_t le_confirmado(void *dst, size_t n)
{
    uint32_t *p = (uint32_t *)dst, crc = 0;
    while (n) {
        size_t k = n > CONF_BYTES ? CONF_BYTES : n, i;
        le_exato(g_conf, k);
        crc = esp_rom_crc32_le(crc, (const uint8_t *)g_conf, k);
        for (i = 0; i < k / 4; ++i) p[i] = g_conf[i];
        p += k / 4;
        n -= k;
        uart_write_bytes(UART_P11, "@@A\n", 4);
    }
    return crc;
}

static int le_linha(char *buf, int cap)
{
    int i = 0;
    while (i < cap - 1) {
        uint8_t c;
        if (uart_read_bytes(UART_P11, &c, 1, portMAX_DELAY) != 1) continue;
        if (c == '\n') break;
        if (c != '\r') buf[i++] = (char)c;
    }
    buf[i] = 0;
    return i;
}

/* ------------------------------------------------------------------ json */
typedef struct { char *p; size_t cap, n; } sb_t;

static void sb_add(sb_t *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#include <stdarg.h>
static void sb_add(sb_t *b, const char *fmt, ...)
{
    va_list ap;
    int r;
    if (b->n >= b->cap) return;
    va_start(ap, fmt);
    r = vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap);
    va_end(ap);
    if (r > 0) b->n += (size_t)r;
    if (b->n > b->cap) b->n = b->cap;
}

/* %.9g preserva um float32 exatamente; NaN/inf como os literais aceitos pelo json do Python */
static void sb_real(sb_t *b, float v)
{
    if (isnan(v)) sb_add(b, "NaN");
    else if (isinf(v)) sb_add(b, v > 0 ? "Infinity" : "-Infinity");
    else sb_add(b, "%.9g", (double)v);
}

static void sb_vet(sb_t *b, const char *nome, const float *v, int n)
{
    int i;
    sb_add(b, "\"%s\":[", nome);
    for (i = 0; i < n; ++i) { if (i) sb_add(b, ","); sb_real(b, v[i]); }
    sb_add(b, "]");
}

/* ------------------------------------------------------------------ memória */
static int prepara_buffers(int n)
{
    size_t bytes;
    if (n == g_n && g_mem) return 0;
    if (g_audio) heap_caps_free(g_audio);
    if (g_mem) heap_caps_free(g_mem);
    g_audio = NULL; g_mem = NULL; g_n = 0;
    bytes = p11_ws_bytes(n, 1);
    /* PSRAM primeiro (o workspace da referência não cabe na SRAM interna); senão interna */
    g_mem = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    g_mem_psram = g_mem != NULL;
    if (!g_mem) g_mem = heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
    g_audio = (float *)heap_caps_malloc(sizeof(float) * (size_t)n,
                                        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!g_audio) g_audio = (float *)heap_caps_malloc(sizeof(float) * (size_t)n, MALLOC_CAP_8BIT);
    if (!g_mem || !g_audio) return -1;
    g_n = n;
    g_mem_bytes = bytes;
    return 0;
}

#ifdef P11_LOWMEM
/* ------------------------------------------------------------------ variante lowmem */
/* Pool montado no boot com todos os blocos livres: DRAM (acesso por byte) deixando
 * LM_RESERVA para o sistema, depois IRAM (só 32 bits, MALLOC_CAP_EXEC no ESP32). */
/* 7 KB: no chip o firmware funcionou com 6 KB livres (só MODEL e SEQ alocam depois, < 3 KB).
 * Com 12 KB (25/10) o último bloco de DRAM ficou em 14 272 B e o qZ (32 KB) não encaixou;
 * no PC, com o mesmo layout, basta esse bloco ter 18 272 B */
#define LM_RESERVA (7 * 1024)
static lm_pool g_pool;
static p11_lm g_lm;
static int g_lm_n = 0;
static int64_t g_lm_init_us = 0;
static int64_t g_marcas[LM_N_T];
static size_t g_pool_8bit = 0, g_pool_32bit = 0;

static void marcador(void *ctx, int e)
{
    (void)ctx;
    if (e >= 0 && e < LM_N_T) g_marcas[e] = esp_timer_get_time();
}

static void monta_pool(void)
{
    void *diram[LM_MAX_BLOCOS];
    int nd = 0, i;
    lm_pool_vazio(&g_pool);
    /* IRAM primeiro: no chip, tirar a IRAM depois da DRAM consumiu a reserva (24/10:
     * sobraram 6 KB de heap de byte em vez de 20 KB). Blocos D/IRAM (que o heap devolve pelo
     * endereço de IRAM) são soltos e voltam como DRAM no laço seguinte: na IRAM só cabem as
     * tabelas de giro, e um bloco D/IRAM de 110 KB fazia falta para os vetores de float. */
    for (;;) {
        size_t maior = heap_caps_get_largest_free_block(MALLOC_CAP_EXEC);
        size_t t;
        void *p;
        if (g_pool.nb >= LM_MAX_BLOCOS || nd >= LM_MAX_BLOCOS || maior < 4096) break;
        t = (maior - 64) & ~(size_t)15;
        p = heap_caps_malloc(t, MALLOC_CAP_EXEC);
        if (!p) break;
        if (esp_ptr_in_diram_iram(p)) { diram[nd++] = p; continue; }
        lm_pool_add(&g_pool, p, t, 1);
        g_pool_32bit += t;
    }
    for (i = 0; i < nd; ++i) heap_caps_free(diram[i]);
    for (;;) {
        size_t livre = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        size_t maior = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
        size_t t;
        void *p;
        if (g_pool.nb >= LM_MAX_BLOCOS || livre < LM_RESERVA + 4096 || maior < 4096) break;
        t = maior;
        if (livre - t < LM_RESERVA) t = livre - LM_RESERVA;
        t = (t - 64) & ~(size_t)15;
        if (t < 4096) break;
        p = heap_caps_malloc(t, MALLOC_CAP_8BIT);
        if (!p) break;
        lm_pool_add(&g_pool, p, t, 0);
        g_pool_8bit += t;
    }
}

/* tabelas (grade, janela, rfft, giros do Hilbert) uma vez por tamanho de janela */
static int lm_prepara(int n)
{
    int i, rc;
    int64_t t0;
    if (g_lm_n == n) return 0;
    for (i = 0; i < g_pool.nb; ++i) g_pool.b[i].usado = g_pool.b[i].marca = 0;
    g_pool.pico = 0;
    g_pool.falhou_bytes = 0;
    g_lm_n = 0;
    t0 = esp_timer_get_time();
    rc = p11_lm_init(&g_lm, &g_pool, n);
    g_lm_init_us = esp_timer_get_time() - t0;
    if (rc != 0) return rc;
    g_lm.marca = marcador;
    g_lm.marca_ctx = NULL;
    g_lm_n = n;
    return 0;
}

static int prepara(int n) { return lm_prepara(n); }
static float *audio_buf(void) { return (float *)g_lm.ws.y; }
#else
static int prepara(int n) { return prepara_buffers(n); }
static float *audio_buf(void) { return g_audio; }
#endif

/* ------------------------------------------------------------------ pipeline */
typedef struct {
    int64_t t_init, t_prep, t_rast, t_vit, t_sal, t_env, t_feat, t_cos;
    int64_t t_hil, t_sos, t_welch;   /* partes do envelope (só na variante lowmem) */
    p11_real min_delta, score, env_mod[2], feat[P11_N_FEAT], z[P11_N_FEAT];
    p11_real cosv[P11_MAX_CLASSES], probs[P11_MAX_CLASSES];
    int T;
    int32_t path[64];
} resultado_t;

#ifdef P11_LOWMEM
/* mesma sequência de p11_host.c::cmd_lowmem, sem dumps; o áudio já está em g_lm.ws.y */
static int roda_janela(const float *audio, int n, resultado_t *r)
{
    int64_t t0, t1;
    float *y;
    memset(r, 0, sizeof(*r));
    if (prepara(n) != 0) return -1;
    y = audio_buf();
    if (audio != y) { int i; for (i = 0; i < n; ++i) y[i] = audio[i]; }   /* y pode estar na IRAM */
    t0 = esp_timer_get_time();
    p11_preparar(y, n, y);
    t1 = esp_timer_get_time(); r->t_prep = t1 - t0;
    if ((g_rc_det = p11_lm_rastrear(&g_lm, &r->min_delta, &r->score, NULL, NULL)) != 0) return -2;
    r->t_rast = g_marcas[LM_T_STFT] - t1;
    r->t_vit = g_marcas[LM_T_VITERBI] - g_marcas[LM_T_STFT];
    r->t_sal = g_marcas[LM_T_SAL] - g_marcas[LM_T_VITERBI];
    t0 = esp_timer_get_time();
    if (p11_lm_envelope(&g_lm, NULL, NULL, NULL, NULL, r->env_mod) != 0) return -3;
    t1 = esp_timer_get_time(); r->t_env = t1 - t0;
    r->t_hil = g_marcas[LM_T_HILBERT] - t0;
    r->t_sos = g_marcas[LM_T_SOS] - g_marcas[LM_T_HILBERT];
    r->t_welch = g_marcas[LM_T_WELCH] - g_marcas[LM_T_SOS];
    t0 = t1;
    p11_lm_features(&g_lm, r->env_mod, r->feat);
    t1 = esp_timer_get_time(); r->t_feat = t1 - t0; t0 = t1;
    p11_cosseno(&g_mod, r->feat, r->z, r->cosv, r->probs);
    t1 = esp_timer_get_time(); r->t_cos = t1 - t0;
    r->T = g_lm.ws.T;
    if (r->T > (int)(sizeof r->path / sizeof r->path[0])) return -4;
    { int t; for (t = 0; t < r->T; ++t) r->path[t] = g_lm.ws.path[t]; }   /* path pode estar na IRAM */
    return 0;
}
#else
static int roda_janela(const float *audio, int n, resultado_t *r)
{
    p11_ws ws;
    int64_t t0, t1;
    memset(r, 0, sizeof(*r));
    if (prepara_buffers(n) != 0) return -1;
    if (audio != g_audio) memcpy(g_audio, audio, sizeof(float) * (size_t)n);

    t0 = esp_timer_get_time();
    if (p11_ws_init(&ws, g_mem, g_mem_bytes, n, 1) != 0) return -2;
    t1 = esp_timer_get_time(); r->t_init = t1 - t0; t0 = t1;
    p11_preparar(g_audio, n, ws.y);
    t1 = esp_timer_get_time(); r->t_prep = t1 - t0; t0 = t1;
    p11_rastrear_quadros(&ws, NULL, NULL);
    t1 = esp_timer_get_time(); r->t_rast = t1 - t0; t0 = t1;
    p11_viterbi(&ws, &r->min_delta, &r->score);
    t1 = esp_timer_get_time(); r->t_vit = t1 - t0; t0 = t1;
    p11_saliencia_caminho(&ws);
    t1 = esp_timer_get_time(); r->t_sal = t1 - t0; t0 = t1;
    if (p11_envelope_mod(&ws, NULL, NULL, NULL, NULL, r->env_mod) != 0) return -3;
    t1 = esp_timer_get_time(); r->t_env = t1 - t0; t0 = t1;
    p11_features(&ws, r->env_mod, r->feat);
    t1 = esp_timer_get_time(); r->t_feat = t1 - t0; t0 = t1;
    p11_cosseno(&g_mod, r->feat, r->z, r->cosv, r->probs);
    t1 = esp_timer_get_time(); r->t_cos = t1 - t0;

    r->T = ws.T;
    if (r->T > (int)(sizeof r->path / sizeof r->path[0])) return -4;
    memcpy(r->path, ws.path, sizeof(int32_t) * (size_t)ws.T);
    return 0;
}
#endif

static void sb_resultado(sb_t *b, const resultado_t *r)
{
    int i;
    sb_add(b, "\"us\":{\"init\":%" PRId64 ",\"prep\":%" PRId64 ",\"rastreio\":%" PRId64
              ",\"viterbi\":%" PRId64 ",\"sal_caminho\":%" PRId64 ",\"envelope\":%" PRId64
              ",\"features\":%" PRId64 ",\"cosseno\":%" PRId64 ",\"hilbert\":%" PRId64
              ",\"sosfilt\":%" PRId64 ",\"welch\":%" PRId64 "},",
           r->t_init, r->t_prep, r->t_rast, r->t_vit, r->t_sal, r->t_env, r->t_feat, r->t_cos,
           r->t_hil, r->t_sos, r->t_welch);
    sb_add(b, "\"T\":%d,\"vit_path\":[", r->T);
    for (i = 0; i < r->T; ++i) sb_add(b, i ? ",%" PRId32 : "%" PRId32, r->path[i]);
    sb_add(b, "],\"vit_min_delta\":"); sb_real(b, r->min_delta);
    sb_add(b, ",\"vit_score\":"); sb_real(b, r->score);
    sb_add(b, ","); sb_vet(b, "env_mod", r->env_mod, 2);
    sb_add(b, ","); sb_vet(b, "features", r->feat, P11_N_FEAT);
    sb_add(b, ","); sb_vet(b, "z", r->z, P11_N_FEAT);
    sb_add(b, ","); sb_vet(b, "cos", r->cosv, g_mod.C);
    sb_add(b, ","); sb_vet(b, "probs", r->probs, g_mod.C);
}

static void sb_memoria(sb_t *b)
{
#ifdef P11_LOWMEM
    /* ws_bytes = pico do pool usado pela p11 (persistente + maior fase), medido no chip */
    sb_add(b, "\"heap_min_livre_interna\":%u,\"heap_min_livre_psram\":%u,"
              "\"pilha_folga_bytes\":%u,\"ws_bytes\":%u,\"ws_na_psram\":0,"
              "\"pool_8bit\":%u,\"pool_32bit\":%u,\"pool_blocos\":%d,\"lm_init_us\":%" PRId64
              ",\"lm_persist\":%u,\"lm_faseA\":%u,\"lm_faseB\":%u,\"lm_faseC\":%u",
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)uxTaskGetStackHighWaterMark(NULL), (unsigned)g_pool.pico,
           (unsigned)g_pool_8bit, (unsigned)g_pool_32bit, g_pool.nb, g_lm_init_us,
           (unsigned)g_lm.bytes_persist, (unsigned)g_lm.bytes_faseA, (unsigned)g_lm.bytes_faseB,
           (unsigned)g_lm.bytes_faseC);
#else
    sb_add(b, "\"heap_min_livre_interna\":%u,\"heap_min_livre_psram\":%u,"
              "\"pilha_folga_bytes\":%u,\"ws_bytes\":%u,\"ws_na_psram\":%d",
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)uxTaskGetStackHighWaterMark(NULL),
           (unsigned)g_mem_bytes, g_mem_psram);
#endif
}

/* ------------------------------------------------------------------ comandos */
static void cmd_info(void)
{
    esp_chip_info_t ci;
    uint32_t flash = 0;
    sb_t b = { g_out, SAIDA_MAX, 0 };
    esp_chip_info(&ci);
    esp_flash_get_size(NULL, &flash);
    sb_add(&b, "@@INFO {\"chip_model\":%d,\"chip_nome\":\"%s\",\"cores\":%d,\"revisao\":%d,"
               "\"features\":%" PRIu32 ",\"cpu_hz\":%d,\"flash_bytes\":%" PRIu32 ","
               "\"psram_init\":%d,\"psram_bytes\":%u,\"idf\":\"%s\",\"flags\":\"%s\","
               "\"variante\":\"%s\",\"compilado\":\"%s %s\",",
           (int)ci.model, CONFIG_IDF_TARGET, ci.cores, ci.revision, ci.features,
           esp_clk_cpu_freq(), flash, (int)esp_psram_is_initialized(),
           (unsigned)(esp_psram_is_initialized() ? esp_psram_get_size() : 0),
           esp_get_idf_version(), P11_FLAGS, P11_VARIANTE, __DATE__, __TIME__);
    sb_add(&b, "\"heap_livre_interna\":%u,\"heap_livre_psram\":%u,\"heap_livre_8bit\":%u,"
               "\"heap_livre_exec\":%u,\"maior_bloco_8bit\":%u,",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_EXEC),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
#ifdef P11_LOWMEM
    {
        int i;
        sb_add(&b, "\"pool_8bit\":%u,\"pool_32bit\":%u,\"pool\":[",
               (unsigned)g_pool_8bit, (unsigned)g_pool_32bit);
        for (i = 0; i < g_pool.nb; ++i)
            sb_add(&b, "%s[%u,%d]", i ? "," : "", (unsigned)g_pool.b[i].tam, g_pool.b[i].so32);
        sb_add(&b, "],");
    }
#endif
    sb_add(&b, "\"ws_bytes_48000\":%u,\"ws_bytes_16000\":%u}\n",
           (unsigned)p11_ws_bytes(48000, 1), (unsigned)p11_ws_bytes(16000, 1));
    escreve(b.p, b.n);
}

static void cmd_model(int nbytes)
{
    uint8_t *buf = (uint8_t *)malloc((size_t)nbytes);
    int32_t C, d;
    double v;
    int i, c;
    char msg[64];
    if (!buf) { escreve_str("@@ERRO MODEL sem memoria\n"); return; }
    le_exato(buf, (size_t)nbytes);
    memcpy(&C, buf, 4);
    memcpy(&d, buf + 4, 4);
    if (d != P11_N_FEAT || C < 1 || C > P11_MAX_CLASSES ||
        nbytes != 8 + 8 * (1 + 2 * P11_N_FEAT + C * P11_N_FEAT)) {
        free(buf);
        escreve_str("@@ERRO MODEL invalido\n");
        return;
    }
    /* mesma conversão de p11_host.c::ler_modelo (double -> p11_real) */
#define LE_D(k) (memcpy(&v, buf + 8 + 8 * (k), 8), v)
    g_mod.C = C;
    g_mod.temp = (p11_real)LE_D(0);
    for (i = 0; i < P11_N_FEAT; ++i) g_mod.mu[i] = (p11_real)LE_D(1 + i);
    for (i = 0; i < P11_N_FEAT; ++i) g_mod.sd[i] = (p11_real)LE_D(1 + P11_N_FEAT + i);
    for (c = 0; c < C; ++c)
        for (i = 0; i < P11_N_FEAT; ++i)
            g_mod.M[c][i] = (p11_real)LE_D(1 + 2 * P11_N_FEAT + c * P11_N_FEAT + i);
#undef LE_D
    free(buf);
    g_tem_modelo = 1;
    snprintf(msg, sizeof msg, "@@OK MODEL C=%d\n", (int)C);
    escreve_str(msg);
}

static void cmd_win(int id, int n)
{
    resultado_t r;
    sb_t b = { g_out, SAIDA_MAX, 0 };
    int rc;
    uint32_t crc;
    if (!g_tem_modelo) { escreve_str("@@ERRO WIN sem modelo\n"); return; }
    if (prepara(n) != 0) {
        /* consome os bytes para manter o protocolo em sincronia */
        uint8_t lixo[256];
        size_t falta = sizeof(float) * (size_t)n;
        /* mesmo ritmo de confirmação do caminho normal */
        while (falta) {
            size_t k = falta > CONF_BYTES ? CONF_BYTES : falta, j;
            for (j = 0; j < k; j += sizeof lixo) le_exato(lixo, k - j > sizeof lixo ? sizeof lixo : k - j);
            falta -= k;
            uart_write_bytes(UART_P11, "@@A\n", 4);
        }
#ifdef P11_LOWMEM
        /* "para X bytes": o pedido do pool que não coube; "livre interna": total do pool */
        snprintf(g_out, SAIDA_MAX, "@@ERRO WIN %d sem memoria para %u bytes de workspace "
                 "(livre interna %u, psram 0)\n", id, (unsigned)g_pool.falhou_bytes,
                 (unsigned)(g_pool_8bit + g_pool_32bit));
#else
        snprintf(g_out, SAIDA_MAX, "@@ERRO WIN %d sem memoria para %u bytes de workspace "
                 "(livre interna %u, psram %u)\n", id, (unsigned)p11_ws_bytes(n, 1),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
#endif
        escreve_str(g_out);
        return;
    }
    /* CRC32 dos bytes recebidos, fora do tempo medido: o host descarta a janela se a serial
     * perdeu ou trocou algum byte */
    crc = le_confirmado(audio_buf(), sizeof(float) * (size_t)n);
    rc = roda_janela(audio_buf(), n, &r);
    if (rc != 0) {
        snprintf(g_out, SAIDA_MAX, "@@ERRO WIN %d rc=%d det=%d\n", id, rc, g_rc_det);
        escreve_str(g_out);
        return;
    }
    sb_add(&b, "@@RES {\"id\":%d,\"n\":%d,\"crc32\":%" PRIu32 ",", id, n, crc);
    sb_resultado(&b, &r);
    sb_add(&b, ",");
    sb_memoria(&b);
    sb_add(&b, "}\n");
    escreve(b.p, b.n);
}

/* Partição "seq": cabeçalho de 16 bytes "P11S", u32 n_jan, u32 n_amostras, f32 escala;
 * u32 rotulo[n_jan]; depois int16 áudio, janela após janela (src/python/pil_esp32.py). */
static void cmd_seq(float cp, float n_logprob, float temp)
{
    const esp_partition_t *pt = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "seq");
    uint8_t cab[16];
    uint32_t n_jan, n_am, *rot;
    float escala;
    int16_t *bloco;
    size_t off;
    p11_hmm_online h;
    uint32_t j, k;
    if (!g_tem_modelo) { escreve_str("@@ERRO SEQ sem modelo\n"); return; }
    if (!pt || esp_partition_read(pt, 0, cab, 16) != ESP_OK || memcmp(cab, "P11S", 4) != 0) {
        escreve_str("@@ERRO SEQ particao vazia\n");
        return;
    }
    memcpy(&n_jan, cab + 4, 4);
    memcpy(&n_am, cab + 8, 4);
    memcpy(&escala, cab + 12, 4);
    rot = (uint32_t *)malloc(4 * n_jan);
    bloco = (int16_t *)malloc(2 * 1024);
    if (!rot || !bloco || prepara((int)n_am) != 0) {
        escreve_str("@@ERRO SEQ sem memoria\n");
        free(rot); free(bloco);
        return;
    }
    esp_partition_read(pt, 16, rot, 4 * n_jan);
    off = 16 + 4 * (size_t)n_jan;
    p11_hmm_online_init(&h, g_mod.C + 1, cp);
    for (j = 0; j < n_jan; ++j) {
        resultado_t r;
        p11_real le[P11_MAX_CLASSES + 1];
        int est, rc;
        int64_t t0 = esp_timer_get_time(), t_le, t_hmm;
        sb_t b = { g_out, SAIDA_MAX, 0 };
        for (k = 0; k < n_am; k += 1024) {
            uint32_t m = n_am - k < 1024 ? n_am - k : 1024, i;
            esp_partition_read(pt, off + 2 * (size_t)k, bloco, 2 * m);
            for (i = 0; i < m; ++i) audio_buf()[k + i] = (float)bloco[i] * escala;
        }
        off += 2 * (size_t)n_am;
        t_le = esp_timer_get_time() - t0;
        rc = roda_janela(audio_buf(), (int)n_am, &r);
        if (rc != 0) { snprintf(g_out, SAIDA_MAX, "@@ERRO SEQ jan %u rc=%d\n", (unsigned)j, rc); escreve_str(g_out); break; }
        t0 = esp_timer_get_time();
        p11_hmm_log_em(r.probs, 1, g_mod.C, temp, n_logprob, le);
        est = p11_hmm_online_passo(&h, le);
        t_hmm = esp_timer_get_time() - t0;
        sb_add(&b, "@@SEQW {\"jan\":%u,\"verdade\":%u,\"decisao\":%d,\"us_leitura_flash\":%" PRId64
                   ",\"us_hmm\":%" PRId64 ",", (unsigned)j, (unsigned)rot[j], est, t_le, t_hmm);
        sb_resultado(&b, &r);
        sb_add(&b, ",");
        sb_memoria(&b);
        sb_add(&b, "}\n");
        escreve(b.p, b.n);
    }
    free(rot); free(bloco);
    escreve_str("@@SEQFIM\n");
}

static void tarefa_bancada(void *arg)
{
    char linha[LINHA_MAX];
    (void)arg;
#ifdef P11_LOWMEM
    monta_pool();   /* depois de a tarefa existir: a pilha dela já saiu do heap */
#endif
    escreve_str("@@PRONTO p11\n");
    for (;;) {
        int a = 0, n = 0;
        float x = 0, y = 0, z = 0;
        if (le_linha(linha, sizeof linha) == 0) continue;
        if (strcmp(linha, "INFO") == 0) cmd_info();
        else if (sscanf(linha, "MODEL %d", &a) == 1) cmd_model(a);
        else if (sscanf(linha, "WIN %d %d", &a, &n) == 2) cmd_win(a, n);
        else if (sscanf(linha, "SEQ %f %f %f", &x, &y, &z) == 3) cmd_seq(x, y, z);
        else if (strcmp(linha, "PING") == 0) escreve_str("@@PONG\n");
        else escreve_str("@@ERRO comando\n");
    }
}

void app_main(void)
{
    uart_config_t cfg = {
        .baud_rate = CONFIG_ESP_CONSOLE_UART_BAUDRATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
#ifdef P11_USAR_ESP_DSP
    /* tabela da ESP-DSP para o maior N potência de 2 usado (4096: meia rfft de 8192) */
    if (dsps_fft2r_init_fc32(NULL, 4096) != ESP_OK) abort();
#endif
    /* buffers pequenos (RAM é o gargalo sem PSRAM); a 921600 baud a tarefa lê bem antes
     * de 4 KB acumularem */
    uart_driver_install(UART_P11, 4 * 1024, 2 * 1024, 0, NULL, 0);
    uart_param_config(UART_P11, &cfg);
    /* interrupção com o FIFO (128 B) em 32 B, não 120: ~1 ms de folga a 921600 baud */
    uart_set_rx_full_threshold(UART_P11, 32);
    /* núcleo 1 no ESP32/ESP32-S3 (o 0 fica com o sistema); prioridade acima do idle */
    xTaskCreatePinnedToCore(tarefa_bancada, "p11", PILHA_TAREFA, NULL, 5, NULL,
                            portNUM_PROCESSORS > 1 ? 1 : 0);
}
