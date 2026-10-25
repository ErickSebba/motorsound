/* p11_lowmem.c -- variante nomeada p11_c_lowmem (ver p11_lowmem.h).
 *
 * As tabelas (grade, janela de Hann, fatores da rfft) são preenchidas com as mesmas contas
 * de p11_ws_init/p11_rfft_init; o laço de sosfilt é o de p11_sosfilt_dec, só que em fluxo
 * (o envelope nunca existe como vetor contíguo); a busca do pico e a mediana são as de
 * p11_envelope_mod. O resto chama p11_core.c sem alteração.
 */
#include "p11_lowmem.h"
#include "p11_soma.h"
#include "p11_constantes.h"

#include <math.h>
#include <string.h>

static const double LM_PI = 3.14159265358979323846264338327950288;
#define LM_ALIN 16u

/* ======================================================================== pool */
void lm_pool_vazio(lm_pool *pl) { memset(pl, 0, sizeof(*pl)); }

int lm_pool_add(lm_pool *pl, void *mem, size_t tam, int so32)
{
    if (pl->nb >= LM_MAX_BLOCOS || mem == NULL) return -1;
    pl->b[pl->nb].p = (unsigned char *)mem;
    pl->b[pl->nb].tam = tam;
    pl->b[pl->nb].usado = 0;
    pl->b[pl->nb].marca = 0;
    pl->b[pl->nb].so32 = so32;
    pl->nb++;
    return 0;
}

size_t lm_em_uso(const lm_pool *pl)
{
    size_t s = 0;
    int i;
    for (i = 0; i < pl->nb; ++i) s += pl->b[i].usado;
    return s;
}

/* melhor encaixe (bloco com a menor sobra suficiente); vetores que aceitam 32 bits tentam
 * antes os blocos so32, para deixar a memória de acesso por byte para quem precisa dela */
void *lm_corta(lm_pool *pl, size_t bytes, int aceita32)
{
    int passo, i;
    for (passo = 0; passo < 2; ++passo) {
        int melhor = -1;
        size_t sobra_melhor = 0;
        uintptr_t ini_melhor = 0;
        if (!aceita32 && passo == 1) break;
        for (i = 0; i < pl->nb; ++i) {
            lm_bloco *b = &pl->b[i];
            uintptr_t ini, fim, lim;
            if (b->so32 && !aceita32) continue;
            if (aceita32 && passo == 0 && !b->so32) continue;
            if (aceita32 && passo == 1 && b->so32) continue;
            ini = ((uintptr_t)(b->p + b->usado) + (LM_ALIN - 1)) & ~(uintptr_t)(LM_ALIN - 1);
            fim = ini + bytes;
            lim = (uintptr_t)(b->p + b->tam);
            if (fim <= lim && (melhor < 0 || (size_t)(lim - fim) < sobra_melhor)) {
                melhor = i;
                sobra_melhor = (size_t)(lim - fim);
                ini_melhor = ini;
            }
        }
        if (melhor >= 0) {
            size_t uso;
            lm_bloco *b = &pl->b[melhor];
#ifdef P11_LM_TRACE
            printf("corta %6u a32=%d -> bloco %d (sobra %u)\n", (unsigned)bytes, aceita32, melhor,
                   (unsigned)sobra_melhor);
#endif
            b->usado = (size_t)(ini_melhor + bytes - (uintptr_t)b->p);
            uso = lm_em_uso(pl);
            if (uso > pl->pico) pl->pico = uso;
            return (void *)ini_melhor;
        }
    }
    pl->falhou_bytes = bytes;
    return NULL;
}

/* corta no bloco i (o plano de p11_lm_init já garantiu que cabe) */
static void *lm_corta_bloco(lm_pool *pl, size_t bytes, int i)
{
    lm_bloco *b = &pl->b[i];
    uintptr_t ini = ((uintptr_t)(b->p + b->usado) + (LM_ALIN - 1)) & ~(uintptr_t)(LM_ALIN - 1);
    size_t uso;
    if (ini + bytes > (uintptr_t)(b->p + b->tam)) { pl->falhou_bytes = bytes; return NULL; }
    b->usado = (size_t)(ini + bytes - (uintptr_t)b->p);
    uso = lm_em_uso(pl);
    if (uso > pl->pico) pl->pico = uso;
    return (void *)ini;
}

void lm_marca(lm_pool *pl)
{
    int i;
    for (i = 0; i < pl->nb; ++i) pl->b[i].marca = pl->b[i].usado;
}

void lm_volta(lm_pool *pl)
{
    int i;
    for (i = 0; i < pl->nb; ++i) pl->b[i].usado = pl->b[i].marca;
}

#define MARCA(e) do { if (lm->marca) lm->marca(lm->marca_ctx, (e)); } while (0)

#define CORTA(ptr, tipo, n, a32) \
    do { (ptr) = (tipo *)lm_corta(lm->pool, sizeof(tipo) * (size_t)(n), (a32)); \
         if (!(ptr)) return -10; } while (0)

/* Pedidos atendidos do maior para o menor, com melhor encaixe. Com a RAM fragmentada do
 * chip, decidir fase a fase não basta: em 25/10 o P_mean (16 KB) do init caiu no bloco de
 * 118 KB e o qZ da fase A ficou sem lugar. Por isso o init planeja junto os pedidos
 * persistentes e os da fase A (bloco de cada um; cada pedido arredondado para LM_ALIN e
 * LM_ALIN de folga por bloco, o que vale para qualquer ordem de corte), e a fase A corta
 * direto no bloco planejado. */
typedef struct { void **dst; size_t bytes; int a32; int bloco; } lm_ped;
#define LM_MAX_PED (LM_MAX_LINHAS + 32)
static lm_ped g_ped[LM_MAX_PED];
static int g_nped;

#define PED(ptr, tipo, cnt, aceita)     do { g_ped[g_nped].dst = (void **)(void *)&(ptr);          g_ped[g_nped].bytes = sizeof(tipo) * (size_t)(cnt); g_ped[g_nped].a32 = (aceita);          g_ped[g_nped].bloco = -1; ++g_nped; } while (0)

static size_t arred(size_t b) { return (b + (LM_ALIN - 1)) & ~(size_t)(LM_ALIN - 1); }

static void ordena_ped(int *ord)
{
    int i, k;
    for (i = 0; i < g_nped; ++i) {
        int x = i;
        k = i - 1;
        while (k >= 0 && g_ped[ord[k]].bytes < g_ped[x].bytes) { ord[k + 1] = ord[k]; --k; }
        ord[k + 1] = x;
    }
}

/* Escolhe o bloco de cada pedido sem cortar nada: busca em profundidade, do maior pedido ao
 * menor, tentando primeiro o bloco de menor sobra (o melhor encaixe guloso é o primeiro ramo).
 * O guloso puro não basta: em 25/10, com DRAM 114624/110528/14272/8240, ele errava por 76 B
 * havendo divisão que cabe. Regras de memória iguais às de lm_corta (so32 só para quem aceita).
 * LM_PLANO_NOS limita a busca (roda uma vez por tamanho de janela). */
#define LM_PLANO_NOS 200000L
static int g_ord[LM_MAX_PED];
static size_t g_livre[LM_MAX_BLOCOS], g_resto[LM_MAX_PED + 1];
static long g_nos;

static int busca(const lm_pool *pl, int i)
{
    int cand[LM_MAX_BLOCOS], nc = 0, j, k;
    lm_ped *q;
    size_t c, tot = 0;
    if (i == g_nped) return 0;
    if (++g_nos > LM_PLANO_NOS) return -1;
    q = &g_ped[g_ord[i]];
    c = arred(q->bytes);
    for (j = 0; j < pl->nb; ++j) tot += g_livre[j];
    if (g_resto[i] > tot) return -1;   /* poda: o que falta não cabe nem somando tudo */
    for (j = 0; j < pl->nb; ++j) {
        if (pl->b[j].so32 && !q->a32) continue;
        if (c <= g_livre[j]) cand[nc++] = j;
    }
    /* ordem: quem aceita 32 bits tenta antes os blocos so32; depois, menor sobra primeiro */
    for (j = 1; j < nc; ++j) {
        int x = cand[j];
        k = j - 1;
        while (k >= 0) {
            int y = cand[k];
            int py = q->a32 ? !pl->b[y].so32 : 0, px = q->a32 ? !pl->b[x].so32 : 0;
            if (py < px || (py == px && g_livre[y] <= g_livre[x])) break;
            cand[k + 1] = y;
            --k;
        }
        cand[k + 1] = x;
    }
    for (k = 0; k < nc; ++k) {
        j = cand[k];
        g_livre[j] -= c;
        q->bloco = j;
        if (busca(pl, i + 1) == 0) return 0;
        g_livre[j] += c;
        if (g_nos > LM_PLANO_NOS) return -1;
    }
    return -1;
}

static int planeja(const lm_pool *pl)
{
    int j;
    for (j = 0; j < pl->nb; ++j) {
        size_t cap = pl->b[j].tam > LM_ALIN ? pl->b[j].tam - LM_ALIN : 0;
        g_livre[j] = cap > arred(pl->b[j].usado) ? cap - arred(pl->b[j].usado) : 0;
    }
    ordena_ped(g_ord);
    g_resto[g_nped] = 0;
    for (j = g_nped - 1; j >= 0; --j) g_resto[j] = g_resto[j + 1] + arred(g_ped[g_ord[j]].bytes);
    g_nos = 0;
    return busca(pl, 0) == 0 ? 0 : -10;
}

/* corta os pedidos que têm bloco planejado (os outros: melhor encaixe), do maior ao menor */
static int atende(lm_pool *pl)
{
    static int ord[LM_MAX_PED];
    int i;
    ordena_ped(ord);
    for (i = 0; i < g_nped; ++i) {
        lm_ped *q = &g_ped[ord[i]];
        void *p = q->bloco >= 0 ? lm_corta_bloco(pl, q->bytes, q->bloco)
                                : lm_corta(pl, q->bytes, q->a32);
        if (!p) { g_nped = 0; return -10; }
        *q->dst = p;
    }
    g_nped = 0;
    return 0;
}

/* ======================================================================== init */
static int n_dec_de(int n) { return (n + P11_DECIM - 1) / P11_DECIM; }

int p11_lm_init(p11_lm *lm, lm_pool *pool, int n)
{
    p11_ws *ws = &lm->ws;
    p11_cpx *tw_meia, *tw_r;
    size_t u0;
    int k, j, h = P11_N_FFT / 2, nd;
    memset(lm, 0, sizeof(*lm));
    lm->pool = pool;
    if (n < 1 || n % LM_N2 != 0 || n / LM_N2 > LM_MAX_LINHAS) return -1;
    lm->n2 = LM_N2;
    lm->n1 = n / LM_N2;
    nd = n_dec_de(n);
    u0 = lm_em_uso(pool);

    ws->n = n;
    ws->n_pad = n < P11_N_FFT ? P11_N_FFT : n;
    ws->T = p11_n_quadros(n);
    ws->usar_tabela = 0;
    ws->n_dec = nd;
    ws->nperseg = nd < 1024 ? nd : 1024;
    if (ws->nperseg % 2) return -2;   /* p11_welch exige nperseg par */

    /* Só as tabelas de giro aceitam IRAM (aceita = 1): o ESP32 não executa lsi/ssi (float) na
     * IRAM, e só p11_tw_le/p11_tw_grava as tocam, em palavras inteiras. Medido no chip em 24/10:
     * y na IRAM deu LoadStoreError em p11_preparar. */
    g_nped = 0;
    PED(ws->y, p11_real, n, 0);
    PED(ws->grade, double, P11_J, 0);
    PED(ws->janela, p11_real, P11_N_FFT, 0);
    PED(ws->P_mean, p11_real, P11_F, 0);
    PED(ws->e_q, p11_real, ws->T, 0);
    PED(ws->path, int32_t, ws->T, 0);
    PED(ws->fbuf, double, ws->T, 0);
    PED(ws->rbuf, p11_real, ws->T, 0);
    PED(ws->env_dec, p11_real, nd, 0);
    PED(tw_meia, p11_cpx, h, 1);
    PED(tw_r, p11_cpx, h, 1);
    PED(lm->tw_r128, p11_cpx, LM_N2, 0);
    PED(lm->tw_q, p11_cpx, lm->n1, 0);
    {   /* fase A entra no plano (mesmos pedidos de p11_lm_rastrear), mas não é cortada aqui */
        void *fant[4];
        int np = g_nped, k;
        PED(fant[0], p11_cpx, P11_F, 0);
        PED(fant[1], p11_cpx, P11_N_FFT / 2, 0);
        PED(fant[2], p11_real, P11_J, 0);
        PED(fant[3], p11_real, (size_t)ws->T * P11_J, 0);
        if (planeja(pool) != 0) { g_nped = 0; pool->falhou_bytes = 0; return -10; }
        for (k = 0; k < 4; ++k) lm->bloco_a[k] = g_ped[np + k].bloco;
        g_nped = np;
    }
    if (atende(pool) != 0) return -10;

    /* grade e janela: mesmas contas de p11_ws_init (calc_grade e np.hanning) */
    for (j = 0; j < P11_J; ++j)
        ws->grade[j] = P11_FMIN * pow(2.0, ((double)j * P11_PASSO_CENTS) / 1200.0);
    for (k = 0; k < P11_N_FFT; ++k) {
        double nn = (double)(1 - P11_N_FFT + 2 * k);
        double w = 0.5 + 0.5 * cos(LM_PI * nn / (double)(P11_N_FFT - 1));
        ws->janela[k] = (p11_real)(float)w;
    }
    /* rfft de 8192: mesmas contas de p11_rfft_init, com as duas tabelas em blocos separados */
    ws->rfft_stft.n = P11_N_FFT;
    if (p11_fft_init(&ws->rfft_stft.metade, h, tw_meia) != 0) return -3;
    ws->rfft_stft.tw_r = tw_r;
    for (k = 0; k < h; ++k) p11_tw_grava(&tw_r[k], -2.0 * LM_PI * (double)k / (double)P11_N_FFT);
    /* fatores do Hilbert em quatro passos: W_N^m = W_N^(N2*q) * W_N^r, m = N2*q + r */
    for (k = 0; k < LM_N2; ++k) {
        double fase = -2.0 * LM_PI * (double)k / (double)n;
        lm->tw_r128[k].re = (p11_real)cos(fase);
        lm->tw_r128[k].im = (p11_real)sin(fase);
    }
    for (k = 0; k < lm->n1; ++k) {
        double fase = -2.0 * LM_PI * (double)(k * LM_N2) / (double)n;
        lm->tw_q[k].re = (p11_real)cos(fase);
        lm->tw_q[k].im = (p11_real)sin(fase);
    }
    lm->bytes_persist = lm_em_uso(pool) - u0;
    lm_marca(pool);
    return 0;
}

/* ======================================================================== fase A */
int p11_lm_rastrear(p11_lm *lm, p11_real *min_delta, p11_real *score,
                    p11_real *dump_e_norm, p11_real *dump_custo)
{
    p11_ws *ws = &lm->ws;
    const int T = ws->T, J = P11_J;
    size_t u0 = lm_em_uso(lm->pool);
    int t;
    lm_volta(lm->pool);
    /* qz guarda o quadro empacotado (H2 complexos) e depois a rfft (F = H2+1 complexos):
     * p11_rfft lê só Z ao escrever out, então out pode ser o próprio z. |X| vai para os
     * primeiros F reais do mesmo vetor: qS[f] (bytes 4f) só sobrescreve qX[f/2], já lido. */
    g_nped = 0;
    PED(ws->qX, p11_cpx, P11_F, 0);       /* acesso por byte: abriga o psi (int16) depois */
    PED(ws->qZ, p11_cpx, P11_N_FFT / 2, 0);
    PED(ws->sal_q, p11_real, J, 0);
    PED(ws->e_norm, p11_real, (size_t)T * J, 0);
    for (t = 0; t < 4; ++t) g_ped[t].bloco = lm->bloco_a[t];   /* blocos planejados no init */
    if (atende(lm->pool) != 0) return -10;
    ws->qz = ws->qX;
    ws->qS = (p11_real *)ws->qX;
    /* psi e delta só são usados por p11_viterbi, depois do último quadro da STFT, quando qX
     * já está livre: moram dentro dele se couberem (T = 8 em 1 s: 28 848 B de 32 776 B) */
    {
        size_t b_psi = (sizeof(int16_t) * (size_t)T * (J + 1) + 15u) & ~(size_t)15u;
        size_t b_del = (sizeof(p11_real) * (size_t)(J + 1) + 15u) & ~(size_t)15u;
        if (b_psi + 2 * b_del <= sizeof(p11_cpx) * (size_t)P11_F) {
            unsigned char *base = (unsigned char *)ws->qX;
            ws->psi = (int16_t *)base;
            ws->delta_a = (p11_real *)(base + b_psi);
            ws->delta_b = (p11_real *)(base + b_psi + b_del);
        } else {
            CORTA(ws->psi, int16_t, (size_t)T * (J + 1), 0);
            CORTA(ws->delta_a, p11_real, J + 1, 0);
            CORTA(ws->delta_b, p11_real, J + 1, 0);
        }
    }
    (void)u0;
    lm->bytes_faseA = lm_em_uso(lm->pool);

    if (p11_rastrear_quadros(ws, NULL, NULL) != 0) return -1;
    MARCA(LM_T_STFT);
    if (dump_e_norm) for (t = 0; t < T * J; ++t) dump_e_norm[t] = ws->e_norm[t];
    if (dump_custo) {
        p11_real er = p11_e_ref(ws);
        for (t = 0; t < T; ++t) p11_custo_linha(ws, er, t, dump_custo + (size_t)t * (J + 1));
    }
    if (p11_viterbi(ws, min_delta, score) != 0) return -2;
    MARCA(LM_T_VITERBI);
    p11_saliencia_caminho(ws);
    MARCA(LM_T_SAL);
    lm_volta(lm->pool);
    ws->qz = ws->qZ = ws->qX = NULL; ws->qS = NULL; ws->sal_q = ws->e_norm = NULL;
    ws->psi = NULL; ws->delta_a = ws->delta_b = NULL;
    return 0;
}

/* ======================================================================== fase B */
static p11_cpx lm_cmul(p11_cpx a, p11_cpx b)
{
    p11_cpx r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

/* W_N^(m mod N) a partir das duas tabelas; inv conjuga */
static p11_cpx giro_n(const p11_lm *lm, long m, int inv)
{
    long n = (long)lm->n1 * lm->n2;
    p11_cpx w;
    m %= n;
    w = lm_cmul(lm->tw_q[m / lm->n2], lm->tw_r128[m % lm->n2]);
    if (inv) w.im = -w.im;
    return w;
}

#ifndef P11_SOS_DOUBLE
#define P11_SOS_DOUBLE 0
#endif
#if P11_REAL_EH_DOUBLE || P11_SOS_DOUBLE
typedef double lm_sos_t;
#define LM_SOS_TAB P11_SOS_F64
#else
typedef float lm_sos_t;
#define LM_SOS_TAB P11_SOS_F32
#endif

static void ordena_real(p11_real *v, int n)
{
    int i, k;
    for (i = 1; i < n; ++i) {
        p11_real x = v[i];
        k = i - 1;
        while (k >= 0 && v[k] > x) { v[k + 1] = v[k]; --k; }
        v[k + 1] = x;
    }
}

int p11_lm_envelope(p11_lm *lm, p11_real *dump_env_abs, p11_real *dump_env_dec,
                    p11_real *dump_welch_f, p11_real *dump_welch_P, p11_real out[2])
{
    p11_ws *ws = &lm->ws;
    const int n = ws->n, n1 = lm->n1, n2 = lm->n2, nd = ws->n_dec;
    int a, b, k, i;
    p11_real fator;
    lm_volta(lm->pool);

    /* ---------------- fase B: Hilbert em quatro passos ---------------- */
    /* y não é usado depois do Hilbert: as primeiras linhas moram dentro do próprio y. A linha
     * a ocupa os reais [2*N2*a, 2*N2*(a+1)) de y, que são as entradas das linhas 2a e 2a+1;
     * carregando de trás para frente (com a linha em row_t antes de gravar), toda entrada é
     * lida antes de ser sobrescrita. */
    {
        int no_y = (int)((sizeof(p11_real) * (size_t)n) / (sizeof(p11_cpx) * (size_t)n2));
        if (no_y > n1) no_y = n1;
        for (a = 0; a < n1; ++a) {
            if (a < no_y) lm->lin[a] = (p11_cpx *)(void *)ws->y + (size_t)a * n2;
            else CORTA(lm->lin[a], p11_cpx, n2, 0);
        }
    }
    CORTA(lm->col_a, p11_cpx, n1, 0);
    CORTA(lm->col_b, p11_cpx, n1, 0);
    CORTA(lm->row_t, p11_cpx, n2, 0);
    {
        p11_cpx *m1, *m2;
        CORTA(m1, p11_cpx, n1, 0);
        CORTA(m2, p11_cpx, n2, 0);
        if (p11_fft_init(&lm->pl_n1, n1, m1) != 0) return -1;
        if (p11_fft_init(&lm->pl_n2, n2, m2) != 0) return -1;
    }
    lm->bytes_faseB = lm_em_uso(lm->pool);

    /* x[N2*n1 + n2] na linha n1, coluna n2 (de trás para frente, ver acima) */
    for (a = n1 - 1; a >= 0; --a) {
        for (b = 0; b < n2; ++b) { lm->row_t[b].re = ws->y[a * n2 + b]; lm->row_t[b].im = 0; }
        for (b = 0; b < n2; ++b) lm->lin[a][b] = lm->row_t[b];
    }
    /* direta: DFT de N1 por coluna, giro W_N^(n2*k1), DFT de N2 por linha;
     * saída (k1, k2) = X[k1 + N1*k2] */
    for (b = 0; b < n2; ++b) {
        for (a = 0; a < n1; ++a) lm->col_a[a] = lm->lin[a][b];
        p11_fft(&lm->pl_n1, lm->col_a, lm->col_b);
        for (a = 0; a < n1; ++a) lm->lin[a][b] = lm_cmul(lm->col_b[a], giro_n(lm, (long)a * b, 0));
    }
    for (a = 0; a < n1; ++a) {
        p11_fft(&lm->pl_n2, lm->lin[a], lm->row_t);
        for (b = 0; b < n2; ++b) lm->lin[a][b] = lm->row_t[b];
    }
    /* h do scipy.signal.hilbert: 1 em 0, 2 em 1..N/2-1, 1 em N/2 (N par), 0 no resto */
    for (a = 0; a < n1; ++a)
        for (b = 0; b < n2; ++b) {
            long kk = (long)a + (long)n1 * b;
            if (n % 2 == 0) {
                if (kk >= 1 && kk < n / 2) { lm->lin[a][b].re *= 2; lm->lin[a][b].im *= 2; }
                else if (kk > n / 2) { lm->lin[a][b].re = 0; lm->lin[a][b].im = 0; }
            } else {
                if (kk >= 1 && kk < (n + 1) / 2) { lm->lin[a][b].re *= 2; lm->lin[a][b].im *= 2; }
                else if (kk >= (n + 1) / 2) { lm->lin[a][b].re = 0; lm->lin[a][b].im = 0; }
            }
        }
    /* inversa: IDFT de N2 por linha, giro W_N^-(k1*n2), IDFT de N1 por coluna;
     * saída (n1, n2) = x[N2*n1 + n2] sem o 1/N */
    for (a = 0; a < n1; ++a) {
        p11_fft_inv(&lm->pl_n2, lm->lin[a], lm->row_t);
        for (b = 0; b < n2; ++b) lm->lin[a][b] = lm_cmul(lm->row_t[b], giro_n(lm, (long)a * b, 1));
    }
    for (b = 0; b < n2; ++b) {
        for (a = 0; a < n1; ++a) lm->col_a[a] = lm->lin[a][b];
        p11_fft_inv(&lm->pl_n1, lm->col_a, lm->col_b);
        for (a = 0; a < n1; ++a) lm->lin[a][b] = lm->col_b[a];
    }
    MARCA(LM_T_HILBERT);
    /* |.|, sosfilt (forma direta II transposta, zi = 0) e [::16] em fluxo: o mesmo laço de
     * p11_sosfilt_dec, na ordem natural das amostras */
    fator = (p11_real)(1.0 / (double)n);
    {
        lm_sos_t z[P11_SOS_N_SECOES][2], sos[P11_SOS_N_SECOES][6];
        int sec, c;
        for (sec = 0; sec < P11_SOS_N_SECOES; ++sec) {
            z[sec][0] = z[sec][1] = 0;
            for (c = 0; c < 6; ++c) sos[sec][c] = (lm_sos_t)LM_SOS_TAB[sec][c];
        }
        for (k = 0; k < n; ++k) {
            const p11_cpx v = lm->lin[k / n2][k % n2];
            p11_real e = P11_HYPOT(v.re * fator, v.im * fator);
            lm_sos_t x = (lm_sos_t)e;
            if (dump_env_abs) dump_env_abs[k] = e;
            for (sec = 0; sec < P11_SOS_N_SECOES; ++sec) {
                lm_sos_t yv = sos[sec][0] * x + z[sec][0];
                z[sec][0] = sos[sec][1] * x - sos[sec][4] * yv + z[sec][1];
                z[sec][1] = sos[sec][2] * x - sos[sec][5] * yv;
                x = yv;
            }
            if (k % P11_DECIM == 0) ws->env_dec[k / P11_DECIM] = (p11_real)x;
        }
    }
    if (dump_env_dec) for (i = 0; i < nd; ++i) dump_env_dec[i] = ws->env_dec[i];
    {
        p11_real m = p11_soma_pares(ws->env_dec, nd) / (p11_real)nd;
        for (i = 0; i < nd; ++i) ws->env_dec[i] -= m;
    }
    MARCA(LM_T_SOS);
    lm_volta(lm->pool);
    for (a = 0; a < n1; ++a) lm->lin[a] = NULL;

    /* ---------------- fase C: Welch (p11_welch) + pico e proeminência ---------------- */
    {
        const int nps = ws->nperseg, nw = nps / 2 + 1;
        ws->mem_rfft_welch = lm_corta(lm->pool, p11_rfft_bytes(nps), 1);
        if (!ws->mem_rfft_welch) return -10;
        CORTA(ws->w_welch, p11_real, nps, 0);
        CORTA(ws->wz, p11_cpx, nps / 2 + 1, 0);
        CORTA(ws->wZ, p11_cpx, nps / 2 + 1, 0);
        CORTA(ws->wX, p11_cpx, nw, 0);
        CORTA(ws->welch_P, p11_real, nw, 0);
        CORTA(ws->welch_f, double, nw, 0);
        CORTA(ws->med_buf, p11_real, nw, 0);
        CORTA(ws->seg_buf, p11_real, nps, 0);
        lm->bytes_faseC = lm_em_uso(lm->pool);
        {
            int r = p11_welch(ws);
            if (r != 0) return r;
        }
        if (dump_welch_f) for (k = 0; k < nw; ++k) dump_welch_f[k] = (p11_real)ws->welch_f[k];
        if (dump_welch_P) for (k = 0; k < nw; ++k) dump_welch_P[k] = ws->welch_P[k];
        /* pico em [2, 100] Hz e proeminência: mesma busca de p11_envelope_mod */
        {
            int nm = 0, ib = -1;
            p11_real pb = 0, med;
            for (k = 0; k < nw; ++k) {
                if (ws->welch_f[k] >= 2.0 && ws->welch_f[k] <= 100.0) {
                    if (ib < 0 || ws->welch_P[k] > pb) { pb = ws->welch_P[k]; ib = k; }
                    ws->med_buf[nm++] = ws->welch_P[k];
                }
            }
            if (nm == 0) { out[0] = 0; out[1] = 0; }
            else {
                ordena_real(ws->med_buf, nm);
                med = (nm % 2) ? ws->med_buf[nm / 2]
                               : (ws->med_buf[nm / 2 - 1] + ws->med_buf[nm / 2]) / (p11_real)2;
                out[0] = (p11_real)ws->welch_f[ib];
                out[1] = (p11_real)10 * P11_LOG10(pb / (med + (p11_real)1e-20) + (p11_real)1e-12);
            }
        }
    }
    MARCA(LM_T_WELCH);
    lm_volta(lm->pool);
    return 0;
}

/* ======================================================================== features */
int p11_lm_features(p11_lm *lm, const p11_real env_mod[2], p11_real feat[P11_N_FEAT])
{
    p11_ws *ws = &lm->ws;
    lm_volta(lm->pool);
    CORTA(ws->qS, p11_real, P11_F, 0);     /* rascunho de F reais de p11_features */
    p11_features(ws, env_mod, feat);
    lm_volta(lm->pool);
    ws->qS = NULL;
    return 0;
}
