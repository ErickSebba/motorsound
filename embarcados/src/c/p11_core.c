/* p11_core.c -- workspace, _preparar, rastreador de rotação (STFT por quadro + soma
 * harmônica + Viterbi), _envelope_mod e as 13 features de p11_decifrae_harmsum.
 *
 * Referência (somente leitura), lida linha a linha em 23/10/2026:
 *   motorsound/runner.py::_preparar
 *   motorsound/rotacao.py::grade, stft_mag, saliencia_harmonica, emissao_baixo,
 *                         viterbi_rotacao, rastrear
 *   pipelines/_decifrae.py::extrair, _envelope_mod
 * Nada do porte do colaborador (colaborador_original/decifrae_core.c) foi copiado
 * literalmente: ele não tinha STFT, rastreador nem envelope; as features foram reescritas
 * para seguir o numpy (ver c_port.md).
 *
 * Precisão: no build double tudo é double. No build float tudo é float, inclusive as
 * acumulações longas (soma harmônica de 48 termos, P_mean sobre T quadros, somas de banda
 * de ~2000 bins, Welch); as decisões de índice que o Python faz em float64 (grade, posição
 * dos harmônicos na feature 4, mediana/IQR de f_rot, tempos do absorb) ficam em double
 * nos dois builds, porque mudam inteiros e são poucas operações.
 */
#include "p11.h"
#include "p11_soma.h"
#include "p11_constantes.h"

#include <math.h>
#include <string.h>

static const double P11_PI = 3.14159265358979323846264338327950288;

/* ======================================================================== layout */
#define P11_ALIN 16u

typedef struct { unsigned char *base; size_t off; } arena_t;

static void *corta(arena_t *a, size_t bytes)
{
    void *p;
    a->off = (a->off + (P11_ALIN - 1)) & ~(size_t)(P11_ALIN - 1);
    p = a->base ? (void *)(a->base + a->off) : NULL;
    a->off += bytes;
    return p;
}

int p11_n_quadros(int n)
{
    int n_pad = n < P11_N_FFT ? P11_N_FFT : n;
    return 1 + (n_pad - P11_N_FFT) / P11_HOP;
}

static int n_dec_de(int n) { return (n + P11_DECIM - 1) / P11_DECIM; }  /* len(env[::16]) */

static int nperseg_de(int n)
{
    int nd = n_dec_de(n);
    return nd < 1024 ? nd : 1024;
}

int p11_n_welch(int n) { return nperseg_de(n) / 2 + 1; }

/* Distribui os ponteiros. base == NULL: só mede. Devolve o total e as partes. */
static size_t layout(p11_ws *ws, unsigned char *base, int n, int usar_tabela,
                     size_t *persist, size_t *faseA, size_t *faseB)
{
    arena_t a;
    size_t ini_uniao, fimA, fimB;
    int T = p11_n_quadros(n), J = P11_J, F = P11_F;
    int nd = n_dec_de(n), nps = nperseg_de(n), nw = nps / 2 + 1;
    a.base = base;
    a.off = 0;

    ws->n = n;
    ws->n_pad = n < P11_N_FFT ? P11_N_FFT : n;
    ws->T = T;
    ws->usar_tabela = usar_tabela;
    ws->n_dec = nd;
    ws->nperseg = nps;

    /* persistente */
    ws->y = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)n);
    ws->grade = (double *)corta(&a, sizeof(double) * (size_t)J);
    ws->janela = (p11_real *)corta(&a, sizeof(p11_real) * P11_N_FFT);
    if (usar_tabela) {
        ws->h_i0 = (uint16_t *)corta(&a, sizeof(uint16_t) * (size_t)P11_N_HARM * J);
        ws->h_a = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)P11_N_HARM * J);
    } else {
        ws->h_i0 = NULL;
        ws->h_a = NULL;
        corta(&a, 0);
    }
    ws->P_mean = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)F);
    ws->e_q = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)T);
    ws->path = (int32_t *)corta(&a, sizeof(int32_t) * (size_t)T);
    ws->fbuf = (double *)corta(&a, sizeof(double) * (size_t)T);
    ws->rbuf = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)T);
    ws->mem_rfft_stft = corta(&a, p11_rfft_bytes(P11_N_FFT));
    *persist = a.off;

    /* fase A */
    ini_uniao = a.off;
    ws->qz = (p11_cpx *)corta(&a, sizeof(p11_cpx) * (P11_N_FFT / 2));
    ws->qZ = (p11_cpx *)corta(&a, sizeof(p11_cpx) * (P11_N_FFT / 2));
    ws->qX = (p11_cpx *)corta(&a, sizeof(p11_cpx) * (size_t)F);
    ws->qS = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)F);
    ws->sal_q = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)J);
    ws->e_norm = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)T * J);
    ws->psi = (int16_t *)corta(&a, sizeof(int16_t) * (size_t)T * (J + 1));
    ws->delta_a = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)(J + 1));
    ws->delta_b = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)(J + 1));
    fimA = a.off;

    /* fase B (recomeça no início da união) */
    a.off = ini_uniao;
    ws->mem_fft_hil = corta(&a, p11_fft_bytes(n));
    ws->hx = (p11_cpx *)corta(&a, sizeof(p11_cpx) * (size_t)n);
    ws->hX = (p11_cpx *)corta(&a, sizeof(p11_cpx) * (size_t)n);
    ws->env_dec = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)nd);
    ws->mem_rfft_welch = corta(&a, p11_rfft_bytes(nps + (nps & 1)));
    ws->w_welch = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)nps);
    ws->wz = (p11_cpx *)corta(&a, sizeof(p11_cpx) * (size_t)(nps / 2 + 1));
    ws->wZ = (p11_cpx *)corta(&a, sizeof(p11_cpx) * (size_t)(nps / 2 + 1));
    ws->wX = (p11_cpx *)corta(&a, sizeof(p11_cpx) * (size_t)nw);
    ws->welch_P = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)nw);
    ws->welch_f = (double *)corta(&a, sizeof(double) * (size_t)nw);
    ws->med_buf = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)nw);
    ws->seg_buf = (p11_real *)corta(&a, sizeof(p11_real) * (size_t)nps);
    fimB = a.off;

    *faseA = fimA - ini_uniao;
    *faseB = fimB - ini_uniao;
    return (fimA > fimB ? fimA : fimB) + P11_ALIN;   /* + folga para alinhar a base */
}

size_t p11_ws_bytes(int n, int usar_tabela)
{
    p11_ws tmp;
    size_t p, a, b;
    return layout(&tmp, NULL, n, usar_tabela, &p, &a, &b);
}

void p11_ws_partes(int n, int usar_tabela, size_t *persist, size_t *faseA, size_t *faseB)
{
    p11_ws tmp;
    layout(&tmp, NULL, n, usar_tabela, persist, faseA, faseB);
}

/* rotacao.grade: n = floor(1200*log2(fmax/fmin)/passo)+1; f = fmin*2**(arange(n)*passo/1200) */
static void calc_grade(double *g)
{
    int j;
    for (j = 0; j < P11_J; ++j)
        g[j] = P11_FMIN * pow(2.0, ((double)j * P11_PASSO_CENTS) / 1200.0);
}

/* df de rotacao.saliencia_harmonica: freqs[1]-freqs[0], freqs = rfftfreq(8192, 1/16000)
 * = arange * (1/(n*d)) */
static double df_stft(void)
{
    double val = 1.0 / ((double)P11_N_FFT * (1.0 / (double)P11_SR));
    return 1.0 * val - 0.0 * val;
}

int p11_ws_init(p11_ws *ws, void *mem, size_t bytes, int n, int usar_tabela)
{
    size_t p, a, b, total;
    unsigned char *base;
    int k, h, j;
    if (n < 1 || mem == NULL) return -1;
    total = p11_ws_bytes(n, usar_tabela);
    if (bytes < total) return -2;
    base = (unsigned char *)(((uintptr_t)mem + (P11_ALIN - 1)) & ~(uintptr_t)(P11_ALIN - 1));
    memset(ws, 0, sizeof(*ws));
    layout(ws, base, n, usar_tabela, &p, &a, &b);

    calc_grade(ws->grade);
    /* np.hanning(M): n = arange(1-M, M, 2); 0.5 + 0.5*cos(pi*n/(M-1)); depois float32 */
    for (k = 0; k < P11_N_FFT; ++k) {
        double nn = (double)(1 - P11_N_FFT + 2 * k);
        double w = 0.5 + 0.5 * cos(P11_PI * nn / (double)(P11_N_FFT - 1));
        ws->janela[k] = (p11_real)(float)w;
    }
    if (usar_tabela) {
        double df = df_stft();
        for (h = 1; h <= P11_N_HARM; ++h)
            for (j = 0; j < P11_J; ++j) {
                double pos = ((double)h * ws->grade[j]) / df;
                double i0 = floor(pos);
                /* pos < F-1 vale para todo (h, j) desta grade (máx. 48*60/1,953 = 1474,6) */
                ws->h_i0[(h - 1) * P11_J + j] = (uint16_t)i0;
                ws->h_a[(h - 1) * P11_J + j] = (p11_real)(pos - i0);
            }
    }
    if (p11_rfft_init(&ws->rfft_stft, P11_N_FFT, ws->mem_rfft_stft) != 0) return -3;
    return 0;
}

/* ======================================================================== _preparar */
void p11_preparar(const float *y_in, int n, p11_real *y_out)
{
    /* numpy: m = float32(sum_pares(y*y) / n); r = float(sqrt(m)) + 1e-12 (double);
     * y * (0.1/r): escalar fraco -> convertido para float32, produto em float32 */
    float s = p11_soma_quad_f32(y_in, n);
    float m = s / (float)n;
    double r = (double)sqrtf(m) + 1e-12;
    float c = (float)(P11_RMS_ALVO / r);
    int i;
    for (i = 0; i < n; ++i) y_out[i] = (p11_real)(y_in[i] * c);
}

/* ======================================================================== rastreio */
int p11_rastrear_quadros(p11_ws *ws, p11_real *dump_S, p11_real *dump_sal)
{
    const int T = ws->T, J = P11_J, F = P11_F, H2 = P11_N_FFT / 2;
    int t, k, f, h, j;
    double df = df_stft();
    for (t = 0; t < T; ++t) {
        const int off = t * P11_HOP;
        p11_real mx;
        /* quadro janelado, empacotado par/ímpar; zero-pad no fim se n < n_fft */
        for (k = 0; k < H2; ++k) {
            int i0 = off + 2 * k, i1 = i0 + 1;
            p11_real x0 = i0 < ws->n ? ws->y[i0] : (p11_real)0;
            p11_real x1 = i1 < ws->n ? ws->y[i1] : (p11_real)0;
            ws->qz[k].re = x0 * ws->janela[2 * k];
            ws->qz[k].im = x1 * ws->janela[2 * k + 1];
        }
        p11_rfft(&ws->rfft_stft, ws->qz, ws->qZ, ws->qX);
        for (f = 0; f < F; ++f) {
            p11_real s = P11_HYPOT(ws->qX[f].re, ws->qX[f].im);   /* np.abs complexo */
            ws->qS[f] = s;
            /* P_mean: soma sequencial em t (como o numpy reduz o eixo não contíguo) */
            if (t == 0) ws->P_mean[f] = s * s;
            else ws->P_mean[f] += s * s;
            if (dump_S) dump_S[(size_t)f * T + t] = s;
        }
        /* soma harmônica: h por fora, j por dentro (mesma ordem de acumulação do Python) */
        for (j = 0; j < J; ++j) ws->sal_q[j] = 0;
        for (h = 1; h <= P11_N_HARM; ++h) {
            for (j = 0; j < J; ++j) {
                int i0;
                p11_real a;
                if (ws->usar_tabela) {
                    i0 = ws->h_i0[(h - 1) * J + j];
                    a = ws->h_a[(h - 1) * J + j];
                } else {
                    double pos = ((double)h * ws->grade[j]) / df;
                    double fi = floor(pos);
                    if (!(pos < (double)(F - 1))) continue;
                    i0 = (int)fi;
                    a = (p11_real)(pos - fi);
                }
                ws->sal_q[j] += (1 - a) * ws->qS[i0] + a * ws->qS[i0 + 1];
            }
        }
        mx = ws->sal_q[0];
        for (j = 1; j < J; ++j) if (ws->sal_q[j] > mx) mx = ws->sal_q[j];
        ws->e_q[t] = mx;
        for (j = 0; j < J; ++j) {
            ws->e_norm[(size_t)t * J + j] = ws->sal_q[j] / (mx + (p11_real)1e-12);
            if (dump_sal) dump_sal[(size_t)t * J + j] = ws->sal_q[j];
        }
    }
    for (f = 0; f < F; ++f) ws->P_mean[f] = ws->P_mean[f] / (p11_real)T;
    return 0;
}

p11_real p11_e_ref(const p11_ws *ws)
{
    p11_real m = ws->e_q[0];
    int t;
    for (t = 1; t < ws->T; ++t) if (ws->e_q[t] > m) m = ws->e_q[t];
    return m;
}

/* emissao_baixo sem pYIN: score = 1.0*(0.55*0*0 + 0.45*e_norm) = 0.45*e_norm (exato) */
static p11_real custo_j(p11_real e) { return -P11_LOG((p11_real)0.45 * e + (p11_real)1e-3); }

static p11_real custo_sil(p11_real eq, p11_real e_ref)
{
    p11_real rel = eq / (e_ref + (p11_real)1e-12);
    p11_real sil;
    if (rel < 0) rel = 0;
    if (rel > 1) rel = 1;
    sil = (p11_real)1.0 * ((p11_real)1.0 - rel);       /* (1 - vprob)*(1 - clip), vprob = 0 */
    return -P11_LOG(sil + (p11_real)1e-3);
}

void p11_custo_linha(const p11_ws *ws, p11_real e_ref, int t, p11_real *linha)
{
    int j;
    for (j = 0; j < P11_J; ++j) linha[j] = custo_j(ws->e_norm[(size_t)t * P11_J + j]);
    linha[P11_J] = custo_sil(ws->e_q[t], e_ref);
}

/* custo de passo: w_step * |o| * passo / 100, na mesma ordem de operações do numpy */
static p11_real c_off(int o)
{
    double ao = (double)(o < 0 ? -o : o);
    return (p11_real)(((P11_VIT_W_STEP * ao) * P11_PASSO_CENTS) / 100.0);
}

int p11_viterbi(p11_ws *ws, p11_real *min_delta, p11_real *score)
{
    const int T = ws->T, J = P11_J, S = J + 1;
    const p11_real w_oct = (p11_real)P11_VIT_W_OCT, w_sil = (p11_real)P11_VIT_W_SIL;
    p11_real coff[2 * P11_MAX_SALTO + 1];
    p11_real *prev = ws->delta_a, *novo = ws->delta_b;
    p11_real e_ref = p11_e_ref(ws);
    int t, j, o, s;
    const int octs[2] = { P11_K_OCT, -P11_K_OCT };

    for (o = -P11_MAX_SALTO; o <= P11_MAX_SALTO; ++o) coff[o + P11_MAX_SALTO] = c_off(o);

    /* delta = custo[0] */
    for (j = 0; j < J; ++j) prev[j] = custo_j(ws->e_norm[j]);
    prev[J] = custo_sil(ws->e_q[0], e_ref);
    for (s = 0; s < S; ++s) ws->psi[s] = 0;

    for (t = 1; t < T; ++t) {
        const p11_real *en = ws->e_norm + (size_t)t * J;
        int16_t *ps = ws->psi + (size_t)t * S;
        int j_best;
        p11_real cand_sil = prev[J] + w_sil;
        for (j = 0; j < J; ++j) {
            p11_real best = (p11_real)INFINITY;
            int arg = 0, q;
            /* offs = -50..50, em ordem, atualização estrita */
            for (o = -P11_MAX_SALTO; o <= P11_MAX_SALTO; ++o) {
                int src = j - o;
                if (src >= 0 && src < J) {
                    p11_real c = prev[src] + coff[o + P11_MAX_SALTO];
                    if (c < best) { best = c; arg = src; }
                }
            }
            /* +k_oct, depois -k_oct */
            for (q = 0; q < 2; ++q) {
                int src = j - octs[q];
                if (src >= 0 && src < J) {
                    p11_real c = prev[src] + w_oct;
                    if (c < best) { best = c; arg = src; }
                }
            }
            if (cand_sil < best) { best = cand_sil; arg = J; }
            novo[j] = best + custo_j(en[j]);
            ps[j] = (int16_t)arg;
        }
        /* SIL: j_best = argmin(prev[:J]) (primeiro) */
        j_best = 0;
        for (j = 1; j < J; ++j) if (prev[j] < prev[j_best]) j_best = j;
        if (prev[J] <= prev[j_best] + w_sil) {
            novo[J] = prev[J] + custo_sil(ws->e_q[t], e_ref);
            ps[J] = (int16_t)J;
        } else {
            novo[J] = prev[j_best] + w_sil + custo_sil(ws->e_q[t], e_ref);
            ps[J] = (int16_t)j_best;
        }
        { p11_real *tmp = prev; prev = novo; novo = tmp; }
    }
    /* final: argmin(delta) (primeiro); backtrack */
    {
        int sb = 0;
        for (s = 1; s < S; ++s) if (prev[s] < prev[sb]) sb = s;
        if (min_delta) *min_delta = prev[sb];
        ws->path[T - 1] = sb;
        for (t = T - 1; t > 0; --t) ws->path[t - 1] = ws->psi[(size_t)t * S + ws->path[t]];
    }
    /* score do caminho (contrato §4), na mesma ordem de soma do delta */
    if (score) {
        p11_real sc;
        int a0 = ws->path[0];
        sc = a0 == J ? custo_sil(ws->e_q[0], e_ref) : custo_j(ws->e_norm[a0]);
        for (t = 1; t < T; ++t) {
            int a = ws->path[t - 1], b = ws->path[t];
            p11_real em = b == J ? custo_sil(ws->e_q[t], e_ref)
                                 : custo_j(ws->e_norm[(size_t)t * J + b]);
            if (a == J && b == J) {
                sc = sc + em;
            } else if (a == J || b == J) {
                sc = (sc + w_sil) + em;
            } else {
                int d = b - a;
                p11_real tr;
                if (d >= -P11_MAX_SALTO && d <= P11_MAX_SALTO) tr = coff[d + P11_MAX_SALTO];
                else if (d == P11_K_OCT || d == -P11_K_OCT) tr = w_oct;
                else tr = (p11_real)INFINITY;   /* não deveria acontecer */
                sc = (sc + tr) + em;
            }
        }
        *score = sc;
    }
    for (t = 0; t < T; ++t) if (ws->path[t] == J) ws->path[t] = -1;
    return 0;
}

void p11_f_rot(const p11_ws *ws, p11_real *f_rot)
{
    int t;
    for (t = 0; t < ws->T; ++t)
        f_rot[t] = ws->path[t] >= 0 ? (p11_real)ws->grade[ws->path[t]] : (p11_real)NAN;
}

/* feature 3: mean(where(vivos, e_norm[t, j_t], 0)). No Python j_t = searchsorted(grade,
 * f_rot[t]) ('left'); como f_rot[t] = grade[path[t]] e a grade é estritamente crescente,
 * searchsorted devolve exatamente path[t]: usamos o índice direto (ver c_port.md). */
void p11_saliencia_caminho(p11_ws *ws)
{
    int t;
    for (t = 0; t < ws->T; ++t)
        ws->rbuf[t] = ws->path[t] >= 0 ? ws->e_norm[(size_t)t * P11_J + ws->path[t]] : (p11_real)0;
    ws->sal_caminho = p11_soma_pares(ws->rbuf, ws->T) / (p11_real)ws->T;
}

/* ======================================================================== envelope */
/* mediana como np.median: ordena; par -> (a+b)/2 */
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

static void ordena_double(double *v, int n)
{
    int i, k;
    for (i = 1; i < n; ++i) {
        double x = v[i];
        k = i - 1;
        while (k >= 0 && v[k] > x) { v[k + 1] = v[k]; --k; }
        v[k + 1] = x;
    }
}

/* scipy.signal.sosfilt (forma direta II transposta, zi = 0, seções em cascata),
 * entrada com passo `stride`, saída só de cada `decim`-ésima amostra (y[::decim]). */
/* P11_SOS_DOUBLE=1 (opção de precisão, desligada por padrão no build float): estado e
 * coeficientes do filtro em double, como o scipy (sosfilt promove para float64). Não muda a
 * conta, só a precisão; no ESP32 o double é emulado em software (custo a medir). */
#ifndef P11_SOS_DOUBLE
#define P11_SOS_DOUBLE 0
#endif
#if P11_REAL_EH_DOUBLE || P11_SOS_DOUBLE
typedef double p11_sos_t;
#define P11_SOS_TAB P11_SOS_F64
#else
typedef float p11_sos_t;
#define P11_SOS_TAB P11_SOS_F32
#endif

void p11_sosfilt_dec(const p11_real *x_in, int stride, int n, int decim, p11_real *out)
{
    p11_sos_t z[P11_SOS_N_SECOES][2];
    p11_sos_t sos[P11_SOS_N_SECOES][6];
    int sec, c, k;
    for (sec = 0; sec < P11_SOS_N_SECOES; ++sec) {
        z[sec][0] = z[sec][1] = 0;
        for (c = 0; c < 6; ++c) sos[sec][c] = (p11_sos_t)P11_SOS_TAB[sec][c];
    }
    for (k = 0; k < n; ++k) {
        p11_sos_t x = (p11_sos_t)x_in[(size_t)k * stride];
        for (sec = 0; sec < P11_SOS_N_SECOES; ++sec) {
            p11_sos_t yv = sos[sec][0] * x + z[sec][0];
            z[sec][0] = sos[sec][1] * x - sos[sec][4] * yv + z[sec][1];
            z[sec][1] = sos[sec][2] * x - sos[sec][5] * yv;
            x = yv;
        }
        if (k % decim == 0) out[k / decim] = (p11_real)x;
    }
}

/* scipy.signal.welch(env, fs=1000, nperseg=min(len,1024)) sobre ws->env_dec (n_dec
 * valores, média já subtraída): janela 'hann' periódica, noverlap = nps//2, detrend
 * constante por segmento, scaling='density', unilateral ×2 exceto DC e Nyquist (nfft par),
 * média entre os (n_dec - noverlap)//passo segmentos. Preenche welch_P e welch_f. */
int p11_welch(p11_ws *ws)
{
    const int nd = ws->n_dec, nps = ws->nperseg, nw = nps / 2 + 1;
    int k, s;
    /* --- welch(env, fs=1000, nperseg=min(len,1024)): hann periódica, noverlap = nps//2,
     *     detrend constante por segmento, densidade, unilateral ×2 exceto DC/Nyquist --- */
    {
        const int nov = nps / 2, passo = nps - nov;
        const int nseg = (nd - nov) / passo;
        double soma_w2 = 0.0, escala;
        p11_real esc_r;
        int par = (nps % 2 == 0);
        /* get_window('hann', nps) = hann(nps+1, sym) sem o último: 0.5 - 0.5cos(2*pi*k/nps) */
        for (k = 0; k < nps; ++k) {
            double w = 0.5 - 0.5 * cos(2.0 * P11_PI * (double)k / (double)nps);
            ws->w_welch[k] = (p11_real)w;
            soma_w2 += (double)ws->w_welch[k] * (double)ws->w_welch[k];
        }
        escala = 1.0 / ((double)P11_FS_ENV * soma_w2);
        esc_r = (p11_real)escala;
        for (k = 0; k < nw; ++k) ws->welch_P[k] = 0;
        if (!par) return -2;   /* nperseg ímpar não ocorre (n múltiplo de 16 -> nps par) */
        if (p11_rfft_init(&ws->rfft_welch, nps, ws->mem_rfft_welch) != 0) return -3;
        for (s = 0; s < nseg; ++s) {
            const p11_real *x = ws->env_dec + (size_t)s * passo;
            p11_real mseg;
            for (k = 0; k < nps; ++k) ws->seg_buf[k] = x[k];
            mseg = p11_soma_pares(ws->seg_buf, nps) / (p11_real)nps;
            for (k = 0; k < nps / 2; ++k) {
                ws->wz[k].re = (x[2 * k] - mseg) * ws->w_welch[2 * k];
                ws->wz[k].im = (x[2 * k + 1] - mseg) * ws->w_welch[2 * k + 1];
            }
            p11_rfft(&ws->rfft_welch, ws->wz, ws->wZ, ws->wX);
            for (k = 0; k < nw; ++k) {
                p11_real p = (ws->wX[k].re * ws->wX[k].re + ws->wX[k].im * ws->wX[k].im) * esc_r;
                if (k > 0 && k < nw - 1) p *= 2;
                ws->welch_P[k] += p;
            }
        }
        for (k = 0; k < nw; ++k) ws->welch_P[k] /= (p11_real)nseg;
        /* f = rfftfreq(nps, 1/fs) */
        {
            double val = 1.0 / ((double)nps * (1.0 / (double)P11_FS_ENV));
            for (k = 0; k < nw; ++k) ws->welch_f[k] = (double)k * val;
        }
    }
    return 0;
}

int p11_envelope_mod(p11_ws *ws, p11_real *dump_env_abs, p11_real *dump_env_dec,
                     p11_real *dump_welch_f, p11_real *dump_welch_P, p11_real out[2])
{
    const int n = ws->n, nd = ws->n_dec, nps = ws->nperseg, nw = nps / 2 + 1;
    int k, i;
    p11_real fator;
    /* --- hilbert (scipy): FFT de n pontos, h = 1, 2..., 1, 0...; |ifft| --- */
    if (p11_fft_init(&ws->fft_hil, n, ws->mem_fft_hil) != 0) return -1;
    for (k = 0; k < n; ++k) { ws->hx[k].re = ws->y[k]; ws->hx[k].im = 0; }
    p11_fft(&ws->fft_hil, ws->hx, ws->hX);
    if (n % 2 == 0) {
        for (k = 1; k < n / 2; ++k) { ws->hX[k].re *= 2; ws->hX[k].im *= 2; }
        for (k = n / 2 + 1; k < n; ++k) { ws->hX[k].re = 0; ws->hX[k].im = 0; }
    } else {
        for (k = 1; k < (n + 1) / 2; ++k) { ws->hX[k].re *= 2; ws->hX[k].im *= 2; }
        for (k = (n + 1) / 2; k < n; ++k) { ws->hX[k].re = 0; ws->hX[k].im = 0; }
    }
    p11_fft_inv(&ws->fft_hil, ws->hX, ws->hx);
    fator = (p11_real)(1.0 / (double)n);
    /* |.| guardado em hX[k].re (hX não é mais usado) */
    for (k = 0; k < n; ++k) {
        p11_real e = P11_HYPOT(ws->hx[k].re * fator, ws->hx[k].im * fator);
        ws->hX[k].re = e;
        if (dump_env_abs) dump_env_abs[k] = e;
    }
    /* --- sosfilt (forma direta II transposta, zi = 0) e dizimação [::16] --- */
    p11_sosfilt_dec(&ws->hX[0].re, 2, n, P11_DECIM, ws->env_dec);
    if (dump_env_dec) for (i = 0; i < nd; ++i) dump_env_dec[i] = ws->env_dec[i];
    /* env - env.mean() */
    {
        p11_real m = p11_soma_pares(ws->env_dec, nd) / (p11_real)nd;
        for (i = 0; i < nd; ++i) ws->env_dec[i] -= m;
    }
    {
        int r = p11_welch(ws);
        if (r != 0) return r;
    }
    if (dump_welch_f) for (k = 0; k < nw; ++k) dump_welch_f[k] = (p11_real)ws->welch_f[k];
    if (dump_welch_P) for (k = 0; k < nw; ++k) dump_welch_P[k] = ws->welch_P[k];
    /* --- pico em [2, 100] Hz e proeminência --- */
    {
        int nm = 0, ib = -1;
        p11_real pb = 0, med;
        for (k = 0; k < nw; ++k) {
            if (ws->welch_f[k] >= 2.0 && ws->welch_f[k] <= 100.0) {
                if (ib < 0 || ws->welch_P[k] > pb) { pb = ws->welch_P[k]; ib = k; }
                ws->med_buf[nm++] = ws->welch_P[k];
            }
        }
        if (nm == 0) { out[0] = 0; out[1] = 0; return 0; }
        ordena_real(ws->med_buf, nm);
        med = (nm % 2) ? ws->med_buf[nm / 2]
                       : (ws->med_buf[nm / 2 - 1] + ws->med_buf[nm / 2]) / (p11_real)2;
        out[0] = (p11_real)ws->welch_f[ib];
        out[1] = (p11_real)10 * P11_LOG10(pb / (med + (p11_real)1e-20) + (p11_real)1e-12);
    }
    return 0;
}

/* ======================================================================== features */
/* np.percentile linear: idx = (n-1)*q; g = idx - floor; lerp do numpy */
static double percentil_linear(const double *v, int n, double q)
{
    double vi = (double)(n - 1) * q;
    double lo = floor(vi);
    int i0 = (int)lo, i1 = i0 + 1;
    double g = vi - lo, a, b, d;
    if (vi >= (double)(n - 1)) { i0 = n - 1; i1 = n - 1; }
    a = v[i0];
    b = v[i1];
    d = b - a;
    return g >= 0.5 ? b - d * (1.0 - g) : a + d * g;
}

void p11_features(const p11_ws *ws, const p11_real env_mod[2], p11_real feat[P11_N_FEAT])
{
    static const int grupos[4][2] = { { 1, 4 }, { 5, 12 }, { 13, 24 }, { 25, 48 } };
    const int T = ws->T, F = P11_F;
    const double df = df_stft();
    double *fr = ws->fbuf;
    p11_real *tmp = ws->qS;          /* F reais de rascunho (região da união, livre aqui) */
    p11_real e_ord[P11_N_HARM];
    int nv = 0, t, k, g;
    double f_med = 0.0, iqr = 0.0;
    p11_real tot, Psum, cent, plan;

    /* 0, 1: mediana e IQR dos f_rot válidos (em double, a partir da grade) */
    for (t = 0; t < T; ++t) if (ws->path[t] >= 0) fr[nv++] = ws->grade[ws->path[t]];
    if (nv > 0) {
        ordena_double(fr, nv);
        f_med = (nv % 2) ? fr[nv / 2] : (fr[nv / 2 - 1] + fr[nv / 2]) / 2.0;
        iqr = percentil_linear(fr, nv, 0.75) - percentil_linear(fr, nv, 0.25);
    }
    feat[0] = (p11_real)f_med;
    feat[1] = (p11_real)iqr;
    feat[2] = (p11_real)(1.0 - (double)nv / (double)T);
    feat[3] = ws->sal_caminho;

    /* banda f in [10, 4000] com f = k*df (double, como rfftfreq) */
    {
        int nb = 0;
        for (k = 0; k < F; ++k) {
            double f = (double)k * df;
            if (f >= 10.0 && f <= 4000.0) tmp[nb++] = ws->P_mean[k];
        }
        tot = p11_soma_pares(tmp, nb) + (p11_real)1e-20;
        /* 12: planura = exp(mean(log(P+1e-20))) / (mean(P)+1e-20) */
        {
            p11_real mP = p11_soma_pares(tmp, nb) / (p11_real)nb;
            p11_real ml;
            for (k = 0; k < nb; ++k) tmp[k] = P11_LOG(tmp[k] + (p11_real)1e-20);
            ml = p11_soma_pares(tmp, nb) / (p11_real)nb;
            plan = P11_EXP(ml) / (mP + (p11_real)1e-20);
        }
    }
    /* 4..8: energia por ordem harmônica */
    for (k = 0; k < P11_N_HARM; ++k) e_ord[k] = 0;
    if (f_med > 0) {
        for (k = 1; k <= P11_N_HARM; ++k) {
            double pos_d = rint(((double)k * f_med) / df);   /* np.round: meio-para-par */
            int pos = (int)pos_d;
            if (pos < F - 1)
                e_ord[k - 1] = ws->P_mean[pos - 1] + ws->P_mean[pos] + ws->P_mean[pos + 1];
        }
    }
    feat[4] = p11_soma_pares(e_ord, P11_N_HARM) / tot;
    for (g = 0; g < 4; ++g) {
        int a = grupos[g][0], b = grupos[g][1];
        p11_real sg = p11_soma_pares(e_ord + (a - 1), b - a + 1);
        feat[5 + g] = (p11_real)10 * P11_LOG10(sg / tot + (p11_real)1e-10);
    }
    feat[9] = env_mod[0];
    feat[10] = env_mod[1];
    /* 11: centróide sobre todos os bins */
    for (k = 0; k < F; ++k) tmp[k] = (p11_real)((double)k * df) * ws->P_mean[k];
    cent = p11_soma_pares(tmp, F);
    Psum = p11_soma_pares(ws->P_mean, F);
    feat[11] = cent / (Psum + (p11_real)1e-20);
    feat[12] = plan;
}
