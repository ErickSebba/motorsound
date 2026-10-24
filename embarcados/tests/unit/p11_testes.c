/* p11_testes.c -- executável auxiliar dos testes unitários (tests/unit/test_p11.py).
 * Cada subcomando lê um binário float64, roda UMA peça do núcleo e grava float64/int32.
 * Não é prova de equivalência (isso é do Agente D): só confere cada peça isolada.
 *
 *   fft     <in.f64 (2n intercalado)> <n> <out.f64>       DFT direta
 *   ifft    <in.f64> <n> <out.f64>                         IDFT sem 1/n
 *   rfft    <in.f64 (n reais)> <n> <out.f64 (2*(n/2+1))>
 *   sosfilt <in.f64> <n> <decim> <out.f64>
 *   welch   <in.f64 (L reais)> <L> <out_f.f64> <out_P.f64>
 *   viterbi <e_norm.f64 (T*J)> <e_q.f64 (T)> <T> <out_path.i32> <out_score.f64 (2)>
 */
#include "p11.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double *ler(const char *p, long n)
{
    FILE *f = fopen(p, "rb");
    double *v = (double *)malloc(sizeof(double) * (size_t)n);
    if (!f || fread(v, sizeof(double), (size_t)n, f) != (size_t)n) { fprintf(stderr, "erro lendo %s\n", p); exit(2); }
    fclose(f);
    return v;
}

static void grava(const char *p, const void *v, size_t tam, size_t n)
{
    FILE *f = fopen(p, "wb");
    fwrite(v, tam, n, f);
    fclose(f);
}

static void grava_reais(const char *p, const p11_real *v, size_t n)
{
    double *d = (double *)malloc(sizeof(double) * n);
    size_t i;
    for (i = 0; i < n; ++i) d[i] = (double)v[i];
    grava(p, d, sizeof(double), n);
    free(d);
}

int main(int argc, char **argv)
{
    if (argc < 2) return 1;
    if ((!strcmp(argv[1], "fft") || !strcmp(argv[1], "ifft")) && argc == 5) {
        int n = atoi(argv[3]), k;
        double *x = ler(argv[2], 2L * n);
        p11_cpx *a = (p11_cpx *)malloc(sizeof(p11_cpx) * n), *b = (p11_cpx *)malloc(sizeof(p11_cpx) * n);
        void *mem = malloc(p11_fft_bytes(n));
        p11_fft_plano pl;
        if (p11_fft_init(&pl, n, mem) != 0) { fprintf(stderr, "n inválido\n"); return 3; }
        for (k = 0; k < n; ++k) { a[k].re = (p11_real)x[2 * k]; a[k].im = (p11_real)x[2 * k + 1]; }
        if (argv[1][0] == 'f') p11_fft(&pl, a, b); else p11_fft_inv(&pl, a, b);
        grava_reais(argv[4], &b[0].re, 2 * (size_t)n);
        return 0;
    }
    if (!strcmp(argv[1], "rfft") && argc == 5) {
        int n = atoi(argv[3]), k, h = n / 2;
        double *x = ler(argv[2], n);
        p11_cpx *z = (p11_cpx *)malloc(sizeof(p11_cpx) * h), *Z = (p11_cpx *)malloc(sizeof(p11_cpx) * h);
        p11_cpx *X = (p11_cpx *)malloc(sizeof(p11_cpx) * (h + 1));
        void *mem = malloc(p11_rfft_bytes(n));
        p11_rfft_plano pl;
        if (p11_rfft_init(&pl, n, mem) != 0) return 3;
        for (k = 0; k < h; ++k) { z[k].re = (p11_real)x[2 * k]; z[k].im = (p11_real)x[2 * k + 1]; }
        p11_rfft(&pl, z, Z, X);
        grava_reais(argv[4], &X[0].re, 2 * (size_t)(h + 1));
        return 0;
    }
    if (!strcmp(argv[1], "sosfilt") && argc == 6) {
        int n = atoi(argv[3]), d = atoi(argv[4]), k, nd = (n + d - 1) / d;
        double *x = ler(argv[2], n);
        p11_real *xr = (p11_real *)malloc(sizeof(p11_real) * n), *y = (p11_real *)malloc(sizeof(p11_real) * nd);
        for (k = 0; k < n; ++k) xr[k] = (p11_real)x[k];
        p11_sosfilt_dec(xr, 1, n, d, y);
        grava_reais(argv[5], y, (size_t)nd);
        return 0;
    }
    if (!strcmp(argv[1], "welch") && argc == 6) {
        int L = atoi(argv[3]), k, nw;
        double *x = ler(argv[2], L);
        p11_ws ws;
        size_t b = p11_ws_bytes(16 * L, 0);     /* n = 16 L -> n_dec = L */
        void *mem = malloc(b);
        p11_real *f;
        if (p11_ws_init(&ws, mem, b, 16 * L, 0) != 0) return 3;
        for (k = 0; k < L; ++k) ws.env_dec[k] = (p11_real)x[k];
        if (p11_welch(&ws) != 0) return 4;
        nw = ws.nperseg / 2 + 1;
        f = (p11_real *)malloc(sizeof(p11_real) * nw);
        for (k = 0; k < nw; ++k) f[k] = (p11_real)ws.welch_f[k];
        {   /* welch_f em double puro */
            grava(argv[4], ws.welch_f, sizeof(double), (size_t)nw);
        }
        grava_reais(argv[5], ws.welch_P, (size_t)nw);
        free(f);
        return 0;
    }
    if (!strcmp(argv[1], "viterbi") && argc == 7) {
        int T = atoi(argv[4]), k;
        double *en = ler(argv[2], (long)T * P11_J), *eq = ler(argv[3], T);
        p11_ws ws;
        int n = P11_N_FFT + (T - 1) * P11_HOP;   /* n que dá exatamente T quadros */
        size_t b = p11_ws_bytes(n, 0);
        void *mem = malloc(b);
        p11_real md, sc;
        double out[2];
        if (p11_ws_init(&ws, mem, b, n, 0) != 0 || ws.T != T) return 3;
        for (k = 0; k < T * P11_J; ++k) ws.e_norm[k] = (p11_real)en[k];
        for (k = 0; k < T; ++k) ws.e_q[k] = (p11_real)eq[k];
        p11_viterbi(&ws, &md, &sc);
        grava(argv[5], ws.path, sizeof(int32_t), (size_t)T);
        out[0] = (double)md; out[1] = (double)sc;
        grava(argv[6], out, sizeof(double), 2);
        return 0;
    }
    fprintf(stderr, "subcomando inválido\n");
    return 1;
}
