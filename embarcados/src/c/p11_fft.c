/* p11_fft.c -- FFT de raiz mista de referência (ver p11_fft.h).
 *
 * Estrutura: a mesma decomposição do KISS FFT (M. Borgerding, licença BSD), reescrita do
 * zero para o p11: fatoração preferindo 4, depois 2, 3, 5, ...; recursão em decimação no
 * tempo; borboletas específicas de raiz 2, 3, 4, 5 e uma genérica O(p^2) para os demais
 * primos. Os fatores de giro são calculados em double e só então convertidos para p11_real.
 * A inversa usa os conjugados dos mesmos fatores (sem tabela extra).
 */
#include "p11_fft.h"

#include <math.h>
#include <string.h>

static const double P11_PI = 3.14159265358979323846264338327950288;

/* ---------------- fatoração ---------------- */
static int fatorar(int n, int *fat)
{
    int p = 4, k = 0;
    double raiz = floor(sqrt((double)n));
    if (n < 1) return -1;
    if (n == 1) { fat[0] = 1; fat[1] = 1; return 1; }
    do {
        while (n % p) {
            switch (p) {
            case 4: p = 2; break;
            case 2: p = 3; break;
            default: p += 2; break;
            }
            if (p > raiz) p = n;          /* não há mais fatores <= sqrt(n): n é primo */
        }
        n /= p;
        if (k >= P11_FFT_MAX_FATORES) return -1;
        fat[2 * k] = p;
        fat[2 * k + 1] = n;
        ++k;
        if (p > P11_FFT_MAX_PRIMO) return -1;
    } while (n > 1);
    return k;
}

/* ---------------- tabelas de giro (acesso em palavras inteiras) ---------------- */
#if P11_REAL_EH_DOUBLE
p11_cpx p11_tw_le(const p11_cpx *p) { return *p; }
#else
p11_cpx p11_tw_le(const p11_cpx *p)
{
    const volatile uint32_t *u = (const volatile uint32_t *)(const void *)p;
    uint32_t a = u[0], b = u[1];
    p11_cpx w;
    memcpy(&w.re, &a, sizeof a);
    memcpy(&w.im, &b, sizeof b);
    return w;
}
#endif

void p11_tw_grava(p11_cpx *p, double fase)
{
    p11_cpx w;
    w.re = (p11_real)cos(fase);
    w.im = (p11_real)sin(fase);
#if P11_REAL_EH_DOUBLE
    *p = w;
#else
    {
        volatile uint32_t *u = (volatile uint32_t *)(void *)p;
        uint32_t a, b;
        memcpy(&a, &w.re, sizeof a);
        memcpy(&b, &w.im, sizeof b);
        u[0] = a;
        u[1] = b;
    }
#endif
}

size_t p11_fft_bytes(int n)
{
    return (size_t)(n > 0 ? n : 0) * sizeof(p11_cpx);
}

int p11_fft_init(p11_fft_plano *pl, int n, void *mem)
{
    int k;
    if (n < 1 || mem == NULL) return -1;
    memset(pl->fatores, 0, sizeof(pl->fatores));
    if (fatorar(n, pl->fatores) < 0) return -1;
    pl->n = n;
    pl->tw = (p11_cpx *)mem;
    for (k = 0; k < n; ++k) p11_tw_grava(&pl->tw[k], -2.0 * P11_PI * (double)k / (double)n);
    return 0;
}

/* ---------------- aritmética complexa ---------------- */
static p11_cpx giro(const p11_fft_plano *pl, size_t idx, int inv)
{
    p11_cpx w = p11_tw_le(&pl->tw[idx]);
    if (inv) w.im = -w.im;
    return w;
}

static p11_cpx cmul(p11_cpx a, p11_cpx b)
{
    p11_cpx r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

/* ---------------- borboletas ---------------- */
static void bfly2(p11_cpx *F, size_t fs, const p11_fft_plano *pl, int m, int inv)
{
    p11_cpx *F2 = F + m;
    int k;
    for (k = 0; k < m; ++k) {
        p11_cpx t = cmul(F2[k], giro(pl, (size_t)k * fs, inv));
        F2[k].re = F[k].re - t.re;
        F2[k].im = F[k].im - t.im;
        F[k].re += t.re;
        F[k].im += t.im;
    }
}

static void bfly4(p11_cpx *F, size_t fs, const p11_fft_plano *pl, int m, int inv)
{
    int k;
    for (k = 0; k < m; ++k) {
        p11_cpx a0 = F[k];
        p11_cpx a1 = cmul(F[k + m], giro(pl, (size_t)k * fs, inv));
        p11_cpx a2 = cmul(F[k + 2 * m], giro(pl, (size_t)k * fs * 2, inv));
        p11_cpx a3 = cmul(F[k + 3 * m], giro(pl, (size_t)k * fs * 3, inv));
        p11_cpx s02, d02, s13, d13;
        s02.re = a0.re + a2.re; s02.im = a0.im + a2.im;
        d02.re = a0.re - a2.re; d02.im = a0.im - a2.im;
        s13.re = a1.re + a3.re; s13.im = a1.im + a3.im;
        d13.re = a1.re - a3.re; d13.im = a1.im - a3.im;
        F[k].re = s02.re + s13.re;          F[k].im = s02.im + s13.im;
        F[k + 2 * m].re = s02.re - s13.re;  F[k + 2 * m].im = s02.im - s13.im;
        if (!inv) {   /* -i * d13 */
            F[k + m].re = d02.re + d13.im;      F[k + m].im = d02.im - d13.re;
            F[k + 3 * m].re = d02.re - d13.im;  F[k + 3 * m].im = d02.im + d13.re;
        } else {      /* +i * d13 */
            F[k + m].re = d02.re - d13.im;      F[k + m].im = d02.im + d13.re;
            F[k + 3 * m].re = d02.re + d13.im;  F[k + 3 * m].im = d02.im - d13.re;
        }
    }
}

static void bfly3(p11_cpx *F, size_t fs, const p11_fft_plano *pl, int m, int inv)
{
    int k;
    /* exp(-+2*pi*i/3): só a parte imaginária é usada; a real é exatamente -1/2 */
    p11_real s3 = giro(pl, fs * (size_t)m, inv).im;
    for (k = 0; k < m; ++k) {
        p11_cpx a0 = F[k];
        p11_cpx a1 = cmul(F[k + m], giro(pl, (size_t)k * fs, inv));
        p11_cpx a2 = cmul(F[k + 2 * m], giro(pl, (size_t)k * fs * 2, inv));
        p11_cpx s, d;
        s.re = a1.re + a2.re; s.im = a1.im + a2.im;
        d.re = a1.re - a2.re; d.im = a1.im - a2.im;
        F[k].re = a0.re + s.re;
        F[k].im = a0.im + s.im;
        {
            p11_real mr = a0.re - (p11_real)0.5 * s.re;
            p11_real mi = a0.im - (p11_real)0.5 * s.im;
            /* (s3 * i) * d = -s3*d.im + i*s3*d.re */
            p11_real rr = -s3 * d.im, ri = s3 * d.re;
            F[k + m].re = mr + rr;      F[k + m].im = mi + ri;
            F[k + 2 * m].re = mr - rr;  F[k + 2 * m].im = mi - ri;
        }
    }
}

static void bfly5(p11_cpx *F, size_t fs, const p11_fft_plano *pl, int m, int inv)
{
    int k;
    p11_cpx ya = giro(pl, fs * (size_t)m, inv);
    p11_cpx yb = giro(pl, fs * 2 * (size_t)m, inv);
    for (k = 0; k < m; ++k) {
        p11_cpx a0 = F[k];
        p11_cpx a1 = cmul(F[k + m], giro(pl, (size_t)k * fs, inv));
        p11_cpx a2 = cmul(F[k + 2 * m], giro(pl, (size_t)k * fs * 2, inv));
        p11_cpx a3 = cmul(F[k + 3 * m], giro(pl, (size_t)k * fs * 3, inv));
        p11_cpx a4 = cmul(F[k + 4 * m], giro(pl, (size_t)k * fs * 4, inv));
        p11_cpx s14, d14, s23, d23, c1, c2;
        s14.re = a1.re + a4.re; s14.im = a1.im + a4.im;
        d14.re = a1.re - a4.re; d14.im = a1.im - a4.im;
        s23.re = a2.re + a3.re; s23.im = a2.im + a3.im;
        d23.re = a2.re - a3.re; d23.im = a2.im - a3.im;
        F[k].re = a0.re + s14.re + s23.re;
        F[k].im = a0.im + s14.im + s23.im;
        c1.re = a0.re + s14.re * ya.re + s23.re * yb.re;
        c1.im = a0.im + s14.im * ya.re + s23.im * yb.re;
        /* r1 = i*(ya.im*d14 + yb.im*d23) */
        {
            p11_real tr = ya.im * d14.re + yb.im * d23.re;
            p11_real ti = ya.im * d14.im + yb.im * d23.im;
            F[k + m].re = c1.re - ti;      F[k + m].im = c1.im + tr;
            F[k + 4 * m].re = c1.re + ti;  F[k + 4 * m].im = c1.im - tr;
        }
        c2.re = a0.re + s14.re * yb.re + s23.re * ya.re;
        c2.im = a0.im + s14.im * yb.re + s23.im * ya.re;
        /* r2 = i*(yb.im*d14 - ya.im*d23) */
        {
            p11_real tr = yb.im * d14.re - ya.im * d23.re;
            p11_real ti = yb.im * d14.im - ya.im * d23.im;
            F[k + 2 * m].re = c2.re - ti;  F[k + 2 * m].im = c2.im + tr;
            F[k + 3 * m].re = c2.re + ti;  F[k + 3 * m].im = c2.im - tr;
        }
    }
}

static void bfly_gen(p11_cpx *F, size_t fs, const p11_fft_plano *pl, int m, int p, int inv)
{
    p11_cpx esc[P11_FFT_MAX_PRIMO];
    int u, q1, q;
    size_t n = (size_t)pl->n;
    for (u = 0; u < m; ++u) {
        int k = u;
        for (q1 = 0; q1 < p; ++q1) { esc[q1] = F[k]; k += m; }
        k = u;
        for (q1 = 0; q1 < p; ++q1) {
            size_t ti = 0;
            p11_cpx acc = esc[0];
            for (q = 1; q < p; ++q) {
                p11_cpx t;
                ti += fs * (size_t)k;
                if (ti >= n) ti -= n * (ti / n);
                t = cmul(esc[q], giro(pl, ti, inv));
                acc.re += t.re;
                acc.im += t.im;
            }
            F[k] = acc;
            k += m;
        }
    }
}

static void trabalho(p11_cpx *out, const p11_cpx *f, size_t fs, const int *fat,
                     const p11_fft_plano *pl, int inv)
{
    p11_cpx *ini = out;
    int p = *fat++;
    int m = *fat++;
    const p11_cpx *fim = out + (size_t)p * m;
    if (m == 1) {
        do { *out = *f; f += fs; } while (++out != fim);
    } else {
        do {
            trabalho(out, f, fs * (size_t)p, fat, pl, inv);
            f += fs;
        } while ((out += m) != fim);
    }
    out = ini;
    switch (p) {
    case 2: bfly2(out, fs, pl, m, inv); break;
    case 3: bfly3(out, fs, pl, m, inv); break;
    case 4: bfly4(out, fs, pl, m, inv); break;
    case 5: bfly5(out, fs, pl, m, inv); break;
    case 1: break;
    default: bfly_gen(out, fs, pl, m, p, inv); break;
    }
}

#if defined(ESP_PLATFORM) && defined(P11_USAR_ESP_DSP)
/* ------------------------------------------------------------------------------------
 * STUB ESP-DSP -- NÃO COMPILADO (sem ESP-IDF nesta máquina; BLOQUEIOS.md B2).
 * Para N potência de 2, troca a recursão por dsps_fft2r_fc32 (raiz 2, no lugar, dados
 * intercalados re/im) + dsps_bit_rev_fc32. Exige P11_REAL=float, dsps_fft2r_init_fc32(NULL,
 * N_max) chamado uma vez na inicialização do firmware e buffers alinhados a 16 bytes.
 * A inversa usa o truque conj(FFT(conj(x))). Os N não potência de 2 (Hilbert com 16000 ou
 * 48000 amostras, Welch com 1000) continuam no código C portável acima: o ESP-DSP não tem
 * raiz 3/5. Qualquer diferença numérica entre os dois caminhos é assunto do Agente D.
 * ------------------------------------------------------------------------------------ */
#include "dsps_fft2r.h"
static int eh_pot2(int n) { return n > 0 && (n & (n - 1)) == 0; }
static int via_esp_dsp(const p11_fft_plano *pl, const p11_cpx *in, p11_cpx *out, int inv)
{
    int k;
    if (!eh_pot2(pl->n)) return 0;
    for (k = 0; k < pl->n; ++k) { out[k].re = in[k].re; out[k].im = inv ? -in[k].im : in[k].im; }
    dsps_fft2r_fc32((float *)out, pl->n);
    dsps_bit_rev_fc32((float *)out, pl->n);
    if (inv) for (k = 0; k < pl->n; ++k) out[k].im = -out[k].im;
    return 1;
}
#endif

void p11_fft(const p11_fft_plano *pl, const p11_cpx *in, p11_cpx *out)
{
#if defined(ESP_PLATFORM) && defined(P11_USAR_ESP_DSP)
    if (via_esp_dsp(pl, in, out, 0)) return;
#endif
    trabalho(out, in, 1, pl->fatores, pl, 0);
}

void p11_fft_inv(const p11_fft_plano *pl, const p11_cpx *in, p11_cpx *out)
{
#if defined(ESP_PLATFORM) && defined(P11_USAR_ESP_DSP)
    if (via_esp_dsp(pl, in, out, 1)) return;
#endif
    trabalho(out, in, 1, pl->fatores, pl, 1);
}

/* ---------------- FFT real ---------------- */
size_t p11_rfft_bytes(int n)
{
    return p11_fft_bytes(n / 2) + (size_t)(n / 2) * sizeof(p11_cpx);
}

int p11_rfft_init(p11_rfft_plano *pl, int n, void *mem)
{
    int k, h = n / 2;
    unsigned char *b = (unsigned char *)mem;
    if (n < 2 || (n & 1)) return -1;
    pl->n = n;
    if (p11_fft_init(&pl->metade, h, b) != 0) return -1;
    pl->tw_r = (p11_cpx *)(b + p11_fft_bytes(h));
    for (k = 0; k < h; ++k) p11_tw_grava(&pl->tw_r[k], -2.0 * P11_PI * (double)k / (double)n);
    return 0;
}

void p11_rfft(const p11_rfft_plano *pl, const p11_cpx *z, p11_cpx *Z, p11_cpx *out)
{
    int h = pl->n / 2, k;
    p11_fft(&pl->metade, z, Z);
    out[0].re = Z[0].re + Z[0].im;  out[0].im = 0;
    out[h].re = Z[0].re - Z[0].im;  out[h].im = 0;
    for (k = 1; k < h; ++k) {
        p11_cpx a = Z[k], b = Z[h - k];          /* b será conjugado */
        /* par = (a + conj b)/2 ; impar = (a - conj b)/(2i) */
        p11_real pr = (p11_real)0.5 * (a.re + b.re);
        p11_real pi = (p11_real)0.5 * (a.im - b.im);
        p11_real dr = (p11_real)0.5 * (a.re - b.re);
        p11_real di = (p11_real)0.5 * (a.im + b.im);
        /* impar = (dr + i di)/i = di - i dr */
        p11_cpx im; im.re = di; im.im = -dr;
        im = cmul(im, p11_tw_le(&pl->tw_r[k]));
        out[k].re = pr + im.re;
        out[k].im = pi + im.im;
    }
}
