/* p11_decisao.c -- classificador por cosseno e HMM de estados de p11_decifrae_harmsum.
 *
 * Referência (somente leitura), lida linha a linha em 23/10/2026:
 *   motorsound/classificador.py::CossenoPrototipos.emissao, predict_proba
 *   pipelines/_decifrae.py::prever (espalha nas colunas das classes; aqui todas presentes)
 *   motorsound/hmm.py::_log_emissoes, decode, filtro_online, _merge, absorb_short,
 *                      rotulos_para_segmentos, segmentos_para_rotulos
 * Diferença deliberada em relação ao porte do colaborador (decifrae_core.c): não há o
 * limiar no denominador do cosseno que ele acrescentou; o Python não tem.
 */
#include "p11.h"
#include "p11_soma.h"

#include <math.h>
#include <string.h>

/* ======================================================================== cosseno */
void p11_cosseno(const p11_modelo *mod, const p11_real x[P11_N_FEAT],
                 p11_real z[P11_N_FEAT], p11_real *cosv, p11_real *probs)
{
    p11_real q[P11_N_FEAT], nrm, tmax, mx, soma;
    int i, c;
    /* z = (x - mu)/sd */
    for (i = 0; i < P11_N_FEAT; ++i) z[i] = (x[i] - mod->mu[i]) / mod->sd[i];
    /* np.linalg.norm(axis=1) = sqrt(add.reduce(z*z)) -> soma em pares (13 >= 8) */
    for (i = 0; i < P11_N_FEAT; ++i) q[i] = z[i] * z[i];
    nrm = P11_SQRT(p11_soma_pares(q, P11_N_FEAT));
    for (i = 0; i < P11_N_FEAT; ++i) q[i] = z[i] / (nrm + (p11_real)1e-9);
    /* cos = q @ M.T (produto interno sequencial; o numpy usa BLAS, ordem não garantida) */
    for (c = 0; c < mod->C; ++c) {
        p11_real s = 0;
        for (i = 0; i < P11_N_FEAT; ++i) s += q[i] * mod->M[c][i];
        cosv[c] = s;
    }
    /* softmax com temperatura: e = cos/max(temp,1e-6); exp(e - max); / soma */
    tmax = mod->temp > (p11_real)1e-6 ? mod->temp : (p11_real)1e-6;
    for (c = 0; c < mod->C; ++c) probs[c] = cosv[c] / tmax;
    mx = probs[0];
    for (c = 1; c < mod->C; ++c) if (probs[c] > mx) mx = probs[c];
    for (c = 0; c < mod->C; ++c) probs[c] = P11_EXP(probs[c] - mx);
    soma = p11_soma_pares(probs, mod->C);
    for (c = 0; c < mod->C; ++c) probs[c] = probs[c] / soma;
}

/* ======================================================================== HMM */
void p11_hmm_log_em(const p11_real *probs, int T, int C, p11_real temp, p11_real n_logprob,
                    p11_real *log_em)
{
    int t, c, S = C + 1;
    for (t = 0; t < T; ++t) {
        const p11_real *p = probs + (size_t)t * C;
        p11_real *l = log_em + (size_t)t * S;
        if (temp != (p11_real)1.0) {
            /* q = clip(p,1e-12,1)^(1/max(temp,1e-6)); p = q/q.sum() */
            p11_real q[P11_MAX_CLASSES], e, s;
            e = (p11_real)1.0 / (temp > (p11_real)1e-6 ? temp : (p11_real)1e-6);
            for (c = 0; c < C; ++c) {
                p11_real v = p[c];
                if (v < (p11_real)1e-12) v = (p11_real)1e-12;
                if (v > (p11_real)1.0) v = (p11_real)1.0;
#if P11_REAL_EH_DOUBLE
                q[c] = pow(v, e);
#else
                q[c] = powf(v, e);
#endif
            }
            s = p11_soma_pares(q, C);
            for (c = 0; c < C; ++c) l[c] = P11_LOG(q[c] / s + (p11_real)1e-12);
        } else {
            for (c = 0; c < C; ++c) l[c] = P11_LOG(p[c] + (p11_real)1e-12);
        }
        l[C] = n_logprob;
    }
}

void p11_hmm_decode(const p11_real *log_em, int T, int S, p11_real change_penalty,
                    uint8_t *psi, p11_real *delta, int32_t *path)
{
    p11_real *prev = delta, *novo = delta + S;
    int t, i, j;
    for (j = 0; j < S; ++j) { prev[j] = log_em[j]; psi[j] = 0; }
    for (t = 1; t < T; ++t) {
        const p11_real *l = log_em + (size_t)t * S;
        for (j = 0; j < S; ++j) {
            /* cand[i] = delta[i] + log_T[i,j]; argmax em i, primeiro índice */
            int ib = 0;
            p11_real vb = prev[0] + (0 == j ? (p11_real)0.0 : -change_penalty);
            for (i = 1; i < S; ++i) {
                p11_real v = prev[i] + (i == j ? (p11_real)0.0 : -change_penalty);
                if (v > vb) { vb = v; ib = i; }
            }
            psi[(size_t)t * S + j] = (uint8_t)ib;
            novo[j] = vb + l[j];
        }
        { p11_real *tmp = prev; prev = novo; novo = tmp; }
    }
    {
        int sb = 0;
        for (j = 1; j < S; ++j) if (prev[j] > prev[sb]) sb = j;
        path[T - 1] = sb;
        for (t = T - 2; t >= 0; --t) path[t] = psi[(size_t)(t + 1) * S + path[t + 1]];
    }
}

/* np.logaddexp (npy_logaddexp) */
static p11_real logaddexp(p11_real x, p11_real y)
{
    if (x == y) {
        return x + (p11_real)0.693147180559945309417232121458176568;   /* LOGE2 */
    } else {
        p11_real d = x - y;
        if (d > 0) return x + P11_LOG1P(P11_EXP(-d));
        else if (d <= 0) return y + P11_LOG1P(P11_EXP(d));
        return d;   /* NaN */
    }
}

/* np.logaddexp.reduce: dobra sequencial da esquerda */
static p11_real lae_reduce(const p11_real *v, int n)
{
    p11_real r = v[0];
    int i;
    for (i = 1; i < n; ++i) r = logaddexp(r, v[i]);
    return r;
}

static int argmax_primeiro(const p11_real *v, int n)
{
    int i, b = 0;
    for (i = 1; i < n; ++i) if (v[i] > v[b]) b = i;
    return b;
}

void p11_hmm_online_init(p11_hmm_online *h, int S, p11_real change_penalty)
{
    memset(h, 0, sizeof(*h));
    h->S = S;
    h->t = 0;
    h->cp = change_penalty;
}

int p11_hmm_online_passo(p11_hmm_online *h, const p11_real *l)
{
    int i, j, S = h->S;
    if (h->t == 0) {
        p11_real z = lae_reduce(l, S);
        for (j = 0; j < S; ++j) h->alpha[j] = l[j] - z;
    } else {
        p11_real a[P11_MAX_CLASSES + 1], col[P11_MAX_CLASSES + 1], z;
        for (j = 0; j < S; ++j) {
            for (i = 0; i < S; ++i) col[i] = h->alpha[i] + (i == j ? (p11_real)0.0 : -h->cp);
            a[j] = lae_reduce(col, S) + l[j];
        }
        z = lae_reduce(a, S);
        for (j = 0; j < S; ++j) h->alpha[j] = a[j] - z;
    }
    h->t++;
    return argmax_primeiro(h->alpha, S);
}

/* ======================================================================== absorb_short */
/* segmentos em seg_buf: [ini, fim, rótulo] por linha, em double como os floats do Python */
static int merge(double *s, int n)
{
    int i, m = 0;
    for (i = 0; i < n; ++i) {
        double a = s[3 * i], b = s[3 * i + 1], r = s[3 * i + 2];
        if (b <= a) continue;
        if (m > 0 && s[3 * (m - 1) + 2] == r) {
            s[3 * (m - 1) + 1] = b;
        } else {
            s[3 * m] = a; s[3 * m + 1] = b; s[3 * m + 2] = r;
            ++m;
        }
    }
    return m;
}

void p11_absorb_short(const int32_t *rot, int n, double passo_s, double min_dur,
                      int32_t estado_n, double *s, int32_t *out)
{
    int m, i;
    /* rotulos_para_segmentos */
    for (i = 0; i < n; ++i) {
        s[3 * i] = (double)i * passo_s;
        s[3 * i + 1] = (double)(i + 1) * passo_s;
        s[3 * i + 2] = (double)rot[i];
    }
    m = merge(s, n);
    /* absorb_short */
    if (min_dur > 0) {
        while (m > 1) {
            int idx = 0, k;
            double dmin = s[1] - s[0];
            for (k = 1; k < m; ++k) {
                double d = s[3 * k + 1] - s[3 * k];
                if (d < dmin) { dmin = d; idx = k; }       /* min(): primeiro em empate */
            }
            if (dmin >= min_dur) break;
            {
                int tem_e = idx > 0, tem_d = idx < m - 1, doador;
                if (tem_e && tem_d) {
                    double de = s[3 * (idx - 1) + 1] - s[3 * (idx - 1)];
                    double dd = s[3 * (idx + 1) + 1] - s[3 * (idx + 1)];
                    doador = de >= dd ? idx - 1 : idx + 1;
                } else {
                    doador = tem_e ? idx - 1 : idx + 1;
                }
                s[3 * idx + 2] = s[3 * doador + 2];
            }
            m = merge(s, m);
        }
    }
    /* segmentos_para_rotulos: int(round(x/passo)) do Python = meio-para-par */
    for (i = 0; i < n; ++i) out[i] = estado_n;
    for (i = 0; i < m; ++i) {
        int i0 = (int)rint(s[3 * i] / passo_s);
        int i1 = (int)rint(s[3 * i + 1] / passo_s);
        int a = i0 > 0 ? i0 : 0, b = i1 < n ? i1 : n, k;
        for (k = a; k < b; ++k) out[k] = (int32_t)s[3 * i + 2];
    }
}
