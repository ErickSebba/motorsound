/* p11_soma.h -- soma em pares com a mesma blocagem do numpy (pairwise_sum de
 * numpy/_core/src/umath/loops_utils.h.src: n < 8 sequencial; n <= 128 oito acumuladores;
 * acima disso divide em n2 = n/2 arredondado para múltiplo de 8). É o que np.sum/np.mean
 * fazem sobre um vetor contíguo. Usar a mesma ordem de soma não muda o algoritmo; só
 * aproxima o arredondamento do C do da referência (no build float, a soma é em float).
 * A recursão tem profundidade log2(n/128) (<= 9 para n = 48000).
 */
#ifndef P11_SOMA_H
#define P11_SOMA_H

#include "p11_tipos.h"

static p11_real p11_soma_pares(const p11_real *a, int n)
{
    if (n < 8) {
        p11_real r = 0;
        int i;
        for (i = 0; i < n; ++i) r += a[i];
        return r;
    } else if (n <= 128) {
        p11_real r[8], res;
        int i, j;
        for (j = 0; j < 8; ++j) r[j] = a[j];
        for (i = 8; i < n - (n % 8); i += 8)
            for (j = 0; j < 8; ++j) r[j] += a[i + j];
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    } else {
        int n2 = n / 2;
        n2 -= n2 % 8;
        return p11_soma_pares(a, n2) + p11_soma_pares(a + n2, n - n2);
    }
}

/* soma em pares de a[i]*a[i] (float32), sem buffer: igual a np.sum(y**2) com y float32 */
static float p11_soma_quad_f32(const float *a, int n)
{
    if (n < 8) {
        float r = 0.0f;
        int i;
        for (i = 0; i < n; ++i) r += a[i] * a[i];
        return r;
    } else if (n <= 128) {
        float r[8], res;
        int i, j;
        for (j = 0; j < 8; ++j) r[j] = a[j] * a[j];
        for (i = 8; i < n - (n % 8); i += 8)
            for (j = 0; j < 8; ++j) r[j] += a[i + j] * a[i + j];
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i] * a[i];
        return res;
    } else {
        int n2 = n / 2;
        n2 -= n2 % 8;
        return p11_soma_quad_f32(a, n2) + p11_soma_quad_f32(a + n2, n - n2);
    }
}

#endif /* P11_SOMA_H */
