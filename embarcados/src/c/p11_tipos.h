/* p11_tipos.h -- tipo real e constantes de projeto de p11_decifrae_harmsum.
 *
 * Origem dos valores: pipelines/_decifrae.py::PreDecifrae/PostDecifrae e
 * motorsound/rotacao.py (VIT_*), lidos em 23/10/2026. Não há nada aqui que mude o
 * algoritmo: só nomes para os números do Python.
 *
 * P11_REAL: escolhido na compilação (-DP11_REAL=float ou -DP11_REAL=double).
 *   float  = o que roda no MCU (ESP32 tem FPU só de precisão simples);
 *   double = só no PC, para separar erro de algoritmo de erro de precisão.
 */
#ifndef P11_TIPOS_H
#define P11_TIPOS_H

#include <stddef.h>
#include <stdint.h>

#ifndef P11_REAL
#define P11_REAL float
#endif
typedef P11_REAL p11_real;

/* P11_REAL_EH_DOUBLE: 1 no build double (decidido pelo CMake); usado para escolher
 * funções da libm (sqrtf × sqrt etc.). */
#ifndef P11_REAL_EH_DOUBLE
#define P11_REAL_EH_DOUBLE 0
#endif

#if P11_REAL_EH_DOUBLE
#define P11_SQRT sqrt
#define P11_LOG log
#define P11_LOG10 log10
#define P11_EXP exp
#define P11_COS cos
#define P11_HYPOT hypot
#define P11_LOG1P log1p
#define P11_FABS fabs
#else
#define P11_SQRT sqrtf
#define P11_LOG logf
#define P11_LOG10 log10f
#define P11_EXP expf
#define P11_COS cosf
#define P11_HYPOT hypotf
#define P11_LOG1P log1pf
#define P11_FABS fabsf
#endif

typedef struct { p11_real re, im; } p11_cpx;

/* ---- parâmetros (PreDecifrae / rotacao.py) ---- */
#define P11_SR          16000
#define P11_N_FFT       8192
#define P11_HOP         1024
#define P11_F           (P11_N_FFT / 2 + 1)      /* 4097 bins */
#define P11_FMIN        15.0
#define P11_FMAX        60.0
#define P11_PASSO_CENTS 2.0
#define P11_J           1201                      /* floor(1200*log2(60/15)/2)+1 */
#define P11_N_HARM      48
#define P11_MAX_SALTO   50
#define P11_K_OCT       600                       /* int(round(1200/2)) */
#define P11_VIT_W_STEP  0.55
#define P11_VIT_W_OCT   7.0
#define P11_VIT_W_SIL   3.2
#define P11_RMS_ALVO    0.1
#define P11_N_FEAT      13
#define P11_DECIM       16                        /* sr // 1000 */
#define P11_FS_ENV      1000
#define P11_MAX_CLASSES 16

#endif /* P11_TIPOS_H */
