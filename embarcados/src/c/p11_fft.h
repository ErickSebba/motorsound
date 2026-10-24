/* p11_fft.h -- interface de FFT do núcleo p11.
 *
 * Implementação de referência própria (p11_fft.c): FFT complexa de raiz mista
 * (borboletas de raiz 2, 3, 4, 5 e genérica), decimação no tempo recursiva, fora do lugar,
 * no estilo do KISS FFT (algoritmo reescrito aqui, não copiado). Serve qualquer N cujo maior
 * fator primo seja <= P11_FFT_MAX_PRIMO. Os N usados pela pipeline: 4096 (meia FFT real de
 * 8192 da STFT), 512/500 (meia FFT real de 1024/1000 do Welch), 16000 e 48000 (Hilbert).
 *
 * Nada aqui aloca memória: o chamador fornece o buffer das tabelas (p11_fft_bytes) e os
 * buffers de dados.
 *
 * Alvo ESP32 (ESP_PLATFORM definido pelo ESP-IDF): ver o bloco no fim de p11_fft.c. Ele
 * desvia N potência de 2 para o ESP-DSP (dsps_fft2r_fc32) e mantém o código C portável para
 * os demais N (o ESP-DSP só tem raiz 2/4). ESSE BLOCO NÃO FOI COMPILADO: não há ESP-IDF
 * nem ESP32 nesta máquina (embarcados/reports/BLOQUEIOS.md, B1 e B2).
 */
#ifndef P11_FFT_H
#define P11_FFT_H

#include "p11_tipos.h"

#define P11_FFT_MAX_FATORES 32
#define P11_FFT_MAX_PRIMO   64

typedef struct {
    int n;
    int fatores[2 * P11_FFT_MAX_FATORES];   /* pares (p, m), como no KISS FFT */
    p11_cpx *tw;                            /* n fatores de giro exp(-2*pi*i*k/n) */
} p11_fft_plano;

/* bytes de tabela para uma FFT complexa de comprimento n */
size_t p11_fft_bytes(int n);
/* devolve 0 se ok; -1 se n inválido ou fator primo > P11_FFT_MAX_PRIMO */
int p11_fft_init(p11_fft_plano *pl, int n, void *mem);
/* X = DFT(x) (sinal -), fora do lugar: in != out. Sem normalização. */
void p11_fft(const p11_fft_plano *pl, const p11_cpx *in, p11_cpx *out);
/* x = IDFT(X) SEM o fator 1/n (sinal +), fora do lugar. */
void p11_fft_inv(const p11_fft_plano *pl, const p11_cpx *in, p11_cpx *out);

/* FFT real de comprimento n par via FFT complexa de n/2 + separação par/ímpar. */
typedef struct {
    int n;
    p11_fft_plano metade;
    p11_cpx *tw_r;                          /* n/2 fatores exp(-2*pi*i*k/n) */
} p11_rfft_plano;

size_t p11_rfft_bytes(int n);
int p11_rfft_init(p11_rfft_plano *pl, int n, void *mem);
/* z: o sinal real x[0..n-1] já empacotado como z[k] = x[2k] + i*x[2k+1] (n/2 complexos).
 * tmp: n/2 complexos de trabalho. out: n/2+1 bins (rfft do numpy). z, tmp e out distintos. */
void p11_rfft(const p11_rfft_plano *pl, const p11_cpx *z, p11_cpx *tmp, p11_cpx *out);

#endif /* P11_FFT_H */
