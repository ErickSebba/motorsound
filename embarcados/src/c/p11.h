/* p11.h -- núcleo C portável de p11_decifrae_harmsum (C99, sem malloc).
 *
 * Referência (somente leitura): motorsound/rotacao.py, pipelines/_decifrae.py,
 * motorsound/classificador.py, motorsound/hmm.py, motorsound/runner.py::_preparar.
 * Contrato de etapas e formatos: embarcados/reports/CONTRATO_INTERFACES.md.
 * Mapeamento função a função e decisões: embarcados/reports/c_port.md.
 *
 * Uso típico (uma janela de N amostras a 16 kHz):
 *   size_t b = p11_ws_bytes(N, 1);  void *mem = <buffer do chamador, b bytes>;
 *   p11_ws ws; p11_ws_init(&ws, mem, b, N, 1);
 *   p11_preparar(audio, N, ws.y);              // etapa y_prep
 *   p11_rastrear_quadros(&ws, NULL, NULL);     // STFT por quadro + soma harmônica + P_mean
 *   p11_viterbi(&ws, &min_delta, &score);      // caminho; f_rot por p11_f_rot
 *   p11_saliencia_caminho(&ws)  ...  p11_envelope_mod(&ws, ...)  p11_features(&ws, ...)
 *   p11_cosseno(&modelo, feat, z, cos, probs);
 * A memória das fases "rastreio" e "envelope" é compartilhada (união) dentro do arena:
 * e_norm/psi deixam de valer depois de p11_envelope_mod. Por isso p11_features usa o
 * valor de saliência guardado em ws->sal_caminho (calculado antes do envelope).
 */
#ifndef P11_H
#define P11_H

#include "p11_tipos.h"
#include "p11_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- workspace ---------------- */
typedef struct {
    int n;               /* amostras da janela */
    int n_pad;           /* max(n, N_FFT): comprimento após o zero-pad de stft_mag */
    int T;               /* quadros da STFT */
    int usar_tabela;     /* 1: tabela (i0, a) por (h, j); 0: calcula na hora (mesma conta) */

    /* --- persistente --- */
    p11_real *y;         /* (n) y_prep */
    double   *grade;     /* (J) em double: f_j = fmin*2^(j*passo/1200) */
    p11_real *janela;    /* (N_FFT) np.hanning(8192).astype(float32) */
    uint16_t *h_i0;      /* (48*J) ou NULL */
    p11_real *h_a;       /* (48*J) ou NULL */
    p11_real *P_mean;    /* (F) (S**2).mean(axis=1) */
    p11_real *e_q;       /* (T) max_j sal */
    int32_t  *path;      /* (T) caminho do Viterbi, -1 = SIL */
    double   *fbuf;      /* (T) f_rot válidos para mediana/IQR */
    p11_real *rbuf;      /* (T) rascunho da feature 3 */
    void     *mem_rfft_stft; /* tabelas da rfft de 8192 */
    p11_real sal_caminho;/* feature 3, calculada por p11_saliencia_caminho */
    p11_rfft_plano rfft_stft;

    /* --- fase A (rastreio) --- */
    p11_cpx  *qz;        /* (N_FFT/2) quadro empacotado */
    p11_cpx  *qZ;        /* (N_FFT/2) trabalho da FFT */
    p11_cpx  *qX;        /* (F) rfft do quadro */
    p11_real *qS;        /* (F) |X| do quadro */
    p11_real *sal_q;     /* (J) soma harmônica do quadro */
    p11_real *e_norm;    /* (T*J) */
    int16_t  *psi;       /* (T*(J+1)) */
    p11_real *delta_a;   /* (J+1) */
    p11_real *delta_b;   /* (J+1) */

    /* --- fase B (envelope; mesma memória da fase A) --- */
    void     *mem_fft_hil;   /* tabelas da FFT de n pontos */
    void     *mem_rfft_welch;
    p11_real *seg_buf;   /* (nperseg) */
    p11_fft_plano fft_hil;
    p11_cpx  *hx;        /* (n) */
    p11_cpx  *hX;        /* (n) */
    p11_real *env_dec;   /* (n_dec) */
    int n_dec;
    p11_rfft_plano rfft_welch;
    int nperseg;
    p11_real *w_welch;   /* (nperseg) hann periódica */
    p11_cpx  *wz, *wZ, *wX;
    p11_real *welch_P;   /* (nperseg/2+1) */
    double   *welch_f;   /* (nperseg/2+1) */
    p11_real *med_buf;   /* (nperseg/2+1) para a mediana */
} p11_ws;

/* número de quadros da STFT para n amostras (1 + (max(n,8192)-8192)//1024) */
int p11_n_quadros(int n);
/* ESTIMATIVA ANALÍTICA dos bytes de workspace para n amostras (inclui alinhamento) */
size_t p11_ws_bytes(int n, int usar_tabela);
/* partes da estimativa (bytes): persistente, fase A, fase B */
void p11_ws_partes(int n, int usar_tabela, size_t *persist, size_t *faseA, size_t *faseB);
int p11_ws_init(p11_ws *ws, void *mem, size_t bytes, int n, int usar_tabela);

/* ---------------- etapas ---------------- */
/* runner._preparar: conta em float32 como no numpy (ver c_port.md); y_out em p11_real */
void p11_preparar(const float *y_in, int n, p11_real *y_out);

/* STFT por quadro + soma harmônica + e_norm + e_q + P_mean. Sem guardar S.
 * dump_S (F*T, ordem (F,T)) e dump_sal (T*J) são opcionais (NULL no MCU). */
int p11_rastrear_quadros(p11_ws *ws, p11_real *dump_S, p11_real *dump_sal);

/* custo de emissão de um quadro (J+1 valores; último = SIL), calculado na hora */
void p11_custo_linha(const p11_ws *ws, p11_real e_ref, int t, p11_real *linha);
/* max_t e_q (referência do custo SIL) */
p11_real p11_e_ref(const p11_ws *ws);

/* Viterbi de rotação; preenche ws->path. min_delta = min(delta) final; score = custo do
 * caminho recalculado pela regra do contrato §4 (mesma ordem de soma). */
int p11_viterbi(p11_ws *ws, p11_real *min_delta, p11_real *score);

/* f_rot[t] = grade[path[t]] ou NaN */
void p11_f_rot(const p11_ws *ws, p11_real *f_rot);

/* feature 3 (precisa de e_norm: chamar antes de p11_envelope_mod) */
void p11_saliencia_caminho(p11_ws *ws);

/* _envelope_mod. Dumps opcionais: env_abs (n), env_dec (n/16), welch_f/welch_P (nperseg/2+1).
 * Devolve em out[0] a frequência do pico e em out[1] a proeminência (dB). */
int p11_envelope_mod(p11_ws *ws, p11_real *dump_env_abs, p11_real *dump_env_dec,
                     p11_real *dump_welch_f, p11_real *dump_welch_P, p11_real out[2]);
int p11_n_welch(int n);   /* nperseg/2+1 */
/* peças do envelope, expostas para teste unitário */
void p11_sosfilt_dec(const p11_real *x, int stride, int n, int decim, p11_real *out);
int p11_welch(p11_ws *ws);   /* sobre ws->env_dec (n_dec), preenche welch_P/welch_f */

/* as 13 features (usa path, grade, P_mean, sal_caminho e env_mod) */
void p11_features(const p11_ws *ws, const p11_real env_mod[2], p11_real feat[P11_N_FEAT]);

/* ---------------- classificador por cosseno ---------------- */
typedef struct {
    int C;
    p11_real temp;
    p11_real mu[P11_N_FEAT];
    p11_real sd[P11_N_FEAT];
    p11_real M[P11_MAX_CLASSES][P11_N_FEAT];
} p11_modelo;

void p11_cosseno(const p11_modelo *mod, const p11_real x[P11_N_FEAT],
                 p11_real z[P11_N_FEAT], p11_real *cosv, p11_real *probs);

/* ---------------- HMM de estados (hmm.py) ---------------- */
/* log-emissões (T, C+1), estado N = coluna C */
void p11_hmm_log_em(const p11_real *probs, int T, int C, p11_real temp, p11_real n_logprob,
                    p11_real *log_em);
/* Viterbi offline. psi: T*S bytes (S = C+1 <= 255); delta: 2*S. path: T */
void p11_hmm_decode(const p11_real *log_em, int T, int S, p11_real change_penalty,
                    uint8_t *psi, p11_real *delta, int32_t *path);

/* filtro forward causal, passo a passo (o que roda no MCU) */
typedef struct {
    int S;
    int t;
    p11_real cp;
    p11_real alpha[P11_MAX_CLASSES + 1];
} p11_hmm_online;
void p11_hmm_online_init(p11_hmm_online *h, int S, p11_real change_penalty);
/* consome uma linha de log_em (S valores) e devolve o estado decidido */
int p11_hmm_online_passo(p11_hmm_online *h, const p11_real *log_em_linha);

/* absorb_short(rotulos_para_segmentos(r, passo_s), min_dur) -> segmentos_para_rotulos.
 * Tempos em double, como o float do Python. seg_buf: 3*n doubles (início, fim, rótulo). */
void p11_absorb_short(const int32_t *rot, int n, double passo_s, double min_dur,
                      int32_t estado_n, double *seg_buf, int32_t *out);

#ifdef __cplusplus
}
#endif
#endif /* P11_H */
