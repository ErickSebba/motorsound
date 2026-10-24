/* p11_host.c -- CLI de PC do núcleo p11 (só host; usa malloc e stdio, o núcleo não).
 *
 *   p11_host caso <dir_caso> <modelo.bin> <dir_saida>
 *   p11_host seq  <probs.f64> <T> <C> <change_penalty> <temp> <n_logprob> <min_dur> <passo_s> <dir_saida>
 *   p11_host bench <dir_caso> <modelo.bin> <n_rep>
 *   p11_host ws <n_amostras>            (ESTIMATIVA ANALÍTICA do workspace, em bytes)
 *
 * Formatos: embarcados/reports/CONTRATO_INTERFACES.md §2-§3. Reais saem em .f32 no build
 * float e .f64 no build double; discretos em .i32. bench mede no PC (QueryPerformanceCounter
 * no Windows, clock_gettime noutros), nunca no ESP32.
 */
#include "p11.h"
#include "p11_lowmem.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#include <time.h>
#define MKDIR(p) mkdir(p, 0777)
#endif

#if P11_REAL_EH_DOUBLE
#define EXT_REAL ".f64"
#define DTYPE_REAL "float64"
#else
#define EXT_REAL ".f32"
#define DTYPE_REAL "float32"
#endif

/* ------------------------------------------------------------------ utilidades */
static void *ler_tudo(const char *path, long *bytes)
{
    FILE *f = fopen(path, "rb");
    void *b;
    long n;
    if (!f) { fprintf(stderr, "erro: não abriu %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "erro lendo %s\n", path); exit(2); }
    fclose(f);
    *bytes = n;
    return b;
}

static void junta(char *dst, size_t cap, const char *dir, const char *nome)
{
    snprintf(dst, cap, "%s/%s", dir, nome);
}

typedef struct {
    FILE *json;
    const char *dir;
    int primeiro;
} saida_t;

/* mkdir -p */
static void cria_dir(const char *dir)
{
    char b[1024];
    size_t i, n = strlen(dir);
    if (n >= sizeof b) return;
    memcpy(b, dir, n + 1);
    for (i = 1; i < n; ++i)
        if (b[i] == '/' || b[i] == 92) {   /* 92 = barra invertida */
            char c = b[i];
            if (b[i - 1] == ':') continue;
            b[i] = 0; MKDIR(b); b[i] = c;
        }
    MKDIR(b);
}

static void abre_saida(saida_t *s, const char *dir)
{
    char p[1024];
    cria_dir(dir);
    junta(p, sizeof p, dir, "etapas.json");
    s->json = fopen(p, "w");
    if (!s->json) { fprintf(stderr, "erro: não criou %s\n", p); exit(2); }
    s->dir = dir;
    s->primeiro = 1;
    fprintf(s->json, "{\n");
}

static void fecha_saida(saida_t *s)
{
    fprintf(s->json, "\n}\n");
    fclose(s->json);
}

static void registra(saida_t *s, const char *etapa, const char *arq, const char *dtype,
                     int nd, const int *shape)
{
    int i;
    fprintf(s->json, "%s  \"%s\": {\"arquivo\": \"%s\", \"dtype\": \"%s\", \"shape\": [",
            s->primeiro ? "" : ",\n", etapa, arq, dtype);
    for (i = 0; i < nd; ++i) fprintf(s->json, "%s%d", i ? ", " : "", shape[i]);
    fprintf(s->json, "]}");
    s->primeiro = 0;
}

static void grava_real(saida_t *s, const char *etapa, const p11_real *v, int nd, const int *shape)
{
    char arq[256], p[1024];
    size_t n = 1;
    int i;
    FILE *f;
    for (i = 0; i < nd; ++i) n *= (size_t)shape[i];
    snprintf(arq, sizeof arq, "%s%s", etapa, EXT_REAL);
    junta(p, sizeof p, s->dir, arq);
    f = fopen(p, "wb");
    fwrite(v, sizeof(p11_real), n, f);
    fclose(f);
    registra(s, etapa, arq, DTYPE_REAL, nd, shape);
}

static void grava_i32(saida_t *s, const char *etapa, const int32_t *v, int nd, const int *shape)
{
    char arq[256], p[1024];
    size_t n = 1;
    int i;
    FILE *f;
    for (i = 0; i < nd; ++i) n *= (size_t)shape[i];
    snprintf(arq, sizeof arq, "%s.i32", etapa);
    junta(p, sizeof p, s->dir, arq);
    f = fopen(p, "wb");
    fwrite(v, sizeof(int32_t), n, f);
    fclose(f);
    registra(s, etapa, arq, "int32", nd, shape);
}

/* modelo §3: int32 C, int32 13, f64 temp, f64 mu[13], f64 sd[13], f64 M[C][13] */
static void ler_modelo(const char *path, p11_modelo *m)
{
    long nb;
    unsigned char *b = (unsigned char *)ler_tudo(path, &nb);
    int32_t C, d;
    const double *v;
    int i, c;
    memcpy(&C, b, 4);
    memcpy(&d, b + 4, 4);
    if (d != P11_N_FEAT || C < 1 || C > P11_MAX_CLASSES ||
        nb != 8 + 8 * (1 + 2 * P11_N_FEAT + (long)C * P11_N_FEAT)) {
        fprintf(stderr, "erro: modelo %s inválido (C=%d, d=%d, %ld bytes)\n", path, C, d, nb);
        exit(2);
    }
    v = (const double *)(b + 8);   /* deslocamento 8: alinhado a 8 no malloc */
    m->C = C;
    m->temp = (p11_real)v[0];
    for (i = 0; i < P11_N_FEAT; ++i) m->mu[i] = (p11_real)v[1 + i];
    for (i = 0; i < P11_N_FEAT; ++i) m->sd[i] = (p11_real)v[1 + P11_N_FEAT + i];
    for (c = 0; c < C; ++c)
        for (i = 0; i < P11_N_FEAT; ++i)
            m->M[c][i] = (p11_real)v[1 + 2 * P11_N_FEAT + c * P11_N_FEAT + i];
    free(b);
}

static float *ler_audio(const char *dir_caso, int *n)
{
    char p[1024];
    long nb;
    float *a;
    junta(p, sizeof p, dir_caso, "audio.f32");
    a = (float *)ler_tudo(p, &nb);
    *n = (int)(nb / 4);
    return a;
}

/* ------------------------------------------------------------------ relógio */
static double agora_s(void)
{
#ifdef _WIN32
    static LARGE_INTEGER fq;
    LARGE_INTEGER c;
    if (!fq.QuadPart) QueryPerformanceFrequency(&fq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)fq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
#endif
}

/* ------------------------------------------------------------------ caso */
static int cmd_caso(const char *dir_caso, const char *mod_path, const char *dir_out)
{
    int n, T, J = P11_J, F = P11_F, C, nw, nd, t;
    float *audio = ler_audio(dir_caso, &n);
    p11_modelo mod;
    p11_ws ws;
    size_t bytes = p11_ws_bytes(n, 1);
    void *mem = malloc(bytes);
    p11_real *S, *sal, *e_norm, *custo, *f_rot, *env_abs, *env_dec, *wf, *wP, *g;
    p11_real min_delta, score, env_mod[2], feat[P11_N_FEAT], z[P11_N_FEAT];
    p11_real cosv[P11_MAX_CLASSES], probs[P11_MAX_CLASSES];
    saida_t s;
    int shp[2];

    ler_modelo(mod_path, &mod);
    C = mod.C;
    if (p11_ws_init(&ws, mem, bytes, n, 1) != 0) { fprintf(stderr, "erro ws_init\n"); return 3; }
    T = ws.T;
    nd = ws.n_dec;
    nw = p11_n_welch(n);
    S = (p11_real *)malloc(sizeof(p11_real) * (size_t)F * T);
    sal = (p11_real *)malloc(sizeof(p11_real) * (size_t)T * J);
    e_norm = (p11_real *)malloc(sizeof(p11_real) * (size_t)T * J);
    custo = (p11_real *)malloc(sizeof(p11_real) * (size_t)T * (J + 1));
    f_rot = (p11_real *)malloc(sizeof(p11_real) * (size_t)T);
    env_abs = (p11_real *)malloc(sizeof(p11_real) * (size_t)n);
    env_dec = (p11_real *)malloc(sizeof(p11_real) * (size_t)nd);
    wf = (p11_real *)malloc(sizeof(p11_real) * (size_t)nw);
    wP = (p11_real *)malloc(sizeof(p11_real) * (size_t)nw);
    g = (p11_real *)malloc(sizeof(p11_real) * (size_t)J);

    abre_saida(&s, dir_out);
    p11_preparar(audio, n, ws.y);
    shp[0] = n; grava_real(&s, "y_prep", ws.y, 1, shp);

    p11_rastrear_quadros(&ws, S, sal);
    shp[0] = F; shp[1] = T; grava_real(&s, "stft_S", S, 2, shp);
    for (t = 0; t < J; ++t) g[t] = (p11_real)ws.grade[t];
    shp[0] = J; grava_real(&s, "grade", g, 1, shp);
    shp[0] = T; shp[1] = J; grava_real(&s, "sal", sal, 2, shp);
    memcpy(e_norm, ws.e_norm, sizeof(p11_real) * (size_t)T * J);
    grava_real(&s, "e_norm", e_norm, 2, shp);
    {
        p11_real er = p11_e_ref(&ws);
        for (t = 0; t < T; ++t) p11_custo_linha(&ws, er, t, custo + (size_t)t * (J + 1));
    }
    shp[0] = T; shp[1] = J + 1; grava_real(&s, "custo", custo, 2, shp);

    p11_viterbi(&ws, &min_delta, &score);
    shp[0] = T; grava_i32(&s, "vit_path", ws.path, 1, shp);
    shp[0] = 1; grava_real(&s, "vit_score", &score, 1, shp);
    grava_real(&s, "vit_min_delta", &min_delta, 1, shp);
    p11_f_rot(&ws, f_rot);
    shp[0] = T; grava_real(&s, "f_rot", f_rot, 1, shp);
    shp[0] = F; grava_real(&s, "P_mean", ws.P_mean, 1, shp);

    p11_saliencia_caminho(&ws);
    if (p11_envelope_mod(&ws, env_abs, env_dec, wf, wP, env_mod) != 0) {
        fprintf(stderr, "erro envelope\n");
        return 3;
    }
    shp[0] = n; grava_real(&s, "env_abs", env_abs, 1, shp);
    shp[0] = nd; grava_real(&s, "env_dec", env_dec, 1, shp);
    shp[0] = nw; grava_real(&s, "welch_f", wf, 1, shp);
    grava_real(&s, "welch_P", wP, 1, shp);
    shp[0] = 2; grava_real(&s, "env_mod", env_mod, 1, shp);

    p11_features(&ws, env_mod, feat);
    shp[0] = P11_N_FEAT; grava_real(&s, "features", feat, 1, shp);
    p11_cosseno(&mod, feat, z, cosv, probs);
    grava_real(&s, "z", z, 1, shp);
    shp[0] = C; grava_real(&s, "cos", cosv, 1, shp);
    grava_real(&s, "probs", probs, 1, shp);
    fecha_saida(&s);

    printf("ok: n=%d T=%d ws=%zu bytes min_delta=%.17g score=%.17g\n", n, T, bytes,
           (double)min_delta, (double)score);
    free(audio); free(mem); free(S); free(sal); free(e_norm); free(custo); free(f_rot);
    free(env_abs); free(env_dec); free(wf); free(wP); free(g);
    return 0;
}

/* ------------------------------------------------------------------ seq */
/* ------------------------------------------------------------------ lowmem */
/* Variante p11_c_lowmem (p11_lowmem.h) com um pool de blocos que imita a RAM do chip:
 * blocos = "tam:so32,tam:so32,..." (so32 = 1 para memória só de 32 bits, como a IRAM).
 * Grava as etapas que a variante expõe (sem stft_S e sal, que ela nunca guarda inteiros). */
static int cmd_lowmem(const char *dir_caso, const char *mod_path, const char *dir_out,
                      const char *blocos)
{
    int n, T, J = P11_J, C, nw, nd, t;
    float *audio = ler_audio(dir_caso, &n);
    p11_modelo mod;
    lm_pool pool;
    p11_lm lm;
    char buf[512], *tok;
    void *mems[LM_MAX_BLOCOS];
    int nb = 0, rc;
    p11_real *e_norm, *custo, *f_rot, *env_abs, *env_dec, *wf, *wP, *g;
    p11_real min_delta, score, env_mod[2], feat[P11_N_FEAT], z[P11_N_FEAT];
    p11_real cosv[P11_MAX_CLASSES], probs[P11_MAX_CLASSES];
    saida_t s;
    int shp[2];

    ler_modelo(mod_path, &mod);
    C = mod.C;
    lm_pool_vazio(&pool);
    snprintf(buf, sizeof buf, "%s", blocos);
    for (tok = strtok(buf, ","); tok && nb < LM_MAX_BLOCOS; tok = strtok(NULL, ",")) {
        size_t tam = (size_t)strtoul(tok, NULL, 10);
        const char *dp = strchr(tok, ':');
        int so32 = dp ? atoi(dp + 1) : 0;
        mems[nb] = malloc(tam);
        lm_pool_add(&pool, mems[nb], tam, so32);
        ++nb;
    }
    rc = p11_lm_init(&lm, &pool, n);
    if (rc != 0) { fprintf(stderr, "erro lm_init %d (faltou %zu bytes)\n", rc, pool.falhou_bytes); return 3; }
    T = lm.ws.T;
    nd = lm.ws.n_dec;
    nw = p11_n_welch(n);
    e_norm = (p11_real *)malloc(sizeof(p11_real) * (size_t)T * J);
    custo = (p11_real *)malloc(sizeof(p11_real) * (size_t)T * (J + 1));
    f_rot = (p11_real *)malloc(sizeof(p11_real) * (size_t)T);
    env_abs = (p11_real *)malloc(sizeof(p11_real) * (size_t)n);
    env_dec = (p11_real *)malloc(sizeof(p11_real) * (size_t)nd);
    wf = (p11_real *)malloc(sizeof(p11_real) * (size_t)nw);
    wP = (p11_real *)malloc(sizeof(p11_real) * (size_t)nw);
    g = (p11_real *)malloc(sizeof(p11_real) * (size_t)J);

    abre_saida(&s, dir_out);
    /* no firmware (float) é no lugar sobre ws.y; aqui lê do áudio float para valer também no
     * build double, em que ws.y é double. A conta é a mesma. */
    p11_preparar(audio, n, lm.ws.y);
    shp[0] = n; grava_real(&s, "y_prep", lm.ws.y, 1, shp);
    for (t = 0; t < J; ++t) g[t] = (p11_real)lm.ws.grade[t];
    shp[0] = J; grava_real(&s, "grade", g, 1, shp);
    rc = p11_lm_rastrear(&lm, &min_delta, &score, e_norm, custo);
    if (rc != 0) { fprintf(stderr, "erro lm_rastrear %d (faltou %zu bytes)\n", rc, pool.falhou_bytes); return 3; }
    shp[0] = T; shp[1] = J; grava_real(&s, "e_norm", e_norm, 2, shp);
    shp[0] = T; shp[1] = J + 1; grava_real(&s, "custo", custo, 2, shp);
    shp[0] = T; grava_i32(&s, "vit_path", lm.ws.path, 1, shp);
    shp[0] = 1; grava_real(&s, "vit_score", &score, 1, shp);
    grava_real(&s, "vit_min_delta", &min_delta, 1, shp);
    p11_f_rot(&lm.ws, f_rot);
    shp[0] = T; grava_real(&s, "f_rot", f_rot, 1, shp);
    shp[0] = P11_F; grava_real(&s, "P_mean", lm.ws.P_mean, 1, shp);
    rc = p11_lm_envelope(&lm, env_abs, env_dec, wf, wP, env_mod);
    if (rc != 0) { fprintf(stderr, "erro lm_envelope %d (faltou %zu bytes)\n", rc, pool.falhou_bytes); return 3; }
    shp[0] = n; grava_real(&s, "env_abs", env_abs, 1, shp);
    shp[0] = nd; grava_real(&s, "env_dec", env_dec, 1, shp);
    shp[0] = nw; grava_real(&s, "welch_f", wf, 1, shp);
    grava_real(&s, "welch_P", wP, 1, shp);
    shp[0] = 2; grava_real(&s, "env_mod", env_mod, 1, shp);
    p11_lm_features(&lm, env_mod, feat);
    shp[0] = P11_N_FEAT; grava_real(&s, "features", feat, 1, shp);
    p11_cosseno(&mod, feat, z, cosv, probs);
    grava_real(&s, "z", z, 1, shp);
    shp[0] = C; grava_real(&s, "cos", cosv, 1, shp);
    grava_real(&s, "probs", probs, 1, shp);
    fecha_saida(&s);
    printf("{\"categoria\": \"estimativa_no_PC_do_pool\", \"n\": %d, \"T\": %d, \"persist\": %zu, "
           "\"pico_faseA\": %zu, \"pico_faseB\": %zu, \"pico_faseC\": %zu, \"pico_total\": %zu}\n",
           n, T, lm.bytes_persist, lm.bytes_faseA, lm.bytes_faseB, lm.bytes_faseC, pool.pico);
    return 0;
}

static int cmd_seq(char **a)
{
    const char *probs_path = a[0];
    int T = atoi(a[1]), C = atoi(a[2]), S = C + 1, i, d;
    p11_real cp = (p11_real)atof(a[3]), temp = (p11_real)atof(a[4]), nlp = (p11_real)atof(a[5]);
    double min_dur = atof(a[6]), passo = atof(a[7]);
    const char *dir_out = a[8];
    long nb;
    double *pd = (double *)ler_tudo(probs_path, &nb);
    p11_real *pr, *le, *delta, *alpha;
    uint8_t *psi;
    int32_t *vit, *onl, *amx, *abs3;
    double *segb;
    p11_hmm_online h;
    saida_t s;
    int shp[3];
    if (nb != (long)T * C * 8) { fprintf(stderr, "erro: probs com %ld bytes != T*C*8\n", nb); return 2; }
    pr = (p11_real *)malloc(sizeof(p11_real) * (size_t)T * C);
    le = (p11_real *)malloc(sizeof(p11_real) * (size_t)T * S);
    alpha = (p11_real *)malloc(sizeof(p11_real) * (size_t)T * S);
    delta = (p11_real *)malloc(sizeof(p11_real) * 2 * (size_t)S);
    psi = (uint8_t *)malloc((size_t)T * S);
    vit = (int32_t *)malloc(sizeof(int32_t) * (size_t)T);
    onl = (int32_t *)malloc(sizeof(int32_t) * (size_t)T);
    amx = (int32_t *)malloc(sizeof(int32_t) * (size_t)T);
    abs3 = (int32_t *)malloc(sizeof(int32_t) * 3 * (size_t)T);
    segb = (double *)malloc(sizeof(double) * 3 * (size_t)T);
    for (i = 0; i < T * C; ++i) pr[i] = (p11_real)pd[i];

    abre_saida(&s, dir_out);
    p11_hmm_log_em(pr, T, C, temp, nlp, le);
    shp[0] = T; shp[1] = S; grava_real(&s, "hmm_log_em", le, 2, shp);
    p11_hmm_decode(le, T, S, cp, psi, delta, vit);
    shp[0] = T; grava_i32(&s, "hmm_viterbi", vit, 1, shp);
    p11_hmm_online_init(&h, S, cp);
    for (i = 0; i < T; ++i) {
        onl[i] = p11_hmm_online_passo(&h, le + (size_t)i * S);
        memcpy(alpha + (size_t)i * S, h.alpha, sizeof(p11_real) * (size_t)S);
    }
    grava_i32(&s, "hmm_online", onl, 1, shp);
    shp[0] = T; shp[1] = S; grava_real(&s, "hmm_online_alpha", alpha, 2, shp);
    for (i = 0; i < T; ++i) {
        int b = 0;
        for (d = 1; d < C; ++d) if (pr[(size_t)i * C + d] > pr[(size_t)i * C + b]) b = d;
        amx[i] = b;
    }
    shp[0] = T; grava_i32(&s, "argmax", amx, 1, shp);
    /* absorb: eixo 0 = decodificação (argmax, viterbi, online), como no ouro */
    p11_absorb_short(amx, T, passo, min_dur, C, segb, abs3);
    p11_absorb_short(vit, T, passo, min_dur, C, segb, abs3 + T);
    p11_absorb_short(onl, T, passo, min_dur, C, segb, abs3 + 2 * T);
    shp[0] = 3; shp[1] = T; grava_i32(&s, "absorb", abs3, 2, shp);
    fecha_saida(&s);
    printf("ok: T=%d C=%d cp=%g min_dur=%g\n", T, C, (double)cp, min_dur);
    free(pd); free(pr); free(le); free(alpha); free(delta); free(psi); free(vit); free(onl);
    free(amx); free(abs3); free(segb);
    return 0;
}

/* ------------------------------------------------------------------ bench */
static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int cmd_bench(const char *dir_caso, const char *mod_path, int n_rep)
{
    enum { E_INIT, E_PREP, E_RAST, E_VIT, E_SAL, E_ENV, E_FEAT, E_COS, E_TOTAL, N_E };
    static const char *nomes[N_E] = { "ws_init", "preparar", "rastrear_quadros", "viterbi",
                                      "saliencia_caminho", "envelope_mod", "features",
                                      "cosseno", "total_sem_init" };
    int n, r, e;
    float *audio = ler_audio(dir_caso, &n);
    p11_modelo mod;
    size_t bytes = p11_ws_bytes(n, 1);
    void *mem = malloc(bytes);
    double *tm = (double *)malloc(sizeof(double) * (size_t)N_E * n_rep);
    double *col = (double *)malloc(sizeof(double) * (size_t)n_rep);
    p11_ws ws;
    ler_modelo(mod_path, &mod);
    for (r = 0; r < n_rep; ++r) {
        double t0, t1, md, sc;
        p11_real min_delta, score, em[2], feat[P11_N_FEAT], z[P11_N_FEAT];
        p11_real cosv[P11_MAX_CLASSES], probs[P11_MAX_CLASSES];
        t0 = agora_s(); p11_ws_init(&ws, mem, bytes, n, 1); t1 = agora_s(); tm[E_INIT * n_rep + r] = t1 - t0;
        t0 = t1; p11_preparar(audio, n, ws.y); t1 = agora_s(); tm[E_PREP * n_rep + r] = t1 - t0;
        t0 = t1; p11_rastrear_quadros(&ws, NULL, NULL); t1 = agora_s(); tm[E_RAST * n_rep + r] = t1 - t0;
        t0 = t1; p11_viterbi(&ws, &min_delta, &score); t1 = agora_s(); tm[E_VIT * n_rep + r] = t1 - t0;
        t0 = t1; p11_saliencia_caminho(&ws); t1 = agora_s(); tm[E_SAL * n_rep + r] = t1 - t0;
        t0 = t1; p11_envelope_mod(&ws, NULL, NULL, NULL, NULL, em); t1 = agora_s(); tm[E_ENV * n_rep + r] = t1 - t0;
        t0 = t1; p11_features(&ws, em, feat); t1 = agora_s(); tm[E_FEAT * n_rep + r] = t1 - t0;
        t0 = t1; p11_cosseno(&mod, feat, z, cosv, probs); t1 = agora_s(); tm[E_COS * n_rep + r] = t1 - t0;
        md = min_delta; sc = score; (void)md; (void)sc;
        tm[E_TOTAL * n_rep + r] = 0;
        for (e = E_PREP; e <= E_COS; ++e) tm[E_TOTAL * n_rep + r] += tm[e * n_rep + r];
    }
    printf("{\"categoria\": \"medicao_no_PC\", \"real\": \"%s\", \"n_amostras\": %d, "
           "\"T\": %d, \"n_rep\": %d, \"ws_bytes\": %zu, \"etapas_ms\": {",
           DTYPE_REAL, n, ws.T, n_rep, bytes);
    for (e = 0; e < N_E; ++e) {
        double soma = 0;
        for (r = 0; r < n_rep; ++r) { col[r] = tm[e * n_rep + r] * 1e3; soma += col[r]; }
        qsort(col, (size_t)n_rep, sizeof(double), cmp_d);
        printf("%s\"%s\": {\"mediana\": %.4f, \"media\": %.4f, \"min\": %.4f, \"max\": %.4f}",
               e ? ", " : "", nomes[e], col[n_rep / 2], soma / n_rep, col[0], col[n_rep - 1]);
    }
    printf("}}\n");
    free(audio); free(mem); free(tm); free(col);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 5 && strcmp(argv[1], "caso") == 0) return cmd_caso(argv[2], argv[3], argv[4]);
    if (argc >= 11 && strcmp(argv[1], "seq") == 0) return cmd_seq(argv + 2);
    if (argc >= 5 && strcmp(argv[1], "bench") == 0) return cmd_bench(argv[2], argv[3], atoi(argv[4]));
    if (argc >= 5 && strcmp(argv[1], "lowmem") == 0)
        return cmd_lowmem(argv[2], argv[3], argv[4], argc >= 6 ? argv[5]
                          : "110592:0,65536:0,49152:0,32768:0,65536:1");
    if (argc >= 3 && strcmp(argv[1], "ws") == 0) {
        int n = atoi(argv[2]);
        size_t p, a, b;
        p11_ws_partes(n, 1, &p, &a, &b);
        printf("{\"categoria\": \"estimativa_analitica\", \"real\": \"%s\", \"n\": %d, \"T\": %d, "
               "\"total_com_tabela\": %zu, \"total_sem_tabela\": %zu, \"persistente\": %zu, "
               "\"faseA_rastreio\": %zu, \"faseB_envelope\": %zu}\n",
               DTYPE_REAL, n, p11_n_quadros(n), p11_ws_bytes(n, 1), p11_ws_bytes(n, 0), p, a, b);
        return 0;
    }
    fprintf(stderr,
            "uso:\n  p11_host caso <dir_caso> <modelo.bin> <dir_saida>\n"
            "  p11_host seq <probs.f64> <T> <C> <change_penalty> <temp> <n_logprob> <min_dur> <passo_s> <dir_saida>\n"
            "  p11_host bench <dir_caso> <modelo.bin> <n_rep>\n  p11_host ws <n_amostras>\n");
    return 1;
}
