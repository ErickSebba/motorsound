#ifndef DECIFRAE_CORE_H
#define DECIFRAE_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECIFRAE_N_FEATURES 13
#define DECIFRAE_MAX_CLASSES 16

/* Parameters copied from PreDecifrae/PostDecifrae in the supplied Python. */
typedef struct {
    int sample_rate;             /* 16000 Hz */
    float fmin_hz;               /* 15 Hz */
    float fmax_hz;               /* 60 Hz */
    float step_cents;            /* 2 cents */
    int n_fft;                   /* 8192 */
    int hop;                     /* 1024 */
    int n_harmonics;             /* 48 */
    int max_jump;                /* 50 */
    int use_viterbi;             /* 1 */
} decifrae_config_t;

/*
 * Output from the frequency tracker and spectral front end.
 * Arrays are supplied by the caller, so this module does not allocate the
 * large STFT/Viterbi work buffers on the stack.
 *
 * f_rot_hz:       n_frames values; use NAN for an invalid/SIL frame.
 * e_norm:         row-major [n_frames][n_grid] salience/emission matrix.
 * grid_hz:        n_grid candidate fundamental frequencies.
 * mean_power:     mean power spectrum, length n_bins, matching Python P=(S**2).mean(axis=1).
 * df_hz:          frequency spacing of mean_power, normally sample_rate/n_fft.
 * modulation_*:   output of the envelope-modulation procedure in the Python source.
 *
 * IMPORTANT: rastrear(), stft_mag(), and _envelope_mod() are not included in
 * the supplied snippet. They must be ported separately and produce arrays with
 * the same conventions as the Python implementation for numerical parity.
 */
typedef struct {
    const float *f_rot_hz;
    const float *e_norm;
    const float *grid_hz;
    size_t n_frames;
    size_t n_grid;
    const float *mean_power;
    size_t n_bins;
    float df_hz;
    float modulation_peak_hz;
    float modulation_peak_prominence_db;
} decifrae_frontend_result_t;

/* Exact feature-vector layout from the supplied Python function extrair(). */
typedef struct {
    float value[DECIFRAE_N_FEATURES];
} decifrae_features_t;

/* Fill the defaults corresponding to PreDecifrae(usar_pyin=False, usar_viterbi=True). */
void decifrae_default_config(decifrae_config_t *cfg);

/*
 * Convert frontend outputs into the 13 features used by extrair().
 * Returns 0 on success, negative value on invalid input.
 */
int decifrae_build_features(const decifrae_frontend_result_t *src,
                            decifrae_features_t *dst,
                            float *rotation_scratch,
                            size_t scratch_len);

/*
 * Cosine-prototype classifier inference.
 *
 * prototypes is row-major [n_classes][13]. Each row MUST use the exact feature
 * scaling/normalization and prototype construction exported by the Python
 * CossenoPrototipos model. This function does not fit prototypes on-device.
 * probabilities is length n_classes. temperature=0.1 matches PostDecifrae.
 * Returns 0 on success, negative value on invalid input.
 */
int decifrae_cosine_predict(const float features[DECIFRAE_N_FEATURES],
                            const float *prototypes,
                            size_t n_classes,
                            float temperature,
                            float *probabilities);

/*
 * Generic HMM/Viterbi decoder for frame- or window-level class probabilities.
 * Inputs transition_log and initial_log must be natural-log probabilities
 * exported from the actual Python sequence model; do not invent these values.
 * emissions is row-major [n_steps][n_states], in probabilities (not logs).
 * path is length n_steps; state IDs are 0..n_states-1.
 * Scratch arrays are supplied by caller to avoid large stack allocations.
 */
int decifrae_hmm_viterbi(const float *emissions,
                         size_t n_steps,
                         size_t n_states,
                         const float *initial_log,
                         const float *transition_log,
                         int *path,
                         float *score_scratch,
                         float *next_score_scratch,
                         int *backpointer_scratch);

#ifdef __cplusplus
}
#endif
#endif
