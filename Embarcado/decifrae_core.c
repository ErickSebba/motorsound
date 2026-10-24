#include "decifrae_core.h"

#include <math.h>
#include <string.h>

#ifndef DECIFRAE_EPS
#define DECIFRAE_EPS 1.0e-20f
#endif

static int is_valid(float x) {
    return isfinite((double)x);
}

static float clamp_min(float x, float min_value) {
    return (x < min_value) ? min_value : x;
}

static void sort_float(float *x, size_t n) {
    /* Insertion sort is simple and safe for the short per-window rotation path. */
    for (size_t i = 1; i < n; ++i) {
        float key = x[i];
        size_t j = i;
        while (j > 0 && x[j - 1] > key) {
            x[j] = x[j - 1];
            --j;
        }
        x[j] = key;
    }
}

static float percentile_sorted(const float *sorted, size_t n, float q) {
    if (n == 0) return 0.0f;
    if (n == 1) return sorted[0];
    float pos = (float)(n - 1) * q;
    size_t lo = (size_t)floorf(pos);
    size_t hi = (size_t)ceilf(pos);
    if (hi >= n) hi = n - 1;
    float frac = pos - (float)lo;
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

void decifrae_default_config(decifrae_config_t *cfg) {
    if (!cfg) return;
    cfg->sample_rate = 16000;
    cfg->fmin_hz = 15.0f;
    cfg->fmax_hz = 60.0f;
    cfg->step_cents = 2.0f;
    cfg->n_fft = 8192;
    cfg->hop = 1024;
    cfg->n_harmonics = 48;
    cfg->max_jump = 50;
    cfg->use_viterbi = 1;
}

int decifrae_build_features(const decifrae_frontend_result_t *src,
                            decifrae_features_t *dst,
                            float *rotation_scratch,
                            size_t scratch_len) {
    if (!src || !dst || !src->f_rot_hz || !src->e_norm || !src->grid_hz ||
        !src->mean_power || src->n_frames == 0 || src->n_grid == 0 ||
        src->n_bins < 3 || !(src->df_hz > 0.0f)) {
        return -1;
    }

    memset(dst, 0, sizeof(*dst));

    /* f_rot median, IQR, and fraction of invalid/SIL frames. */
    size_t n_valid = 0;
    for (size_t t = 0; t < src->n_frames; ++t) {
        if (is_valid(src->f_rot_hz[t])) ++n_valid;
    }
    if (n_valid > 0) {
        /* Caller-owned scratch avoids heap allocation inside the real-time loop. */
        if (!rotation_scratch || scratch_len < n_valid) return -2;
        size_t p = 0;
        for (size_t t = 0; t < src->n_frames; ++t) {
            if (is_valid(src->f_rot_hz[t])) rotation_scratch[p++] = src->f_rot_hz[t];
        }
        sort_float(rotation_scratch, n_valid);
        dst->value[0] = percentile_sorted(rotation_scratch, n_valid, 0.50f);
        dst->value[1] = percentile_sorted(rotation_scratch, n_valid, 0.75f) -
                        percentile_sorted(rotation_scratch, n_valid, 0.25f);
    } else {
        dst->value[0] = 0.0f;
        dst->value[1] = 0.0f;
    }
    dst->value[2] = 1.0f - (float)n_valid / (float)src->n_frames;

    /* Harmonic-family salience: preserve numpy.searchsorted's insertion index. */
    double sal_sum = 0.0;
    for (size_t t = 0; t < src->n_frames; ++t) {
        float fr = src->f_rot_hz[t];
        if (!is_valid(fr)) continue; /* Python uses zero salience for invalid frames. */
        size_t lo = 0, hi = src->n_grid;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (src->grid_hz[mid] < fr) lo = mid + 1;
            else hi = mid;
        }
        size_t j = lo;
        if (j >= src->n_grid) j = src->n_grid - 1;
        sal_sum += src->e_norm[t * src->n_grid + j];
    }
    dst->value[3] = (float)(sal_sum / (double)src->n_frames);

    /* Mean power in 10 Hz--4 kHz and energy around the first 48 harmonics. */
    double total = 0.0;
    double centroid_num = 0.0;
    double power_sum = 0.0;
    double log_power_sum = 0.0;
    size_t n_band = 0;
    for (size_t i = 0; i < src->n_bins; ++i) {
        float f = (float)i * src->df_hz;
        float p = src->mean_power[i];
        if (p < 0.0f || !is_valid(p)) return -3;
        power_sum += p;
        centroid_num += (double)f * p;
        if (f >= 10.0f && f <= 4000.0f) {
            total += p;
            log_power_sum += log((double)p + DECIFRAE_EPS);
            ++n_band;
        }
    }
    total += DECIFRAE_EPS;

    float f_med = dst->value[0];
    double e_ord[48] = {0.0};
    if (f_med > 0.0f) {
        for (int k = 1; k <= 48; ++k) {
            long pos_l = lroundf(((float)k * f_med) / src->df_hz);
            if (pos_l >= 1 && (size_t)pos_l < src->n_bins - 1) {
                size_t pos = (size_t)pos_l;
                e_ord[k - 1] = (double)src->mean_power[pos - 1] +
                               (double)src->mean_power[pos] +
                               (double)src->mean_power[pos + 1];
            }
        }
    }
    double harm_sum = 0.0;
    for (int k = 0; k < 48; ++k) harm_sum += e_ord[k];
    dst->value[4] = (float)(harm_sum / total);

    const int group_start[4] = {0, 4, 12, 24};
    const int group_end[4]   = {4, 12, 24, 48};
    for (int g = 0; g < 4; ++g) {
        double group_sum = 0.0;
        for (int k = group_start[g]; k < group_end[g]; ++k) group_sum += e_ord[k];
        dst->value[5 + g] = 10.0f * log10f((float)(group_sum / total) + 1.0e-10f);
    }

    /* Features 9 and 10 are supplied by the C port of _envelope_mod(). */
    dst->value[9] = src->modulation_peak_hz;
    dst->value[10] = src->modulation_peak_prominence_db;

    dst->value[11] = (float)(centroid_num / (power_sum + DECIFRAE_EPS));
    if (n_band > 0) {
        double arithmetic_mean = total / (double)n_band;
        double geometric_mean = exp(log_power_sum / (double)n_band);
        dst->value[12] = (float)(geometric_mean / (arithmetic_mean + DECIFRAE_EPS));
    } else {
        dst->value[12] = 0.0f;
    }
    return 0;
}

int decifrae_cosine_predict(const float features[DECIFRAE_N_FEATURES],
                            const float *prototypes,
                            size_t n_classes,
                            float temperature,
                            float *probabilities) {
    if (!features || !prototypes || !probabilities || n_classes == 0 ||
        n_classes > DECIFRAE_MAX_CLASSES || !(temperature > 0.0f)) return -1;

    float x_norm_sq = 0.0f;
    for (size_t i = 0; i < DECIFRAE_N_FEATURES; ++i) {
        if (!is_valid(features[i])) return -2;
        x_norm_sq += features[i] * features[i];
    }
    float x_norm = sqrtf(x_norm_sq);
    float logits[DECIFRAE_MAX_CLASSES];
    float max_logit = -INFINITY;
    for (size_t c = 0; c < n_classes; ++c) {
        const float *r = prototypes + c * DECIFRAE_N_FEATURES;
        float dot = 0.0f, r_norm_sq = 0.0f;
        for (size_t i = 0; i < DECIFRAE_N_FEATURES; ++i) {
            if (!is_valid(r[i])) return -3;
            dot += features[i] * r[i];
            r_norm_sq += r[i] * r[i];
        }
        float denom = x_norm * sqrtf(r_norm_sq);
        float sim = (denom > DECIFRAE_EPS) ? dot / denom : 0.0f;
        logits[c] = sim / temperature;
        if (logits[c] > max_logit) max_logit = logits[c];
    }

    float sum = 0.0f;
    for (size_t c = 0; c < n_classes; ++c) {
        probabilities[c] = expf(logits[c] - max_logit);
        sum += probabilities[c];
    }
    if (!(sum > 0.0f) || !is_valid(sum)) return -4;
    for (size_t c = 0; c < n_classes; ++c) probabilities[c] /= sum;
    return 0;
}

int decifrae_hmm_viterbi(const float *emissions,
                         size_t n_steps,
                         size_t n_states,
                         const float *initial_log,
                         const float *transition_log,
                         int *path,
                         float *score_scratch,
                         float *next_score_scratch,
                         int *backpointer_scratch) {
    if (!emissions || !initial_log || !transition_log || !path ||
        !score_scratch || !next_score_scratch || !backpointer_scratch ||
        n_steps == 0 || n_states == 0 || n_states > DECIFRAE_MAX_CLASSES) return -1;

    /* Backpointers are row-major [n_steps][n_states], provided by caller. */
    for (size_t s = 0; s < n_states; ++s) {
        float p = emissions[s];
        score_scratch[s] = initial_log[s] + logf(clamp_min(p, 1.0e-30f));
        backpointer_scratch[s] = -1;
    }

    for (size_t t = 1; t < n_steps; ++t) {
        for (size_t s = 0; s < n_states; ++s) {
            float best = -INFINITY;
            int best_prev = 0;
            for (size_t prev = 0; prev < n_states; ++prev) {
                float candidate = score_scratch[prev] + transition_log[prev * n_states + s];
                if (candidate > best) {
                    best = candidate;
                    best_prev = (int)prev;
                }
            }
            float p = emissions[t * n_states + s];
            next_score_scratch[s] = best + logf(clamp_min(p, 1.0e-30f));
            backpointer_scratch[t * n_states + s] = best_prev;
        }
        memcpy(score_scratch, next_score_scratch, n_states * sizeof(float));
    }

    size_t best_state = 0;
    for (size_t s = 1; s < n_states; ++s) {
        if (score_scratch[s] > score_scratch[best_state]) best_state = s;
    }
    path[n_steps - 1] = (int)best_state;
    for (size_t t = n_steps - 1; t > 0; --t) {
        int prev = backpointer_scratch[t * n_states + (size_t)path[t]];
        if (prev < 0 || (size_t)prev >= n_states) return -2;
        path[t - 1] = prev;
    }
    return 0;
}
