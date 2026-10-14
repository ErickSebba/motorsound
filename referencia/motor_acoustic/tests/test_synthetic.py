"""Testes em sinal sintético. Rodar: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motoracoustic.config import PipelineConfig  # noqa: E402
from motoracoustic.evaluate import detection_delays, f0_error_metrics  # noqa: E402
from motoracoustic.features import envelope_modulation, file_features  # noqa: E402
from motoracoustic.hmm import ForwardFilter, sticky_transition, viterbi_states  # noqa: E402
from motoracoustic.preprocess import preprocess  # noqa: E402
from motoracoustic.ridge import track_rotation, track_rotation_framewise  # noqa: E402
from motoracoustic.synth import add_noise, motor_tone, synthetic_classes  # noqa: E402

SR = 16000
CFG = PipelineConfig()


def _ramp(dur=8.0, f_a=35.0, f_b=55.0):
    n = int(SR * dur)
    return np.linspace(f_a, f_b, n)


def _ref_at_frames(f0_true, times):
    return np.interp(times, np.arange(len(f0_true)) / SR, f0_true)


def test_ridge_segue_rampa_com_fundamental_fraca():
    f0 = _ramp()
    x = preprocess(motor_tone(f0, SR, rng=1), SR, CFG)
    r = track_rotation(x, SR, CFG)
    m = f0_error_metrics(r["f0"], _ref_at_frames(f0, r["t"]))
    assert m["within_2pct"] > 0.95, m
    assert m["octave_err"] < 0.02, m


def test_ridge_sobrevive_a_ruido_de_0dB():
    f0 = _ramp()
    x = add_noise(motor_tone(f0, SR, rng=2), snr_db=0, rng=3)
    r = track_rotation(preprocess(x, SR, CFG), SR, CFG)
    m = f0_error_metrics(r["f0"], _ref_at_frames(f0, r["t"]))
    assert m["within_2pct"] > 0.9, m


def test_viterbi_ganha_do_argmax_em_ruido_forte():
    f0 = _ramp()
    x = preprocess(add_noise(motor_tone(f0, SR, rng=4), snr_db=-6, rng=5), SR, CFG)
    r = track_rotation(x, SR, CFG)
    fw = track_rotation_framewise(x, SR, CFG)
    ref = _ref_at_frames(f0, r["t"])
    mv = f0_error_metrics(r["f0"], ref)
    ma = f0_error_metrics(fw, ref)
    assert mv["within_2pct"] >= ma["within_2pct"], (mv, ma)


def test_modulacao_detecta_chaveamento():
    x = motor_tone(np.full(SR * 2, 40.0), SR, am_hz=27.8, am_depth=0.6, rng=6)
    f, s = envelope_modulation(x, SR, CFG)
    assert abs(f - 27.8) < 2.0, f


def test_classes_sinteticas_sao_separaveis():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for seed in range(4):
        for lab, sig in synthetic_classes(SR, 6.0, rng=seed).items():
            sig = add_noise(sig, 5, rng=seed + 100)
            F = file_features(preprocess(sig, SR, CFG), SR, CFG)
            X.append(F); y += [lab] * len(F)
    X = np.vstack(X); y = np.array(y)
    half = len(y) // 2
    perm = np.random.default_rng(0).permutation(len(y))
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(X[perm[:half]], y[perm[:half]])
    assert m.score(X[perm[half:]], y[perm[half:]]) > 0.9


def test_transicao_e_estocastica_por_linha():
    T = sticky_transition(3, 0.98)
    assert np.allclose(T.sum(axis=1), 1)


def test_hmm_remove_flicker_e_mede_atraso():
    rng = np.random.default_rng(0)
    yt = np.array([0] * 40 + [1] * 40)
    P = np.full((80, 2), 0.2)
    P[np.arange(80), yt] = 0.8
    flip = rng.random(80) < 0.15           # 15 % de janelas erradas
    P[flip] = P[flip][:, ::-1]
    raw = P.argmax(1)
    vit = viterbi_states(P, 0.95)
    onl = ForwardFilter(2, 0.95).run(P)
    assert (np.diff(vit) != 0).sum() <= (np.diff(raw) != 0).sum()
    assert np.isfinite(detection_delays(yt, onl, 0.5)[0])
