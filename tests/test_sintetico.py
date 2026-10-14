import numpy as np
import pytest

from motorsound import rotacao
from ruidos import lee_shen, noise, sintetico


def test_tensao_lee_shen_forma():
    v = lee_shen.tensao_perturbada(2.0, 1000, "sag", alpha=0.3, t1=0.5, t2=1.5)
    assert v[100] == 1.0 and np.isclose(v[1000], 0.7) and v[1800] == 1.0
    vb = lee_shen.tensao_perturbada(0.2, 16000, "broken", fase_ms=0.0)
    assert set(np.round(np.unique(vb), 3)) == {0.25, 1.25}
    # troca a cada 18 ms
    trocas = np.nonzero(np.diff(vb))[0]
    assert np.allclose(np.diff(trocas) / 16000, 0.018, atol=1 / 16000)


def test_aplicar_disturbio_via_noise():
    y = np.sin(2 * np.pi * 600 * np.arange(16000) / 16000)
    out = noise.aplicar(y, 16000, {"tipo": "lee_shen", "disturbio": "sag", "alpha": 0.5,
                                   "t1": 0.3, "t2": 0.7}, seed=0)
    assert out.shape == y.shape and np.all(np.isfinite(out))


def test_gerador_rotacao_nominal_constante():
    y, f = sintetico.gerar(4.0, tipo="nominal", snr_db=None, seed=0, f_nom=26.3)
    assert len(y) == len(f) == 4 * 16000
    np.testing.assert_allclose(f, 26.3, rtol=1e-9)
    # o pico mais forte do espectro é o 24º harmônico
    S = np.abs(np.fft.rfft(y))
    fr = np.fft.rfftfreq(len(y), 1 / 16000)
    assert abs(fr[np.argmax(S)] - 24 * 26.3) < 1.0


@pytest.mark.parametrize("f_nom", [24.5, 26.3, 27.5])
def test_rastreador_recupera_nominal_sem_ruido(f_nom):
    y, f = sintetico.gerar(8.0, tipo="nominal", snr_db=None, seed=1, f_nom=f_nom)
    f_est, _, _ = rotacao.rastrear(y, 16000)
    assert np.isfinite(f_est).all()
    assert np.median(np.abs(f_est / f_nom - 1)) < 0.02


@pytest.mark.xfail(reason="limitação conhecida: com a grade padrão de 10 cents, 25,0 e 25,7 Hz "
                          "caem em 4/3·f (f e 4f/3 explicam os mesmos harmônicos múltiplos de "
                          "4 e o desempate é decidido pela quantização da grade)", strict=False)
@pytest.mark.parametrize("f_nom", [25.0, 25.7])
def test_rastreador_nominal_grade_10_cents_falha_4_3(f_nom):
    y, f = sintetico.gerar(8.0, tipo="nominal", snr_db=None, seed=1, f_nom=f_nom)
    f_est, _, _ = rotacao.rastrear(y, 16000, passo_cents=10.0, max_salto=10)
    assert np.median(np.abs(f_est / f_nom - 1)) < 0.02


@pytest.mark.parametrize("f_nom", [25.0, 25.7])
def test_rastreador_grade_2_cents_resolve(f_nom):
    y, f = sintetico.gerar(8.0, tipo="nominal", snr_db=None, seed=1, f_nom=f_nom)
    f_est, _, _ = rotacao.rastrear(y, 16000, passo_cents=2.0, max_salto=50)
    assert np.median(np.abs(f_est / f_nom - 1)) < 0.02
