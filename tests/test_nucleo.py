import numpy as np
import pytest

from motorsound import cache, hmm, metricas, rotacao, stats
from motorsound.config import PreConfig


def test_cache_ida_e_volta_e_divergencia(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    k = cache.chave("px", PreConfig(sr=16000), None, None, "idmt/train/good/pure", [0, 1])
    assert cache.carregar(k) is None
    X = np.arange(6.0).reshape(2, 3)
    cache.salvar(k, X)
    np.testing.assert_array_equal(cache.carregar(k), X)
    # outra configuração → outro arquivo (não devolve o antigo)
    k2 = cache.chave("px", PreConfig(sr=8000), None, None, "idmt/train/good/pure", [0, 1])
    assert cache.carregar(k2) is None
    # metadado adulterado no mesmo caminho → erro, nunca silêncio
    cache.caminho(k2).parent.mkdir(parents=True, exist_ok=True)
    cache.caminho(k).replace(cache.caminho(k2))
    with pytest.raises(cache.CacheMismatchError):
        cache.carregar(k2)


def test_sequencia_atraso_e_espurias():
    v = ["a"] * 10 + ["b"] * 10
    p = ["a"] * 12 + ["b"] * 8                  # detecta a troca com 2 janelas de atraso
    r = metricas.sequencia(v, p, passo_s=1.0)
    assert r["n_trocas"] == 1 and r["atraso_mediano_s"] == 2.0
    assert r["perdidas"] == 0 and r["espurias"] == 0
    p2 = ["a"] * 5 + ["b"] + ["a"] * 4 + ["b"] * 10   # uma troca espúria (e sua volta)
    r2 = metricas.sequencia(v, p2, passo_s=1.0)
    assert r2["espurias"] == 2


def test_hmm_suaviza_e_online_e_causal():
    P = np.array([[.9, .1], [.9, .1], [.4, .6], [.9, .1], [.1, .9], [.1, .9], [.1, .9]])
    assert hmm.decode(P, ["a", "b"], change_penalty=2.0) == ["a"] * 4 + ["b"] * 3
    on = hmm.filtro_online(P, ["a", "b"], change_penalty=2.0)
    on_curto = hmm.filtro_online(P[:4], ["a", "b"], change_penalty=2.0)
    assert on[:4] == on_curto                     # decisão em t não depende do futuro


def test_rastreador_sintetico():
    sr = 16000
    t = np.arange(3 * sr) / sr
    f0 = 26.0
    y = sum(np.sin(2 * np.pi * k * f0 * t) / (1 + (k % 12 != 0) * 5) for k in range(1, 40))
    f, _, _ = rotacao.rastrear(y, sr)
    assert abs(np.nanmedian(f) - f0) / f0 < 0.02


def test_holm_monotono():
    a = stats.holm([0.01, 0.04, 0.03, np.nan])
    assert np.isnan(a[3]) and a[0] <= a[2] <= a[1] <= 1
