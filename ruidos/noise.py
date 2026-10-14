# Origem: Artigo2(Acordes)/Decifra-Violão/noise.py (commit 93609f7)
# Mudanças: `detune` removido (sem sentido para motor); `add_percussive` passa a ter
#   amplitude RELATIVA ao RMS do sinal (no violão era absoluta, 0,4, o que dependia do
#   nível do áudio); `reverb_tail` não normaliza mais o pico (o runner normaliza o
#   RMS depois); acrescentados `add_brown_noise` (ruído grave, substituto de ambiente
#   fabril) e `variar_ganho` (degraus de ganho, imitando a descrição do stress test do
#   IDMT); `aplicar` recebe um dict, que é como o runner descreve a perturbação.
"""Bateria de ruídos aplicada aos segmentos de validação (treino sempre limpo)."""
import numpy as np


def _rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def _escala(y, ruido, snr_db):
    p_sig = np.mean(y ** 2) + 1e-12
    return ruido / (_rms(ruido) + 1e-12) * np.sqrt(p_sig / 10 ** (snr_db / 10.0))


def add_white_noise(y, snr_db=20.0, seed=None):
    rng = np.random.default_rng(seed)
    return y + _escala(y, rng.normal(0, 1, len(y)), snr_db)


def _colorido(n, expoente, rng):
    spec = np.fft.rfft(rng.normal(0, 1, n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    return np.fft.irfft(spec / f ** expoente, n)


def add_pink_noise(y, snr_db=20.0, seed=None):
    rng = np.random.default_rng(seed)
    return y + _escala(y, _colorido(len(y), 0.5, rng), snr_db)


def add_brown_noise(y, snr_db=20.0, seed=None):
    rng = np.random.default_rng(seed)
    return y + _escala(y, _colorido(len(y), 1.0, rng), snr_db)


def add_percussive(y, sr, hits_por_s=1.5, amp_rel=8.0, seed=None):
    """Batidas de banda larga (~50 ms) com pico = amp_rel × RMS do sinal."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    n = len(y)
    dur = sr // 20
    r = _rms(y)
    for _ in range(max(1, int(round(hits_por_s * n / sr)))):
        pos = int(rng.integers(0, max(1, n - dur)))
        click = rng.normal(0, 1, dur) * np.exp(-np.linspace(0, 8, dur))
        click = click / (np.abs(click).max() + 1e-12) * amp_rel * r
        out[pos:pos + dur] += click[: n - pos]
    return out


def reverb_tail(y, sr, decay=0.35, delay_ms=80):
    out = y.copy()
    d0 = int(sr * delay_ms / 1000.0)
    for k in range(1, 4):
        d = d0 * k
        if d < len(y):
            out[d:] += (decay ** k) * y[: len(y) - d]
    return out


def variar_ganho(y, sr, db=6.0, trocas_por_s=0.5, seed=None):
    """Degraus de ganho aleatórios em ±db (rampa de 10 ms)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    k = max(1, int(round(trocas_por_s * n / sr)))
    pos = np.sort(rng.integers(0, n, k))
    g = np.ones(n)
    nivel = 1.0
    for p in pos:
        nivel = 10 ** (rng.uniform(-db, db) / 20)
        g[p:] = nivel
    w = max(1, sr // 100)
    g = np.convolve(g, np.ones(w) / w, mode="same")
    return y * g


def aplicar(y, sr, perturbacao: dict, seed=None):
    """perturbacao = {"tipo": ..., **parâmetros, "seed": opcional}."""
    p = {k: v for k, v in perturbacao.items() if k not in ("tipo", "seed")}
    t = perturbacao["tipo"]
    if t == "limpo":
        return y
    if t == "branco":
        return add_white_noise(y, seed=seed, **p)
    if t == "rosa":
        return add_pink_noise(y, seed=seed, **p)
    if t == "marrom":
        return add_brown_noise(y, seed=seed, **p)
    if t == "percussivo":
        return add_percussive(y, sr, seed=seed, **p)
    if t == "reverb":
        return reverb_tail(y, sr, **p)
    if t == "ganho":
        return variar_ganho(y, sr, seed=seed, **p)
    if t == "lee_shen":
        from .lee_shen import aplicar_disturbio
        return aplicar_disturbio(y, sr, seed=seed, **p)
    raise ValueError(f"perturbação desconhecida: {t}")


# Catálogo varrido na validação ruidosa (critério decidido em 14/10/2026)
CONDICOES_VAL = [
    {"tipo": "branco", "snr_db": 20.0}, {"tipo": "branco", "snr_db": 10.0},
    {"tipo": "branco", "snr_db": 0.0}, {"tipo": "branco", "snr_db": -5.0},
    {"tipo": "rosa", "snr_db": 10.0}, {"tipo": "rosa", "snr_db": 0.0},
    {"tipo": "marrom", "snr_db": 0.0},
    {"tipo": "percussivo", "hits_por_s": 1.5, "amp_rel": 8.0},
    {"tipo": "reverb", "decay": 0.35},
    {"tipo": "ganho", "db": 6.0},
]
