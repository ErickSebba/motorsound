# Origem: Artigo1(cross-talk)/Decifra-Baixo/fase1_modular/modulo_ruidos.py (commit 2c9d64c)
#   funções _u, _time_axis, _gen_sag, _gen_swell, _gen_interruption, _gen_flicker,
#   _gen_harmonic, _gen_notch, _gen_spike e as faixas de ParamsAleatorios.
# Mudanças:
#   * volta ao domínio original: o distúrbio age sobre a TENSÃO de alimentação normalizada
#     v(t) (1,0 = nominal), não sobre a nota. O motor do IDMT (2ACT 42BLF01, 24 VDC) é
#     alimentado em CC, então a "portadora" sen(2πft) das equações vira o nível CC 1,0
#     (argumento `cc=True`, padrão). Com isso sag/swell/interruption/flicker devolvem
#     exatamente o envelope da equação de Lee & Shen; em harmônicos, notch e spike o
#     termo de base vira 1,0 e os termos de ripple/entalhe/pico continuam na frequência
#     da rede `f` (padrão 60 Hz, o valor do original elétrico). Com `cc=False` as funções
#     devolvem a forma de onda CA original, sem alteração da matemática;
#   * em `_gen_harmonic` com cc=True a normalização pelo pico foi retirada (normalizar
#     tiraria o nível CC de 1,0); com cc=False ela continua;
#   * flicker: a faixa de beta volta à do original elétrico (5–20 Hz) em vez da musical
#     (2–8 Hz); a profundidade volta a 0,1–0,2;
#   * random → numpy Generator com semente (reprodutível);
#   * acrescentado `_gen_chaveamento` (classe "broken" do IDMT) e o modelo de 1ª ordem da
#     rotação (`rotacao_relativa`), que não existiam no original;
#   * acrescentado `aplicar_disturbio`, que leva v(t) a um sinal de áudio de motor já
#     gravado (ver docstring: modulação de amplitude + de fase instantânea por time-warp).
"""Distúrbios de qualidade de energia de Lee & Shen (2011) na tensão de alimentação.

Referência: Lee, C.-Y.; Shen, Y.-X. "Optimal feature selection for power-quality
disturbances classification", IEEE Trans. Power Delivery 26(4), 2011 (equações dos
distúrbios, como no `modulo_ruidos.py` do Artigo 1).

Classe "broken" do IDMT: tensão chaveada entre 15 % e 75 % da alimentação a cada 18 ms.
Esses números vêm do `PLANO_PIPELINE_MOTOR.md` §4 e do `CONTEXTO_ESTUDO_MOTORSOUND.md`;
**não foram confirmados** nem no artigo EUSIPCO (docs/literatura/grollmisch2019.md, texto
completo lido, não traz os números) nem no PDF de descrição da base (docs/DATASETS.md §1.2,
que diz só "change of supply voltage and loading weight"). Ficam como suposição.
Convenção de escala adotada: 1,0 p.u. = tensão da classe "good" (60 % da alimentação,
mesma fonte não confirmada), logo o chaveamento vai de 0,15/0,60 = 0,25 p.u. a
0,75/0,60 = 1,25 p.u.
"""
from __future__ import annotations

import numpy as np
from numpy import arange
from scipy.signal import lfilter

F_REDE = 60.0          # frequência da rede no original elétrico (Hz)
TAU_MEC = 0.10         # constante de tempo mecânica suposta (s); datasheet do 42BLF01 não consultado
TIPOS = ("nominal", "sag", "swell", "interruption", "flicker", "harmonicos", "notch",
         "spike", "broken")


# ── funções auxiliares (copiadas) ────────────────────────────────────────────────────

def _u(x: np.ndarray) -> np.ndarray:
    """Função degrau de Heaviside. u(x) = 1 se x > 0, 0 caso contrário."""
    return np.heaviside(x, 0)


def _time_axis(duracao: float, sr: int) -> np.ndarray:
    return arange(0, duracao, 1.0 / sr)


def _base(f, t, cc):
    """Termo de base: 1,0 (alimentação CC) ou sen(2πft) (original CA)."""
    return np.ones_like(t) if cc else np.sin(2 * np.pi * f * t)


# ── geradores (matemática de Lee & Shen preservada; base trocada por _base) ──────────

def _gen_normal(f, duracao, sr, A, cc=True):
    t = _time_axis(duracao, sr)
    return A * _base(f, t, cc)


def _gen_sag(f, duracao, sr, A, alpha, t1, t2, cc=True):
    """x(t) = A·[1 − α(u(t−t1) − u(t−t2))]·base."""
    t = _time_axis(duracao, sr)
    envelope = 1 - alpha * (_u(t - t1) - _u(t - t2))
    return A * envelope * _base(f, t, cc)


def _gen_swell(f, duracao, sr, A, alpha, t1, t2, cc=True):
    """x(t) = A·[1 + α(u(t−t1) − u(t−t2))]·base."""
    t = _time_axis(duracao, sr)
    envelope = 1 + alpha * (_u(t - t1) - _u(t - t2))
    return A * envelope * _base(f, t, cc)


def _gen_interruption(f, duracao, sr, A, alpha, t1, t2, cc=True):
    """Sag com α ≈ 0,9–1,0 (distinção só paramétrica)."""
    return _gen_sag(f, duracao, sr, A, alpha, t1, t2, cc)


def _gen_flicker(f, duracao, sr, A, alpha, beta, cc=True):
    """x(t) = A·[1 + α·sen(2πβt)]·base."""
    t = _time_axis(duracao, sr)
    modulator = 1 + alpha * np.sin(2 * np.pi * beta * t)
    return A * modulator * _base(f, t, cc)


def _gen_harmonic(f, duracao, sr, A, alpha, cc=True):
    """x(t) = a1·base + Σ α_k·sen(2π(2k+1)ft), a1 = √(1 − Σα²). Em CC: ripple na rede."""
    t = _time_axis(duracao, sr)
    a1 = np.sqrt(1 - sum(a ** 2 for a in alpha))
    s = a1 * _base(f, t, cc)
    for k, a in enumerate(alpha):
        ordem = 2 * (k + 1) + 1   # 3, 5, 7
        s += a * np.sin(f * 2 * np.pi * t * ordem)
    if cc:
        return A * s
    pico = np.max(np.abs(s))
    return A * s / pico if pico > 0 else s


def _gen_notch(f, duracao, sr, A, k, t1, t2, cc=True):
    """x(t) = base − k·Σ[u(t−(t1+n/f)) − u(t−(t2+n/f))]·sign(base)."""
    t = _time_axis(duracao, sr)
    base = _base(f, t, cc)
    notch_ = np.zeros(len(t))
    n_ciclos = int(duracao * f) + 1
    for n in range(n_ciclos):
        notch_ += k * (_u(t - (t1 + n / f)) - _u(t - (t2 + n / f)))
    y = base + np.multiply(notch_, -np.sign(base))
    return A * y


def _gen_spike(f, duracao, sr, A, k, t1, t2, cc=True):
    """x(t) = base + k·Σ[u(t−(t1+n/f)) − u(t−(t2+n/f))]·sign(base)."""
    t = _time_axis(duracao, sr)
    base = _base(f, t, cc)
    spike_ = np.zeros(len(t))
    n_ciclos = int(duracao * f) + 1
    for n in range(n_ciclos):
        spike_ += k * (_u(t - (t1 + n / f)) - _u(t - (t2 + n / f)))
    y = base + np.multiply(spike_, +np.sign(base))
    return A * y


def _gen_chaveamento(duracao, sr, v_alta=0.75 / 0.60, v_baixa=0.15 / 0.60,
                     periodo_ms=18.0, fase_ms=0.0):
    """Classe "broken" do IDMT (novo): tensão alterna entre v_alta e v_baixa a cada
    `periodo_ms` (meio período da onda quadrada = 18 ms; período completo 36 ms)."""
    t = _time_axis(duracao, sr)
    k = np.floor((t * 1000.0 + fase_ms) / periodo_ms).astype(np.int64)
    return np.where(k % 2 == 0, v_alta, v_baixa)


# ── sorteio de parâmetros (faixas de ParamsAleatorios, domínio elétrico) ─────────────

def sortear_parametros(tipo: str, duracao: float, rng) -> dict:
    """Faixas do original. Janelas t1/t2 em segundos, como no módulo do Artigo 1."""
    p: dict = {}
    if tipo in ("sag", "swell", "interruption"):
        tamanho = rng.uniform(0.2, 0.6) * duracao
        p["t1"] = float(rng.uniform(0.05 * duracao, (duracao - tamanho) * 0.95))
        p["t2"] = p["t1"] + float(tamanho)
        p["alpha"] = float({"sag": rng.uniform(0.1, 0.9), "swell": rng.uniform(0.1, 0.8),
                            "interruption": rng.uniform(0.9, 1.0)}[tipo])
    elif tipo == "flicker":
        p["alpha"] = float(rng.uniform(0.1, 0.2))
        p["beta"] = float(rng.uniform(5.0, 20.0))
    elif tipo == "harmonicos":
        p["alpha"] = [float(rng.uniform(0.05, 0.20)), float(rng.uniform(0.03, 0.15)),
                      float(rng.uniform(0.02, 0.10))]
    elif tipo in ("notch", "spike"):
        periodo = 1.0 / F_REDE
        p["k"] = float(rng.uniform(0.05, 0.3))
        p["t1"] = float(rng.uniform(0, 0.3 * periodo))
        p["t2"] = p["t1"] + float(rng.uniform(0.01, 0.08) * periodo)
    elif tipo == "broken":
        p["fase_ms"] = float(rng.uniform(0, 36.0))
    return p


def tensao_perturbada(dur_s: float, fs_ctrl: int, tipo: str, seed=None,
                      devolver_params: bool = False, **params):
    """Tensão de alimentação normalizada v(t) (1,0 = nominal) com o distúrbio `tipo`.

    Parâmetros omitidos são sorteados com `seed` nas faixas do original. Tipos: ver TIPOS.
    `f` (frequência da rede para harmônicos/notch/spike) tem padrão F_REDE.
    """
    if tipo not in TIPOS:
        raise ValueError(f"distúrbio desconhecido: {tipo}")
    rng = np.random.default_rng(seed)
    p = {**sortear_parametros(tipo, dur_s, rng), **params}
    f = p.pop("f", F_REDE)
    n = len(_time_axis(dur_s, fs_ctrl))
    if tipo == "nominal":
        v = _gen_normal(f, dur_s, fs_ctrl, 1.0)
    elif tipo == "sag":
        v = _gen_sag(f, dur_s, fs_ctrl, 1.0, p["alpha"], p["t1"], p["t2"])
    elif tipo == "swell":
        v = _gen_swell(f, dur_s, fs_ctrl, 1.0, p["alpha"], p["t1"], p["t2"])
    elif tipo == "interruption":
        v = _gen_interruption(f, dur_s, fs_ctrl, 1.0, p["alpha"], p["t1"], p["t2"])
    elif tipo == "flicker":
        v = _gen_flicker(f, dur_s, fs_ctrl, 1.0, p["alpha"], p["beta"])
    elif tipo == "harmonicos":
        v = _gen_harmonic(f, dur_s, fs_ctrl, 1.0, p["alpha"])
    elif tipo == "notch":
        v = _gen_notch(f, dur_s, fs_ctrl, 1.0, p["k"], p["t1"], p["t2"])
    elif tipo == "spike":
        v = _gen_spike(f, dur_s, fs_ctrl, 1.0, p["k"], p["t1"], p["t2"])
    else:  # broken
        v = _gen_chaveamento(dur_s, fs_ctrl, **p)
    v = np.asarray(v, dtype=float)[:n]
    if tipo not in ("broken",):
        p["f"] = f
    return (v, p) if devolver_params else v


def rotacao_relativa(v, fs, tau_s=TAU_MEC):
    """Modelo de 1ª ordem τ·dr/dt = v − r, discretizado exatamente (ZOH), com r(0) = v(0).
    É a rotação normalizada (r = f_rot/K)."""
    v = np.asarray(v, dtype=float)
    if tau_s <= 0:
        return v.copy()
    a = np.exp(-1.0 / (fs * tau_s))
    r, _ = lfilter([1 - a], [1, -a], v, zi=[a * v[0]])
    return np.clip(r, 0.0, None)


def aplicar_disturbio(y, sr, tipo="sag", seed=None, tau_s=TAU_MEC, m_am=0.3,
                      disturbio=None, **params):
    """Leva o distúrbio de tensão a um áudio de motor já existente (uso em noise.py).

    Pelo `noise.aplicar`, a chave "tipo" do dict é consumida ("lee_shen"); o distúrbio vem
    então em "disturbio": {"tipo": "lee_shen", "disturbio": "broken", ...}.

    Modelo simples e declarado:
      r(t)  = rotacao_relativa(v, tau_s)        (rotação normalizada, 1,0 = nominal)
      fase  : time-warp y(τ(t)) com τ(t) = ∫ r dt → TODAS as componentes do áudio
              (inclusive o ruído de fundo da gravação) têm a frequência escalada por r(t);
      ampl. : a(t) = r(t)·(1 − m_am + m_am·v(t))  (nível ∝ rotação, mais uma parcela que
              segue a tensão instantânea, como no gerador sintético).
    Quando τ passa do fim do sinal (swell), o áudio é estendido com sua cópia invertida no
    tempo (espelho), o que preserva o espectro de um som estacionário.
    """
    if disturbio is not None:
        tipo = disturbio
    y = np.asarray(y, dtype=float)
    n = len(y)
    v = tensao_perturbada(n / sr, sr, tipo, seed, **params)
    if len(v) < n:
        v = np.pad(v, (0, n - len(v)), mode="edge")
    v = v[:n]
    r = rotacao_relativa(v, sr, tau_s)
    tau = np.concatenate([[0.0], np.cumsum(r[:-1])])         # em amostras
    estendido = np.concatenate([y, y[::-1]])
    while tau[-1] >= len(estendido) - 1:
        estendido = np.concatenate([estendido, estendido[::-1]])
    yw = np.interp(tau, np.arange(len(estendido)), estendido)
    return yw * r * (1.0 - m_am + m_am * v)
