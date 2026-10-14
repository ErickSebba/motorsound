"""Métricas do artigo.

Por janela  : acurácia, F1 macro, matriz de confusão, por condição de ruído.
Por sequência (comparável a Strantzalis et al. 2022, Tab. 7-8):
  - atraso de detecção: tempo entre a troca real de estado e a primeira
    janela a partir da qual a decisão fica no estado novo;
  - trocas espúrias por minuto (falso alarme).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def window_metrics(y_true, y_pred, labels) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "cm": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def by_condition(y_true, y_pred, cond) -> dict:
    y_true, y_pred, cond = map(np.asarray, (y_true, y_pred, cond))
    return {c: float(np.mean(y_true[cond == c] == y_pred[cond == c]))
            for c in sorted(set(cond))}


def detection_delays(y_true: np.ndarray, y_pred: np.ndarray, hop_s: float) -> list[float]:
    """Atraso (s) em cada troca de estado da verdade."""
    delays = []
    changes = np.flatnonzero(np.diff(y_true) != 0) + 1
    bounds = list(changes) + [len(y_true)]
    for i, c in enumerate(changes):
        end = bounds[i + 1]
        new = y_true[c]
        seg = y_pred[c:end] == new
        # primeira posição a partir da qual fica certo até o fim do segmento
        ok = np.flatnonzero(~seg)
        k = (ok[-1] + 1) if len(ok) else 0
        delays.append(np.inf if k >= len(seg) else k * hop_s)
    return delays


def spurious_switches_per_min(y_true: np.ndarray, y_pred: np.ndarray, hop_s: float) -> float:
    true_sw = int(np.sum(np.diff(y_true) != 0))
    pred_sw = int(np.sum(np.diff(y_pred) != 0))
    minutes = len(y_true) * hop_s / 60
    return max(0, pred_sw - true_sw) / minutes


def f0_error_metrics(f0_est: np.ndarray, f0_ref: np.ndarray) -> dict:
    """Para validar o ridge onde houver referência (sintético)."""
    n = min(len(f0_est), len(f0_ref))
    e, r = np.asarray(f0_est[:n], float), np.asarray(f0_ref[:n], float)
    ok = np.isfinite(e)
    if not ok.any():   # pYIN declara tudo "unvoiced" em ruído forte
        return {"mae_hz": float("nan"), "within_2pct": 0.0, "octave_err": 0.0,
                "unvoiced": 1.0}
    rel = np.abs(e[ok] - r[ok]) / r[ok]
    ratio = e[ok] / r[ok]
    octave = np.mean((np.abs(ratio - 0.5) < 0.05) | (np.abs(ratio - 2) < 0.1))
    return {"mae_hz": float(np.mean(np.abs(e[ok] - r[ok]))),
            "within_2pct": float(np.mean(rel < 0.02)),
            "octave_err": float(octave),
            "unvoiced": float(1 - ok.mean())}
