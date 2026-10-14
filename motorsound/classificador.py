# Origem: Artigo2(Acordes)/Decifra-Violão/classifier.py::CosineChordClassifier
#         (commit 93609f7)
# Mudanças: os templates binários de acorde viram protótipos de classe aprendidos no
#   treino (centróide das features padronizadas por z-score do treino); a emissão é o
#   vetor de cossenos contra os protótipos, e `probabilidades` aplica softmax com
#   temperatura (como viterbi._softmax). Sem penalidade de complexidade (não se aplica).
"""Classificador por cosseno contra protótipos de classe (barato no MCU: C produtos
internos de dimensão d)."""
from __future__ import annotations

import numpy as np


class CossenoPrototipos:
    def __init__(self, temp: float = 0.1):
        self.temp = temp

    def fit(self, X, y, classes):
        X = np.asarray(X, dtype=float)
        self.classes = list(classes)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sd
        y = np.asarray(y)
        M = np.stack([Z[y == c].mean(axis=0) for c in self.classes])
        self.M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        return self

    def emissao(self, X):
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return Z @ self.M.T

    def predict_proba(self, X):
        e = self.emissao(X) / max(self.temp, 1e-6)
        e = np.exp(e - e.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
