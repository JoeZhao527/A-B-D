"""Factor scores: Thurstone regression scores  W = (R + lambda I)^-1 S,  F = Z W,
where S is the structure matrix (= pattern for orthogonal rotations).

lambda (ScoresConfig.ridge, default 0.1 on the correlation scale) is a ridge penalty on the
inverse. With lambda = 0 this is the textbook estimator, which needs R to be well conditioned:
a trait set with near-duplicate or additive categories (LIWC's pronoun = ppron + ipron, the
function-word total, ...) makes R nearly singular (the paper's linguistic traits: smallest eigenvalue 2e-7,
condition number 7e7) and the weights explode into huge opposing coefficients on the
duplicates, so the scores no longer track the loadings (two orthogonal factors came out
with scores correlated .89). The ridge caps the amplification of a degenerate direction at
1/lambda while changing well-conditioned directions (eigenvalue >= 1) by < lambda; the
loadings, k and rotation are untouched (they never invert R). Scores are insensitive to
lambda in 0.03-0.3 (per-factor r >= .99 across that range on the paper's data)."""
from __future__ import annotations

import numpy as np

from .config import ScoresConfig


def regression_scores(Z: np.ndarray, R: np.ndarray, structure: np.ndarray,
                      cfg: ScoresConfig | None = None):
    cfg = cfg or ScoresConfig()
    if cfg.method != "regression":
        raise ValueError(f"unknown scores method {cfg.method!r}")
    lam = float(cfg.ridge or 0.0)
    if lam > 0:
        W = np.linalg.solve(R + lam * np.eye(R.shape[0]), structure)
    else:
        W = np.linalg.pinv(R) @ structure
    F = Z @ W
    if cfg.standardize:
        F = (F - F.mean(0)) / F.std(0, ddof=1)
    return F, W
