"""Principal-axis factoring with iterated communalities (SMC start)."""
from __future__ import annotations

import numpy as np

from .config import ExtractConfig


def paf(R: np.ndarray, k: int, cfg: ExtractConfig | None = None):
    """-> (unrotated loadings p x k, communalities p, n_iter)."""
    cfg = cfg or ExtractConfig()
    p = len(R)
    h2 = 1 - 1.0 / np.diag(np.linalg.pinv(R + 1e-6 * np.eye(p)))     # SMC initial communalities
    h2 = np.clip(h2, 0, 1)
    Rr = R.copy()
    n_iter = 0
    for n_iter in range(1, cfg.max_iter + 1):
        np.fill_diagonal(Rr, h2)
        w, V = np.linalg.eigh(Rr)
        idx = np.argsort(w)[::-1][:k]
        w_k = np.clip(w[idx], 0, None)
        L = V[:, idx] * np.sqrt(w_k)
        h2_new = (L ** 2).sum(1)
        if np.max(np.abs(h2_new - h2)) < cfg.tol:
            h2 = h2_new
            break
        h2 = h2_new
    return L, h2, n_iter


def extract(R: np.ndarray, k: int, cfg: ExtractConfig | None = None):
    cfg = cfg or ExtractConfig()
    if cfg.method != "paf":
        raise ValueError(f"unknown extraction method {cfg.method!r}")
    return paf(R, k, cfg)
