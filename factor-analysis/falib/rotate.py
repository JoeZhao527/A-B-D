"""Rotations. Every function returns (pattern, structure, Phi, T):
pattern = unique loadings, structure = correlations of variables with factors,
Phi = factor correlation matrix (identity for orthogonal rotations), T = rotation
matrix such that pattern = L @ T. Columns are sign-flipped so each factor's largest
absolute loading is positive, and ordered by pattern sum of squares (descending).
"""
from __future__ import annotations

import numpy as np

from .config import RotateConfig


def _kaiser_normalize(L: np.ndarray):
    h = np.sqrt((L ** 2).sum(1))
    h[h == 0] = 1.0
    return L / h[:, None], h


def varimax(L: np.ndarray, normalize: bool = True, max_iter: int = 200, tol: float = 1e-6):
    """Kaiser-normalized varimax; -> (rotated loadings, T)."""
    p, k = L.shape
    Ln, h = _kaiser_normalize(L) if normalize else (L.copy(), np.ones(p))
    T = np.eye(k)
    d_old = 0.0
    for _ in range(max_iter):
        Lr = Ln @ T
        B = Ln.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p)
        U, s, Vt = np.linalg.svd(B)
        T = U @ Vt
        d = s.sum()
        if abs(d - d_old) < tol:
            break
        d_old = d
    return (Ln @ T) * h[:, None], T


def promax(L: np.ndarray, power: int = 4, normalize: bool = True):
    """Hendrickson & White (1964), as in R's promax(): varimax, then an oblique
    least-squares fit to the |loading|^power target. -> (pattern, T, Phi)."""
    V, Tv = varimax(L, normalize=normalize)
    Q = V * np.abs(V) ** (power - 1)
    U = np.linalg.lstsq(V, Q, rcond=None)[0]
    d = np.sqrt(np.diag(np.linalg.pinv(U.T @ U)))
    U = U * d
    pattern = V @ U
    T = Tv @ U
    Phi = np.linalg.pinv(T.T @ T)
    return pattern, T, Phi


def _tidy(pattern, Phi, T):
    """Sign-flip (largest |loading| positive) and order factors by SS loadings."""
    k = pattern.shape[1]
    pattern, Phi, T = pattern.copy(), Phi.copy(), T.copy()
    for j in range(k):
        if pattern[np.argmax(np.abs(pattern[:, j])), j] < 0:
            pattern[:, j] *= -1
            T[:, j] *= -1
            Phi[j, :] *= -1
            Phi[:, j] *= -1
    ss = (pattern ** 2).sum(0)
    o = np.argsort(ss)[::-1]
    return pattern[:, o], Phi[np.ix_(o, o)], T[:, o]


def rotate(L: np.ndarray, cfg: RotateConfig | None = None):
    cfg = cfg or RotateConfig()
    k = L.shape[1]
    if cfg.method == "none" or k == 1:
        pattern, T, Phi = L.copy(), np.eye(k), np.eye(k)
    elif cfg.method == "varimax":
        pattern, T = varimax(L, cfg.kaiser_normalize)
        Phi = np.eye(k)
    elif cfg.method == "promax":
        pattern, T, Phi = promax(L, cfg.promax_power, cfg.kaiser_normalize)
    else:
        raise ValueError(f"unknown rotation {cfg.method!r}")
    pattern, Phi, T = _tidy(pattern, Phi, T)
    structure = pattern @ Phi
    return pattern, structure, Phi, T
