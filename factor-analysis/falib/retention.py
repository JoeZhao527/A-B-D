"""How many factors: Horn's parallel analysis, the Kaiser rule, Velicer's MAP.

All three are always computed so a run's report can show where they agree; the
config's `rule` decides which one sets k.
"""
from __future__ import annotations

import numpy as np

from .config import RetentionConfig


def eigenvalues(R: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(R))[::-1]


def smc_reduced(R: np.ndarray) -> np.ndarray:
    Rr = R.copy()
    h2 = 1 - 1.0 / np.diag(np.linalg.pinv(R + 1e-6 * np.eye(len(R))))
    np.fill_diagonal(Rr, np.clip(h2, 0, 1))
    return Rr


def parallel_analysis(X: np.ndarray, cfg: RetentionConfig, seed: int = 0) -> dict:
    """X: standardized data (n x p). Random eigenvalues from normal data of the same
    shape (or column permutations of X); k = # observed eigenvalues above the
    `pa_percentile` of the random ones."""
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    obs = eigenvalues(smc_reduced(R) if cfg.pa_basis == "smc" else R)
    rng = np.random.RandomState(seed)
    rand = np.zeros((cfg.pa_n_random, p))
    for b in range(cfg.pa_n_random):
        if cfg.pa_data == "permute":
            Xr = np.column_stack([rng.permutation(X[:, j]) for j in range(p)])
        else:
            Xr = rng.standard_normal((n, p))
        Rr = np.corrcoef(Xr, rowvar=False)
        rand[b] = eigenvalues(smc_reduced(Rr) if cfg.pa_basis == "smc" else Rr)
    thr = np.percentile(rand, cfg.pa_percentile, axis=0)
    k = int((obs > thr).sum())
    return {"k": k, "observed": obs, "threshold": thr}


def kaiser(R: np.ndarray) -> int:
    return int((eigenvalues(R) > 1).sum())


def velicer_map(R: np.ndarray, power: int = 2, k_max: int | None = None) -> dict:
    """Average squared (or 4th-power) partial correlation after partialling out the
    first k principal components; k = argmin."""
    p = len(R)
    k_max = min(p - 2, k_max if k_max is not None else p // 2)
    w, V = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    curve = []
    for k in range(0, k_max + 1):
        L = V[:, :k] * np.sqrt(np.clip(w[:k], 0, None)) if k else np.zeros((p, 0))
        Rs = R - L @ L.T
        d = np.sqrt(np.clip(np.diag(Rs), 1e-12, None))
        P = Rs / np.outer(d, d)
        np.fill_diagonal(P, 0)
        curve.append(float((np.abs(P) ** power).sum() / (p * (p - 1))))
    return {"k": int(np.argmin(curve)), "curve": np.array(curve)}


def choose_k(X: np.ndarray, cfg: RetentionConfig, seed: int = 0) -> dict:
    """Compute every rule; return {"k", "rule", "parallel", "kaiser", "map"}."""
    R = np.corrcoef(X, rowvar=False)
    p = R.shape[0]
    k_max = cfg.k_max if cfg.k_max is not None else p // 2
    pa = parallel_analysis(X, cfg, seed)
    ka = kaiser(R)
    mp = velicer_map(R, cfg.map_power, k_max)
    if cfg.rule == "parallel":
        k = pa["k"]
    elif cfg.rule == "kaiser":
        k = ka
    elif cfg.rule == "map":
        k = mp["k"]
    elif cfg.rule == "fixed":
        if cfg.k_fixed is None:
            raise ValueError("rule=fixed needs k_fixed")
        k = cfg.k_fixed
    else:
        raise ValueError(f"unknown retention rule {cfg.rule!r}")
    k = int(min(max(k, cfg.k_min), k_max))
    return {"k": k, "rule": cfg.rule, "parallel": pa, "kaiser": ka, "map": mp}
