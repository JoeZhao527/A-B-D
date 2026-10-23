"""Is the correlation matrix factorable? Bartlett's sphericity test and KMO."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as chi2dist


def bartlett(R: np.ndarray, n: int) -> dict:
    p = len(R)
    sign, logdet = np.linalg.slogdet(R)
    stat = -((n - 1) - (2 * p + 5) / 6.0) * logdet
    df = p * (p - 1) / 2
    return {"chi2": float(stat), "df": int(df), "p": float(chi2dist.sf(stat, df))}


def kmo(R: np.ndarray, ridge: float = 1e-6) -> dict:
    """Overall KMO and per-variable MSA (ridge-stabilised inverse for n ~ p)."""
    p = len(R)
    Rinv = np.linalg.pinv(R + ridge * np.eye(p))
    D = np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
    partial = -Rinv / D
    np.fill_diagonal(partial, 0)
    Roff = R.copy()
    np.fill_diagonal(Roff, 0)
    r2, q2 = Roff ** 2, partial ** 2
    overall = r2.sum() / (r2.sum() + q2.sum())
    msa = r2.sum(0) / (r2.sum(0) + q2.sum(0))
    return {"overall": float(overall), "msa": msa}
