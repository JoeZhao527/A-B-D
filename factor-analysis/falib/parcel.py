"""Parcels: merge traits that are copies of one construct into one indicator.

Following Little et al. (2002, 2013), a parcel is legitimate when its members are
unidimensional. We make that (nearly) true by construction — hierarchical clustering
on 1 - |r| cut so that every pair inside a cluster correlates at least
`r_threshold` (complete linkage) — and then verify it: first-PC share and
Cronbach's alpha of each merged cluster must pass `min_pc1` / `min_alpha`, or the
cluster is left unmerged.

Input matrix: models x traits (rows = models). Output: models x parcels, and a
table with one row per parcel (members, anchor, checks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .config import ParcelConfig


def zscore(df: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    sd = df.std(axis=0, ddof=ddof).replace(0, 1.0)
    return (df - df.mean(axis=0)) / sd


def cronbach_alpha(X: pd.DataFrame) -> float:
    k = X.shape[1]
    if k < 2:
        return float("nan")
    tot = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - X.var(ddof=1).sum() / tot)) if tot > 0 else float("nan")


def first_pc(X: pd.DataFrame):
    """(share of variance on the first PC, loadings on it) of the z-scored columns."""
    R = np.corrcoef(zscore(X).values, rowvar=False)
    w, V = np.linalg.eigh(R)
    i = int(np.argmax(w))
    v = V[:, i]
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return float(w[i] / w.sum()), v


def family_of(name: str) -> str:
    return name.split(":", 1)[0] if ":" in name else ""


def cluster_traits(X: pd.DataFrame, cfg: ParcelConfig) -> list[list[str]]:
    """Groups of column names; singletons included. Deterministic (column order)."""
    cols = list(X.columns)
    if len(cols) < 2:
        return [[c] for c in cols]
    R = X.corr().abs().values
    if cfg.scope == "within_family":
        fam = np.array([family_of(c) for c in cols])
        R = np.where(fam[:, None] == fam[None, :], R, 0.0)
    D = 1.0 - R
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method=cfg.linkage)
    lab = fcluster(Z, t=1.0 - cfg.r_threshold, criterion="distance")
    groups: dict[int, list[str]] = {}
    for c, g in zip(cols, lab):
        groups.setdefault(int(g), []).append(c)
    # order groups by first member's column position
    return sorted(groups.values(), key=lambda g: cols.index(g[0]))


def build_parcels(X: pd.DataFrame, cfg: ParcelConfig, anchor_stat: pd.Series | None = None):
    """-> (models x parcels DataFrame, parcels table DataFrame).

    Parcel score = mean of the sign-aligned z-scored members, re-standardized.
    Name = anchor (+N for merged parcels). anchor_stat (e.g. ICC per trait) picks the
    anchor; default = highest first-PC loading."""
    if not cfg.enabled:
        groups = [[c] for c in X.columns]
    else:
        groups = cluster_traits(X, cfg)
    Xz = zscore(X)
    out, rows = {}, []
    for members in groups:
        if len(members) == 1:
            c = members[0]
            out[c] = Xz[c]
            rows.append({"parcel": c, "anchor": c, "n_members": 1, "members": c, "signs": "+",
                         "min_pair_r": np.nan, "alpha": np.nan, "pc1_share": np.nan, "merged": False})
            continue
        sub = Xz[members]
        pc1, load = first_pc(sub)
        signs = np.where(load < 0, -1.0, 1.0) if cfg.align_sign else np.ones(len(members))
        aligned = sub * signs
        alpha = cronbach_alpha(aligned)
        min_r = float(aligned.corr().values[np.triu_indices(len(members), 1)].min())
        ok = (pc1 >= cfg.min_pc1) and (alpha >= cfg.min_alpha)
        if anchor_stat is not None and any(m in anchor_stat.index for m in members):
            anchor = max(members, key=lambda m: anchor_stat.get(m, -np.inf))
        else:
            anchor = members[int(np.argmax(np.abs(load)))]
        if not ok:                       # unidimensionality failed: keep members separate
            for c in members:
                out[c] = Xz[c]
                rows.append({"parcel": c, "anchor": c, "n_members": 1, "members": c, "signs": "+",
                             "min_pair_r": min_r, "alpha": alpha, "pc1_share": pc1, "merged": False})
            continue
        name = f"{anchor}+{len(members) - 1}"
        score = aligned.mean(axis=1)
        out[name] = (score - score.mean()) / (score.std(ddof=1) or 1.0)
        rows.append({"parcel": name, "anchor": anchor, "n_members": len(members),
                     "members": "|".join(members), "signs": "".join("+" if s > 0 else "-" for s in signs),
                     "min_pair_r": min_r, "alpha": alpha, "pc1_share": pc1, "merged": True})
    P = pd.DataFrame(out, index=X.index)
    return P, pd.DataFrame(rows)
