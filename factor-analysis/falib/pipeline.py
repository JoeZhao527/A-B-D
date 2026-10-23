"""One factor-analysis run: trait matrix -> parcels -> adequacy -> k -> extraction ->
rotation -> scores, returned as an FAResult that can write itself to a directory."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .adequacy import bartlett, kmo
from .config import FAConfig
from .extract import extract
from .parcel import build_parcels, zscore
from .retention import choose_k
from .rotate import rotate
from .scores import regression_scores

FAMILY_PREFIX = {"functional": "F", "linguistic": "L", "liwc": "L"}


def prefixed_names(mtm: pd.DataFrame) -> list[str]:
    """'F:<feature>' / 'L:<feature>' from the trait matrix's family column."""
    if "family" not in mtm.columns:
        return [str(i) for i in mtm.index]
    return [f"{FAMILY_PREFIX.get(str(f), str(f)[:1].upper())}:{i}" for i, f in zip(mtm.index, mtm["family"])]


@dataclass
class FAResult:
    config: FAConfig
    meta: dict
    X: pd.DataFrame                 # models x traits (raw adjusted means, NaN-free)
    matrix: pd.DataFrame            # models x parcels (z-scored)
    parcels: pd.DataFrame           # one row per parcel
    R: pd.DataFrame                 # parcel correlation matrix
    adequacy: dict
    retention: dict
    k: int
    unrotated: pd.DataFrame
    communalities: pd.Series
    n_iter: int
    pattern: pd.DataFrame
    structure: pd.DataFrame
    phi: pd.DataFrame
    T: np.ndarray
    scores: pd.DataFrame
    weights: pd.DataFrame
    variance: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ---------------------------------------------------------------- reporting
    def factor_names(self):
        return list(self.pattern.columns)

    def summary_md(self, cutoff: float | None = None) -> str:
        cutoff = self.config.loading_cutoff if cutoff is None else cutoff
        m = self.meta
        L = []
        L.append(f"# Factor analysis — {m.get('family', 'all traits')}\n")
        L.append(f"traits from `{m.get('matrix', '')}`; n = {m['n_models']} models; "
                 f"p = {m['p_traits']} traits -> {m['p_parcels']} parcels "
                 f"(parcel {'on' if self.config.parcel.enabled else 'off'}, |r| >= {self.config.parcel.r_threshold}, "
                 f"{self.config.parcel.linkage}); rotation {self.config.rotate.method}; "
                 f"k = {self.k} by {self.retention['rule']}\n")
        merged = self.parcels[self.parcels["merged"]]
        if len(merged):
            L.append("## Parcels (merged)\n")
            L.append("| parcel | members | min pair r | alpha | PC1 |\n|---|---|---|---|---|")
            for r in merged.itertuples():
                L.append(f"| {r.parcel} | {r.members.replace('|', ', ')} | {r.min_pair_r:.2f} | {r.alpha:.2f} | {r.pc1_share:.0%} |")
            L.append("")
        a = self.adequacy
        L.append("## Adequacy\n")
        L.append(f"Bartlett chi2 = {a['bartlett']['chi2']:.1f}, df = {a['bartlett']['df']}, p = {a['bartlett']['p']:.3g}; "
                 f"KMO = {a['kmo_overall']:.3f}\n")
        rt = self.retention
        L.append("## Number of factors\n")
        pa = rt["parallel"]
        L.append("| | " + " | ".join(f"F{i+1}" for i in range(min(8, len(pa['observed'])))) + " |")
        L.append("|---|" + "---|" * min(8, len(pa["observed"])))
        L.append("| observed eigenvalue | " + " | ".join(f"{v:.2f}" for v in pa["observed"][:8]) + " |")
        L.append("| parallel-analysis threshold | " + " | ".join(f"{v:.2f}" for v in pa["threshold"][:8]) + " |")
        L.append(f"\nparallel analysis k = {pa['k']}; Kaiser k = {rt['kaiser']}; MAP k = {rt['map']['k']}; "
                 f"**used k = {self.k} ({rt['rule']})**\n")
        L.append("## Variance\n")
        L.append("| factor | SS pattern | share of p | SS structure |\n|---|---|---|---|")
        for f, r in self.variance.iterrows():
            L.append(f"| {f} | {r['ss_pattern']:.2f} | {r['share_pattern']:.1%} | {r['ss_structure']:.2f} |")
        L.append("")
        L.append(f"## Pattern loadings (|loading| >= {cutoff})\n")
        for f in self.factor_names():
            s = self.pattern[f][self.pattern[f].abs() >= cutoff].sort_values(key=lambda x: -x.abs())
            L.append(f"- **{f}**: " + ", ".join(f"{i} {v:+.2f}" for i, v in s.items()))
        L.append("")
        if self.k > 1:
            L.append("## Factor correlations\n")
            L.append("| | " + " | ".join(self.factor_names()) + " |")
            L.append("|---|" + "---|" * self.k)
            for f in self.factor_names():
                L.append(f"| {f} | " + " | ".join(f"{self.phi.loc[f, g]:.2f}" for g in self.factor_names()) + " |")
            L.append("")
        low = self.communalities[self.communalities < 0.5]
        if len(low):
            L.append("communalities < .5: " + ", ".join(f"{i} {v:.2f}" for i, v in low.items()) + "\n")
        L.append("## Models\n")
        for f in self.factor_names():
            s = self.scores[f].sort_values(ascending=False)
            L.append(f"- **{f}** top: " + ", ".join(f"{i.split('/')[-1]} {v:+.2f}" for i, v in s.head(5).items()))
            L.append(f"  bottom: " + ", ".join(f"{i.split('/')[-1]} {v:+.2f}" for i, v in s.tail(5).items()))
        return "\n".join(L) + "\n"

    def to_dir(self, out: str) -> None:
        os.makedirs(out, exist_ok=True)
        cfg = self.config.to_dict()
        cfg["meta"] = self.meta
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        self.parcels.to_csv(os.path.join(out, "parcels.csv"), index=False)
        self.matrix.to_csv(os.path.join(out, "matrix.csv"))
        with open(os.path.join(out, "adequacy.json"), "w") as f:
            json.dump({"bartlett": self.adequacy["bartlett"], "kmo_overall": self.adequacy["kmo_overall"],
                       "msa": {k: float(v) for k, v in self.adequacy["msa"].items()}}, f, indent=2)
        pa = self.retention["parallel"]
        rt = pd.DataFrame({"eigenvalue": pa["observed"], "pa_threshold": pa["threshold"]},
                          index=pd.RangeIndex(1, len(pa["observed"]) + 1, name="component"))
        mp = self.retention["map"]["curve"]
        rt["map_avg_partial"] = pd.Series(mp, index=pd.RangeIndex(0, len(mp))).reindex(rt.index)
        rt.to_csv(os.path.join(out, "retention.csv"))
        with open(os.path.join(out, "retention.json"), "w") as f:
            json.dump({"k": self.k, "rule": self.retention["rule"], "parallel_k": pa["k"],
                       "kaiser_k": self.retention["kaiser"], "map_k": self.retention["map"]["k"]}, f, indent=2)
        self.pattern.round(4).to_csv(os.path.join(out, "loadings_pattern.csv"))
        self.structure.round(4).to_csv(os.path.join(out, "loadings_structure.csv"))
        self.phi.round(4).to_csv(os.path.join(out, "factor_corr.csv"))
        self.communalities.round(4).rename("communality").to_csv(os.path.join(out, "communalities.csv"))
        self.variance.round(4).to_csv(os.path.join(out, "variance.csv"))
        self.scores.to_csv(os.path.join(out, "factor_scores.csv"))
        with open(os.path.join(out, "summary.md"), "w") as f:
            f.write(self.summary_md())


def run_fa(mtm: pd.DataFrame, config: FAConfig | None = None, anchor_stat: pd.Series | None = None,
           meta: dict | None = None) -> FAResult:
    """mtm: the trait matrix as written by trait-tests (index = feature, optional
    'family' column, one column per model). anchor_stat: per prefixed trait name
    (e.g. ICC) to choose parcel anchors."""
    config = config or FAConfig()
    meta = dict(meta or {})
    names = prefixed_names(mtm)
    X = mtm.drop(columns=[c for c in ("family",) if c in mtm.columns]).T.astype(float)
    X.columns = names
    X.index.name = "model"
    if config.drop_models_with_nan:
        X = X.dropna(axis=0, how="any")
    n, p_traits = X.shape
    if n < 3:
        raise ValueError("fewer than 3 models without missing traits")

    P, parcels = build_parcels(X, config.parcel, anchor_stat)
    Z = zscore(P, ddof=1)
    Zv = Z.values
    p = Zv.shape[1]
    R = np.corrcoef(Zv, rowvar=False)

    b = bartlett(R, n)
    km = kmo(R)
    adequacy = {"bartlett": b, "kmo_overall": km["overall"], "msa": pd.Series(km["msa"], index=Z.columns)}

    ret = choose_k(Zv, config.retention, config.seed)
    k = ret["k"]
    L, h2, n_iter = extract(R, k, config.extract)
    pattern, structure, Phi, T = rotate(L, config.rotate)
    F, W = regression_scores(Zv, R, structure, config.scores)

    fn = [f"F{j + 1}" for j in range(k)]
    pat = pd.DataFrame(pattern, index=Z.columns, columns=fn)
    var = pd.DataFrame({"ss_pattern": (pattern ** 2).sum(0), "share_pattern": (pattern ** 2).sum(0) / p,
                        "ss_structure": (structure ** 2).sum(0)}, index=fn)
    meta.update({"n_models": int(n), "p_traits": int(p_traits), "p_parcels": int(p), "k": int(k)})
    return FAResult(
        config=config, meta=meta, X=X, matrix=Z, parcels=parcels,
        R=pd.DataFrame(R, index=Z.columns, columns=Z.columns), adequacy=adequacy, retention=ret, k=k,
        unrotated=pd.DataFrame(L, index=Z.columns, columns=fn), communalities=pd.Series(h2, index=Z.columns),
        n_iter=n_iter, pattern=pat, structure=pd.DataFrame(structure, index=Z.columns, columns=fn),
        phi=pd.DataFrame(Phi, index=fn, columns=fn), T=T,
        scores=pd.DataFrame(F, index=Z.index, columns=fn), weights=pd.DataFrame(W, index=Z.columns, columns=fn),
        variance=var)
