"""T1/T2/T3 trait tests on a per-trajectory feature table (functional or LIWC).

  T1 aggregation reliability  — split-half of a cell's trajectories, correlate the
     two half-means across cells, Spearman-Brown corrected, averaged over splits.
     "Is this feature's cell-mean reproducible under resampling?"           (>= 0.7)
  T2 cross-task consistency   — mean pairwise Pearson r of models' cell-means across
     task pairs (backbone models present in both). "Does the between-model pattern
     hold across tasks?"                                                    (>= 0.3)
  T3 between-model ICC        — one-way random-effects ICC(1) across models (ANOVA
     method of moments, unequal group sizes) on the feature with the task_type and
     harness main effects removed: ICC = sigma2_model / (sigma2_model + sigma2_within),
     the between-model share of the variance left after the context.       (>= 0.10)

cell = (model, task_type). backbone = models spanning >= min_ctx task types. A feature
is a TRAIT iff T1>=t1 and T2>=t2 and T3>=t3 — reproducible, cross-task stable, AND
discriminating between models.

T3 removes the context offsets with an OLS that carries the model term
(y ~ C(task_type) + C(harness) + C(model)), so the harness dummies cannot absorb
model differences; on an adjusted table (traitlib.adjust) those offsets are ~0 and
T3 reduces to the plain ANOVA ICC of the adjusted values. T3 uses this closed-form
ANOVA estimator rather than a REML fit, so the result does not depend on the
optimizer or the BLAS thread count.
"""
import multiprocessing as mp
import warnings
import zlib

import numpy as np
import pandas as pd

from .filters import backbone_models, cell_key, harness_filter
from .io import feature_cols

warnings.filterwarnings("ignore")


# ---- T1: split-half aggregation reliability (Spearman-Brown), no covariates ----
def t1_reliability(df, feat, min_cell=5, n_splits=10):
    # deterministic per-feature RNG (crc32 of the feature name) so split-half
    # results are reproducible and INDEPENDENT of execution order.
    rng = np.random.RandomState(zlib.crc32(feat.encode("utf-8")) & 0xFFFFFFFF)
    groups = [g[feat].values for _, g in df.groupby("_cell") if len(g) >= min_cell]
    groups = [g for g in groups if np.nanstd(g) >= 0 and len(g) >= min_cell]
    if len(groups) < 5:
        return np.nan
    rs = []
    for _ in range(n_splits):
        a, b = [], []
        for g in groups:
            idx = rng.permutation(len(g)); h = len(g) // 2
            if h < 1:
                continue
            a.append(np.nanmean(g[idx[:h]])); b.append(np.nanmean(g[idx[h:2 * h]]))
        a, b = np.asarray(a), np.asarray(b)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if np.isfinite(r):
            rs.append(2 * r / (1 + r) if (1 + r) != 0 else np.nan)
    return float(np.nanmean(rs)) if rs else np.nan


# ---- T2: cross-task consistency of model cell-means (backbone only) ----
def t2_consistency(df, feat, backbone, min_cell=5, min_models=4):
    sub = df[df["model"].isin(backbone)]
    cell_mean = (sub.groupby(["model", "task_type"])
                 .agg(v=(feat, "mean"), n=(feat, "size")).reset_index())
    cell_mean = cell_mean[cell_mean["n"] >= min_cell]
    mat = cell_mean.pivot(index="model", columns="task_type", values="v")
    tts = list(mat.columns)
    rs = []
    for i in range(len(tts)):
        for j in range(i + 1, len(tts)):
            pair = mat[[tts[i], tts[j]]].dropna()
            if len(pair) >= min_models and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
                rs.append(pair.corr().iloc[0, 1])
    return float(np.nanmean(rs)) if rs else np.nan


# ---- T3: between-model ICC(1), context removed ----
# Fixed-effect controls partialled out before the between-model ICC. harness is the
# agent scaffold: its system-prompt format drives how much a model narrates planning/
# reflection (same model swings ~35x in A_prop_Planning across harnesses), so it must
# be removed or the "trait" is a scaffold artifact. Identifiable because most models
# span >=2 harnesses; harnesses collinear with a single model are dropped upstream.
T3_CONTROLS = ("task_type", "harness")
# kept for traitlib.adjust, which still fits the mixed model for the adjustment offsets
T3_METHODS = ("bfgs", "cg", "powell", "nm", "lbfgs")


def t3_formula(controls=T3_CONTROLS):
    return "y ~ " + " + ".join(f"C({c})" for c in controls)


def anova_icc(y, groups):
    """One-way random-effects ICC(1) by the ANOVA method of moments for unequal group
    sizes (Donner 1986): s2_between = (MSB - MSW) / n0, ICC = s2_between / (s2_between + MSW)."""
    g = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(groups)}).dropna()
    N, k = len(g), g["g"].nunique()
    if k < 2 or N <= k:
        return np.nan
    gm = g.groupby("g")["y"]
    means, ni, grand = gm.mean(), gm.size(), g["y"].mean()
    msb = (ni * (means - grand) ** 2).sum() / (k - 1)
    msw = ((g["y"] - g["g"].map(means)) ** 2).sum() / (N - k)
    n0 = (N - (ni ** 2).sum() / N) / (k - 1)
    s2b = max((msb - msw) / n0, 0.0) if n0 > 0 else 0.0
    return float(s2b / (s2b + msw)) if (s2b + msw) > 0 else np.nan


def _control_offsets(d, controls):
    """Per-row task/harness offsets from y ~ C(controls) + C(model): the model term keeps
    the offsets from absorbing model differences. Centred, so the grand mean is kept."""
    import statsmodels.formula.api as smf
    params = smf.ols(t3_formula(controls) + " + C(model)", d).fit().params
    off = np.zeros(len(d))
    for c in controls:
        coef = {k.split("[T.", 1)[1][:-1]: float(v) for k, v in params.items() if k.startswith(f"C({c})[T.")}
        off += d[c].astype(str).map(coef).fillna(0.0).to_numpy()
    return off - off.mean()


def t3_icc(df, feat, controls=T3_CONTROLS):
    d = (df[[feat, "model", *controls]].dropna().copy()
         .rename(columns={feat: "y"}))
    if d["y"].std() == 0 or d["model"].nunique() < 3:
        return np.nan
    try:
        y = d["y"].to_numpy() - _control_offsets(d, controls)
    except Exception:
        return np.nan
    return anova_icc(y, d["model"])


# ---- per-feature worker (fork-based pool; the table is shared read-only via globals) ----
_W = {}


def _score_one(f):
    df, backbone, min_cell, controls = _W["df"], _W["backbone"], _W["min_cell"], _W["controls"]
    col = df[f]
    if col.notna().sum() == 0 or col.std(skipna=True) in (0, np.nan):
        return f, None
    return f, (t1_reliability(df, f, min_cell), t2_consistency(df, f, backbone, min_cell),
               t3_icc(df, f, controls), float(col.mean(skipna=True)), int(col.notna().sum()))


def run_tests(df, *, min_cell=5, min_ctx=3, min_models_per_harness=2,
              t1=0.7, t2=0.3, t3=0.10, controls=T3_CONTROLS, workers=1, log=print):
    """Score every feature column of `df`; returns the results table
    (feature, block, mean, n_nonnull, T1_reliability, T2_consistency, T3_icc_model,
    is_trait) sorted by is_trait then T3. workers > 1 scores features in parallel
    (results are identical: every feature's statistics are computed independently
    and T1's RNG is seeded per feature)."""
    df = df.copy()
    b_traj, b_mod, b_h = len(df), df["model"].nunique(), df["harness"].nunique()
    df, drop_h = harness_filter(df, min_models_per_harness)
    if min_models_per_harness > 1:
        log(f"harness filter (>= {min_models_per_harness} models/harness): "
            f"dropped {b_traj - len(df)} traj across {len(drop_h)} collinear harness(es); "
            f"models {b_mod} -> {df['model'].nunique()}, harnesses {b_h} -> "
            f"{df['harness'].nunique()}")
        if drop_h:
            log(f"  dropped harnesses: {drop_h}")
    df["_cell"] = cell_key(df)
    backbone = backbone_models(df, min_ctx)
    log(f"trajectories={len(df)}  models={df['model'].nunique()}  "
        f"tasks={df['task_type'].nunique()}  cells={df['_cell'].nunique()}  "
        f"backbone(>= {min_ctx} tasks)={len(backbone)}")

    feats = feature_cols(df)
    log(f"features to test: {len(feats)} (workers={workers})")
    _W.update(df=df, backbone=backbone, min_cell=min_cell, controls=controls)
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.imap(_score_one, feats, chunksize=1)
            scored = []
            for k, r in enumerate(results):
                scored.append(r)
                if (k + 1) % 25 == 0:
                    log(f"  ...{k + 1}/{len(feats)}")
    else:
        scored = []
        for k, f in enumerate(feats):
            scored.append(_score_one(f))
            if (k + 1) % 25 == 0:
                log(f"  ...{k + 1}/{len(feats)}")
    _W.clear()
    rows = []
    for f, res in scored:
        if res is None:
            continue
        r1, r2, r3, mean, n_nonnull = res
        is_trait = (np.isfinite(r1) and r1 >= t1 and
                    np.isfinite(r2) and r2 >= t2 and
                    np.isfinite(r3) and r3 >= t3)
        rows.append({"feature": f,
                     "block": f.split("::", 1)[0] if "::" in f else f.split("_", 1)[0],
                     "mean": mean, "n_nonnull": n_nonnull,
                     "T1_reliability": r1, "T2_consistency": r2,
                     "T3_icc_model": r3, "is_trait": bool(is_trait)})
    return (pd.DataFrame(rows)
            .sort_values(["is_trait", "T3_icc_model"], ascending=[False, False])
            .reset_index(drop=True))
