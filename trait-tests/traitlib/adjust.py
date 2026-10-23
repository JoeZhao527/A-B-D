"""Context adjustment: remove the task_type and harness main effects from every
feature before anything downstream looks at model means.

For each feature y (one value per trajectory) we fit the T3 model
    y = mu + alpha[task_type] + beta[harness] + u[model] + e,   u ~ N(0, s2_model)
(statsmodels MixedLM, REML, the same optimizer chain as T3) and subtract the fitted
fixed-effect offsets:
    y_adj = y - (alpha_hat[task_type] + beta_hat[harness] - mean of that offset)
The offset is centred over the table so the grand mean of y_adj equals that of y:
adjusted values read as "at the sample's average task/harness mix". Fitting the
offsets WITH the model random intercept is essential — an OLS on task + harness
alone would absorb part of the model effect into the harness dummies.

Fallback when no optimizer converges: OLS with model as a fixed effect
(y ~ C(task_type) + C(harness) + C(model)); the task/harness coefficients are then
unshrunken but still estimated alongside the model term.

Assumptions (documented, not fixable with this data): the context effect is
additive and identical for all models (no model x harness interaction); a harness
used by a single model must be dropped beforehand (filters.harness_filter).
"""
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd

from .io import feature_cols
from .tests import T3_METHODS, t3_formula

warnings.filterwarnings("ignore")

CONTROLS = ("task_type", "harness")


def level_offsets(params, d, controls):
    """Per-row sum of the fitted dummy coefficients for the row's levels of each control
    (reference levels contribute 0). Works for MixedLM and OLS params alike."""
    off = np.zeros(len(d))
    for c in controls:
        coef = {k.split("[T.", 1)[1][:-1]: float(v) for k, v in params.items()
                if k.startswith(f"C({c})[T.")}
        off += d[c].astype(str).map(coef).fillna(0.0).to_numpy()
    return off


def fit_feature(d, controls=CONTROLS):
    """d: columns y, model, *controls (no NaN). Returns (params, info)."""
    import statsmodels.formula.api as smf
    info = {"n": len(d), "path": None, "method": None,
            "sigma2_model": np.nan, "sigma2_resid": np.nan, "icc_prefit": np.nan}
    md = smf.mixedlm(t3_formula(controls), d, groups=d["model"])
    for method in T3_METHODS:
        try:
            m = md.fit(method=method, reml=True)
            if not getattr(m, "converged", False):
                continue
            g, r = float(m.cov_re.iloc[0, 0]), float(m.scale)
            if g + r <= 0:
                continue
            info.update(path="reml", method=method, sigma2_model=g, sigma2_resid=r,
                        icc_prefit=g / (g + r))
            return m.fe_params, info
        except Exception:
            continue
    # fallback: fixed-effect OLS with the model term
    ols = smf.ols(t3_formula(controls) + " + C(model)", d).fit()
    info.update(path="ols_fe", method="ols")
    return ols.params, info


def adjust_feature(df, feat, controls=CONTROLS, center=True):
    """Returns (y_adj aligned to df.index, effects rows, summary row)."""
    d = df[[feat, "model", *controls]].dropna().rename(columns={feat: "y"})
    y_adj = pd.Series(np.nan, index=df.index, dtype=float)
    base = {"feature": feat, "n": len(d)}
    if len(d) == 0 or d["y"].std() == 0 or d["model"].nunique() < 3:
        y_adj.loc[d.index] = d["y"].to_numpy()
        return y_adj, [], {**base, "path": "identity", "method": None,
                           "sigma2_model": np.nan, "sigma2_resid": np.nan,
                           "icc_prefit": np.nan, "r2_context": 0.0,
                           "mean_raw": float(d["y"].mean()) if len(d) else np.nan,
                           "mean_adj": float(d["y"].mean()) if len(d) else np.nan}
    params, info = fit_feature(d, controls)
    off = level_offsets(params, d, controls)
    if center:
        off = off - off.mean()
    adj = d["y"].to_numpy() - off
    y_adj.loc[d.index] = adj
    effects = [{"feature": feat, "term": c, "level": k.split("[T.", 1)[1][:-1],
                "coef": float(v), "path": info["path"]}
               for c in controls for k, v in params.items() if k.startswith(f"C({c})[T.")]
    vy = float(np.var(d["y"].to_numpy()))
    summary = {**base, **info,
               "r2_context": float(np.var(off) / vy) if vy > 0 else 0.0,
               "mean_raw": float(d["y"].mean()), "mean_adj": float(np.mean(adj))}
    return y_adj, effects, summary


# ---- parallel driver (fork-based; the table is shared read-only via a module global) ----
_DF = None
_CTRL = None


def _work(feat):
    y_adj, effects, summary = adjust_feature(_DF, feat, _CTRL)
    return feat, y_adj.to_numpy(), effects, summary


def adjust_table(df, feats=None, controls=CONTROLS, workers=1, log=print):
    """Adjust every feature column of df (metadata untouched).
    Returns (df_adj, effects_df, summary_df)."""
    global _DF, _CTRL
    feats = list(feats) if feats is not None else feature_cols(df)
    for c in controls:
        if df[c].isna().any():
            raise ValueError(f"control column {c!r} has NaN — cannot adjust")
    _DF, _CTRL = df, tuple(controls)
    out = df.copy()
    effects, summaries = [], []
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            results = pool.imap_unordered(_work, feats)
            for k, (feat, vals, eff, summ) in enumerate(results, 1):
                out[feat] = vals; effects += eff; summaries.append(summ)
                if k % 25 == 0:
                    log(f"  ...{k}/{len(feats)}")
    else:
        for k, feat in enumerate(feats, 1):
            feat, vals, eff, summ = _work(feat)
            out[feat] = vals; effects += eff; summaries.append(summ)
            if k % 25 == 0:
                log(f"  ...{k}/{len(feats)}")
    _DF = _CTRL = None
    eff_df = pd.DataFrame(effects, columns=["feature", "term", "level", "coef", "path"])
    summ_df = (pd.DataFrame(summaries)
               .set_index("feature").loc[feats].reset_index())   # input order
    return out, eff_df, summ_df
