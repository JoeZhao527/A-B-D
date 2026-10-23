"""Exploratory factor analysis over the model x trait matrix.

Reads trait-tests/out/model_trait_matrix.csv (context-adjusted model means of the
traits that passed the three-layer test), optionally merges near-duplicate traits
into parcels, checks adequacy, picks the number of factors, extracts (principal
axis), rotates and scores the models. Everything is written to factor-analysis/out/
(see falib/pipeline.py: FAResult.to_dir).

The defaults are the paper's settings (one FA per family, no parcels, varimax):

  python factor-analysis/run_fa.py
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from falib.config import FAConfig                              # noqa: E402
from falib.pipeline import run_fa                              # noqa: E402


def load_anchor_stat(tdir):
    """ICC per prefixed trait name, from the trait-test tables (missing -> None)."""
    out = {}
    for fam, pre in (("functional", "F"), ("liwc", "L")):
        p = os.path.join(tdir, f"trait_tests_{fam}.csv")
        if os.path.exists(p):
            t = pd.read_csv(p)
            out.update({f"{pre}:{k}": v for k, v in zip(t["feature"], t["T3_icc_model"])})
    return pd.Series(out) if out else None


def build_config(a):
    cfg = FAConfig()
    cfg.parcel.enabled = a.parcel
    cfg.parcel.r_threshold = a.parcel_r
    cfg.parcel.linkage = a.linkage
    cfg.parcel.scope = a.parcel_scope
    cfg.parcel.min_pc1 = a.min_pc1
    cfg.parcel.min_alpha = a.min_alpha
    cfg.retention.rule = "fixed" if a.k else a.rule
    cfg.retention.k_fixed = a.k
    cfg.retention.pa_percentile = a.pa_percentile
    cfg.retention.pa_n_random = a.pa_n_random
    cfg.retention.pa_basis = a.pa_basis
    cfg.retention.pa_data = a.pa_data
    cfg.rotate.method = a.rotation
    cfg.rotate.promax_power = a.promax_power
    cfg.scores.ridge = a.ridge
    cfg.seed = a.seed
    cfg.loading_cutoff = a.loading_cutoff
    return cfg


FAMILY_PREFIX = {"functional": "Fun", "linguistic": "Lin"}


def run_separate(mtm, cfg, anchor, matrix, out):
    """One factor analysis per family (functional, linguistic), each written to out/<family>/,
    plus the merged model x factor table out/factor_scores.csv whose columns are prefixed
    Fun_F1.. / Lin_F1.. so all factor scores are in one table."""
    results, scores, variance, summaries = {}, [], [], []
    for fam in ("functional", "linguistic"):
        sub = mtm[mtm["family"] == fam]
        if sub.empty:
            print(f"[{fam}] no traits — skipped")
            continue
        res = run_fa(sub, cfg, anchor_stat=anchor, meta={"matrix": matrix, "family": fam})
        res.to_dir(os.path.join(out, fam))
        results[fam] = res
        scores.append(res.scores.add_prefix(f"{FAMILY_PREFIX[fam]}_"))
        v = res.variance.copy(); v.insert(0, "family", fam); v.index = [f"{FAMILY_PREFIX[fam]}_{i}" for i in v.index]
        variance.append(v)
        summaries.append(f"# {fam}\n\n" + res.summary_md())
    os.makedirs(out, exist_ok=True)
    merged = pd.concat(scores, axis=1)            # models x (Fun_F1.. Lin_F1..); outer join on model
    merged.index.name = "model"
    merged.to_csv(os.path.join(out, "factor_scores.csv"))
    pd.concat(variance).round(4).to_csv(os.path.join(out, "variance.csv"))
    conf = cfg.to_dict()
    conf["meta"] = {"matrix": matrix, "mode": "separate",
                    "families": {fam: {k: r.meta[k] for k in ("n_models", "p_traits", "p_parcels", "k")}
                                 for fam, r in results.items()},
                    "score_columns": list(merged.columns)}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(conf, f, indent=2)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("# Factor analysis (separate per family)\n\n" + "\n\n".join(summaries))
    return results


def main(a):
    mtm = pd.read_csv(a.matrix).set_index("feature")
    matrix = os.path.relpath(os.path.abspath(a.matrix), ROOT)   # recorded in config.json / summary.md
    cfg = build_config(a)
    anchor = load_anchor_stat(os.path.dirname(os.path.abspath(a.matrix))) if not a.no_icc_anchor else None
    out = a.out
    if a.separate:
        results = run_separate(mtm, cfg, anchor, matrix, out)
        for fam, res in results.items():
            print(f"\n===== {fam} =====\n" + res.summary_md())
        print(f"wrote {out}/ (functional/, linguistic/, merged factor_scores.csv)")
        return
    res = run_fa(mtm, cfg, anchor_stat=anchor, meta={"matrix": matrix})
    res.to_dir(out)
    print(res.summary_md())
    print(f"wrote {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=os.path.join(ROOT, "trait-tests", "out", "model_trait_matrix.csv"),
                    help="model x trait matrix (the trait_tests_*.csv next to it give the ICC anchors)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"), help="output directory")
    ap.add_argument("--separate", action=argparse.BooleanOptionalAction, default=True,
                    help="one factor analysis per family (functional, linguistic), as in the paper; "
                         "writes out/{functional,linguistic}/ and a merged factor_scores.csv (Fun_F*, Lin_F*). "
                         "--no-separate runs one joint analysis over all traits")
    g = ap.add_argument_group("parcels")
    g.add_argument("--parcel", action=argparse.BooleanOptionalAction, default=False,
                   help="merge near-duplicate traits into parcels (off in the paper)")
    g.add_argument("--parcel-r", type=float, default=0.8)
    g.add_argument("--linkage", choices=["complete", "average"], default="complete")
    g.add_argument("--parcel-scope", choices=["all", "within_family"], default="all")
    g.add_argument("--min-pc1", type=float, default=0.70)
    g.add_argument("--min-alpha", type=float, default=0.80)
    g.add_argument("--no-icc-anchor", action="store_true", help="anchor = highest PC1 loading instead of ICC")
    g = ap.add_argument_group("number of factors")
    g.add_argument("--rule", choices=["parallel", "kaiser", "map"], default="parallel")
    g.add_argument("--k", type=int, default=None, help="fix the number of factors")
    g.add_argument("--pa-percentile", type=float, default=95.0)
    g.add_argument("--pa-n-random", type=int, default=500)
    g.add_argument("--pa-basis", choices=["pca", "smc"], default="pca")
    g.add_argument("--pa-data", choices=["normal", "permute"], default="normal")
    g = ap.add_argument_group("rotation")
    g.add_argument("--rotation", choices=["varimax", "promax", "none"], default="varimax")
    g.add_argument("--promax-power", type=int, default=4)
    ap.add_argument("--ridge", type=float, default=0.1, help="ridge on R for the regression factor scores (0 = plain inverse)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loading-cutoff", type=float, default=0.40)
    main(ap.parse_args())
