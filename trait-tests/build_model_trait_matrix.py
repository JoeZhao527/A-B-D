"""Build the model x trait matrix: rows = trait features (functional first, then the
LIWC linguistic traits), columns = models, cell = the model's mean level of the
feature over its trajectories on the harness-filtered population the trait tests
use. Identifier columns `family` and `feature` precede the model columns.

Defaults read the adjusted feature tables and trait-test results from out/; pass
explicit paths to build from other tables (e.g. the raw upstream tables for an
unadjusted comparison).

Usage:
  python build_model_trait_matrix.py
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traitlib.filters import harness_filter                 # noqa: E402
from traitlib.io import load_features, out_paths            # noqa: E402


def family_matrix(features_csv, tests_csv, min_models_per_harness=2):
    tt = pd.read_csv(tests_csv)
    traits = (tt[tt["is_trait"].astype(str) == "True"]
              .sort_values("T3_icc_model", ascending=False)["feature"].tolist())
    df, _ = harness_filter(load_features(features_csv), min_models_per_harness)
    mat = df.groupby("model")[traits].mean().T      # rows=feature, cols=model
    mat.index.name = "feature"
    return mat


def main(args):
    p = out_paths(mkdir=True)
    f_feat = args.functional_features or p["features_adj:functional"]
    l_feat = args.liwc_features or p["features_adj:liwc"]
    f_tt = args.functional_tests or p["trait_tests:functional"]
    l_tt = args.liwc_tests or p["trait_tests:liwc"]
    out = args.out or p["matrix"]

    func = family_matrix(f_feat, f_tt, args.min_models_per_harness)
    print(f"functional: {func.shape[0]} traits x {func.shape[1]} models")
    ling = family_matrix(l_feat, l_tt, args.min_models_per_harness)
    print(f"linguistic (LIWC): {ling.shape[0]} traits x {ling.shape[1]} models")

    models = sorted(set(func.columns) | set(ling.columns))
    func = func.reindex(columns=models)
    ling = ling.reindex(columns=models)
    res = pd.concat([func, ling], axis=0)
    res.insert(0, "family", ["functional"] * len(func) + ["linguistic"] * len(ling))
    res = res.reset_index()
    res.to_csv(out, index=False)
    print(f"wrote {res.shape[0]} traits x {len(models)} models (+2 id cols) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--functional-features")
    ap.add_argument("--liwc-features")
    ap.add_argument("--functional-tests")
    ap.add_argument("--liwc-tests")
    ap.add_argument("--out")
    ap.add_argument("--min-models-per-harness", type=int, default=2)
    main(ap.parse_args())
