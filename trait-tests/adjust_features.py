"""Remove the task_type + harness main effects from every feature of a table.

Reads an upstream feature table (functional or LIWC), applies the harness filter
(>= N models per harness), fits y ~ C(task_type) + C(harness) + (1|model) per
feature and writes the same table with adjusted feature values, plus the fitted
offsets and per-feature fit diagnostics. See traitlib/adjust.py.

Usage (from the repository root):
  python trait-tests/adjust_features.py --features behavior-annotation/out/features_group.csv \
      --out trait-tests/out/features_functional_adj.csv \
      --effects trait-tests/out/adjust_effects_functional.csv \
      --summary trait-tests/out/adjust_summary_functional.csv --workers 8
"""
import argparse
import os
import sys

if any(a == "--workers" or a.startswith("--workers=") for a in sys.argv[1:]):
    # forked workers on a shared core budget: BLAS thread pools would spin against each other
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traitlib.adjust import CONTROLS, adjust_table          # noqa: E402
from traitlib.filters import harness_filter                 # noqa: E402
from traitlib.io import feature_cols, load_features         # noqa: E402


def main(args):
    t0 = time.time()
    df = load_features(args.features)
    controls = tuple(args.controls.split(","))
    b_traj, b_mod, b_h = len(df), df["model"].nunique(), df["harness"].nunique()
    df, drop_h = harness_filter(df, args.min_models_per_harness)
    print(f"harness filter (>= {args.min_models_per_harness} models/harness): "
          f"dropped {b_traj - len(df)} traj across {len(drop_h)} harness(es); "
          f"models {b_mod} -> {df['model'].nunique()}, harnesses {b_h} -> {df['harness'].nunique()}")
    if drop_h:
        print(f"  dropped harnesses: {drop_h}")
    feats = feature_cols(df)
    print(f"adjusting {len(feats)} features on {len(df)} trajectories; controls={controls}; "
          f"workers={args.workers}")
    out, eff, summ = adjust_table(df, feats, controls, workers=args.workers)
    for p in (args.out, args.effects, args.summary):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    out.to_csv(args.out, index=False)
    eff.to_csv(args.effects, index=False)
    summ.to_csv(args.summary, index=False)
    paths = summ["path"].value_counts().to_dict()
    print(f"fit paths: {paths}")
    print(f"context R2 (share of trajectory variance removed): median "
          f"{summ['r2_context'].median():.3f}, max {summ['r2_context'].max():.3f}")
    print(f"wrote {args.out}\n      {args.effects}\n      {args.summary}\n"
          f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effects", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--controls", default=",".join(CONTROLS))
    ap.add_argument("--min-models-per-harness", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    main(ap.parse_args())
