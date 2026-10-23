"""T1/T2/T3 trait tests on a per-trajectory feature table (functional or LIWC).

Same three gates for both behaviour families; see traitlib/tests.py for the
definitions. A feature is a TRAIT iff T1>=t1 and T2>=t2 and T3>=t3.

Usage (from the repository root):
  python trait-tests/run_trait_tests.py --features trait-tests/out/features_functional_adj.csv \
      --out trait-tests/out/trait_tests_functional.csv --workers 8
"""
import argparse
import os
import sys

if any(a == "--workers" or a.startswith("--workers=") for a in sys.argv[1:]):
    # forked workers on a shared core budget: BLAS thread pools would spin against each other
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traitlib.io import load_features                       # noqa: E402
from traitlib.tests import run_tests                        # noqa: E402


def main(args):
    df = load_features(args.features)
    res = run_tests(df, min_cell=args.min_cell, min_ctx=args.min_ctx,
                    min_models_per_harness=args.min_models_per_harness,
                    t1=args.t1, t2=args.t2, t3=args.t3, workers=args.workers)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_csv(args.out, index=False)

    def npass(col, thr):
        return int((res[col] >= thr).sum())
    print(f"\n=== gate pass counts (of {len(res)} tested) ===")
    print(f"  T1 >= {args.t1}: {npass('T1_reliability', args.t1)}")
    print(f"  T2 >= {args.t2}: {npass('T2_consistency', args.t2)}")
    print(f"  T3 >= {args.t3}: {npass('T3_icc_model', args.t3)}")
    traits = res[res["is_trait"]]
    print(f"  ALL THREE (trait): {len(traits)}")
    print(f"\n=== TRAIT features (T1>={args.t1} & T2>={args.t2} & T3>={args.t3}) ===")
    import pandas as pd
    with pd.option_context("display.max_rows", 60, "display.width", 140):
        show = traits[["feature", "block", "mean", "T1_reliability",
                       "T2_consistency", "T3_icc_model"]]
        print(show.to_string(index=False) if len(show) else "  (none)")
    print(f"\nwrote {len(res)} rows -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cell", type=int, default=5)
    ap.add_argument("--min-ctx", type=int, default=3)
    ap.add_argument("--min-models-per-harness", type=int, default=2,
                    help="drop trajectories on harnesses used by fewer than this many "
                         "models (collinear with their single model); 1 disables")
    ap.add_argument("--t1", type=float, default=0.7)
    ap.add_argument("--t2", type=float, default=0.3)
    ap.add_argument("--t3", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=1, help="score features in parallel (identical results)")
    main(ap.parse_args())
