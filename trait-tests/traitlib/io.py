"""Paths and table conventions for the trait-tests stage.

A *feature table* is one row per trajectory: metadata columns (META_COLS) + one
column per behaviour feature. Two families exist, produced upstream:
  functional  behavior-annotation/out/features_group.csv   (extract_features.py)
  liwc        liwc-traits/out/liwc_features.csv            (build_liwc_features.py)

Everything this stage writes goes to trait-tests/out/:
  features_<family>_adj.csv   adjusted feature table (regenerable; git-ignored)
  adjust_effects_<family>.csv fitted task_type / harness offsets per feature
  adjust_summary_<family>.csv per-feature fit diagnostics
  trait_tests_<family>.csv    T1/T2/T3 per feature + is_trait
  model_trait_matrix.csv      trait x model means (rows: functional then linguistic)
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # trait-tests/
FAMILIES = ("functional", "liwc")

META_COLS = ("trajectory_id", "model", "harness", "source", "task", "task_type",
             "n_steps", "n_coded_steps", "n_spans", "reasoning_tier", "prose_words",
             "log_prose", "benchmark", "task_subtype", "has_user_sim")


def feature_cols(df):
    """Every column that is neither metadata nor an internal helper ('_cell')."""
    return [c for c in df.columns if c not in META_COLS and not c.startswith("_")]


def load_features(path):
    df = pd.read_csv(path)
    if "task_type" not in df.columns and "task" in df.columns:
        df = df.rename(columns={"task": "task_type"})
    return df


OUT_DIR = os.path.join(HERE, "out")


def out_paths(mkdir=False):
    d = OUT_DIR
    if mkdir:
        os.makedirs(d, exist_ok=True)
    p = {"dir": d, "matrix": os.path.join(d, "model_trait_matrix.csv")}
    for fam in FAMILIES:
        p[f"features_adj:{fam}"] = os.path.join(d, f"features_{fam}_adj.csv")
        p[f"effects:{fam}"] = os.path.join(d, f"adjust_effects_{fam}.csv")
        p[f"summary:{fam}"] = os.path.join(d, f"adjust_summary_{fam}.csv")
        p[f"trait_tests:{fam}"] = os.path.join(d, f"trait_tests_{fam}.csv")
    return p
