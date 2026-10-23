"""Assemble the LIWC-category feature table for the three-layer trait test.

The LIWC-22 output (liwc_raw_output.csv) has one row per trajectory in the SAME
order as the export (trajectories_for_liwc.csv), but it does not carry the string
trajectory id. export_prose_for_liwc.py writes traj_meta.csv alongside the export,
in the same row order, so the metadata are joined back by row position (validated:
corr(LIWC WC, prose_words) = 0.999). The result is a features CSV (trajectory_id,
model, harness, task_type + LIWC category rates) that trait-tests/
(adjust_features.py, run_trait_tests.py) can score.

Non-content LIWC columns (text-length stats and punctuation) are dropped; the ~100
psychological / linguistic category rates and the four summary dimensions
(Analytic, Clout, Authentic, Tone, BigWords) are kept as candidate features.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import argparse

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--liwc", default=os.path.join(HERE, "data", "liwc_raw_output.csv"),
                 help="LIWC-22 output CSV (row order = the export order)")
_ap.add_argument("--meta", default=os.path.join(HERE, "data", "traj_meta.csv"),
                 help="traj_meta.csv written by export_prose_for_liwc.py (same row order)")
_ap.add_argument("--out", default=os.path.join(HERE, "out", "liwc_features.csv"))
_a = _ap.parse_args()
LIWC, META, OUT = _a.liwc, _a.meta, _a.out

# LIWC columns that are text mechanics / punctuation, not behavioural style
DROP = {"Segment", "WC", "WPS", "Dic",
        "AllPunc", "Period", "Comma", "QMark", "Exclam", "Apostro", "OtherP", "Emoji"}

liwc = pd.read_csv(LIWC)
feat_cols = [c for c in liwc.columns if c not in DROP]
liwc = liwc[feat_cols].reset_index(drop=True)

# metadata in the SAME order as the export (already restricted to --min-words)
m = pd.read_csv(META).rename(columns={"id": "trajectory_id"})

assert len(m) == len(liwc), f"row mismatch: meta {len(m)} vs liwc {len(liwc)}"
meta = m[["trajectory_id", "model", "harness", "task_type"]].reset_index(drop=True)
out = pd.concat([meta, liwc], axis=1)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
out.to_csv(OUT, index=False)
print(f"wrote {OUT}: {out.shape[0]} trajectories x {len(feat_cols)} LIWC features "
      f"(+4 meta cols)")
print("features:", ", ".join(feat_cols[:12]), "...")
