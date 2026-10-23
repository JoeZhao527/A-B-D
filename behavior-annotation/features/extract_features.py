"""Per-trajectory behavior-feature extraction from Act-onomy span annotations.

Turns the batch annotation output (annotations_all.jsonl, or the per-chunk shards
data/batch/annotations_chunk_*.jsonl) into a flat feature table: ONE ROW PER
TRAJECTORY, columns = metadata (model / harness / source / task) + one column per
behavior-style feature. This is the table the trait tests (trait-tests/) consume.

Behavior unit is selectable:
  --unit group   labels = the 10 Act-onomy groups   (compact, robust)
  --unit sub     labels = the 46 sub-actions         (fine-grained, sparser)
The label UNIVERSE is fixed from codebook.py (not from the observed data), so the
column set is identical across runs regardless of which labels a shard happens to use.

Feature blocks (no outcome block: not all trajectories are graded, so success and
step counts are excluded):
  A frequency    A_prop_<u>            share of coded spans per label
                 A_density            spans per annotated step
                 A_entropy            normalized Shannon entropy of the label mix
  B cooccurrence B_lift_<a>|<b>       within-step co-occurrence lift  P(a,b)/(P(a)P(b))
                 B_units_per_step     mean distinct labels co-active in a step
  C transition   C_trans_<a>><b>      P(b present in the next coded step | a present now)
                 C_self_loop          P(a label present now is present in the next coded step)
                 C_step_jaccard       mean Jaccard overlap of consecutive coded steps' label sets
                 C_trans_entropy      mean normalized entropy of the next-step label dist
  D stage        D_centroid_<u>       mean normalized position (0..1) of the label
                 D_early/mid/late_<u> share within each third of the trajectory
                 D_shift_<u>          late-share minus early-share (front/back loading)

Metadata come from turns.parquet (authoritative harness/source), joined on
trajectory_id; model/task fall back to the annotation record if a trajectory is
missing from the metadata table.

Caveats:
  * Per trajectory the derived rates are computed from THAT trajectory's counts
    only (no cross-trajectory pooling) -- short trajectories give sparse/noisy
    transition & lift blocks. Aggregate downstream if you need stable rates.
  * The C block is ACROSS STEPS ONLY: transitions are counted between the label
    SETS of consecutive coded steps (steps without labels are skipped), so they
    follow real time (step order) and do not depend on the order of spans within a
    step. Within-step span order (order_spans) only makes the per-step label lists
    reproducible.

Usage (from the repository root):
  python behavior-annotation/features/extract_features.py \
      --annotations behavior-annotation/data/annotations_all.jsonl \
      --turns behavior-annotation/data/turns.parquet --unit group \
      --out behavior-annotation/out/features_group.csv
  python behavior-annotation/features/extract_features.py --unit sub --blocks AD
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import codebook as CB  # noqa: E402


# --------------------------------------------------------------------------- #
# Label universe (fixed from the taxonomy so columns are stable across runs)
# --------------------------------------------------------------------------- #
def unit_universe(unit: str) -> list[str]:
    if unit == "group":
        return list(CB.groups())
    if unit == "sub":
        subs = []
        for g in CB.groups():
            subs.extend(CB.subs(g))          # 46, names are globally unique
        return subs
    raise ValueError(f"unit must be 'group' or 'sub', got {unit!r}")


def label_of(span: dict, unit: str):
    return span.get("grp") if unit == "group" else span.get("sub")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def order_spans(spans: list[dict], step_text) -> list[dict]:
    """Order a step's spans by the position of their (whitespace-normalized) quote
    in the step text, so the span stream is reproducible instead of following the
    annotator's arbitrary emission order. Stable: ties and unlocatable quotes keep
    emission order; unlocatable quotes sort to the end. No step_text -> unchanged."""
    if step_text is None or len(spans) < 2:
        return spans
    nst = _norm(step_text)
    end = len(nst) + 1

    def pos(s):
        i = nst.find(_norm(s.get("quote")))
        return i if i >= 0 else end
    return sorted(spans, key=pos)


# --------------------------------------------------------------------------- #
# Per-trajectory sufficient statistics
# --------------------------------------------------------------------------- #
def traj_raw(turn_labels: list[list[str]], universe: list[str]) -> dict:
    """Sufficient statistics for one trajectory.

    turn_labels: per annotated step, the list of in-universe labels (empty lists
    kept -- a no-behavior step still counts as a step for density/position).
    """
    uset = set(universe)
    turn_labels = [[l for l in labs if l in uset] for labs in turn_labels]
    n = len(turn_labels)

    r = {
        "gc": Counter(), "n_tags": 0, "n_turns": n,
        "turn_present": Counter(), "pair_present": Counter(), "gpt_sum": 0,
        "trans": Counter(), "cur_present": Counter(), "n_trans": 0, "self_loops": 0,
        "jaccard_sum": 0.0,
        "pos_sum": {g: 0.0 for g in universe}, "pos_n": Counter(),
        "early": Counter(), "mid": Counter(), "late": Counter(),
        "early_tot": 0, "mid_tot": 0, "late_tot": 0,
    }
    if n == 0:
        return r

    stream: list[str] = []
    b1, b2 = n / 3.0, 2.0 * n / 3.0
    for i, labs in enumerate(turn_labels):
        stream.extend(labs)
        pos = i / (n - 1) if n > 1 else 0.5
        third = "early" if i < b1 else ("mid" if i < b2 else "late")
        for g in labs:
            r["pos_sum"][g] += pos
            r["pos_n"][g] += 1
            r[third][g] += 1
        r[f"{third}_tot"] += len(labs)
        s = sorted(set(labs))
        r["gpt_sum"] += len(s)
        for g in s:
            r["turn_present"][g] += 1
        for a, b in combinations(s, 2):
            r["pair_present"][(a, b)] += 1

    r["gc"] = Counter(stream)
    r["n_tags"] = len(stream)
    # across-step transitions: label SET of a coded step -> label set of the next
    # coded step. trans[(a, b)] = # step pairs with a now and b next; cur_present[a] =
    # # step pairs with a now (the row denominator); self_loops = # (pair, a) with a
    # persisting into the next step; jaccard_sum = sum of set overlaps.
    coded = [sorted(set(labs)) for labs in turn_labels if labs]
    r["n_trans"] = max(len(coded) - 1, 0)
    for A, B in zip(coded, coded[1:]):
        sb = set(B)
        for a in A:
            r["cur_present"][a] += 1
            for b in B:
                r["trans"][(a, b)] += 1
            if a in sb:
                r["self_loops"] += 1
        r["jaccard_sum"] += len(set(A) & sb) / len(set(A) | sb)
    return r


# --------------------------------------------------------------------------- #
# Derive features from the sufficient statistics
# --------------------------------------------------------------------------- #
def _norm_entropy(counts, k) -> float:
    tot = sum(counts.values())
    if tot == 0:
        return 0.0
    h = -sum((c / tot) * math.log2(c / tot) for c in counts.values() if c)
    return h / math.log2(k) if k > 1 else 0.0


def derive(raw: dict, universe: list[str], blocks: str) -> dict:
    f: dict = {}
    nt, ntu, ntr = raw["n_tags"], raw["n_turns"], raw["n_trans"]

    if "A" in blocks:
        for g in universe:
            f[f"A_prop_{g}"] = raw["gc"][g] / nt if nt else 0.0
        f["A_density"] = nt / ntu if ntu else 0.0
        f["A_entropy"] = _norm_entropy(raw["gc"], len(universe))

    if "B" in blocks:
        # denom = # steps that carry >=1 label (a code can only co-occur there)
        n_coded = raw.get("_n_coded", 0)
        for a, b in combinations(universe, 2):
            pa, pb = raw["turn_present"][a], raw["turn_present"][b]
            # pair_present is keyed on the SORTED pair (traj_raw walks sorted(set(labs))),
            # while `universe` is in taxonomy order: sort before the lookup or half the
            # pairs silently read 0.
            pab = raw["pair_present"][tuple(sorted((a, b)))]
            f[f"B_lift_{a}|{b}"] = (pab * n_coded) / (pa * pb) if (pa and pb and n_coded) else 0.0
        f["B_units_per_step"] = raw["gpt_sum"] / n_coded if n_coded else 0.0

    if "C" in blocks:
        # rows are conditional PRESENCE probabilities P(b next | a now); a row does not
        # sum to 1 (several labels can follow), so read each cell on its own.
        rowsum = raw["cur_present"]
        for a in universe:
            for b in universe:
                f[f"C_trans_{a}>{b}"] = raw["trans"][(a, b)] / rowsum[a] if rowsum[a] else 0.0
        tot = sum(rowsum.values())            # = sum over step pairs of |current set|
        f["C_self_loop"] = raw["self_loops"] / tot if tot else 0.0
        f["C_step_jaccard"] = raw["jaccard_sum"] / ntr if ntr else 0.0
        te = 0.0
        if tot:
            for a in universe:
                if rowsum[a]:
                    nxt = Counter({b: raw["trans"][(a, b)] for b in universe if raw["trans"][(a, b)]})
                    te += (rowsum[a] / tot) * _norm_entropy(nxt, len(universe))
        f["C_trans_entropy"] = te

    if "D" in blocks:
        for g in universe:
            f[f"D_centroid_{g}"] = raw["pos_sum"][g] / raw["pos_n"][g] if raw["pos_n"][g] else np.nan
            pe = raw["early"][g] / raw["early_tot"] if raw["early_tot"] else 0.0
            pm = raw["mid"][g] / raw["mid_tot"] if raw["mid_tot"] else 0.0
            pl = raw["late"][g] / raw["late_tot"] if raw["late_tot"] else 0.0
            f[f"D_early_{g}"], f[f"D_mid_{g}"], f[f"D_late_{g}"] = pe, pm, pl
            f[f"D_shift_{g}"] = pl - pe
    return f


# --------------------------------------------------------------------------- #
# Trajectory assembly
# --------------------------------------------------------------------------- #
def feats_for_trajectory(turn_labels, universe, blocks):
    raw = traj_raw(turn_labels, universe)
    raw["_n_coded"] = sum(1 for labs in turn_labels if any(l in set(universe) for l in labs))
    feat = derive(raw, universe, blocks)
    return feat, raw


def load_turns(turns_path: str):
    """-> (meta, step_text). meta: trajectory_id -> {model, harness, source, task_type,
    n_steps}. step_text: (trajectory_id, step_index) -> step_text (for within-step
    ordering)."""
    cols = ["trajectory_id", "model", "harness", "source", "task_type", "n_steps",
            "step_index", "step_text"]
    df = pd.read_parquet(turns_path, columns=cols)
    meta = (df.drop_duplicates("trajectory_id")
              .set_index("trajectory_id")[
                  ["model", "harness", "source", "task_type", "n_steps"]]
              .to_dict("index"))
    step_text = {(t, int(s)): x for t, s, x in
                 zip(df["trajectory_id"], df["step_index"], df["step_text"])}
    return meta, step_text


def load_annotations(paths: list[str]) -> dict:
    """-> {trajectory_id: {step_index: (spans, model, task)}}  (last write wins)."""
    by_traj: dict = defaultdict(dict)
    n_lines = 0
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                by_traj[r["trajectory_id"]][int(r["step_index"])] = (
                    r.get("spans") or [], r.get("model_under_study"), r.get("task_type"))
                n_lines += 1
    print(f"read {n_lines} step records over {len(by_traj)} trajectories from {len(paths)} file(s)")
    return by_traj


def build(annotations, turns, unit, blocks):
    universe = unit_universe(unit)
    has_turns = bool(turns and os.path.exists(turns))
    meta, step_text = load_turns(turns) if has_turns else ({}, {})
    if not meta:
        print(f"[warn] no turns table ({turns}); harness/source blank, "
              f"within-step order falls back to emission order")
    by_traj = load_annotations(annotations)

    rows, missing_meta, no_text = [], 0, 0
    for tid, steps in by_traj.items():
        ordered_si = sorted(steps)
        ordered = [steps[si] for si in ordered_si]                 # (spans, model, task)
        turn_labels = []
        for si, (spans, _m, _t) in zip(ordered_si, ordered):
            st = step_text.get((tid, si))
            if st is None and len(spans) >= 2:
                no_text += 1
            turn_labels.append([label_of(s, unit) for s in order_spans(spans, st)])
        feat, raw = feats_for_trajectory(turn_labels, universe, blocks)

        m = meta.get(tid)
        if m is None:
            missing_meta += 1
        # metadata: prefer the authoritative turns table; fall back to the record
        rec_model = next((mdl for _, mdl, _ in ordered if mdl), None)
        rec_task = next((tt for _, _, tt in ordered if tt), None)
        row = {
            "trajectory_id": tid,
            "model": (m or {}).get("model") or rec_model,
            "harness": (m or {}).get("harness"),
            "source": (m or {}).get("source") or str(tid).split("/", 1)[0],
            "task": (m or {}).get("task_type") or rec_task,
            "n_steps": len(ordered),
            "n_coded_steps": raw["_n_coded"],
            "n_spans": raw["n_tags"],
        }
        row.update(feat)
        rows.append(row)

    if missing_meta:
        print(f"[warn] {missing_meta} trajectories not in metadata table "
              f"(harness blank; source from id prefix)")
    if no_text:
        print(f"[warn] {no_text} multi-span steps had no step_text "
              f"(within-step order fell back to emission order)")
    df = pd.DataFrame(rows).sort_values("trajectory_id").reset_index(drop=True)
    return df


def resolve_annotations(arg) -> list[str]:
    if arg:
        paths = []
        for a in arg:
            paths.extend(sorted(glob.glob(a)) or [a])
        return paths
    here = os.path.join(os.path.dirname(__file__), "..", "data")
    allf = os.path.join(here, "annotations_all.jsonl")
    if os.path.exists(allf):
        return [allf]
    return sorted(glob.glob(os.path.join(here, "batch", "annotations_chunk_*.jsonl")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", nargs="*", default=None,
                    help="jsonl path(s)/glob(s); default annotations_all.jsonl or batch shards")
    ap.add_argument("--turns", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "turns.parquet"),
        help="metadata source (harness/source/model/task) keyed by trajectory_id")
    ap.add_argument("--unit", choices=["group", "sub"], default="group")
    ap.add_argument("--blocks", default="auto",
                    help="which feature blocks (subset of ABCD). 'auto' = ABCD for "
                         "group, AD for sub (B/C explode to ~3k cols at sub level)")
    ap.add_argument("--out", default=None, help="output csv (default ../out/features_<unit>.csv)")
    a = ap.parse_args()

    blocks = a.blocks.upper()
    if a.blocks == "auto":
        blocks = "ABCD" if a.unit == "group" else "AD"
    assert set(blocks) <= set("ABCD"), "blocks must be a subset of ABCD"

    ann = resolve_annotations(a.annotations)
    if not ann:
        sys.exit("no annotation files found")
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "out", f"features_{a.unit}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    print(f"unit={a.unit} blocks={blocks} universe={len(unit_universe(a.unit))}")
    df = build(ann, a.turns, a.unit, blocks)
    df.to_csv(out, index=False)
    nfeat = sum(1 for c in df.columns if c[:2] in ("A_", "B_", "C_", "D_")
                or c.split("_", 1)[0] in ("A", "B", "C", "D"))
    meta_cols = ["trajectory_id", "model", "harness", "source", "task",
                 "n_steps", "n_coded_steps", "n_spans"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    print(f"wrote {len(df)} trajectories x {len(feat_cols)} features "
          f"(+{len(meta_cols)} metadata cols) -> {out}")
    print(f"coverage: harness set on {df['harness'].notna().sum()}/{len(df)}, "
          f"models={df['model'].nunique()} harnesses={df['harness'].nunique()} "
          f"tasks={df['task'].nunique()} sources={df['source'].nunique()}")


if __name__ == "__main__":
    main()
