"""Merge the per-chunk annotation shards into one JSONL for downstream analysis.

Reads every annotations_chunk_*.jsonl under --in-dir, de-duplicates by
(trajectory_id, step_index) (last write wins — a resumed shard may re-emit a
trajectory), writes a single sorted annotations_all.jsonl, and validates coverage
against the segmented turns.parquet so any step a shard missed (crash mid-flush,
walltime, an un-run chunk) is reported and can be re-annotated.

Usage (from the repository root):
  python behavior-annotation/src/batch_collect.py --in-dir behavior-annotation/data/batch \
      --turns behavior-annotation/data/turns.parquet \
      --out behavior-annotation/data/annotations_all.jsonl
"""
import argparse
import collections
import glob
import json
import os

import pandas as pd


def run(args):
    shards = sorted(glob.glob(os.path.join(args.in_dir, "annotations_chunk_*.jsonl")))
    print(f"shards: {len(shards)}")
    recs = {}                       # (tid, step_index) -> record  (last wins)
    dup = 0
    for sh in shards:
        n = 0
        for line in open(sh, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["trajectory_id"], r["step_index"])
            if key in recs:
                dup += 1
            recs[key] = r
            n += 1
        print(f"  {os.path.basename(sh)}: {n} lines")
    print(f"unique (traj,step): {len(recs)}  (deduped {dup} repeats)")

    # coverage vs the segmented corpus
    td = pd.read_parquet(args.turns, columns=["trajectory_id", "step_index"])
    want = set(zip(td["trajectory_id"], td["step_index"].astype(int)))
    have = set(recs)
    missing = want - have
    extra = have - want
    print(f"\ncoverage: corpus steps={len(want)}  annotated={len(have)}  "
          f"missing={len(missing)}  extra(not in corpus)={len(extra)}")
    if missing:
        by_traj = collections.Counter(t for t, _ in missing)
        print(f"  missing spans {len(missing)} steps across {len(by_traj)} trajectories; "
              f"top incomplete:")
        for tid, c in by_traj.most_common(10):
            print(f"    {c:5d} steps missing  {tid}")

    # write merged, ordered
    rows = sorted(recs.values(), key=lambda r: (r["trajectory_id"], r["step_index"]))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} steps -> {args.out}")

    # stats
    nspan = sum(len(r.get("spans") or []) for r in rows)
    nempty = sum(1 for r in rows if not (r.get("spans") or []))
    grp = collections.Counter(s["grp"] for r in rows for s in (r.get("spans") or []))
    src = collections.Counter()
    for r in rows:
        src[str(r["trajectory_id"]).split("/", 1)[0]] += 1
    print(f"stats: spans={nspan}  spans/step={nspan/max(1,len(rows)):.2f}  "
          f"steps-with->=1-span={len(rows)-nempty} "
          f"({100*(len(rows)-nempty)/max(1,len(rows)):.0f}% yield)")
    print("group distribution:", dict(grp.most_common()))
    print("steps by source-prefix:", dict(src.most_common()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "batch"))
    ap.add_argument("--turns", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "turns.parquet"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "annotations_all.jsonl"))
    run(ap.parse_args())
