"""Batch Act-onomy annotation worker for one shard of the full corpus.

Each shard is one call with its own --chunk-id (0..num-chunks-1). The full
turns.parquet trajectory list is shuffled with a FIXED seed and sliced into
num-chunks contiguous chunks, so shuffling mixes long/short trajectories and each
chunk gets a statistically even step load without explicit bin-packing. A whole
trajectory always stays in one chunk (the annotation window needs its prior steps).

Robustness for a multi-hour run:
  * incremental write — each COMPLETED trajectory's steps are appended and flushed
    immediately (not one write at the very end), so a crash/walltime loses at most
    the in-flight trajectories;
  * resume — on (re)start, trajectory_ids already present in the chunk's output file
    are skipped, so re-running the same chunk continues where it stopped.

--level sub stops at sub-action (leaf left null), ~30% fewer LLM calls.
Reuses the annotation primitives from annotate.py (single source of truth).

Usage (from the repository root; one call per chunk, e.g. one per GPU node):
  python behavior-annotation/src/batch_annotate.py --turns behavior-annotation/data/turns.parquet \
      --num-chunks 36 --chunk-id 0 --level sub --k 8 --workers 32 \
      --base-url http://localhost:8000/v1 --model openai/gpt-oss-120b \
      --out-dir behavior-annotation/data/batch
"""
import argparse
import collections
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import annotate as A   # noqa: E402  (annotate_grp / choose / build_window / chat)
import codebook as CB  # noqa: E402


def annotate_step(base, model, st, obs, win, level):
    """spans + group, then per span the sub-action (and the leaf unless level==sub)."""
    spans = A.annotate_grp(base, model, st, obs, win)
    res = []
    for s in spans:
        sub = A.choose(base, model, s["quote"], st, s["grp"], None,
                       CB.subs(s["grp"]), "sub")
        leaf = None
        if level != "sub":
            leaf = A.choose(base, model, s["quote"], st, s["grp"], sub,
                            CB.leaves(s["grp"], sub), "leaf")
        res.append({"quote": s["quote"], "grp": s["grp"], "sub": sub, "leaf": leaf})
    return res


def chunk_trajectory_ids(all_tids, num_chunks, chunk_id, seed):
    """FIXED-seed shuffle then contiguous slice -> this chunk's trajectory ids."""
    tids = list(all_tids)
    random.Random(seed).shuffle(tids)
    size = math.ceil(len(tids) / num_chunks)
    return tids[chunk_id * size:(chunk_id + 1) * size]


def run(args):
    df = pd.read_parquet(args.turns, columns=[
        "trajectory_id", "model", "task_type", "step_index", "step_text",
        "obs_before"])
    all_tids = list(dict.fromkeys(df["trajectory_id"]))     # unique, stable order
    mine = set(chunk_trajectory_ids(all_tids, args.num_chunks, args.chunk_id,
                                    args.seed))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"annotations_chunk_{args.chunk_id}.jsonl")

    done = set()                                            # resume: skip finished
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["trajectory_id"])
            except Exception:
                pass
    pending = [t for t in all_tids if t in mine and t not in done]
    print(f"[chunk {args.chunk_id}/{args.num_chunks}] level={args.level} "
          f"my_traj={len(mine)} done={len(done)} pending={len(pending)}", flush=True)
    if not pending:
        print("nothing to do", flush=True)
        return

    dfp = df[df["trajectory_id"].isin(set(pending))]
    tasks, expect = [], {}
    for tid, g in dfp.groupby("trajectory_id", sort=False):
        g = g.sort_values("step_index")
        texts = list(g["step_text"])
        expect[tid] = len(g)
        for i, r in enumerate(g.itertuples(index=False)):
            tasks.append((tid, int(r.step_index), r.step_text, r.obs_before,
                          A.build_window(texts[:i], args.k), r.model, r.task_type))
    print(f"  {len(tasks)} steps over {len(pending)} trajectories", flush=True)

    def work(t):
        tid, si, st, obs, win, mdl, tt = t
        try:
            spans = annotate_step(args.base_url, args.model, st, obs, win, args.level)
        except Exception as e:              # one bad step must not kill the shard
            spans = []
            print(f"  step {tid}#{si} failed: {type(e).__name__}", flush=True)
        return {"trajectory_id": tid, "step_index": si, "model_under_study": mdl,
                "task_type": tt, "spans": spans}

    # ex.map yields results in task order; tasks are grouped by trajectory, so a
    # trajectory's steps arrive contiguously — buffer them and flush the whole
    # trajectory (atomically, sorted by step) the moment it is complete.
    buf = collections.defaultdict(list)
    t0, n, ntraj = time.time(), 0, 0
    with open(out_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(work, tasks):
                n += 1
                tid = res["trajectory_id"]
                buf[tid].append(res)
                if len(buf[tid]) == expect[tid]:
                    for r in sorted(buf.pop(tid), key=lambda x: x["step_index"]):
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    ntraj += 1
                if n % 500 == 0:
                    rate = n / (time.time() - t0)
                    print(f"  {n}/{len(tasks)} steps  {rate:.1f}/s  "
                          f"eta {(len(tasks)-n)/rate:.0f}s  ({ntraj} traj flushed)",
                          flush=True)
    dt = time.time() - t0
    print(f"[chunk {args.chunk_id}] DONE {len(tasks)} steps / {ntraj} traj in {dt:.0f}s "
          f"({len(tasks)/max(1,dt):.1f} steps/s) -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "turns.parquet"))
    ap.add_argument("--num-chunks", type=int, required=True)
    ap.add_argument("--chunk-id", type=int, required=True)
    ap.add_argument("--level", choices=["sub", "leaf"], default="sub")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "batch"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--base-url",
                    default=os.environ.get("BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    a = ap.parse_args()
    assert 0 <= a.chunk_id < a.num_chunks, "chunk-id out of range"
    run(a)
