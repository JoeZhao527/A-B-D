"""Segment each trajectory into AGENT STEPS for Act-onomy-style annotation.

Turn model: an agent step = one ATIF `source=agent` step (= one model call:
reasoning_content + message + tool_calls[]) of the trajectory's Harbor
ATIF-v1.8 document. A step is an annotation target iff it is ANNOTATABLE —
`is_annotatable_step` below, the same definition the dataset's selection
used (>= 5 annotatable steps): not a demo (few-shot) step, and carrying
reasoning text, a message, or a tool call. Non-annotatable agent steps (demo,
or empty model turns) are skipped, never merged into a neighbour. Whether a
step is thought+action / thought-only / action-only / multi-action is not
distinguished — the annotator picks verbatim spans from the rendered step text.

Observations are NOT annotation targets. Everything that arrives between two
annotatable steps is the later step's preceding context (`obs_before`, for the
annotation window): observation results of the earlier steps (`[env] …`; image
results are the `<screenshot>` placeholder — pixels are not in the dataset),
user steps (`[user] …`, or `[env] …` when `extra.wild.scripted`: scripted user
text is environment) and system steps of kind `env_event`. System steps of
kind system_prompt / lifecycle / score_event / other are never context
(outcome signals must not leak into behaviour annotation).

Input: the wild-trajectory dataset (one ATIF document per row, read by atif_io). No LLM.
Output: data/turns.parquet, one row per (trajectory, annotatable agent step).
Usage: python turn_segment.py [--limit N] [--sources s1,s2] [--hf-dir DIR]
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from atif_io import iter_docs, meta, wild as _wild  # noqa: E402

HF = os.path.join(ROOT, "data", "wild-trajectory")
OUT = os.path.join(HERE, "..", "data", "turns.parquet")

OBS_CAP = 1200          # keep obs context compact (it is only window context)
# content of an image observation result (the pixels are not in the dataset);
# rendered as the <screenshot> placeholder.
IMAGE_PLACEHOLDER = "[image observation"


def is_annotatable_step(step):
    """An agent step is ANNOTATABLE when it is not a demo (few-shot / warm-up)
    step and carries something to code: reasoning text, a message, or at least
    one tool call."""
    if step.get("source") != "agent":
        return False
    if _wild(step).get("is_demo"):
        return False
    return bool((step.get("reasoning_content") or "").strip()
                or (step.get("message") or "").strip()
                or step.get("tool_calls"))


def render_args(tc):
    a = tc.get("arguments")
    if _wild(tc).get("args_not_object") and isinstance(a, dict) and "_wild_value" in a:
        a = a["_wild_value"]            # non-object args (e.g. a raw command string)
    return a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)


def render_step(st):
    """The agent step as annotation text: [think] / [msg] / one [action] per
    tool call. Reasoning that is only an API summary is tagged [think:summary]."""
    w = _wild(st)
    lines = []
    r = st.get("reasoning_content") or ""
    if r.strip():
        lines.append(f"{'[think:summary]' if w.get('summary_only') else '[think]'} {r}")
    m = st.get("message") or ""
    if m.strip():
        lines.append(f"[msg] {m}")
    for tc in st.get("tool_calls") or []:
        lines.append(f"[action] {tc.get('function_name')}: {render_args(tc)}")
    return "\n".join(lines)


def render_result(r):
    """One observation result as context. Screenshot-style results: the pixels
    are not in the dataset; keep a placeholder (plus any error flag) so the
    window shows an observation happened."""
    w = _wild(r)
    c = r.get("content") or ""
    if not isinstance(c, str):
        c = json.dumps(c, ensure_ascii=False)
    if w.get("modality") in ("image", "mixed"):
        tag = "<screenshot>" if not w.get("error") else "<screenshot: error>"
        c = tag if c.startswith(IMAGE_PLACEHOLDER) else f"{tag} {c}".strip()
    return f"[env] {c}"


def render_user(st):
    who = "env" if _wild(st).get("scripted") else "user"
    return f"[{who}] {st.get('message') or ''}"


def make_step(st, ctx):
    obs = "\n".join(ctx)
    return {
        "step_text": render_step(st),
        "obs_before": obs[:OBS_CAP],
        "obs_before_full_len": len(obs),
        "tools": [tc.get("function_name") for tc in st.get("tool_calls") or []
                  if tc.get("function_name")],
        "has_thought": bool((st.get("reasoning_content") or "").strip()
                            or (st.get("message") or "").strip()),
        "has_action": bool(st.get("tool_calls")),
        "step_id": st.get("step_id"),
    }


def segment(doc):
    """ATIF document -> list of annotatable agent steps, each with the context
    that arrived since the previous one (in step order)."""
    steps, pending = [], []
    for st in doc.get("steps") or []:
        src = st.get("source")
        w = _wild(st)
        if src == "agent":
            if is_annotatable_step(st):
                steps.append(make_step(st, pending))
                pending = []
        elif src == "user":
            if not w.get("is_demo"):
                pending.append(render_user(st))
        elif w.get("kind") == "env_event":       # system: only env events are context
            pending.append(f"[env] {st.get('message') or ''}")
        # observation results sit on the step that owns their cause (or on a
        # system step when they precede any agent step) and arrive after it,
        # so they are context for the NEXT annotatable step — on every step
        # kind, annotatable or not.
        pending += [render_result(r)
                    for r in ((st.get("observation") or {}).get("results") or [])
                    if not _wild(r).get("is_demo")]
    return steps


def trajectories(hf_dir, sources=None):
    """(metadata, ATIF document) per trajectory, in dataset order."""
    for doc in iter_docs(hf_dir):
        m = meta(doc)
        if sources and m["source"] not in sources:
            continue
        yield m, doc


def run(limit=None, sources=None, sample=None, sample_by="task_type", out=OUT,
        max_steps=None, hf_dir=HF):
    def emit(m, steps, rows):
        for i, s in enumerate(steps):
            rows.append({"trajectory_id": m["trajectory_id"], "model": m["model"],
                         "task_type": m["task_type"], "source": m["source"],
                         "harness": m["harness"], "step_index": i,
                         "n_steps": len(steps), **s})

    rows, n_traj = [], 0
    if sample:
        # the ≤max_steps filter needs step counts, so segment first, cache the
        # steps, filter, then stratified-sample trajectory_ids from that pool.
        pool = []            # (metadata, steps) passing the filter
        for m, doc in trajectories(hf_dir, sources):
            steps = segment(doc)
            if max_steps is not None and len(steps) > max_steps:
                continue
            pool.append((m, steps))
        strata = pd.DataFrame({"trajectory_id": [m["trajectory_id"] for m, _ in pool],
                               sample_by: [m[sample_by] for m, _ in pool]})
        keep = []
        for _, g in strata.groupby(sample_by):
            keep += list(g["trajectory_id"].sample(min(sample, len(g)), random_state=0))
        by_id = {m["trajectory_id"]: (m, steps) for m, steps in pool}
        for tid in keep:
            m, steps = by_id[tid]
            emit(m, steps, rows)
            n_traj += 1
        cap = f" (≤{max_steps} steps)" if max_steps is not None else ""
        print(f"stratified sample: {sample} per {sample_by}{cap} -> {n_traj} "
              f"trajectories (eligible pool {len(pool)})")
    else:
        for m, doc in trajectories(hf_dir, sources):
            emit(m, segment(doc), rows)
            n_traj += 1
            if limit and n_traj >= limit:
                break
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(out, index=False)
    print(f"{n_traj} trajectories -> {len(df_out)} agent-steps  ({out})")
    if sample and len(df_out):
        print(f"sample composition by {sample_by} (trajectories):")
        print(df_out.groupby(sample_by)["trajectory_id"].nunique().to_string())
    if len(df_out):
        st = df_out.groupby("trajectory_id").size()
        print(f"steps/trajectory: min={st.min()} med={int(st.median())} "
              f"p90={int(st.quantile(.9))} max={st.max()} mean={st.mean():.1f}")
        print(f"steps with action: {df_out['has_action'].mean()*100:.1f}%  "
              f"thought: {df_out['has_thought'].mean()*100:.1f}%  "
              f"both: {(df_out.has_action & df_out.has_thought).mean()*100:.1f}%  "
              f"thought-only: {(df_out.has_thought & ~df_out.has_action).mean()*100:.1f}%  "
              f"action-only: {(~df_out.has_thought & df_out.has_action).mean()*100:.1f}%")
    return df_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sources", type=str, default=None,
                    help="comma-separated source names to restrict to")
    ap.add_argument("--sample", type=int, nargs="?", const=5, default=None,
                    help="stratified: sample N trajectories per --sample-by "
                         "(bare flag = 5); writes data/turns_sample.parquet")
    ap.add_argument("--sample-by", default="task_type",
                    help="stratum column for --sample (default task_type)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="with --sample, only draw from trajectories whose agent-step "
                         "count is ≤ this (e.g. 10)")
    ap.add_argument("--hf-dir", default=HF,
                    help=f"wild-trajectory dataset dir with data/*.parquet (default {HF})")
    ap.add_argument("--out", default=None,
                    help="output parquet (default data/turns.parquet, or "
                         "data/turns_sample.parquet with --sample)")
    a = ap.parse_args()
    out = a.out or (OUT if a.sample is None else os.path.join(
        os.path.dirname(OUT), "turns_sample.parquet"))
    run(a.limit, a.sources.split(",") if a.sources else None,
        a.sample, a.sample_by, out, max_steps=a.max_steps, hf_dir=a.hf_dir)
