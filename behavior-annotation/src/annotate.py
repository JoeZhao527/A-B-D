"""Hierarchical Act-onomy annotation of agent steps with a small local model
(gpt-oss-120b via a vLLM OpenAI-compatible endpoint), using STRUCTURED OUTPUT so
the model cannot mis-format or pick an invalid label.

Per agent step:
  call 1 : json_schema forces {"spans":[{"quote":str,"grp":<enum 10>}]} (<=3).
           The group is enum-locked; the quote is free text and is validated
           afterwards as a verbatim (whitespace-normalized) substring of the
           step, with a bounded retry.
  call 2 : per span, json_schema forces {"choice":<enum: subactions of its grp>}.
  call 3 : per span, json_schema forces {"choice":<enum: leaves of its sub>}.
Candidate sets shrink each level, and structured output guarantees valid JSON
and valid labels (so the only thing left to check is that the quote is real).

Context: the previous K steps (default 8) plus the step's observation are given
as read-only context in call 1; prior-step labels are not shown.

This module holds the annotation primitives used by batch_annotate.py; run on its own it
annotates the first --limit trajectories of turns.parquet, for a quick check.
Usage (from the repository root):
  python behavior-annotation/src/annotate.py --dry-run --limit 1
  python behavior-annotation/src/annotate.py --base-url http://localhost:8000/v1 \
      --model openai/gpt-oss-120b --limit 30
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
import codebook as CB  # noqa: E402

WINDOW_STEP_CAP = 300
OBS_CAP = 600
STEP_CAP = 4000
RETRY = 2
REASONING_EFFORT = "low"


def _schema(name, obj):
    return {"type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": obj}}


def spans_schema():
    return _schema("spans", {
        "type": "object", "additionalProperties": False,
        "properties": {"spans": {
            "type": "array", "maxItems": 3,
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"quote": {"type": "string"},
                                     "grp": {"type": "string", "enum": CB.groups()}},
                      "required": ["quote", "grp"]}}},
        "required": ["spans"]})


def choice_schema(choices):
    return _schema("choice", {
        "type": "object", "additionalProperties": False,
        "properties": {"choice": {"type": "string", "enum": choices}},
        "required": ["choice"]})


def chat(base, model, messages, response_format, max_tokens):
    body = {"model": model, "messages": messages, "temperature": 0.0,
            "max_tokens": max_tokens, "response_format": response_format,
            "reasoning_effort": REASONING_EFFORT}
    r = requests.post(f"{base}/chat/completions", timeout=600, json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def matched(quote, host):
    return bool(quote) and _norm(quote) in _norm(host)


def build_window(prior_texts, k):
    ctx = prior_texts[-k:] if k else []
    start = len(prior_texts) - len(ctx)
    return "\n".join(f"(prev step {start+i}) {t[:WINDOW_STEP_CAP]}"
                     for i, t in enumerate(ctx)) or "(none)"


# ---------------- level 1: spans + group ----------------
def grp_messages(step_text, obs, window):
    sysp = (
        "You label an AI agent's runtime behavior with a fixed taxonomy of 10 GROUPS:\n"
        + CB.group_menu() +
        "\n\nFrom the STEP text, pick UP TO 3 short quotes (3-15 words each), each a "
        "DISTINCT behavioral move, and give each its group. Quotes must be copied "
        "verbatim from STEP (exact substrings); use CONTEXT/OBSERVATION only to "
        "understand the step, never quote them. If the step shows no clear behavior, "
        "return an empty list.")
    usr = (f"CONTEXT (prior steps — do not quote):\n{window}\n\n"
           f"OBSERVATION before this step (do not quote):\n{obs[:OBS_CAP]}\n\n"
           f"STEP (quote ONLY from here):\n{step_text[:STEP_CAP]}")
    return [{"role": "system", "content": sysp}, {"role": "user", "content": usr}]


def annotate_grp(base, model, step_text, obs, window):
    for _ in range(RETRY + 1):
        out = parse_json(chat(base, model, grp_messages(step_text, obs, window),
                              spans_schema(), 2000))
        spans = (out or {}).get("spans", []) if isinstance(out, dict) else []
        good = [{"quote": s["quote"], "grp": s["grp"]} for s in spans
                if isinstance(s, dict) and matched(s.get("quote"), step_text)]
        if good:
            return good[:3]
        if isinstance(out, dict) and out.get("spans") == []:
            return []                      # model deliberately said "no behavior"
    return []


# ---------------- level 2/3: per-span closed-set choice ----------------
def choose(base, model, quote, step_text, grp, sub, choices, level):
    tag = "SUBACTION" if level == "sub" else "LEAF"
    ctx = f"group={grp}" + (f", subaction={sub}" if sub else "")
    sysp = (f"Choose the single best {tag} for the quoted behavior from the allowed "
            f"list. Return {{\"choice\": <one value>}}.")
    usr = (f"STEP:\n{step_text[:STEP_CAP]}\n\nQUOTE: \"{quote}\"  ({ctx})\n\n"
           f"Allowed {tag}s:\n" + "\n".join(f"- {c}" for c in choices))
    out = parse_json(chat(base, model, [{"role": "system", "content": sysp},
                                        {"role": "user", "content": usr}],
                          choice_schema(choices), 600))
    pick = (out or {}).get("choice") if isinstance(out, dict) else None
    return pick if pick in choices else (choices[0] if choices else None)


def annotate_step(base, model, step_text, obs, window):
    spans = annotate_grp(base, model, step_text, obs, window)
    res = []
    for s in spans:
        sub = choose(base, model, s["quote"], step_text, s["grp"], None,
                     CB.subs(s["grp"]), "sub")
        leaf = choose(base, model, s["quote"], step_text, s["grp"], sub,
                      CB.leaves(s["grp"], sub), "leaf")
        res.append({"quote": s["quote"], "grp": s["grp"], "sub": sub, "leaf": leaf})
    return res


def run(args):
    turns = args.turns or os.path.join(os.path.dirname(__file__), "..", "data", "turns.parquet")
    df = pd.read_parquet(turns)
    if args.traj:                       # first N trajectories (all their steps)
        keep = set(df["trajectory_id"].drop_duplicates().head(args.traj))
        df = df[df["trajectory_id"].isin(keep)]
    elif args.limit:                    # or first N steps
        df = df.head(args.limit)
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", args.out)

    # Every step's input is knowable upfront: the window is the prior K steps'
    # TEXT (labels are not fed), so all steps are independent -> build the whole
    # task list and run it through a thread pool (vLLM serves the requests
    # concurrently; each step's 3 calls stay sequential within its own thread).
    tasks = []
    for tid, g in df.groupby("trajectory_id", sort=False):
        g = g.sort_values("step_index")
        texts = list(g["step_text"])
        for i, r in enumerate(g.itertuples(index=False)):
            if args.dry_run:
                _dry(r.step_text, r.obs_before, build_window(texts[:i], args.k))
                return
            tasks.append((tid, int(r.step_index), r.step_text, r.obs_before,
                          build_window(texts[:i], args.k), r.model, r.task_type))

    def work(t):
        tid, si, st, obs, win, mdl, tt = t
        try:
            spans = annotate_step(args.base_url, args.model, st, obs, win)
        except Exception as e:          # one bad step must not kill the batch
            spans = []
            print(f"  step {tid}#{si} failed: {type(e).__name__}", flush=True)
        return {"trajectory_id": tid, "step_index": si, "model_under_study": mdl,
                "task_type": tt, "spans": spans}

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(work, tasks), 1):
            results.append(res)
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(tasks)} steps  {rate:.1f} steps/s  "
                      f"eta {(len(tasks)-i)/rate:.0f}s", flush=True)
    dt = time.time() - t0
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ntr = df["trajectory_id"].nunique()
    print(f"annotated {len(tasks)} steps over {ntr} trajectories in {dt:.0f}s "
          f"({len(tasks)/dt:.1f} steps/s, {dt/max(1,ntr):.2f}s/traj, workers={args.workers}) "
          f"-> {out_path}")


def _dry(step_text, obs, window):
    print("=== CALL 1 messages (json_schema: spans[{quote, grp:enum10}], maxItems 3) ===")
    for m in grp_messages(step_text, obs, window):
        print(f"[{m['role']}]\n{m['content'][:1200]}\n")
    print("=== CALL 1 response_format ==="); print(json.dumps(spans_schema(), indent=1)[:600])
    print("\n=== CALL 2 (per span) example: choice-schema over Planning's subactions ===")
    print(json.dumps(choice_schema(CB.subs("Planning")), indent=1)[:500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=32, help="concurrent annotation threads")
    ap.add_argument("--traj", type=int, default=None, help="annotate first N trajectories")
    ap.add_argument("--limit", type=int, default=None, help="or first N steps")
    ap.add_argument("--turns", default=None, help="turns parquet (default data/turns.parquet)")
    ap.add_argument("--out", default="annotations_smoke.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    run(ap.parse_args())
