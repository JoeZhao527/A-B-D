"""Prose extraction for the linguistic features, used by export_prose_for_liwc.py.

For each trajectory: pull the agent's own natural language from the ATIF
document's `source=agent` steps (`message` +, separately, `reasoning_content`),
excluding demo steps and summary-only / redacted reasoning, and clean it to prose.

Run on its own, it writes the cleaned prose with word counts and metadata (one row
per trajectory) to a parquet file and prints per-source word-count statistics, for
inspecting the cleaning. Usage (from the repository root):
  python liwc-traits/build_corpus.py --sample-per-source 80 --show 12
"""
import argparse
import collections
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from clean import clean_prose, word_count  # noqa: E402
from atif_io import iter_docs, meta as atif_meta, wild as _wild  # noqa: E402

DEFAULT_SUBSET = os.path.join(os.path.dirname(HERE), "data", "wild-trajectory")
META_COLS = ["trajectory_id", "source", "benchmark", "task_type", "task_subtype",
             "model", "harness", "reasoning_tier", "has_user_sim"]


def iter_rows(subset_dir, sample_per_source=0):
    """(metadata, ATIF document) per trajectory in dataset order; with
    sample_per_source, only the first N trajectories of every source."""
    per_src = collections.Counter()
    for doc in iter_docs(subset_dir):
        m = atif_meta(doc)
        if sample_per_source and per_src[m["source"]] >= sample_per_source:
            continue
        per_src[m["source"]] += 1
        yield {c: m[c] for c in META_COLS}, doc


def extract_prose(atif_json):
    """(assistant prose, reasoning prose, n_asst) — agent-authored NL only.

    One ATIF `source=agent` step = one model call. Assistant prose is the
    `message` of the non-demo agent steps (a non-empty message makes the step
    annotatable); reasoning prose is `reasoning_content`
    minus what is not the model's own wording: demo (few-shot) steps and
    reasoning that is only an API summary or redacted (extra.wild.summary_only
    / redacted). Agent steps carry no `scripted` flag (that marks user events).
    n_asst counts agent steps with a non-empty message (the `n_asst_events`
    column)."""
    doc = json.loads(atif_json) if isinstance(atif_json, str) else (atif_json or {})
    asst, reason, n_asst = [], [], 0
    for st in doc.get("steps") or []:
        if st.get("source") != "agent":
            continue
        w = _wild(st)
        if w.get("is_demo"):
            continue
        msg = st.get("message")
        if msg and msg.strip():
            asst.append(msg); n_asst += 1
        rc = st.get("reasoning_content")
        if rc and rc.strip() and not (w.get("summary_only") or w.get("redacted")):
            reason.append(rc)
    return "\n".join(asst), "\n".join(reason), n_asst


def build(subset_dir, out_path, sample_per_source=0, show=0):
    rows = []
    samples = []
    for meta, atif_json in iter_rows(subset_dir, sample_per_source):
        raw_a, raw_r, n_asst = extract_prose(atif_json)
        clean_a = clean_prose(raw_a)
        clean_r = clean_prose(raw_r)
        # agent natural language lives in the message OR the reasoning depending
        # on source/model, so the analysis unit is the COMBINED prose.
        prose_all = (clean_a + " " + clean_r).strip()
        rec = {**meta,
               "n_asst_events": n_asst,
               "clean_asst_words": word_count(clean_a),
               "clean_reason_words": word_count(clean_r),
               "prose_words": word_count(prose_all),
               "prose_asst": clean_a,
               "prose_reason": clean_r,
               "prose_all": prose_all}
        rows.append(rec)
        if show and len(samples) < show and word_count(clean_a) >= 20:
            samples.append((meta["source"], meta["model"], raw_a[:300], clean_a[:400]))

    # write corpus
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)

    # ---- report (COMBINED asst+reason prose = the analysis unit) ----
    import statistics
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r["source"]].append((r["clean_asst_words"], r["clean_reason_words"],
                                    r["prose_words"]))
    print(f"\ncorpus: {len(rows)} trajectories -> {out_path}")
    print(f"{'source':<22}{'n':>6}{'med_asst':>9}{'med_reas':>9}{'med_ALL':>9}"
          f"{'>=50w':>7}{'>=100w':>8}{'zero':>6}")
    def line(name, recs):
        n = len(recs)
        ma = int(statistics.median([x[0] for x in recs])) if n else 0
        mr = int(statistics.median([x[1] for x in recs])) if n else 0
        mall = int(statistics.median([x[2] for x in recs])) if n else 0
        ge50 = 100*sum(x[2] >= 50 for x in recs)/n
        ge100 = 100*sum(x[2] >= 100 for x in recs)/n
        zero = 100*sum(x[2] == 0 for x in recs)/n
        print(f"{name:<22}{n:>6}{ma:>9}{mr:>9}{mall:>9}{ge50:>6.0f}%{ge100:>7.0f}%{zero:>5.0f}%")
    for s in sorted(by_src, key=lambda x: -len(by_src[x])):
        line(s, by_src[s])
    line("ALL", [x for w in by_src.values() for x in w])

    for i, (src, mdl, raw, cln) in enumerate(samples):
        print(f"\n--- sample {i+1} [{src} | {mdl}] ---")
        print(f"  RAW  : {raw!r}")
        print(f"  CLEAN: {cln!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "data", "corpus.parquet"))
    ap.add_argument("--sample-per-source", type=int, default=0)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    build(a.subset, os.path.abspath(a.out), a.sample_per_source, a.show)
