"""Reader for the wild-trajectory dataset (huggingface.co/datasets/A-B-D/wild-trajectory).

One row per trajectory; the columns are the top-level fields of its Harbor
ATIF-v1.8 document:
  schema_version, session_id, trajectory_id   string
  agent                                       struct {name, version, model_name, extra}
  steps, final_metrics, extra                 JSON string
`agent.extra` is a JSON string as well. `to_doc` turns a row back into the ATIF
document. Everything beyond the ATIF vocabulary (task, outcome, flags, source,
...) is under `extra["wild"]` at the level it belongs to; `meta` collects the
fields the pipeline groups by.
"""
import glob
import json
import os

import pyarrow.parquet as pq

WILD = "wild"
COLUMNS = ["schema_version", "session_id", "trajectory_id", "agent", "steps",
           "final_metrics", "extra"]
JSON_COLUMNS = ("steps", "final_metrics", "extra")


def wild(obj):
    """The `extra["wild"]` block of an ATIF object ({} when absent)."""
    return ((obj or {}).get("extra") or {}).get(WILD) or {}


def data_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "data", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no data/*.parquet under {data_dir}")
    return files


def to_doc(row):
    """Parquet row -> ATIF document (only the columns present in `row`)."""
    doc = {}
    for k in COLUMNS:
        if k not in row:
            continue
        v = row[k]
        if k == "agent" and v is not None:
            v = dict(v)
            v["extra"] = json.loads(v["extra"]) if v.get("extra") is not None else None
        elif k in JSON_COLUMNS and v is not None:
            v = json.loads(v)
        doc[k] = v
    return doc


def iter_docs(data_dir, columns=COLUMNS, batch_size=256):
    """Yield the ATIF documents of the dataset in file order. Pass a subset of
    `columns` (e.g. without "steps") to skip parsing what is not needed."""
    for f in data_files(data_dir):
        for b in pq.ParquetFile(f).iter_batches(batch_size=batch_size, columns=list(columns)):
            for row in b.to_pylist():
                yield to_doc(row)


def meta(doc):
    """Per-trajectory metadata used for grouping (model / harness / task / source)."""
    w = wild(doc)
    task, flags = w.get("task") or {}, w.get("flags") or {}
    agent = doc.get("agent") or {}
    return {"trajectory_id": doc.get("trajectory_id"),
            "source": (w.get("source") or {}).get("slug"),
            "benchmark": task.get("benchmark"),
            "task_type": task.get("task_type"),
            "task_subtype": task.get("task_subtype"),
            "model": agent.get("model_name"),
            "harness": agent.get("name"),
            "reasoning_tier": flags.get("reasoning_tier"),
            "has_user_sim": bool(flags.get("has_user_sim")),
            "score_norm": (w.get("outcome") or {}).get("score_norm")}
