"""Export agent prose from the wild-trajectory dataset to a CSV that LIWC-22 can score.

dataset -> extract agent prose -> clean -> trajectories_for_liwc.csv, using
  - extract_prose (agent messages + reasoning, excluding demo steps)        [build_corpus.py]
  - clean_prose  (strip code/URLs/paths/terminal/markup, keep NL prose)     [clean.py]
Trajectories with < --min-words words of prose are dropped.

Outputs (into --out-dir):
  trajectories_for_liwc.csv   id,text        (feed to LIWC-22 --mode wc)
  traj_meta.csv               id,model,task_type,harness,prose_words   (for the join back)

LIWC-22 dropped the string id, so downstream alignment is by row order; keep the
row order stable (do not re-sort either file). Usage (from the repository root):
  python liwc-traits/export_prose_for_liwc.py --subset data/wild-trajectory \
      --out-dir liwc-traits/data --min-words 50
  python liwc-traits/export_prose_for_liwc.py --sample-per-source 5    # quick check
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clean import clean_prose, word_count            # noqa: E402
from build_corpus import iter_rows, extract_prose, DEFAULT_SUBSET  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--sample-per-source", type=int, default=0)
    a = ap.parse_args()

    ft = open(os.path.join(a.out_dir, "trajectories_for_liwc.csv"), "w", newline="")
    fm = open(os.path.join(a.out_dir, "traj_meta.csv"), "w", newline="")
    wt, wm = csv.writer(ft), csv.writer(fm)
    wt.writerow(["id", "text"])
    wm.writerow(["id", "model", "task_type", "harness", "prose_words"])

    kept = seen = 0
    for meta, traj in iter_rows(a.subset, a.sample_per_source):
        seen += 1
        asst, reason, _ = extract_prose(traj)
        prose = (clean_prose(asst) + " " + clean_prose(reason)).strip()
        pw = word_count(prose)
        if pw < a.min_words:
            continue
        tid = meta["trajectory_id"]
        wt.writerow([tid, prose.replace("\r", " ").replace("\n", " ")])
        wm.writerow([tid, meta["model"], meta["task_type"], meta["harness"], pw])
        kept += 1
    ft.close(); fm.close()
    print(f"scanned {seen} trajectories, exported {kept} (prose_words>={a.min_words}) "
          f"-> {a.out_dir}/trajectories_for_liwc.csv")


if __name__ == "__main__":
    main()
