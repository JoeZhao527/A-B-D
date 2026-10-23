"""Load the Act-onomy codebook (10 action / 46 sub / 120 leaf) from the CSV, and
serve the candidate menus the hierarchical annotator shows the model at each
level. Group-level one-line definitions are hard-coded (the CSV carries only
leaf definitions); sub/leaf menus come from the CSV.
"""
import collections
import csv
import functools
import os

CSV = os.path.join(os.path.dirname(__file__), "..", "codebook",
                   "act-onomy_taxonomy.csv")

# 10 top-level groups (names exactly as in the v4.2 codebook) + one-line defs.
GROUP_DEFS = collections.OrderedDict([
    ("Grounding", "Interacting with users, physical world, digital interfaces, peer agents; augmenting with external computation"),
    ("Retrieval", "Pulling info from skill libraries, corpora, knowledge bases, the open web"),
    ("Reasoning", "Internal cognition: generating, analysing, explaining, distilling, inferring, comparing/ranking, contextualising, combining, filtering"),
    ("Planning", "Decomposing tasks, formulating workflows, selecting strategies, modifying plans"),
    ("Evaluate", "Checking outputs against gold, goals/requirements/constraints, or quality criteria without ground truth"),
    ("Deciding", "Picking among options, accepting/rejecting, deciding under uncertainty"),
    ("Executing", "Externalised action: executing planned strategy, executing debug, terminating"),
    ("Reflection", "In-episode reflection on errors, self-outcomes, external feedback (no policy change)"),
    ("Learning", "Persistent change to reasoning, grounding, knowledge, LLM parameters, instructions"),
    ("Memory", "Storing, updating, discarding, consolidating, reading working / episodic / semantic memory"),
])


@functools.lru_cache(maxsize=1)
def _load():
    tree = collections.OrderedDict()          # grp -> {sub -> [leaf, ...]}
    defs = {}                                  # (grp, sub, leaf) -> definition
    for r in csv.DictReader(open(CSV)):
        g, s, leaf, d = r["Action"], r["Subaction"], r["Instance"], r["definition"]
        tree.setdefault(g, collections.OrderedDict()).setdefault(s, [])
        if leaf not in tree[g][s]:
            tree[g][s].append(leaf)
        defs[(g, s, leaf)] = d
    return tree, defs


def groups():
    return list(GROUP_DEFS.keys())


def subs(grp):
    return list(_load()[0].get(grp, {}).keys())


def leaves(grp, sub):
    return list(_load()[0].get(grp, {}).get(sub, []))


def definition(grp, sub, leaf):
    return _load()[1].get((grp, sub, leaf), "")


def group_menu():
    """Level-1 menu: the 10 groups with one-line definitions."""
    return "\n".join(f"- {g}: {d}" for g, d in GROUP_DEFS.items())


def sub_menu(grp):
    """Level-2 menu: subactions under a group (with a representative leaf def)."""
    tree = _load()[0]
    out = []
    for s in tree.get(grp, {}):
        leaf0 = tree[grp][s][0]
        out.append(f"- {s}  (e.g. {definition(grp, s, leaf0)[:80]})")
    return "\n".join(out)


def leaf_menu(grp, sub):
    """Level-3 menu: leaves under a subaction, with definitions."""
    return "\n".join(f"- {leaf}: {definition(grp, sub, leaf)[:100]}"
                     for leaf in leaves(grp, sub))


if __name__ == "__main__":
    t, d = _load()
    print(f"groups={len(groups())} subs={sum(len(subs(g)) for g in groups())} "
          f"leaves={len(d)}")
    print("\n=== level-1 group menu ===")
    print(group_menu())
    print("\n=== level-2 sub menu for 'Planning' ===")
    print(sub_menu("Planning"))
    print("\n=== level-3 leaf menu for Planning / Decompose task ===")
    print(leaf_menu("Planning", subs("Planning")[0]))
