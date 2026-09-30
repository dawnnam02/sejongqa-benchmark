"""Bootstrap confidence intervals of Table III (All) and the paired system differences in the paper
(python scripts/paired_ci.py; needs numpy).

Same resampling as the scorer used for the paper (code/27_채점.py): 2,000 resamples; units are minimal
pairs (Identity, Temporal) or items (Multi-hop); resampling is stratified by category; the generator is
numpy default_rng([20260929, crc32(group name)]), so every system uses the same resamples for a group
and differences are paired. Group names are the Korean labels of the original scorer because they seed
the generator. Percentile intervals (2.5, 97.5). Values are printed in percent and compared with the paper.
"""
import collections
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import em  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B, SEED = 2000, 20260929
PAIR_CATS = ("인물판정", "시간추론")
ALL = "전체"
GROUPS = {"Identity": "인물판정", "Temporal": "시간추론", "Multi-hop": "멀티홉", "All": ALL,
          "same person": "인물판정 · 같은 사람", "different persons": "인물판정 · 다른 사람",
          "2-page": "멀티홉 · 2-page chain (1 relation)", "7-page": "멀티홉 · 7-page chain (6 relations)"}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


GOLD = load(os.path.join(HERE, "data", "answers.jsonl"))


def rows(system):
    pred = {a["id"]: a.get("answer", "") for a in load(os.path.join(HERE, "runs", system, "answers.jsonl"))}
    return [{"id": g["id"], "category": g["category"], "sub": g.get("sub") or "", "pair_id": g.get("pair_id"),
             "em": em(pred[g["id"]], g["answer"], g.get("answer_aliases"))} for g in GOLD if g["id"] in pred]


def group(rs, name):
    if name == ALL:
        return rs
    if name.startswith("인물판정 · "):
        return [r for r in rs if r["category"] == "인물판정" and r["sub"] == name.split(" · ")[1]]
    if name.startswith("멀티홉 · "):
        k = int(name.split("-page")[0][-1])
        return [r for r in rs if r["category"] == "멀티홉" and r["sub"] == "%dhop" % k]
    return [r for r in rs if r["category"] == name]


class Boot:
    def __init__(self, rs, name):
        cl = collections.OrderedDict()
        for r in sorted(rs, key=lambda r: (r["category"], r["pair_id"] or r["id"], r["id"])):
            key = r["pair_id"] if r["category"] in PAIR_CATS and r["pair_id"] else r["id"]
            cl.setdefault((r["category"], key), []).append(r)
        self.keys = list(cl)
        members = list(cl.values())
        self.sums = np.array([sum(x["em"] for x in c) for c in members], dtype=float)
        self.cnts = np.array([len(c) for c in members], dtype=float)
        rng = np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8"))])
        strata = collections.OrderedDict()
        for i, (cat, _) in enumerate(self.keys):
            strata.setdefault(cat, []).append(i)
        parts = [np.asarray(ix)[rng.integers(0, len(ix), size=(B, len(ix)))] for ix in strata.values()]
        self.idx = np.concatenate(parts, axis=1)

    def point(self):
        return self.sums.sum() / self.cnts.sum()

    def dist(self):
        return self.sums[self.idx].sum(1) / self.cnts[self.idx].sum(1)


def ci(d):
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return lo, hi


def one(system, label):
    name = GROUPS[label]
    b = Boot(group(rows(system), name), name)
    return (b.point(),) + ci(b.dist())


def diff(sa, sb, label):
    name = GROUPS[label]
    ra, rb = rows(sa), rows(sb)
    ids = {r["id"] for r in ra} & {r["id"] for r in rb}
    ga = group([r for r in ra if r["id"] in ids], name)
    gb = group([r for r in rb if r["id"] in ids], name)
    A, Bt = Boot(ga, name), Boot(gb, name)
    assert A.keys == Bt.keys and np.array_equal(A.idx, Bt.idx)
    return (A.point() - Bt.point(),) + ci(A.dist() - Bt.dist())


# paper values in percent: (point, low, high); None = not reported
TABLE_III = {  # Table III: All with 95% CI
    ("closed", "All"): (1.8, 1.0, 2.8), ("vanilla", "All"): (37.4, 34.5, 40.5), ("vanilla_bm25", "All"): (40.6, 37.7, 43.6),
    ("graphrag", "All"): (17.7, 15.3, 20.3), ("graphrag_basic", "All"): (15.2, 12.7, 17.6), ("gold", "All"): (74.7, 71.7, 77.4),
    ("gold_gpt41", "All"): (80.9, 78.3, 83.5),  # Section V: Gold Evidence re-read with gpt-4.1
}
DIFFS = {
    ("gold", "vanilla", "All"): (37.2, 33.6, 40.7), ("gold", "vanilla_bm25", "All"): (34.1, None, None),
    ("gold", "vanilla_bm25", "same person"): (13.3, 5.3, 21.3), ("gold", "vanilla_bm25", "Temporal"): (54.0, 47.7, 60.0),
    ("vanilla_bm25", "vanilla", "All"): (3.2, 0.9, 5.4),
    ("vanilla", "graphrag", "All"): (19.7, 16.5, 22.7), ("vanilla", "graphrag_basic", "All"): (22.3, 19.2, 25.4),
    ("graphrag", "graphrag_basic", "Identity"): (9.0, 1.7, 16.7), ("graphrag", "graphrag_basic", "All"): (2.6, -0.4, 5.8),
    ("graphrag_ctx", "graphrag", "All"): (0.5, -2.5, 3.5),
    ("gold_gpt41", "gold", "All"): (6.3, 3.7, 8.9), ("gold_gpt41", "gold", "Temporal"): (11.3, 6.3, 16.7),
}
# Section V: every item type shows a gap to Gold Evidence whose CI excludes zero for every retrieval pipeline
GAP_TYPES = ("same person", "different persons", "Temporal", "Multi-hop")
GAP_SYSTEMS = ("vanilla", "vanilla_bm25", "graphrag", "graphrag_basic")


def check(label, got, want):
    got = [100 * x for x in got]
    ok = all(w is None or abs(g - w) < 0.051 for g, w in zip(got, want))
    shown = "%6.1f [%6.1f, %6.1f]" % tuple(got) if want[1] is not None else "%6.1f" % got[0]
    print("%s %-45s %s   paper %s" % ("PASS" if ok else "FAIL", label, shown, want))
    return ok


def main():
    bad = 0
    for (s, g), want in TABLE_III.items():
        bad += not check("EM %s %s" % (s, g), one(s, g), want)
    for (a, b, g), want in DIFFS.items():
        bad += not check("EM %s - %s %s" % (a, b, g), diff(a, b, g), want)
    lows = [(s, g, diff("gold", s, g)[1]) for s in GAP_SYSTEMS for g in GAP_TYPES]
    ok = all(lo > 0 for _, _, lo in lows)
    bad += not ok
    print("%s Gold minus each pipeline, 4 item types x 4 pipelines: lowest lower CI bound %.1f (paper: all exclude zero)" % ("PASS" if ok else "FAIL", 100 * min(lo for _, _, lo in lows)))
    print("failures:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
