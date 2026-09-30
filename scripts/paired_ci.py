"""Bootstrap confidence intervals of Table III and the paired system differences in the paper
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
TABLE_III = {
    ("closed", "Identity"): (2.7, 1.0, 5.0), ("closed", "Temporal"): (1.0, 0.0, 2.3), ("closed", "Multi-hop"): (1.9, 0.5, 3.5), ("closed", "All"): (1.8, 1.0, 2.8),
    ("vanilla", "Identity"): (67.3, 61.0, 73.3), ("vanilla", "Temporal"): (16.7, 12.3, 21.7), ("vanilla", "Multi-hop"): (30.1, 25.6, 34.9), ("vanilla", "All"): (37.4, 34.5, 40.5),
    ("graphrag", "Identity"): (36.0, 29.3, 42.3), ("graphrag", "Temporal"): (8.7, 5.3, 12.0), ("graphrag", "Multi-hop"): (10.4, 7.5, 13.6), ("graphrag", "All"): (17.7, 15.3, 20.3),
    ("gold", "Identity"): (91.0, 86.7, 94.7), ("gold", "Temporal"): (71.7, 65.3, 77.3), ("gold", "Multi-hop"): (64.0, 59.5, 68.8), ("gold", "All"): (74.7, 71.7, 77.4),
    ("vanilla_bm25", "All"): (40.6, 37.7, 43.6),
}
DIFFS = {
    ("graphrag", "vanilla", "All"): (-19.7, -22.7, -16.5), ("graphrag", "vanilla", "Identity"): (-31.3, -38.0, -24.3),
    ("graphrag", "vanilla", "Temporal"): (-8.0, -12.7, -3.3), ("graphrag", "vanilla", "Multi-hop"): (-19.7, -24.3, -15.2),
    ("gold", "graphrag", "All"): (56.9, 53.4, 60.3), ("gold", "vanilla", "All"): (37.2, 33.6, 40.7),
    ("vanilla", "vanilla_bm25", "All"): (-3.2, -5.4, -0.9),
    ("graphrag", "vanilla", "2-page"): (-43.5, None, None), ("graphrag", "vanilla", "7-page"): (-1.2, -7.1, 4.7),
    ("graphrag_ctx", "graphrag", "All"): (0.5, -2.5, 3.5),
}


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
    print("failures:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
