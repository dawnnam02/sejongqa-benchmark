"""Checks for numbers that appear only in the final paper (standard library only).

    python scripts/final_paper_checks.py

1. Exact McNemar test, GraphRAG Local vs. Basic, on EM (Section IV-B): all 975 questions
   (p = 0.084), Identity (p = 0.012), Temporal and Relation Chain (not significant).
2. Identity pairs whose two questions differ only in the person's name and the attached particle
   (Section III-B: 141 of 150; the other 9 also differ in referring expressions or context).
   Method: replace each name (diagnostics/identity_pairs.tsv, name_a / name_b) by a placeholder,
   map particle variants to one form, remove spaces, and compare the two questions.

Prints each recomputed value next to the paper's value and exits non-zero on a mismatch.
"""
import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from score import em  # noqa: E402

fails = 0


def ck(label, got, want):
    global fails
    ok = got == want
    fails += not ok
    print("%s %-62s %s (paper %s)" % ("PASS" if ok else "FAIL", label, got, want))


def jl(*p):
    return [json.loads(l) for l in open(os.path.join(HERE, *p), encoding="utf-8") if l.strip()]


gold = {g["id"]: g for g in jl("data", "answers.jsonl")}
questions = {q["id"]: q for q in jl("data", "questions.jsonl")}


def em_by_id(run):
    out = {}
    for r in jl("runs", run, "answers.jsonl"):
        g = gold[r["id"]]
        out[r["id"]] = em(r.get("answer", ""), g["answer"], g["answer_aliases"])
    return out


def mcnemar_exact(a, b, ids):
    a_only = sum(1 for i in ids if a[i] and not b[i])
    b_only = sum(1 for i in ids if b[i] and not a[i])
    n, k = a_only + b_only, min(a_only, b_only)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return a_only, b_only, p


local, basic = em_by_id("graphrag"), em_by_id("graphrag_basic")
groups = {
    "All": lambda g: True,
    "Identity": lambda g: g["category"] == "인물판정",
    "Temporal": lambda g: g["category"] == "시간추론",
    "Relation Chain": lambda g: g["category"] == "멀티홉",
}
print("== 1. Exact McNemar test on EM, GraphRAG Local vs. Basic (Local only / Basic only / p)")
expect_p = {"All": 0.084, "Identity": 0.012}
for name, f in groups.items():
    ids = [i for i, g in gold.items() if f(g)]
    a_only, b_only, p = mcnemar_exact(local, basic, ids)
    print("     %-15s n=%d  Local only %d, Basic only %d, p = %.3f" % (name, len(ids), a_only, b_only, p))
    if name in expect_p:
        ck("McNemar p, %s" % name, round(p, 3), expect_p[name])
    else:
        ck("McNemar not significant at 0.05, %s" % name, p >= 0.05, True)

print("== 2. Identity pairs differing only in the name and its particle")
PART = [("이라는", "라는"), ("이라고", "라고"), ("이라", "라"), ("이란", "란"), ("이며", "며"), ("이자", "자"),
        ("으로", "로"), ("은", "는"), ("이", "가"), ("을", "를"), ("과", "와")]


def norm_q(q, name):
    t = q.replace(name, "\u0001")
    for a, b in PART:
        t = t.replace("\u0001" + a, "\u0001" + b)
    return re.sub(r"\s+", "", t)


rows = list(csv.DictReader(open(os.path.join(HERE, "diagnostics", "identity_pairs.tsv"), encoding="utf-8"), delimiter="\t"))
differ = [r["pair"] for r in rows
          if norm_q(questions[r["pair"] + "a"]["question"], r["name_a"]) != norm_q(questions[r["pair"] + "b"]["question"], r["name_b"])]
ck("Identity pairs", len(rows), 150)
ck("pairs differing only in name and particle", len(rows) - len(differ), 141)
print("     the other pairs:", " ".join(differ))

print("failures:", fails)
sys.exit(1 if fails else 0)
