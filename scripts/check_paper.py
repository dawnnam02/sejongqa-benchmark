"""Recompute every number of the paper from the released files (standard library only).

    python scripts/check_paper.py

Prints each recomputed value next to the paper's value and exits non-zero on a mismatch.
Percentages are rounded to one decimal place.
"""
import collections
import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from score import em, ABSTAIN  # noqa: E402

fails = 0


def ck(label, got, want):
    global fails
    ok = got == want
    fails += not ok
    print("%s %-66s %s (paper %s)" % ("PASS" if ok else "FAIL", label, got, want))


def jl(*p):
    return [json.loads(l) for l in open(os.path.join(HERE, *p), encoding="utf-8") if l.strip()]


def pct(x, n):
    return round(100.0 * x / n, 1)


Q = {q["id"]: q for q in jl("data", "questions.jsonl")}
G = {g["id"]: g for g in jl("data", "answers.jsonl")}
E = {e["id"]: e for e in jl("data", "evidence.jsonl")}
IDS = list(G)
CAT = {"Alias": lambda g: g["sub"] == "같은 사람", "Confusable": lambda g: g["sub"] == "다른 사람",
       "Temporal": lambda g: g["category"] == "시간추론", "Relation Chain": lambda g: g["category"] == "멀티홉"}
CHAIN = {k: (lambda g, k=k: g["category"] == "멀티홉" and g["sub"] == "%dhop" % k) for k in (2, 3, 5, 7)}

# ---------------------------------------------------------------- Section III
print("== III-A  corpus and evidence articles")
arts = list(csv.DictReader(open(os.path.join(HERE, "data", "article_ids.tsv"), encoding="utf-8"), delimiter="\t"))
ck("retrieval corpus articles", len(arts), 4273)
ck("retrieval corpus characters", sum(int(a["n_chars"]) for a in arts), 2214948)
ev_all = {i for g in G.values() for i in g["evidence_ids"]}
ck("unique evidence articles / all in the corpus", (len(ev_all), ev_all <= {a["id"] for a in arts}), (773, True))

print("== III-B, Table I  composition")
ck("questions", len(IDS), 975)
rows = [("Alias", CAT["Alias"], 150, 75, 29.8, 1.7), ("Confusable", CAT["Confusable"], 150, 75, 23.2, 1.1),
        ("Temporal", CAT["Temporal"], 300, 150, 36.7, 2.0)]
rows += [("%d-document chain" % k, CHAIN[k], n, None, ln, ev) for k, n, ln, ev in
         ((2, 92, 30.0, 1.2), (3, 98, 36.2, 2.4), (5, 100, 61.2, 4.8), (7, 85, 84.4, 7.1))]
rows += [("All", lambda g: True, 975, 300, 39.6, 2.5)]
for name, f, n, npair, ln, ev in rows:
    ids = [i for i in IDS if f(G[i])]
    pairs = len({G[i]["pair_id"] for i in ids if G[i]["pair_id"]}) or None
    got = (len(ids), pairs, round(sum(len(Q[i]["question"]) for i in ids) / len(ids), 1),
           round(sum(len(G[i]["evidence_ids"]) for i in ids) / len(ids), 1))
    ck("Table I %s: count, pairs, avg. length, avg. evidence" % name, got, (n, npair, ln, ev))
fam = collections.Counter(G[i]["relation_family"] for i in IDS if CAT["Temporal"](G[i]))
ck("Temporal expressions: day, month, year, sequence", tuple(fam[k] for k in ("하루", "달", "해", "순서")), (62, 76, 90, 72))

PART = [("이라는", "라는"), ("이라고", "라고"), ("이라", "라"), ("이란", "란"), ("이며", "며"), ("이자", "자"),
        ("으로", "로"), ("은", "는"), ("이", "가"), ("을", "를"), ("과", "와")]


def norm_q(q, name):
    t = q.replace(name, "\u0001")
    for a, b in PART:
        t = t.replace("\u0001" + a, "\u0001" + b)
    return re.sub(r"\s+", "", t)


names = list(csv.DictReader(open(os.path.join(HERE, "data", "identity_names.tsv"), encoding="utf-8"), delimiter="\t"))
same = [r["pair"] for r in names
        if norm_q(Q[r["pair"] + "a"]["question"], r["name_a"]) == norm_q(Q[r["pair"] + "b"]["question"], r["name_b"])]
ck("Identity pairs differing only in name and particle", (len(same), len(names)), (141, 150))

# ---------------------------------------------------------------- runs
def load(run, ids=None):
    out = {}
    for r in jl("runs", run, "answers.jsonl"):
        g = G[r["id"]]
        a = r.get("answer", "")
        out[r["id"]] = {"em": em(a, g["answer"], g["answer_aliases"]), "abst": ABSTAIN in a,
                        "ctx": set(r.get("context_docs") or [])}
    if os.path.exists(os.path.join(HERE, "runs", run, "judged.jsonl")):
        for r in jl("runs", run, "judged.jsonl"):
            out[r["id"]]["judge"] = r["judge"]
    return out


R = {"Gold-context": load("gold"), "Local": load("graphrag"), "Basic": load("graphrag_basic")}
GPT41 = load("gold_gpt41")


def em_of(run, f):
    ids = [i for i in IDS if f(G[i])]
    return pct(sum(run[i]["em"] for i in ids), len(ids))


def pair_em(run, cat):
    pairs = collections.defaultdict(list)
    for i in IDS:
        if G[i]["category"] == cat:
            pairs[G[i]["pair_id"]].append(run[i]["em"])
    return pct(sum(all(v) for v in pairs.values()), len(pairs))


print("== IV-B, Table II  EM by type, Pair EM, abstention, LLM-Eq (all 975)")
T2 = {"Gold-context": (88.7, 93.3, 71.7, 64.0, 74.7, 86.7, 58.7, 1.3, 79.8),
      "Local": (38.0, 34.0, 8.7, 10.4, 17.7, 22.7, 1.3, 39.4, 19.8),
      "Basic": (31.3, 22.7, 8.7, 10.9, 15.2, 14.0, 1.3, 42.3, 16.5)}
for name, run in R.items():
    got = tuple(em_of(run, CAT[c]) for c in ("Alias", "Confusable", "Temporal", "Relation Chain"))
    got += (em_of(run, lambda g: True), pair_em(run, "인물판정"), pair_em(run, "시간추론"),
            pct(sum(run[i]["abst"] for i in IDS), 975), pct(sum(run[i]["judge"] for i in IDS), 975))
    ck("Table II %s" % name, got, T2[name])
ck("gpt-4.1 Gold-context: EM, Pair EM Identity, Pair EM Temporal",
   (em_of(GPT41, lambda g: True), pair_em(GPT41, "인물판정"), pair_em(GPT41, "시간추론")), (80.9, 88.7, 74.0))


def mcnemar(a, b, ids):
    x = sum(1 for i in ids if a[i]["em"] and not b[i]["em"])
    y = sum(1 for i in ids if b[i]["em"] and not a[i]["em"])
    n, k = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0


L, B, GC = R["Local"], R["Basic"], R["Gold-context"]
ck("McNemar Local vs. Basic, all questions: p", round(mcnemar(L, B, IDS), 3), 0.084)
ck("McNemar Local vs. Basic, Identity: p", round(mcnemar(L, B, [i for i in IDS if G[i]["category"] == "인물판정"]), 3), 0.012)
ck("McNemar Local vs. Basic, Temporal / Relation Chain: p >= 0.05",
   tuple(mcnemar(L, B, [i for i in IDS if CAT[c](G[i])]) >= 0.05 for c in ("Temporal", "Relation Chain")), (True, True))

sample = json.load(open(os.path.join(HERE, "runs", "graphrag_ctx", "sample.json"), encoding="utf-8"))["ids"]
CTX = load("graphrag_ctx")
ck("200-question re-read: sample size, EM original, EM re-read",
   (len(sample), pct(sum(L[i]["em"] for i in sample), len(sample)), pct(sum(CTX[i]["em"] for i in sample), len(sample))),
   (200, 14.0, 14.5))

# ---------------------------------------------------------------- Section IV-C
print("== IV-C, Table III  evidence article delivery")
hit = {n: {i: bool(set(G[i]["evidence_ids"]) & R[n][i]["ctx"]) for i in IDS} for n in ("Local", "Basic")}
full = {n: {i: set(G[i]["evidence_ids"]) <= R[n][i]["ctx"] for i in IDS} for n in ("Local", "Basic")}
temp = [i for i in IDS if CAT["Temporal"](G[i])]


def anchor_target(i):
    e = E[i]
    ids = []
    for k in ("anchor", "target"):
        v = e[k]
        ids.append(v if isinstance(v, str) else (v.get("source_id") or v.get("id")))
    return set(ids)


both = {n: pct(sum(anchor_target(i) <= R[n][i]["ctx"] for i in temp), len(temp)) for n in ("Local", "Basic")}
for n, want in (("Local", (31.6, 12.7, 5.0)), ("Basic", (37.2, 12.4, 5.7))):
    ck("Table III %s: Evidence Hit Rate, all delivered, Temporal both" % n,
       (pct(sum(hit[n].values()), 975), pct(sum(full[n].values()), 975), both[n]), want)
fl = [i for i in IDS if full["Local"][i]]
rest = [i for i in IDS if not full["Local"][i]]
ck("Local, all evidence delivered: n, EM; remaining EM",
   (len(fl), pct(sum(L[i]["em"] for i in fl), len(fl)), pct(sum(L[i]["em"] for i in rest), len(rest))), (124, 74.2, 9.5))
ck("Gold-context EM on the same two groups",
   (pct(sum(GC[i]["em"] for i in fl), len(fl)), pct(sum(GC[i]["em"] for i in rest), len(rest))), (91.9, 72.2))


def one_side(run):
    pairs = collections.defaultdict(list)
    for i in IDS:
        if CAT["Alias"](G[i]):
            pairs[G[i]["pair_id"]].append(run[i]["em"])
    return sum(sum(v) == 1 for v in pairs.values())


ck("Alias pairs with exactly one question correct: Local, Gold-context", (one_side(L), one_side(GC)), (21, 5))
ck("Relation Chain EM 2 -> 7 documents: Local", (em_of(L, CHAIN[2]), em_of(L, CHAIN[7])), (30.4, 3.5))
ck("Relation Chain EM 2 -> 7 documents: Gold-context", (em_of(GC, CHAIN[2]), em_of(GC, CHAIN[7])), (90.2, 42.4))
ck("5- and 7-document chains with all evidence delivered: Local, Basic",
   tuple(sum(full[n][i] for i in IDS if CHAIN[5](G[i]) or CHAIN[7](G[i])) for n in ("Local", "Basic")), (0, 0))

print("failures:", fails)
sys.exit(1 if fails else 0)
