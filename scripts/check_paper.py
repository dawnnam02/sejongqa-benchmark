"""Recompute the numbers reported in the paper from the released files.

    python scripts/check_paper.py

Each line shows the recomputed value next to the value printed in the paper.
The script exits with status 1 if any value differs. Percentages are rounded to
one decimal place. Standard library only.

Run folders: runs/closed = Parametric Knowledge, runs/gold = Gold-context,
runs/vanilla = Vanilla RAG, runs/graphrag = Local Search,
runs/graphrag_basic = Basic Search, runs/gold_gpt41 = Gold-context with gpt-4.1,
runs/graphrag_ctx = 200-question re-read of Local Search contexts.
"""
import collections
import csv
import json
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
    print("%s %-70s %s (paper %s)" % ("PASS" if ok else "FAIL", label, got, want))


def jl(*p):
    # text mode reads both LF and CRLF files
    with open(os.path.join(HERE, *p), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def tsv(*p):
    with open(os.path.join(HERE, *p), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pct(x, n):
    return round(100.0 * x / n, 1)


def year(article_id):
    # article IDs start with the regnal year: 00 (accession year) .. 32
    return int(article_id[:2])


Q = {q["id"]: q for q in jl("data", "questions.jsonl")}
G = {g["id"]: g for g in jl("data", "answers.jsonl")}
E = {e["id"]: e for e in jl("data", "evidence.jsonl")}
IDS = list(G)
CAT = {"Alias": lambda g: g["sub"] == "같은 사람", "Confusable": lambda g: g["sub"] == "다른 사람",
       "Temporal": lambda g: g["category"] == "시간추론", "Relation Chain": lambda g: g["category"] == "멀티홉"}
CHAIN = {k: (lambda g, k=k: g["category"] == "멀티홉" and g["sub"] == "%dhop" % k) for k in (2, 3, 5, 7)}

# ---------------------------------------------------------------- Section III-A
print("== III-A  source corpus, retrieval corpus, gold articles")
arts = [a["id"] for a in tsv("data", "article_ids.tsv")]
excluded = [a["id"] for a in tsv("data", "excluded_article_ids.tsv")]
early = [a for a in arts if year(a) <= 9]
ck("source corpus = retrieval articles 1418-1427 + excluded articles",
   (len(early) + len(excluded), len(early), len(excluded)), (11275, 4245, 7030))
ck("excluded articles: overlap with retrieval corpus, all from 1418-1427",
   (len(set(excluded) & set(arts)), all(year(a) <= 9 for a in excluded)), (0, True))
ck("retrieval corpus articles", len(arts), 4273)
ck("retrieval corpus articles from later years (regnal years 10-32)", sum(year(a) >= 10 for a in arts), 28)
gold_all = {i for g in G.values() for i in g["evidence_ids"]}
ck("unique gold articles; all in the retrieval corpus; all from 1418-1427",
   (len(gold_all), gold_all <= set(arts), all(year(a) <= 9 for a in gold_all)), (773, True, True))

# ---------------------------------------------------------------- Table I
print("== III-B, Table I  composition by question type")
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
    ck("Table I %s: questions, pairs, avg. len., avg. gold" % name, got, (n, npair, ln, ev))
fam = collections.Counter(G[i]["relation_family"] for i in IDS if CAT["Temporal"](G[i]))
ck("Table I note b, Temporal expressions: day, month, year, order",
   tuple(fam[k] for k in ("하루", "달", "해", "순서")), (62, 76, 90, 72))

# Korean particles that change with the preceding name (with / without final consonant)
PART = [("이라는", "라는"), ("이라고", "라고"), ("이라", "라"), ("이란", "란"), ("이며", "며"), ("이자", "자"),
        ("으로", "로"), ("은", "는"), ("이", "가"), ("을", "를"), ("과", "와")]


def norm_q(q, name):
    t = q.replace(name, "\u0001")
    for a, b in PART:
        t = t.replace("\u0001" + a, "\u0001" + b)
    return re.sub(r"\s+", "", t)


names = tsv("data", "identity_names.tsv")
same = [r["pair"] for r in names
        if norm_q(Q[r["pair"] + "a"]["question"], r["name_a"]) == norm_q(Q[r["pair"] + "b"]["question"], r["name_b"])]
ck("Identity pairs differing only in the name and its particle", (len(same), len(names)), (141, 150))


# ---------------------------------------------------------------- runs
def load(run):
    out = {}
    for r in jl("runs", run, "answers.jsonl"):
        g = G[r["id"]]
        a = r.get("answer", "")
        out[r["id"]] = {"em": em(a, g["answer"], g["answer_aliases"]), "abst": ABSTAIN in a,
                        "ctx": set(r.get("context_docs") or [])}
    return out


R = {"Parametric Knowledge": load("closed"), "Gold-context": load("gold"), "Vanilla RAG": load("vanilla"),
     "Local Search": load("graphrag"), "Basic Search": load("graphrag_basic")}
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


# ---------------------------------------------------------------- Table II
print("== IV-B, Table II  EM by type, Pair EM (Identity, Temporal), abstention; all 975 questions")
T2 = {"Parametric Knowledge": (4.0, 1.3, 1.0, 1.9, 1.8, 0.7, 0.0, 23.1),
      "Gold-context": (88.7, 93.3, 71.7, 64.0, 74.7, 86.7, 58.7, 1.3),
      "Vanilla RAG": (36.0, 24.7, 9.0, 15.2, 17.9, 18.0, 0.7, 33.3),
      "Local Search": (38.0, 34.0, 8.7, 10.4, 17.7, 22.7, 1.3, 39.4),
      "Basic Search": (31.3, 22.7, 8.7, 10.9, 15.2, 14.0, 1.3, 42.3)}
for name, run in R.items():
    got = tuple(em_of(run, CAT[c]) for c in ("Alias", "Confusable", "Temporal", "Relation Chain"))
    got += (em_of(run, lambda g: True), pair_em(run, "인물판정"), pair_em(run, "시간추론"),
            pct(sum(run[i]["abst"] for i in IDS), 975))
    ck("Table II %s" % name, got, T2[name])
ck("Table II note c, gpt-4.1 Gold-context: EM, Pair EM Identity, Pair EM Temporal",
   (em_of(GPT41, lambda g: True), pair_em(GPT41, "인물판정"), pair_em(GPT41, "시간추론")), (80.9, 88.7, 74.0))

L, GC, V = R["Local Search"], R["Gold-context"], R["Vanilla RAG"]
with open(os.path.join(HERE, "runs", "graphrag_ctx", "sample.json"), encoding="utf-8") as f:
    sample = json.load(f)["ids"]
CTX = load("graphrag_ctx")
ck("Local Search 200-question re-read: sample size, EM original, EM re-read",
   (len(sample), pct(sum(L[i]["em"] for i in sample), len(sample)), pct(sum(CTX[i]["em"] for i in sample), len(sample))),
   (200, 14.0, 14.5))

# ---------------------------------------------------------------- Table III
print("== IV-C, Table III  gold article retrieval rates")
SEARCH = ("Vanilla RAG", "Local Search", "Basic Search")
# an article counts as retrieved if any of its chunks reached the answer generator
hit = {n: {i: bool(set(G[i]["evidence_ids"]) & R[n][i]["ctx"]) for i in IDS} for n in SEARCH}
full = {n: {i: set(G[i]["evidence_ids"]) <= R[n][i]["ctx"] for i in IDS} for n in SEARCH}
temp = [i for i in IDS if CAT["Temporal"](G[i])]


def anchor_target(i):
    e = E[i]
    ids = []
    for k in ("anchor", "target"):
        v = e[k]
        ids.append(v if isinstance(v, str) else (v.get("source_id") or v.get("id")))
    return set(ids)


both = {n: pct(sum(anchor_target(i) <= R[n][i]["ctx"] for i in temp), len(temp)) for n in SEARCH}
for n, want in (("Vanilla RAG", (37.2, 12.4, 5.7)), ("Local Search", (31.6, 12.7, 5.0)),
                ("Basic Search", (37.2, 12.4, 5.7))):
    ck("Table III %s: Retrieval Hit Rate, all gold retrieved, Temporal both" % n,
       (pct(sum(hit[n].values()), 975), pct(sum(full[n].values()), 975), both[n]), want)

print("== IV-C  analyses by retrieval status and question type")
fv = [i for i in IDS if full["Vanilla RAG"][i]]
rv = [i for i in IDS if not full["Vanilla RAG"][i]]
ck("Vanilla RAG, all gold retrieved: questions, EM; EM on the rest",
   (len(fv), pct(sum(V[i]["em"] for i in fv), len(fv)), pct(sum(V[i]["em"] for i in rv), len(rv))), (121, 79.3, 9.3))
ck("Gold-context EM on the Vanilla RAG all-gold-retrieved questions", pct(sum(GC[i]["em"] for i in fv), len(fv)), 86.8)
fl = [i for i in IDS if full["Local Search"][i]]
rest = [i for i in IDS if not full["Local Search"][i]]
ck("Local Search, all gold retrieved: questions, EM; EM on the rest",
   (len(fl), pct(sum(L[i]["em"] for i in fl), len(fl)), pct(sum(L[i]["em"] for i in rest), len(rest))), (124, 74.2, 9.5))
ck("Gold-context EM on the same two Local Search groups",
   (pct(sum(GC[i]["em"] for i in fl), len(fl)), pct(sum(GC[i]["em"] for i in rest), len(rest))), (91.9, 72.2))


def one_side(run, cat):
    pairs = collections.defaultdict(list)
    for i in IDS:
        if CAT[cat](G[i]):
            pairs[G[i]["pair_id"]].append(run[i]["em"])
    return sum(sum(v) == 1 for v in pairs.values())


ck("Alias pairs with exactly one correct: Gold-context, Vanilla RAG, Local Search",
   (one_side(GC, "Alias"), one_side(V, "Alias"), one_side(L, "Alias")), (5, 20, 21))
ck("Confusable pairs with exactly one correct: Gold-context, Vanilla RAG, Local Search",
   (one_side(GC, "Confusable"), one_side(V, "Confusable"), one_side(L, "Confusable")), (8, 17, 19))
ck("Temporal Pair EM, Gold-context: gpt-4.1-mini, gpt-4.1",
   (pair_em(GC, "시간추론"), pair_em(GPT41, "시간추론")), (58.7, 74.0))
ck("Relation Chain EM, 2- and 7-document chains: Gold-context", (em_of(GC, CHAIN[2]), em_of(GC, CHAIN[7])), (90.2, 42.4))
ck("Relation Chain EM, 2- and 7-document chains: Vanilla RAG", (em_of(V, CHAIN[2]), em_of(V, CHAIN[7])), (38.0, 0.0))
ck("Relation Chain EM, 2- and 7-document chains: Local Search", (em_of(L, CHAIN[2]), em_of(L, CHAIN[7])), (30.4, 3.5))
ck("5- and 7-document chains with all gold retrieved: Vanilla RAG, Local Search, Basic Search",
   tuple(sum(full[n][i] for i in IDS if CHAIN[5](G[i]) or CHAIN[7](G[i])) for n in SEARCH), (0, 0, 0))

print("failures:", fails)
sys.exit(1 if fails else 0)
