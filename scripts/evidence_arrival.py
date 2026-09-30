"""Table IV (gold articles in the reader's context) and the Gold-solved re-count of Section V
(python scripts/evidence_arrival.py; standard library).

Context articles per question:
  Vanilla RAG   runs/vanilla/retrieval.jsonl "final" (top 8 fused chunks)
  BM25 only     runs/vanilla/retrieval.jsonl "bm25"[:8]
  Dense only    runs/vanilla/retrieval.jsonl "dense"[:8]  (a ranking diagnostic; no QA run)
  GraphRAG      runs/graphrag/answers.jsonl and runs/graphrag_basic/answers.jsonl "context_docs"
A chunk ID is "<article ID>#<n>". EM uses scripts/score.py. Each line prints the recomputed value and
the value in the paper; the script exits with a non-zero status on a mismatch.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import em  # noqa: E402

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD = [0]


def jl(*a):
    return [json.loads(l) for l in open(os.path.join(H, *a), encoding="utf-8") if l.strip()]


def pct(k, n):
    return round(100.0 * k / n, 1)


def ck(label, got, want):
    ok = got == want
    BAD[0] += not ok
    print("%s %-58s %s (paper %s)" % ("PASS" if ok else "FAIL", label, got, want))


G = {r["id"]: r for r in jl("data", "answers.jsonl")}
EV = {r["id"]: r for r in jl("data", "evidence.jsonl")}
IDS = list(G)
RUNS = ("closed", "vanilla", "vanilla_bm25", "graphrag", "graphrag_basic", "gold")
ANS = {s: {a["id"]: a for a in jl("runs", s, "answers.jsonl")} for s in RUNS}
EM = {s: {i: em(ANS[s][i]["answer"], G[i]["answer"], G[i]["answer_aliases"]) for i in IDS} for s in RUNS}
RET = {r["id"]: r for r in jl("runs", "vanilla", "retrieval.jsonl")}
art = lambda chunks: {c.split("#")[0] for c in chunks}
CTX = {
    "Vanilla": ("vanilla", lambda i: art(RET[i]["final"])),
    "BM25": ("vanilla_bm25", lambda i: art(RET[i]["bm25"][:8])),
    "Dense": (None, lambda i: art(RET[i]["dense"][:8])),
    "Local": ("graphrag", lambda i: set(ANS["graphrag"][i].get("context_docs") or [])),
    "Basic": ("graphrag_basic", lambda i: set(ANS["graphrag_basic"][i].get("context_docs") or [])),
}
PAPER = {  # any %, all %, Temporal both %, EM all arrived, Gold same, EM other, Gold same, n all arrived
    "Vanilla": (80.2, 34.1, 19.0, 78.0, 88.6, 16.5, 67.5, 332),
    "BM25": (84.7, 37.4, 23.7, 75.9, 87.7, 19.5, 66.9, 365),
    "Dense": (37.6, 13.2, 5.3, None, None, None, None, None),
    "Local": (31.6, 12.7, 5.0, 74.2, 91.9, 9.5, 72.2, 124),
    "Basic": (37.2, 12.4, 5.7, 69.4, 86.8, 7.5, 73.0, 121),
}
print("== Table IV")
tr = [i for i in IDS if G[i]["category"] == "시간추론"]
long_ = [i for i in IDS if G[i]["category"] == "멀티홉" and G[i]["hops"] in (5, 7)]
for name, (run, fn) in CTX.items():
    anyn = alln = both = longall = 0
    arrived = []
    for i in IDS:
        d, ev = fn(i), set(G[i]["evidence_ids"])
        h = len(ev & d)
        anyn += h > 0
        alln += h == len(ev)
        if h == len(ev):
            arrived.append(i)
    for i in tr:
        d = fn(i)
        both += EV[i]["anchor"]["source_id"] in d and EV[i]["target"]["source_id"] in d
    longall = sum(set(G[i]["evidence_ids"]) <= fn(i) for i in long_)
    got = (pct(anyn, 975), pct(alln, 975), pct(both, 300))
    if run:
        rest = [i for i in IDS if i not in set(arrived)]
        got += (pct(sum(EM[run][i] for i in arrived), len(arrived)), pct(sum(EM["gold"][i] for i in arrived), len(arrived)),
                pct(sum(EM[run][i] for i in rest), len(rest)), pct(sum(EM["gold"][i] for i in rest), len(rest)), len(arrived))
    else:
        got += (None, None, None, None, None)
    ck("%s: any, all, Temporal both, EM/Gold (all arrived), EM/Gold (other), n" % name, got, PAPER[name])
    ck("%s: 5-/7-page items with all gold articles" % name, longall, 0)

print("== Section V: questions that Gold Evidence answers exactly")
ok_ = [i for i in IDS if EM["gold"][i]]
ck("Gold-solved questions", len(ok_), 728)
ck("EM on them: Closed, Vanilla, BM25, Local, Basic",
   tuple(pct(sum(EM[s][i] for i in ok_), len(ok_)) for s in RUNS[:-1]), (2.2, 47.7, 50.8, 22.5, 19.2))
pairs = collections.defaultdict(list)
for i in tr:
    pairs[G[i]["pair_id"]].append(i)
okp = [p for p, v in pairs.items() if all(EM["gold"][i] for i in v)]
ck("Temporal pairs Gold answers on both sides", len(okp), 88)
ck("both sides on them: Vanilla, BM25, Local, Basic",
   tuple(pct(sum(all(EM[s][i] for i in pairs[p]) for p in okp), len(okp)) for s in RUNS[1:-1]), (8.0, 4.5, 1.1, 1.1))
print("failures:", BAD[0])
sys.exit(1 if BAD[0] else 0)
