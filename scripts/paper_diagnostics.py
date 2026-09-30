"""Diagnostic counts stated in the text of the paper that the other scripts do not cover
(python scripts/paper_diagnostics.py; standard library).

  Section V   Identity: same-person pairs answered exactly on one side only; same-person pairs (both
              answered) whose two names get different answers; different-person pairs with identical
              Hangul names, identical gold article sets, or a surname-and-period alias type (III-B).
              Temporal: answers that are clauses or lists, Gold Evidence EM and LLM-Eq on them.
              Multi-hop: 3- to 7-page items answered by Vanilla RAG without the first hop's article in
              the context; BM25 only EM on 7-page chains.
  Section VI  top-20 arrival (BM25, dense), Cohen's kappa of "any gold article arrives" between dense
              top 8 and the GraphRAG searches, and the 109-question Vanilla RAG / GraphRAG Local
              comparison under EM, LLM-Eq and answer-string containment.
  Section IV  re-read of 200 stored GraphRAG Local contexts (IV-A); EM with and without community
              reports in the Local context (IV-C).

Context articles per question are defined as in scripts/evidence_arrival.py. EM, normalization and
abstention come from scripts/score.py. LLM-Eq counts an item as correct if it is an EM match or the
judge (runs/<run>/judged.jsonl) accepted it. Answer-string containment counts an item as correct if it is
an EM match, or the answer is not an abstention and its normalized form contains the normalized gold
answer or an alias (empty strings ignored), as in scripts/verify_paper.py (IV-A).
Each line prints the recomputed value and the value in the paper; the script exits with a non-zero
status on a mismatch.
"""
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import ABSTAIN, em, norm  # noqa: E402

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD = [0]


def jl(*a):
    return [json.loads(l) for l in open(os.path.join(H, *a), encoding="utf-8") if l.strip()]


def pct(k, n):
    return round(100.0 * k / n, 1)


def ck(label, got, want):
    ok = got == want
    BAD[0] += not ok
    print("%s %-66s %s (paper %s)" % ("PASS" if ok else "FAIL", label, got, want))


Q = {r["id"]: r for r in jl("data", "questions.jsonl")}
G = {r["id"]: r for r in jl("data", "answers.jsonl")}
EV = {r["id"]: r for r in jl("data", "evidence.jsonl")}
IDS = list(G)
RUNS = ("vanilla", "vanilla_bm25", "graphrag", "graphrag_basic", "gold", "graphrag_ctx")
ANS = {s: {a["id"]: a for a in jl("runs", s, "answers.jsonl")} for s in RUNS}
JUD = {s: {j["id"]: float(j["judge"]) for j in jl("runs", s, "judged.jsonl")} for s in RUNS}


def A(s, i):
    return ANS[s][i].get("answer", "")


def E(s, i):
    return em(A(s, i), G[i]["answer"], G[i]["answer_aliases"])


def AB(s, i):
    return ABSTAIN in A(s, i)


def LQ(s, i):
    return max(E(s, i), JUD[s].get(i, 0.0))


def CONT(s, i):
    if E(s, i):
        return 1.0
    p = A(s, i)
    if ABSTAIN in p or not norm(p):
        return 0.0
    return float(any(norm(x) and norm(x) in norm(p) for x in [G[i]["answer"]] + list(G[i]["answer_aliases"])))


RET = {r["id"]: r for r in jl("runs", "vanilla", "retrieval.jsonl")}
art = lambda chunks: {c.split("#")[0] for c in chunks}
VAN = lambda i: art(RET[i]["final"])
DEN8 = lambda i: art(RET[i]["dense"][:8])
LOC = lambda i: set(ANS["graphrag"][i].get("context_docs") or [])
BAS = lambda i: set(ANS["graphrag_basic"][i].get("context_docs") or [])

# ---------------------------------------------------------------- Section V: Identity
print("== Section V: Identity")
pairs = collections.defaultdict(list)
for i in IDS:
    if G[i]["category"] == "인물판정":
        pairs[G[i]["pair_id"]].append(i)
same = [v for v in pairs.values() if G[v[0]]["sub"] == "같은 사람"]
diff = [v for v in pairs.values() if G[v[0]]["sub"] == "다른 사람"]
ck("same-person pairs / different-person pairs", (len(same), len(diff)), (75, 75))
ck("same-person pairs with exactly one side EM: Vanilla, Local, Gold",
   tuple(sum(E(s, a) != E(s, b) for a, b in same) for s in ("vanilla", "graphrag", "gold")), (16, 21, 5))
nb = [(a, b) for a, b in same if not AB("vanilla", a) and not AB("vanilla", b)]
ck("Vanilla: same-person pairs (no abstention) with different answers",
   "%d/%d" % (sum(norm(A("vanilla", a)) != norm(A("vanilla", b)) for a, b in nb), len(nb)), "11/60")
# III-B descriptors of the different-person pairs (alias_type is shared by both items of a pair)
ck("different-person pairs with identical Hangul names (alias type 동명)",
   sum(G[v[0]]["alias_type"].startswith("동명") for v in diff), 4)
ck("different-person pairs with the same gold article set",
   sum(set(G[a]["evidence_ids"]) == set(G[b]["evidence_ids"]) for a, b in diff), 62)
ck("different-person pairs sharing only surname and period (동성동시기)",
   sum(G[v[0]]["alias_type"].startswith("동성동시기") for v in diff), 13)

# ---------------------------------------------------------------- Section V: Temporal
print("== Section V: Temporal answers that are clauses or lists")


def clause_or_list(i):
    # normalized answer of 8+ characters, an enumeration mark or conjunction, or a verbal ending
    a = G[i]["answer"]
    n = norm(a)
    return (len(n) >= 8 or bool(re.search(r"[·,]|\s및\s|[가-힣](와|과)\s", a))
            or bool(re.search(r"(다|라|함|음|줌|둠)$", n)))


tr = [i for i in IDS if G[i]["category"] == "시간추론"]
dl = [i for i in tr if clause_or_list(i)]
ck("Temporal answers that are clauses or lists", len(dl), 61)
ck("Gold Evidence (gpt-4.1-mini) on them: EM, LLM-Eq",
   (pct(sum(E("gold", i) for i in dl), len(dl)), pct(sum(LQ("gold", i) for i in dl), len(dl))), (47.5, 70.5))

# ---------------------------------------------------------------- Section V: Multi-hop
print("== Section V: Multi-hop")
mh = [i for i in IDS if G[i]["category"] == "멀티홉" and G[i]["hops"] in (3, 5, 7) and E("vanilla", i)]
skip = [i for i in mh if not (set(EV[i]["chain"][0]["source_ids"]) & VAN(i))]
ck("3- to 7-page items Vanilla answers / without a first-hop article", (len(mh), len(skip)), (45, 28))
h7 = [i for i in IDS if G[i]["category"] == "멀티홉" and G[i]["hops"] == 7]
ck("BM25 only EM on 7-page chains", pct(sum(E("vanilla_bm25", i) for i in h7), len(h7)), 12.9)

# ---------------------------------------------------------------- Section VI
print("== Section VI: arrival")


def any_arrival(fn):
    return [int(bool(set(G[i]["evidence_ids"]) & fn(i))) for i in IDS]


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return round((po - pe) / (1 - pe), 2)


ck("any gold article in top 20: BM25, dense",
   (pct(sum(any_arrival(lambda i: art(RET[i]["bm25"][:20]))), 975), pct(sum(any_arrival(lambda i: art(RET[i]["dense"][:20]))), 975)),
   (93.3, 49.3))
D8, L, B = any_arrival(DEN8), any_arrival(LOC), any_arrival(BAS)
ck("Cohen's kappa (any arrival): dense top 8 vs Basic, vs Local", (kappa(D8, B), kappa(D8, L)), (0.71, 0.24))

print("== Section VI: questions whose gold articles all reach both Vanilla RAG and GraphRAG Local")
both = [i for i in IDS if set(G[i]["evidence_ids"]) <= VAN(i) and set(G[i]["evidence_ids"]) <= LOC(i)]
# 95% CI as in scripts/evidence_arrival.py: minimal pairs (Identity, Temporal) or items (Multi-hop) as
# units, stratified by category, 2,000 resamples, random.Random(20260929), percentile (sorted[49], sorted[1949])
units = collections.defaultdict(lambda: collections.defaultdict(list))
for i in both:
    c = G[i]["category"]
    units[c][G[i]["pair_id"] if c in ("인물판정", "시간추론") and G[i].get("pair_id") else i].append(i)


def boot_ci(f):
    rng = random.Random(20260929)
    ds = []
    for _ in range(2000):
        s_ = n_ = 0
        for c, u in units.items():
            us = list(u.values())
            for grp in (rng.choice(us) for _ in us):
                for i in grp:
                    s_ += f("vanilla", i) - f("graphrag", i)
                    n_ += 1
        ds.append(100.0 * s_ / n_)
    ds.sort()
    return (round(ds[49], 1), round(ds[1949], 1))


def compare(f):
    v = sum(f("vanilla", i) for i in both)
    loc = sum(f("graphrag", i) for i in both)
    return pct(v, len(both)), pct(loc, len(both)), round(100.0 * (v - loc) / len(both), 1)


ck("questions", len(both), 109)
ck("EM: Vanilla, Local, difference, 95% CI", compare(E) + boot_ci(E), (85.3, 74.3, 11.0, 3.4, 19.2))
ck("LLM-Eq difference, 95% CI", compare(LQ)[2:] + boot_ci(LQ), (6.4, -1.8, 14.3))
ck("answer-string containment difference", compare(CONT)[2], 0.9)
lw = [i for i in both if not E("graphrag", i)]
ck("Local misses / of which containing the gold answer", (len(lw), int(sum(CONT("graphrag", i) for i in lw))), (28, 14))

# ---------------------------------------------------------------- Section IV
print("== Section IV-A: re-read of 200 stored GraphRAG Local contexts")
smp = json.load(open(os.path.join(H, "runs", "graphrag_ctx", "sample.json"), encoding="utf-8"))["ids"]
ck("sample size", len(smp), 200)
ck("abstention %: native, re-read",
   (pct(sum(AB("graphrag", i) for i in smp), len(smp)), pct(sum(AB("graphrag_ctx", i) for i in smp), len(smp))), (44.5, 28.5))
ck("EM %: native, re-read",
   (pct(sum(E("graphrag", i) for i in smp), len(smp)), pct(sum(E("graphrag_ctx", i) for i in smp), len(smp))), (14.0, 14.5))

print("== Section IV-C: community reports in the GraphRAG Local context")
rows = jl("runs", "graphrag", "context_index.jsonl")
rep = [r["id"] for r in rows if r.get("report_ids")]
rest = [r["id"] for r in rows if not r.get("report_ids")]
ck("questions with reports / EM with reports / EM without",
   (len(rep), pct(sum(E("graphrag", i) for i in rep), len(rep)), pct(sum(E("graphrag", i) for i in rest), len(rest))),
   (809, 16.9, 21.7))

print("failures:", BAD[0])
sys.exit(1 if BAD[0] else 0)
