"""Recompute the paper's numbers from the released files (python scripts/verify_paper.py; standard library).

Covers composition (Table I), corpus, author sample, run results other than confidence intervals
(Table III point values, Section V), input sizes and costs, index counts, the automatic graph flags and
the graph-check judgments. Confidence intervals: scripts/paired_ci.py. Section VI linkage counts:
scripts/vi_recompute.py. Each line prints the recomputed value and the value in the paper.
"""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import ABSTAIN, em, norm  # noqa: E402

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv.field_size_limit(10 ** 9)
BAD = [0]


def P(*a):
    return os.path.join(H, *a)


def jl(*a):
    return [json.loads(l) for l in open(P(*a), encoding="utf-8") if l.strip()]


def tsv(*a):
    with open(P(*a), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def js(*a):
    return json.load(open(P(*a), encoding="utf-8"))


def ck(label, got, want):
    ok = got == want if not isinstance(want, float) else abs(got - want) < 0.051
    BAD[0] += not ok
    print("%s %-62s %s (paper %s)" % ("PASS" if ok else "FAIL", label, got, want))


def pct(k, n):
    return round(100.0 * k / n, 1)


Q = {r["id"]: r for r in jl("data", "questions.jsonl")}
G = {r["id"]: r for r in jl("data", "answers.jsonl")}
EV = {r["id"]: r for r in jl("data", "evidence.jsonl")}
IDS = list(G)
RUNS = ("closed", "vanilla", "graphrag", "gold", "vanilla_bm25")
ANS = {s: {a["id"]: a for a in jl("runs", s, "answers.jsonl")} for s in RUNS + ("graphrag_ctx",)}
JUD = {s: {j["id"]: j["judge"] for j in jl("runs", s, "judged.jsonl")} for s in RUNS}
EM = {s: {i: em(ANS[s][i]["answer"], G[i]["answer"], G[i]["answer_aliases"]) for i in ANS[s]} for s in ANS}


def cat(i):
    return G[i]["category"]


def sub(i):
    return G[i]["sub"]


# ---------------- Section III, Table I ----------------
print("== III / Table I")
ck("items", len(IDS), 975)
pj = [i for i in IDS if cat(i) == "인물판정"]
tr = [i for i in IDS if cat(i) == "시간추론"]
mh = [i for i in IDS if cat(i) == "멀티홉"]
ck("Identity pairs same / different", (len({G[i]["pair_id"] for i in pj if sub(i) == "같은 사람"}), len({G[i]["pair_id"] for i in pj if sub(i) == "다른 사람"})), (75, 75))
ck("Temporal pairs", len({G[i]["pair_id"] for i in tr}), 150)
ac = collections.Counter(G[i]["alias_class"] for i in pj if sub(i) == "같은 사람")
ck("same-person pairs by alias class (office, princely, courtesy, spelling, posthumous)",
   tuple(ac[k] // 2 for k in ("관직·관계 호칭", "봉호·군호", "호·이칭·약칭", "표기 이형", "시호·묘호")), (29, 21, 13, 10, 2))
fam = collections.Counter(EV[i]["family"] for i in tr)
ck("Temporal items by family (day, month, year, order)", (fam["하루"], fam["달"], fam["해"], fam["순서"]), (62, 76, 90, 72))
pairs = tsv("diagnostics", "identity_pairs.tsv")
hg = lambda s: s.split("(")[0].strip()
same_hangul = [r for r in pairs if hg(r["name_a"]) == hg(r["name_b"]) and r["name_a"] != r["name_b"]]
ck("pairs with identical Hangul names (same / different person)",
   (sum(r["sub"] == "같은 사람" for r in same_hangul), sum(r["sub"] == "다른 사람" for r in same_hangul)), (8, 3))
diff_ending = 0
for p in {G[i]["pair_id"] for i in tr}:
    a, b = Q[p + "a"]["question"], Q[p + "b"]["question"]
    ra, rb = EV[p + "a"]["relation"], EV[p + "b"]["relation"]
    diff_ending += a.replace(ra, "", 1) != b.replace(rb, "", 1)
ck("Temporal pairs whose wording differs beyond the relation word", diff_ending, 34)
ck("Multi-hop items per chain length (2, 3, 5, 7)", tuple(sum(sub(i) == "%dhop" % k for i in mh) for k in (2, 3, 5, 7)), (92, 98, 100, 85))
ck("two-page items with one evidence article", sum(sub(i) == "2hop" and G[i]["n_evidence"] == 1 for i in mh), 76)
cap = {"2hop": 50, "3hop": 50, "5hop": 70, "7hop": 90}
ck("questions over the length limit", sum(len(Q[i]["question"]) > cap.get(sub(i), 50) for i in IDS), 0)
rows_t1 = [("Identity same", lambda i: cat(i) == "인물판정" and sub(i) == "같은 사람", (29.8, 1.69)),
           ("Identity different", lambda i: cat(i) == "인물판정" and sub(i) == "다른 사람", (23.2, 1.05)),
           ("Temporal", lambda i: cat(i) == "시간추론", (36.7, 2.00)),
           ("2-page", lambda i: sub(i) == "2hop", (30.0, 1.20)), ("3-page", lambda i: sub(i) == "3hop", (36.2, 2.44)),
           ("5-page", lambda i: sub(i) == "5hop", (61.2, 4.84)), ("7-page", lambda i: sub(i) == "7hop", (84.4, 7.12)),
           ("All", lambda i: True, (39.6, 2.51))]
for lab, f, want in rows_t1:
    v = [i for i in IDS if f(i)]
    ck("Table I %s mean length, mean evidence" % lab,
       (round(sum(len(Q[i]["question"]) for i in v) / len(v), 1), round(sum(G[i]["n_evidence"] for i in v) / len(v), 2)), want)

# ---------------- corpus ----------------
print("== III-A corpus")
arts = tsv("data", "article_ids.tsv")
ck("corpus articles", len(arts), 4273)
ck("corpus characters", sum(int(r["n_chars"]) for r in arts), 2214948)
ck("articles within accession year to year 9 / later", (sum(int(r["id"][:2]) <= 9 for r in arts), sum(int(r["id"][:2]) > 9 for r in arts)), (4245, 28))
gold_arts = {a for i in IDS for a in G[i]["evidence_ids"]}
ck("gold articles / all within the period", (len(gold_arts), all(int(a[:2]) <= 9 for a in gold_arts)), (773, True))
ck("questions with a later article in Vanilla RAG's chunks", sum(any(int(d[:2]) > 9 for d in ANS["vanilla"][i]["context_docs"]) for i in IDS), 107)
cs = js("data", "corpus_stats.json")
ck("Vanilla chunks / max characters", (cs["vanilla_chunks"]["n_chunks"], cs["vanilla_chunks"]["max_chars"] <= 600), (6861, True))
ck("Vanilla chunk count in the retrieval manifest", js("runs", "vanilla", "manifest_retrieval.json")["n_chunks"], 6861)

# ---------------- author sample ----------------
print("== III-C author sample")
au = tsv("data", "author_sample.tsv")
ac = collections.Counter(r["category"] for r in au)
ck("sample: Identity pairs, Temporal pairs, Multi-hop items", (ac["인물판정"] // 2, ac["시간추론"] // 2, ac["멀티홉"]), (15, 15, 40))
ck("sample judged correct", sum(r["verdict"] == "accepted" for r in au), 100)
aid = [r["id"] for r in au]
ck("Gold Evidence exact on the sample", int(sum(EM["gold"][i] for i in aid)), 78)
miss = [i for i in aid if not EM["gold"][i]]
contain = abst = form = other = 0
for i in miss:
    p = ANS["gold"][i]["answer"]
    np_ = norm(p)
    golds = [x for x in [G[i]["answer"]] + G[i]["answer_aliases"] if norm(x)]
    if ABSTAIN in p:
        abst += 1
    elif np_ and any(np_ in norm(x) or norm(x) in np_ for x in golds):
        contain += 1
    elif JUD["gold"].get(i) == 1.0:
        form += 1
    else:
        other += 1
ck("misses: contain/contained, abstention, form only (LLM-Eq), different", (contain, abst, form, other), (7, 2, 2, 11))

vs = js("data", "validation_summary.json")
fp = vs["first_pass"]
ck("first-pass exact agreement (%): Identity, Temporal, Multi-hop",
   tuple(pct(*fp[c]["exact_agreement"]) for c in ("Identity", "Temporal", "Multi-hop")), (92.7, 75.7, 89.6))
ck("first-pass exact agreement counts", tuple(tuple(fp[c]["exact_agreement"]) for c in ("Identity", "Temporal", "Multi-hop")), ((278, 300), (227, 300), (336, 375)))
ck("first-pass main-answer agreement (%)", tuple(pct(*fp[c]["main_answer_agreement"]) for c in ("Identity", "Temporal", "Multi-hop")), (98.0, 89.3, 100.0))
ck("Temporal pairs to one solver / re-solved separately (recorded)",
   (vs["temporal_solver_assignment"]["pairs_with_both_items_to_one_solver"], vs["temporal_solver_assignment"]["of_these_re_solved_separately"]), (35, 13))

# ---------------- Section IV ----------------
print("== IV inputs and costs")
man = {s: js("runs", s, "manifest.json") for s in ("closed", "vanilla", "gold", "graphrag")}
for s, want in (("closed", 124), ("vanilla", 2276), ("gold", 1608), ("graphrag", 7470)):
    ck("mean prompt tokens %s" % s, round(man[s]["usage_gpt-4.1-mini"]["prompt_tokens"] / 975), want)
ck("GraphRAG mean output tokens", round(man["graphrag"]["usage_gpt-4.1-mini"]["completion_tokens"] / 975, 1), 4.4)
ck("cost of the three reader runs incl. query embeddings (US$)",
   round(man["closed"]["usd"] + man["vanilla"]["usd"] + man["gold"]["usd"] + js("runs", "vanilla", "manifest_retrieval.json")["usd"], 2), 1.58)
ck("GraphRAG query cost (US$)", round(man["graphrag"]["usd"], 2), 2.92)
ck("indexing cost (US$)", js("graphrag", "index_stats.json")["indexing_cost_usd"], 45.89)
ck("returned reader version", man["vanilla"]["model_returned"], "gpt-4.1-mini-2025-04-14")
ck("run date", man["vanilla"]["run_date"], "2026-09-29")
ci_rows = jl("runs", "graphrag", "context_index.jsonl")
rep = [r["id"] for r in ci_rows if r.get("report_ids")]
nrep = [r["id"] for r in ci_rows if not r.get("report_ids")]
ck("contexts with community reports (n, %)", (len(rep), pct(len(rep), 975)), (809, 83.0))
ck("GraphRAG EM with / without reports", (pct(sum(EM["graphrag"][i] for i in rep), len(rep)), pct(sum(EM["graphrag"][i] for i in nrep), len(nrep))), (16.9, 21.7))
smp = js("runs", "graphrag_ctx", "sample.json")["ids"]
ck("200-item re-read EM: common instruction / native", (pct(sum(EM["graphrag_ctx"][i] for i in smp), 200), pct(sum(EM["graphrag"][i] for i in smp), 200)), (14.5, 14.0))
st = js("graphrag", "index_stats.json")["counts"]
ck("index text units / entities / relationships / communities",
   (len(tsv("graph", "text_units_ids.tsv")), len(tsv("graph", "entities_ids.tsv")), len(tsv("graph", "relationships_ids.tsv")), st["communities"]),
   (4600, 18593, 55001, 4131))
ents = {r["title"]: r for r in tsv("graph", "entities_ids.tsv")}
ck("text units of the node for the reigning king (주상)", len(ents["주상"]["text_unit_ids"].split(",")), 1665)

# ---------------- Table III point values and Section V ----------------
print("== Table III and Section V")


def rate(s, ids, f=None):
    return pct(sum((f or EM[s].get)(i) for i in ids), len(ids))


def pair_em(s, ids):
    pe = collections.defaultdict(list)
    for i in ids:
        pe[G[i]["pair_id"]].append(EM[s][i])
    return pct(sum(all(v) for v in pe.values()), len(pe))


T3 = {"closed": (0.7, 0.0, 0.3, 2.6, 23.1), "vanilla": (54.0, 5.3, 29.7, 39.4, 16.3),
      "graphrag": (22.7, 1.3, 12.0, 19.8, 39.4), "gold": (86.7, 58.7, 72.7, 79.8, 1.3)}
for s, want in T3.items():
    got = (pair_em(s, pj), pair_em(s, tr), pair_em(s, pj + tr), rate(s, IDS, lambda i: JUD[s][i]),
           rate(s, IDS, lambda i: ABSTAIN in ANS[s][i]["answer"]))
    ck("%s Pair EM (Id, Temp, All), LLM-Eq, abstention" % s, got, want)
ck("BM25 alone: Pair EM, Temporal EM, Temporal Pair EM", (pair_em("vanilla_bm25", pj + tr), rate("vanilla_bm25", tr), pair_em("vanilla_bm25", tr)), (31.3, 17.7, 2.7))
same_p = [i for i in pj if sub(i) == "같은 사람"]
diff_p = [i for i in pj if sub(i) == "다른 사람"]
ck("Identity EM same / different: Vanilla, Gold", (rate("vanilla", same_p), rate("vanilla", diff_p), rate("gold", same_p), rate("gold", diff_p)), (72.0, 62.7, 88.7, 93.3))
ck("Vanilla right & Gold wrong / Gold right & Vanilla wrong",
   (sum(EM["vanilla"][i] and not EM["gold"][i] for i in IDS), sum(EM["gold"][i] and not EM["vanilla"][i] for i in IDS)), (18, 381))


def recall(s, ids):
    anyv = cov = 0.0
    for i in ids:
        E, C = set(G[i]["evidence_ids"]), set(ANS[s][i].get("context_docs") or [])
        anyv += bool(E & C)
        cov += len(E & C) / len(E)
    return pct(anyv, len(ids)), round(100 * cov / len(ids), 1)


ck("evidence recall any / coverage: GraphRAG", recall("graphrag", IDS), (31.6, 20.9))
ck("evidence recall any / coverage: Vanilla", recall("vanilla", IDS), (80.2, 53.0))
ck("coverage BM25 alone", recall("vanilla_bm25", IDS)[1], 57.3)
dep = {k: [i for i in mh if sub(i) == "%dhop" % k] for k in (2, 3, 5, 7)}
ck("Multi-hop EM 2-page -> 7-page: Vanilla", (rate("vanilla", dep[2]), rate("vanilla", dep[7])), (73.9, 4.7))
ck("Multi-hop EM 2-page -> 7-page: GraphRAG", (rate("graphrag", dep[2]), rate("graphrag", dep[7])), (30.4, 3.5))
ck("Multi-hop EM 2-page -> 7-page: Gold", (rate("gold", dep[2]), rate("gold", dep[7])), (90.2, 42.4))
ck("Vanilla coverage 2-page -> 7-page", (recall("vanilla", dep[2])[1], recall("vanilla", dep[7])[1]), (87.1, 15.6))
allret = lambda s, i: set(G[i]["evidence_ids"]) <= set(ANS[s][i]["context_docs"])
ck("5-/7-page items with all gold articles retrieved (Vanilla)", sum(allret("vanilla", i) for i in dep[5] + dep[7]), 0)
long_ = dep[5] + dep[7]
over8 = [i for i in long_ if G[i]["n_evidence"] > 8]
ck("5-/7-page items with more than eight evidence articles", (len(over8), len(long_)), (12, 185))
ret = {r["id"]: r for r in jl("runs", "vanilla", "retrieval.jsonl")}
top20 = lambda i: {c.split("#")[0] for c in ret[i]["rrf"][:20]}
ck("other 5-/7-page items with all gold articles in the fused top 20", sum(set(G[i]["evidence_ids"]) <= top20(i) for i in long_ if i not in over8), 0)
va_miss = [i for i in IDS if not EM["vanilla"][i]]
ck("Vanilla misses: total / no gold article / some / all",
   (len(va_miss), sum(not set(G[i]["evidence_ids"]) & set(ANS["vanilla"][i]["context_docs"]) for i in va_miss),
    sum(bool(set(G[i]["evidence_ids"]) & set(ANS["vanilla"][i]["context_docs"])) and not allret("vanilla", i) for i in va_miss),
    sum(allret("vanilla", i) for i in va_miss)), (610, 171, 366, 73))
vf = {r["id"]: r["answer_string_in_retrieved_chunks"] == "True" for r in tsv("diagnostics", "vanilla_answer_string.tsv")}
va_all = [i for i in va_miss if allret("vanilla", i)]
ck("of these 73, answer string in the retrieved chunks", (sorted(vf) == sorted(va_all), sum(vf[i] for i in va_all)), (True, 53))


def same_answer_pairs(s, ids, want_same):
    pe = collections.defaultdict(list)
    for i in ids:
        pe[G[i]["pair_id"]].append(ANS[s][i]["answer"])
    n = 0
    for v in pe.values():
        if len(v) == 2 and not any(ABSTAIN in x for x in v):
            n += (norm(v[0]) == norm(v[1])) == want_same
    return n


ck("one answer to both sides: different-person pairs (Vanilla, Gold)", (same_answer_pairs("vanilla", diff_p, True), same_answer_pairs("gold", diff_p, True)), (5, 1))
ck("one answer to both sides: temporal pairs (Vanilla, Gold)", (same_answer_pairs("vanilla", tr, True), same_answer_pairs("gold", tr, True)), (47, 7))
ck("two answers: same-person pairs (Vanilla, Gold)", (same_answer_pairs("vanilla", same_p, False), same_answer_pairs("gold", same_p, False)), (11, 5))

# ---------------- Section VI (beyond scripts/vi_recompute.py) ----------------
print("== VI")
mt = tsv("diagnostics", "multihop.tsv")
cov_n = lambda k, f: round(100 * sum(float(r[f]) for r in mt if r["sub"] == "%dhop" % k) / sum(r["sub"] == "%dhop" % k for r in mt), 1)
ck("chain entities with nodes, 2-page / 7-page (%)", (cov_n(2, "node_cov"), cov_n(7, "node_cov")), (94.6, 86.6))
ck("hop evidence articles in the context, 2-page / 7-page (%)", (cov_n(2, "hop_ctx"), cov_n(7, "hop_ctx")), (33.7, 3.9))
ast = tsv("diagnostics", "answer_string.tsv")
ck("answer string in context: share (%), Multi-hop count",
   (pct(sum(r["answer_string_in_context"] == "True" for r in ast), 975), sum(r["answer_string_in_context"] == "True" and r["group"].startswith("멀티홉") for r in ast)), (48.1, 154))
no2 = [i for i in IDS if not any(len("".join(x.split())) >= 2 for x in [G[i]["answer"]] + G[i]["answer_aliases"])]
ck("items without a gold string of two or more characters", sorted(no2), ["PJ-065a", "PJ-065b", "TR-133a"])
af = js("diagnostics", "graph_checks", "auto_flags.json")["counts"]
ck("automatic flags: Hanja-gloss pairs, address nodes, non-verbatim names, merge candidates",
   (af["hanja_split_pairs"], af["address_nodes"], af["non_verbatim_names"], af["same_name_merge_candidates"]), (1162, 14, 2566, 23))
titles = set(ents)
han = lambda t: any("一" <= ch <= "鿿" for ch in t)
import re  # noqa: E402
base = {re.sub(r"\(.*?\)|[一-鿿]", "", t).strip() for t in titles if han(t)}
ck("Hanja-gloss pairs recomputed from graph/entities_ids.tsv", len([b for b in base if b and b in titles]), 1162)
gc = lambda f: jl("diagnostics", "graph_checks", f)
rel = collections.Counter(r["판정"] for r in gc("relationships30.jsonl"))
ck("relationships: fully supported / core relation kept (of 30)", (rel["지지"], rel["지지"] + rel["부분 지지"]), (22, 29))
ck("Hanja-split pairs referring to one entity (of 20)", sum(r["판정"] == "같은 대상이 갈라짐" for r in gc("hanja_split20.jsonl")), 16)
nv = collections.Counter(r["판정"] for r in gc("non_verbatim_names20.jsonl"))
ck("non-verbatim names: spelling variants / descriptive (of 20)", (nv["표기 변형"], nv["서술형 이름"]), (11, 9))
sm = collections.Counter(r["판정"] for r in gc("same_name_merge20.jsonl"))
ck("merge candidates with distinct referents: persons / non-person terms", (sm["다른 사람이 합쳐짐"], sm["사람 아님"]), (12, 3))
n98 = sum(len(gc(f)) for f in ("relationships30.jsonl", "hanja_split20.jsonl", "non_verbatim_names20.jsonl", "same_name_merge20.jsonl", "table_iv_candidates.jsonl"))
ck("judged graph elements (90 sampled + 8 candidates)", n98, 98)
first = {}
for f in ("relationships30.jsonl", "hanja_split20.jsonl", "non_verbatim_names20.jsonl", "same_name_merge20.jsonl", "table_iv_candidates.jsonl"):
    for r in gc(f):
        first[r.get("sample_id") or r.get("case_id")] = r["판정"]
rc = gc("recheck35.jsonl")
ck("re-checked items / disagreements", (len(rc), sum(first[r["id"]] != r["판정"] for r in rc)), (35, 0))
print("failures:", BAD[0])
sys.exit(1 if BAD[0] else 0)
