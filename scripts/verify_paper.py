"""Recompute the paper's numbers from the released files (python scripts/verify_paper.py; standard library).

Covers composition (Table I), corpus, author sample, run results other than confidence intervals
(Table III point values, Section V), input sizes and costs, index counts, the automatic graph flags and
the graph-check judgments. Table IV and the Gold-solved re-count: scripts/evidence_arrival.py. Full-period
corpus checks (need a local copy of the translation): scripts/full_period_check.py. Confidence intervals: scripts/paired_ci.py. Section VI linkage counts:
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
    print("%s %-62s %s (expected %s)" % ("PASS" if ok else "FAIL", label, got, want))


def pct(k, n):
    return round(100.0 * k / n, 1)


Q = {r["id"]: r for r in jl("data", "questions.jsonl")}
G = {r["id"]: r for r in jl("data", "answers.jsonl")}
EV = {r["id"]: r for r in jl("data", "evidence.jsonl")}
IDS = list(G)
RUNS = ("closed", "vanilla", "graphrag", "gold", "vanilla_bm25", "graphrag_basic")
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
rows_t1 = [("Identity same", lambda i: cat(i) == "인물판정" and sub(i) == "같은 사람", (29.8, 1.7)),
           ("Identity different", lambda i: cat(i) == "인물판정" and sub(i) == "다른 사람", (23.2, 1.1)),
           ("Temporal", lambda i: cat(i) == "시간추론", (36.7, 2.0)),
           ("2-page", lambda i: sub(i) == "2hop", (30.0, 1.2)), ("3-page", lambda i: sub(i) == "3hop", (36.2, 2.4)),
           ("5-page", lambda i: sub(i) == "5hop", (61.2, 4.8)), ("7-page", lambda i: sub(i) == "7hop", (84.4, 7.1)),
           ("All", lambda i: True, (39.6, 2.5))]
for lab, f, want in rows_t1:
    v = [i for i in IDS if f(i)]
    ck("Table I %s mean length, mean evidence" % lab,
       (round(sum(len(Q[i]["question"]) for i in v) / len(v), 1), round(sum(G[i]["n_evidence"] for i in v) / len(v), 1)), want)
ck("different-person pairs with identical Hangul names (Hangul only / Hangul and Hanja)",
   tuple(sum(G[i]["alias_type"] == t for i in pj if sub(i) == "다른 사람" and i.endswith("a")) for t in ("동명(한글 같음)", "동명(한글·한자 같음)")), (3, 1))

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
led = tsv("data", "ledger_ids.tsv")
ck("construction ledger: articles, used, not used", (len(led), sum(r["used_in_wiki"] == "yes" for r in led), sum(r["used_in_wiki"] == "no" for r in led)), (11275, 4213, 7062))
ck("ledger articles used = corpus articles within the period that the ledger marks used",
   len({r["id"] for r in led if r["used_in_wiki"] == "yes"} - {r["id"] for r in arts}), 0)

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
man = {s: js("runs", s, "manifest.json") for s in ("closed", "vanilla", "gold", "graphrag", "graphrag_basic", "vanilla_bm25")}
for s, want in (("closed", 124), ("vanilla", 2276), ("gold", 1608), ("graphrag", 7470), ("graphrag_basic", 5301)):
    ck("mean prompt tokens %s" % s, round(man[s]["usage_gpt-4.1-mini"]["prompt_tokens"] / 975), want)
ck("GraphRAG mean output tokens", round(man["graphrag"]["usage_gpt-4.1-mini"]["completion_tokens"] / 975, 1), 4.4)
ck("cost of the three reader runs incl. query embeddings (US$)",
   round(man["closed"]["usd"] + man["vanilla"]["usd"] + man["gold"]["usd"] + js("runs", "vanilla", "manifest_retrieval.json")["usd"], 2), 1.58)
ck("GraphRAG query cost (US$)", round(man["graphrag"]["usd"], 2), 2.92)
ck("GraphRAG Basic query cost (US$), BM25-only cost (US$)", (round(man["graphrag_basic"]["usd"], 2), round(man["vanilla_bm25"]["usd"], 2)), (2.08, 0.84))
ck("GraphRAG Basic: text units, context tokens, search", (man["graphrag_basic"]["top_k"]["k"], man["graphrag_basic"]["top_k"]["max_context_tokens"], man["graphrag_basic"]["search"]), (10, 12000, "basic (native prompt)"))
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


same_p = [i for i in pj if sub(i) == "같은 사람"]
diff_p = [i for i in pj if sub(i) == "다른 사람"]
# Table III (v3K): EM same person, different persons, Temporal, Multi-hop, All; Pair EM Identity, Temporal; LLM-Eq; abstention
T3 = {"closed": (4.0, 1.3, 1.0, 1.9, 1.8, 0.7, 0.0, 2.6, 23.1), "vanilla": (72.0, 62.7, 16.7, 30.1, 37.4, 54.0, 5.3, 39.4, 16.3),
      "vanilla_bm25": (75.3, 68.7, 17.7, 33.9, 40.6, 60.0, 2.7, 42.6, 13.9), "graphrag": (38.0, 34.0, 8.7, 10.4, 17.7, 22.7, 1.3, 19.8, 39.4),
      "graphrag_basic": (31.3, 22.7, 8.7, 10.9, 15.2, 14.0, 1.3, 16.5, 42.3), "gold": (88.7, 93.3, 71.7, 64.0, 74.7, 86.7, 58.7, 79.8, 1.3)}
for s, want in T3.items():
    got = (rate(s, same_p), rate(s, diff_p), rate(s, tr), rate(s, mh), rate(s, IDS), pair_em(s, pj), pair_em(s, tr),
           rate(s, IDS, lambda i: JUD[s][i]), rate(s, IDS, lambda i: ABSTAIN in ANS[s][i]["answer"]))
    ck("%s EM (same, diff, Temp, MH, All), Pair EM (Id, Temp), LLM-Eq, abst." % s, got, want)
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
ck("evidence recall any: GraphRAG Basic, BM25 only", (recall("graphrag_basic", IDS)[0], recall("vanilla_bm25", IDS)[0]), (37.2, 84.7))
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
bas = tsv("diagnostics", "basic_answer_string.tsv")
bh = [r["id"] for r in bas if r["answer_string_in_context"] == "True"]
ck("answer string in GraphRAG Basic context: count, EM among them", (len(bh), pct(sum(EM["graphrag_basic"][i] for i in bh), len(bh))), (355, 41.7))
ast = tsv("diagnostics", "answer_string.tsv")
ck("answer string in context: share (%), Multi-hop count",
   (pct(sum(r["answer_string_in_context"] == "True" for r in ast), 975), sum(r["answer_string_in_context"] == "True" and r["group"].startswith("멀티홉") for r in ast)), (48.1, 154))
ah = [r["id"] for r in ast if r["answer_string_in_context"] == "True"]
ck("answer string in GraphRAG Local context: count, EM among them", (len(ah), pct(sum(EM["graphrag"][i] for i in ah), len(ah))), (469, 36.7))
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
ck("illustrative candidates confirmed (of 8)", sum(r["판정"] == "확인" for r in gc("table_iv_candidates.jsonl")), 6)
ck("sampled merge candidates with distinct referents (of 20)", sm["다른 사람이 합쳐짐"] + sm["사람 아님"], 15)
first = {}
for f in ("relationships30.jsonl", "hanja_split20.jsonl", "non_verbatim_names20.jsonl", "same_name_merge20.jsonl", "table_iv_candidates.jsonl"):
    for r in gc(f):
        first[r.get("sample_id") or r.get("case_id")] = r["판정"]
rc = gc("recheck35.jsonl")
ck("re-checked items / disagreements", (len(rc), sum(first[r["id"]] != r["판정"] for r in rc)), (35, 0))
# IV-A: answer form (wrong answers containing a gold answer string; EM among answered questions)
for s_, want in (("graphrag", (41, 21.9, 173, 591, 29.3)), ("vanilla", (21, 39.6, 365, 816, 44.7))):
    wc = sum(1 for i in IDS if not EM[s_][i] and ABSTAIN not in ANS[s_][i]["answer"] and norm(ANS[s_][i]["answer"])
             and any(norm(x) and norm(x) in norm(ANS[s_][i]["answer"]) for x in [G[i]["answer"]] + list(G[i]["answer_aliases"])))
    k = sum(EM[s_][i] for i in IDS)
    answered = sum(ABSTAIN not in ANS[s_][i]["answer"] for i in IDS)
    ck("answer form %s: wrong but containing gold, lenient %%, EM among answered" % s_,
       (wc, pct(k + wc, len(IDS)), int(k), answered, pct(k, answered)), want)
vs = js("data", "validation_summary.json")["temporal_one_solver_pairs"]
ck("Temporal pairs whole to one solver / re-solved / not (III-C)", (vs["pairs_whole_to_one_solver"], vs["re_solved_separately"], vs["not_re_solved"]), (35, 13, 22))
ck("first-pass exact agreement: not re-solved / separated pairs", (tuple(vs["first_pass_exact_not_re_solved"]), tuple(vs["first_pass_exact_separated_pairs"])), ((36, 44), (172, 230)))
print("failures:", BAD[0])
sys.exit(1 if BAD[0] else 0)
