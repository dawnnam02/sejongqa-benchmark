"""Recompute the Section VI counts from the released per-question tables (python scripts/vi_recompute.py).

Inputs: diagnostics/{identity_items,identity_pairs,temporal,multihop,answer_string}.tsv (no article text).
Values in the tables are Korean labels from the original analysis:
  temporal gr_ctx/va_ctx: 둘 다 = both articles, 기준만 = anchor only, 목표만 = target only, 둘 다 없음 = neither.
  multihop sub: 2hop/3hop/5hop/7hop = 2-, 3-, 5-, 7-page chains. identity_pairs sub: 같은 사람 = same person, 다른 사람 = different persons.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "diagnostics")


def rows(name):
    with open(os.path.join(D, name), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


T = lambda v: v == "True"
F = lambda v: float(v) == 1.0


def recompute():
    out = {}
    it = rows("identity_items.tsv")
    out["identity_items"] = len(it)
    out["node"] = sum(bool(r["nodes"]) for r in it)
    out["linked"] = sum(T(r["grounded"]) for r in it)
    out["in_ctx_any"] = sum(T(r["in_ctx"]) for r in it)
    sn = [r for r in it if T(r["same_node"])]
    rest = [r for r in it if T(r["grounded"]) and not T(r["same_node"])]
    out["same_node"] = len(sn)
    out["same_node_gold_article_in_ctx"] = sum(F(r["gr_gold_article_in_ctx"]) for r in sn)
    out["same_node_em"] = sum(F(r["gr_em"]) for r in sn)
    out["linked_not_same_node"] = len(rest)
    out["linked_not_same_node_em"] = sum(F(r["gr_em"]) for r in rest)
    pr = rows("identity_pairs.tsv")
    diff = [r for r in pr if r["sub"] == "다른 사람"]
    merged = [r for r in diff if r["shared_grounded_both"] == "True"]
    out["diff_pairs_one_node_both_gold"] = len(merged)
    out["diff_pairs_one_node_both_gold_graphrag_pair_em"] = sum(F(r["gr_pair"]) for r in merged)
    out["diff_pairs_one_node_both_gold_vanilla_pair_em"] = sum(F(r["va_pair"]) for r in merged)
    tr = rows("temporal.tsv")
    for lab, key in (("둘 다", "both"), ("둘 다 없음", "neither")):
        g = [r for r in tr if r["gr_ctx"] == lab]
        out["temporal_" + key] = len(g)
        out["temporal_" + key + "_em"] = sum(F(r["gr_em"]) for r in g)
        out["temporal_" + key + "_vanilla"] = sum(r["va_ctx"] == lab for r in tr)
    mh = rows("multihop.tsv")
    for d in ("2hop", "3hop", "5hop", "7hop"):
        v = [r for r in mh if r["sub"] == d]
        out[d + "_items"] = len(v)
        out[d + "_path_connected"] = sum(T(r["path"]) for r in v)
    ans = rows("answer_string.tsv")
    y = [r for r in ans if T(r["answer_string_in_context"])]
    out["answer_string_in_context"] = len(y)
    out["answer_string_in_context_em"] = sum(F(r["gr_em"]) for r in y)
    out["answer_string_absent_em"] = sum(F(r["gr_em"]) for r in ans if not T(r["answer_string_in_context"]))
    return out


EXPECTED = {"identity_items": 300, "node": 290, "linked": 256, "in_ctx_any": 113, "same_node": 102,
            "same_node_gold_article_in_ctx": 53, "same_node_em": 52, "linked_not_same_node": 154, "linked_not_same_node_em": 42,
            "diff_pairs_one_node_both_gold": 3, "diff_pairs_one_node_both_gold_graphrag_pair_em": 0,
            "diff_pairs_one_node_both_gold_vanilla_pair_em": 1,
            "temporal_both": 15, "temporal_both_em": 7, "temporal_both_vanilla": 57,
            "temporal_neither": 190, "temporal_neither_em": 11, "temporal_neither_vanilla": 60,
            "2hop_items": 92, "2hop_path_connected": 34, "3hop_path_connected": 31, "5hop_path_connected": 4, "7hop_path_connected": 0,
            "answer_string_in_context": 469, "answer_string_in_context_em": 172, "answer_string_absent_em": 1}

if __name__ == "__main__":
    got = recompute()
    bad = 0
    for k, v in EXPECTED.items():
        ok = got.get(k) == v
        bad += not ok
        print("%s %-50s %s (paper %s)" % ("PASS" if ok else "FAIL", k, got.get(k), v))
    sys.exit(1 if bad else 0)
