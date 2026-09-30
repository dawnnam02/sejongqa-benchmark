"""Regression tests for scripts/score.py (python scripts/test_score.py).

Checks the normalization rules and documents known limitations, then verifies that the released
answers reproduce the EM numerators of Table III.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import em, norm, score  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    # (pred, gold, aliases, expected, note)
    ("", "경복궁", [], 0.0, "empty answer is wrong"),
    ("!!", "경복궁", [], 0.0, "punctuation-only answer is wrong"),
    ("경복궁(景福宮)", "경복궁", [], 1.0, "parenthesized Hanja is ignored"),
    ("景福宮", "연희궁", ["衍禧宮"], 0.0, "Hanja-only alias is dropped; Hanja-only answer does not match"),
    ("양녕대군", "이제", ["양녕대군"], 1.0, "alias match"),
    ("근거 부족", "경복궁", [], 0.0, "abstention is wrong"),
    ("1422년(임인년)", "1422", [], 0.0, "known limitation: unit suffix is not normalized"),
    ("4~5일", "45일", [], 1.0, "known limitation: range sign is removed, so 4~5 and 45 collide"),
]


def main():
    bad = 0
    for pred, gold, aliases, exp, note in CASES:
        got = em(pred, gold, aliases)
        ok = got == exp
        bad += not ok
        print("%s %-60s pred=%r gold=%r -> %s" % ("PASS" if ok else "FAIL", note, pred, gold, got))
    expected = {"closed": 18, "vanilla": 365, "gold": 728, "graphrag": 173, "vanilla_bm25": 396,
                "graphrag_ctx": 29}
    for s, k in expected.items():
        p = os.path.join(HERE, "runs", s, "answers.jsonl")
        if os.path.exists(p):
            got = score(p, boot=10)["ALL"]["em_k"]
            ok = got == k
            bad += not ok
            print("%s Table III %s EM numerator %d (expected %d)" % ("PASS" if ok else "FAIL", s, got, k))
    # 200-question re-read: native GraphRAG on the same 200 items
    sp = os.path.join(HERE, "runs", "graphrag_ctx", "sample.json")
    if os.path.exists(sp):
        import json
        ids = set(json.load(open(sp, encoding="utf-8"))["ids"])
        gp = os.path.join(HERE, "runs", "graphrag", "answers.jsonl")
        gold = {g["id"]: g for g in (json.loads(l) for l in open(os.path.join(HERE, "data", "answers.jsonl"), encoding="utf-8"))}
        k = sum(em(a.get("answer", ""), gold[a["id"]]["answer"], gold[a["id"]].get("answer_aliases"))
                for a in (json.loads(l) for l in open(gp, encoding="utf-8")) if a["id"] in ids)
        ok = k == 28
        bad += not ok
        print("%s native GraphRAG on the 200 re-read items: EM numerator %d (expected 28)" % ("PASS" if ok else "FAIL", k))
    # Section VI counts from the released diagnostics tables
    if os.path.exists(os.path.join(HERE, "diagnostics", "identity_items.tsv")):
        import vi_recompute
        got = vi_recompute.recompute()
        for key, v in vi_recompute.EXPECTED.items():
            ok = got.get(key) == v
            bad += not ok
            if not ok:
                print("FAIL Section VI", key, got.get(key), "expected", v)
        print("Section VI counts checked:", len(vi_recompute.EXPECTED))
    # The extraction prompt is released with its two example articles masked; with a local corpus
    # (env SEJONGQA_CORPUS = rebuilt articles.csv) the restored prompt must match the paper's hash
    import rebuild_corpus
    with open(rebuild_corpus.PROMPT, encoding="utf-8", newline="") as f:
        eg = f.read()
    ok = all(("<<ARTICLE %s>>" % k) in eg for k in rebuild_corpus.MASKED)
    bad += not ok
    print("%s extract_graph.txt masks the example articles %s" % ("PASS" if ok else "FAIL", rebuild_corpus.MASKED))
    cp = os.environ.get("SEJONGQA_CORPUS")
    if cp and os.path.exists(cp):
        h, ok = rebuild_corpus.restore_prompt(rebuild_corpus.load_csv(cp), write=False)
        bad += not ok
        print("%s restored extract_graph prompt sha256 %s (expected prefix %s)" % ("PASS" if ok else "FAIL", h[:16], rebuild_corpus.PROMPT_SHA256_PREFIX))
    else:
        print("SKIP restored-prompt hash check (set SEJONGQA_CORPUS to a rebuilt articles.csv)")
    print("failures:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
