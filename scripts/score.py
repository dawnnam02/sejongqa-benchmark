"""Score a SejongQA answer file (standard library only).

    python scripts/score.py runs/graphrag/answers.jsonl
    python scripts/score.py runs/graphrag_ctx/answers.jsonl      # 200-question subset

Reports EM by question type (Alias, Confusable, Temporal, Relation Chain, All),
Pair EM for the Identity and Temporal contrastive pairs, and the abstention rate,
as in Table II of the paper.

Input: one JSON object per line with at least "id" and "answer".
Only the question IDs present in the answer file are scored. If the file covers
fewer than all 975 questions, the output says so; a pair enters Pair EM only if
both of its questions are present.

EM normalization (the rule used in the paper):
    NFKC -> remove parenthesized text -> remove Chinese characters
    -> remove punctuation and whitespace -> lowercase.
An answer is correct if it equals the gold answer or any alias after normalization.
Gold candidates that become empty after normalization are ignored (unless all are).
Abstention: the answer contains "근거 부족" ("insufficient evidence"); it is never
equal to a gold answer, so abstentions count as incorrect.
"""
import argparse
import collections
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABSTAIN = "근거 부족"

# question types as data labels -> names used in the paper
TYPES = [("Alias", lambda g: g["sub"] == "같은 사람"),
         ("Confusable", lambda g: g["sub"] == "다른 사람"),
         ("Temporal", lambda g: g["category"] == "시간추론"),
         ("Relation Chain", lambda g: g["category"] == "멀티홉"),
         ("All", lambda g: True)]
PAIRS = [("Identity", "인물판정"), ("Temporal", "시간추론")]


def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[一-鿿]", "", s)
    return re.sub(r"[\s\W_]+", "", s).lower()


def em(pred, gold, aliases):
    golds = [gold] + list(aliases or [])
    golds = [g for g in golds if norm(g)] or golds
    p = norm(pred)
    return float(bool(p) and any(p == norm(g) for g in golds))


def load(path):
    # text mode reads both LF and CRLF files
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def score(answers_path, gold_path=None):
    gold = {g["id"]: g for g in load(gold_path or os.path.join(HERE, "data", "answers.jsonl"))}
    pred = {a["id"]: a.get("answer", "") for a in load(answers_path)}
    unknown = sorted(set(pred) - set(gold))
    if unknown:
        raise SystemExit("unknown question ids in %s: %s" % (answers_path, unknown[:5]))
    ids = [i for i in gold if i in pred]
    ok = {i: em(pred[i], gold[i]["answer"], gold[i].get("answer_aliases")) for i in ids}
    out = {"n_gold": len(gold), "n_scored": len(ids), "types": {}, "pairs": {}}
    for name, f in TYPES:
        sel = [i for i in ids if f(gold[i])]
        out["types"][name] = {"n": len(sel), "em": int(sum(ok[i] for i in sel)),
                              "abstain": sum(ABSTAIN in pred[i] for i in sel)}
    for name, cat in PAIRS:
        pairs = collections.defaultdict(list)
        for i in ids:
            if gold[i]["category"] == cat and gold[i].get("pair_id"):
                pairs[gold[i]["pair_id"]].append(ok[i])
        full = [v for v in pairs.values() if len(v) == 2]
        out["pairs"][name] = {"n": len(full), "correct": sum(all(v) for v in full)}
    return out


def rate(k, n):
    return "%3d/%3d = %5.1f%%" % (k, n, 100.0 * k / n) if n else "%3d/  0 =     -" % k


def main():
    ap = argparse.ArgumentParser(
        description="Score a SejongQA answer file: EM by question type, Pair EM, abstention rate.")
    ap.add_argument("answers", help="answer file (JSONL with 'id' and 'answer'), e.g. runs/gold/answers.jsonl")
    ap.add_argument("--gold", help="gold answer file (default: data/answers.jsonl)")
    a = ap.parse_args()
    res = score(a.answers, a.gold)
    if res["n_scored"] < res["n_gold"]:
        print("subset: scoring the %d of %d questions present in the answer file"
              % (res["n_scored"], res["n_gold"]))
    print("%-16s %-21s %s" % ("Type", "EM", "Abstention"))
    for name, d in res["types"].items():
        print("%-16s %-21s %s" % (name, rate(d["em"], d["n"]), rate(d["abstain"], d["n"])))
    for name, d in res["pairs"].items():
        print("%-16s %s" % ("Pair EM " + name, rate(d["correct"], d["n"])))


if __name__ == "__main__":
    main()
