"""SejongQA scorer (self-contained, standard library only).

Usage:
    python scripts/score.py runs/vanilla/answers.jsonl [--judged runs/vanilla/judged.jsonl]

Reproduces the EM, Pair EM, LLM-Eq and abstention columns of Table III from the released
answers. EM normalization (same as reference/27_채점.py):
    NFKC -> remove parenthesized text -> remove Hanja -> remove punctuation and spaces -> lowercase.
    An answer is correct if it equals the gold answer or any alias after normalization.
    Gold candidates that become empty after normalization are ignored (unless all are empty).
Pair EM: a minimal pair (Identity or Temporal, same pair_id) counts only if both items are EM-correct.
Abstention: the answer contains the Korean string "근거 부족" ("insufficient evidence"); it counts as an error.
95% CI: 2,000 bootstrap resamples, seed 20260929, pairs (Identity/Temporal) or items (Multi-hop) as units,
stratified by category. Intervals can differ in the last digit from the paper, which used reference/27_채점.py.
"""
import argparse
import collections
import json
import os
import random
import re
import unicodedata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABSTAIN = "근거 부족"


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
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def score(answers_path, judged_path=None, gold_path=None, boot=2000, seed=20260929):
    gold = {g["id"]: g for g in load(gold_path or os.path.join(HERE, "data", "answers.jsonl"))}
    pred = {a["id"]: a.get("answer", "") for a in load(answers_path)}
    judge = {j["id"]: float(j["judge"]) for j in load(judged_path)} if judged_path else {}
    rows = []
    for i, g in gold.items():
        p = pred.get(i, "")
        rows.append({"id": i, "cat": g["category"], "pair": g.get("pair_id"),
                     "em": em(p, g["answer"], g.get("answer_aliases")),
                     "abstain": float(ABSTAIN in p), "judge": judge.get(i)})
    out = {}
    cats = sorted({r["cat"] for r in rows}) + ["ALL"]
    for c in cats:
        v = [r for r in rows if c == "ALL" or r["cat"] == c]
        pe = collections.defaultdict(list)
        for r in v:
            if r["pair"]:
                pe[r["pair"]].append(r["em"])
        pairs = [all(x) for x in pe.values() if len(x) == 2]
        d = {"n": len(v), "em_k": int(sum(r["em"] for r in v)), "abstain_k": int(sum(r["abstain"] for r in v))}
        if pairs:
            d.update(pairs_n=len(pairs), pair_em_k=sum(pairs))
        if judge:
            d["llm_eq_k"] = int(sum(r["judge"] or 0 for r in v))
        out[c] = d
    # bootstrap CI for EM
    units = collections.defaultdict(list)
    for r in rows:
        units[(r["cat"], r["pair"] or r["id"])].append(r["em"])
    bycat = collections.defaultdict(list)
    for (c, _), ems in units.items():
        bycat[c].append(ems)
    rng = random.Random(seed)
    samples = collections.defaultdict(list)
    for _ in range(boot):
        tot_s = tot_n = 0
        for c, us in bycat.items():
            s = n = 0
            for ems in (rng.choice(us) for _ in range(len(us))):
                s += sum(ems)
                n += len(ems)
            samples[c].append(s / n)
            tot_s += s
            tot_n += n
        samples["ALL"].append(tot_s / tot_n)
    for c in cats:
        xs = sorted(samples[c])
        out[c]["em_ci"] = [round(xs[int(0.025 * boot)], 3), round(xs[int(0.975 * boot) - 1], 3)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--judged")
    ap.add_argument("--gold")
    a = ap.parse_args()
    res = score(a.answers, a.judged, a.gold)
    for c, d in res.items():
        line = "%-10s EM %d/%d = %.1f%% CI [%.1f, %.1f]" % (c, d["em_k"], d["n"], 100 * d["em_k"] / d["n"], 100 * d["em_ci"][0], 100 * d["em_ci"][1])
        if "pairs_n" in d:
            line += " | Pair EM %d/%d = %.1f%%" % (d["pair_em_k"], d["pairs_n"], 100 * d["pair_em_k"] / d["pairs_n"])
        if "llm_eq_k" in d:
            line += " | LLM-Eq %d/%d = %.1f%%" % (d["llm_eq_k"], d["n"], 100 * d["llm_eq_k"] / d["n"])
        line += " | abstain %d/%d = %.1f%%" % (d["abstain_k"], d["n"], 100 * d["abstain_k"] / d["n"])
        print(line)


if __name__ == "__main__":
    main()
