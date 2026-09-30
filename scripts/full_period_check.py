"""Section VII corpus checks over all articles of the covered period (needs numpy and a local copy of the
translation; the article texts are not redistributed).

    python scripts/full_period_check.py my_articles.jsonl

my_articles.jsonl has the format described in scripts/rebuild_corpus.py (article_ref, date_heading,
body_text), covering at least the accession year through regnal year 9. The script
  1) rebuilds the 4,273-article corpus and its 6,861 chunks (at most 600 characters, as in
     code/22_Vanilla색인.py) and checks that BM25 over character bigrams (rank_bm25 0.2.2 BM25Okapi
     formula, as in code/25_Vanilla검색.py) reproduces the released top-20 BM25 rankings;
  2) indexes all 11,275 articles of the period plus the 28 later corpus articles the same way and reports
     BM25 top-8 evidence recall (no QA run);
  3) screens the 72 immediately-before/after items: for each, it lists articles outside the corpus dated
     between the anchor and the target that contain the questioned person's Hangul name (the first word of
     the question). The screen cannot see articles that name the person by an alias. In the paper, the 10
     items that this screen flags or cannot parse were then read by an LLM agent (one reader, not blind);
     no answer changed, and for TR-003b the adjacent article in the full period differs (08-02-10[02]) while
     the answer is the same.
"""
import collections
import csv
import json
import math
import os
import re
import sys

import numpy as np

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD = [0]


def jl(*a):
    return [json.loads(l) for l in open(os.path.join(H, *a), encoding="utf-8") if l.strip()]


def ck(label, got, want):
    ok = got == want
    BAD[0] += not ok
    print("%s %-60s %s (paper %s)" % ("PASS" if ok else "FAIL", label, got, want))


def split(text, limit=600):
    head, _, body = text.partition("\n\n")
    paras = [p.strip() for p in re.split(r"\n+", body) if p.strip()]
    pieces = []
    room = limit - len(head) - 2
    for p in paras:
        while len(p) > room:
            cut = max(p.rfind("다.", 0, room), p.rfind(". ", 0, room))
            cut = cut + 2 if cut > room // 2 else room
            pieces.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            pieces.append(p)
    chunks, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) + 1 > room:
            chunks.append(buf)
            buf = p
        else:
            buf = (buf + "\n" + p).strip()
    if buf:
        chunks.append(buf)
    return [head + "\n\n" + c for c in chunks] or [text]


def bigrams(text):
    g = []
    for w in re.findall(r"[0-9A-Za-z가-힣一-鿿]+", text):
        g += [w[i:i + 2] for i in range(len(w) - 1)] or [w]
    return g


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75, eps=0.25):
        self.k1, self.b, self.N = k1, b, len(corpus)
        self.dl = np.array([len(d) for d in corpus], dtype=float)
        self.avg = sum(len(d) for d in corpus) / self.N
        post, nd = collections.defaultdict(lambda: ([], [])), {}
        for i, d in enumerate(corpus):
            for w, f in collections.Counter(d).items():
                post[w][0].append(i)
                post[w][1].append(f)
                nd[w] = nd.get(w, 0) + 1
        self.post = {w: (np.array(a, dtype=np.int64), np.array(f, dtype=float)) for w, (a, f) in post.items()}
        self.idf, tot, neg = {}, 0.0, []
        for w, f in nd.items():
            v = math.log(self.N - f + 0.5) - math.log(f + 0.5)
            self.idf[w] = v
            tot += v
            if v < 0:
                neg.append(w)
        e = eps * tot / len(self.idf)
        for w in neg:
            self.idf[w] = e

    def scores(self, q):
        s = np.zeros(self.N)
        for w in q:
            if w in self.post:
                idx, tf = self.post[w]
                s[idx] += (self.idf.get(w) or 0) * (tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[idx] / self.avg)))
        return s


def main(src):
    corpus_ids = [r["id"] for r in csv.DictReader(open(os.path.join(H, "data", "article_ids.tsv"), encoding="utf-8"), delimiter="\t")]
    have = {}
    for line in open(src, encoding="utf-8"):
        a = json.loads(line)
        have.setdefault(a["article_ref"], a)
    text = lambda k: have[k]["date_heading"] + "\n\n" + have[k]["body_text"] + "\n"
    period = sorted(k for k in have if k[:2] <= "09")
    full = sorted(set(period) | set(corpus_ids))
    missing = [k for k in full if k not in have]
    if missing:
        sys.exit("missing articles in %s: %d (first: %s)" % (src, len(missing), missing[:5]))
    ck("articles in the period / with the later corpus articles", (len(period), len(full)), (11275, 11303))
    G = {r["id"]: r for r in jl("data", "answers.jsonl")}
    Q = {r["id"]: r for r in jl("data", "questions.jsonl")}
    RET = {r["id"]: r for r in jl("runs", "vanilla", "retrieval.jsonl")}
    ids = list(G)
    in_corpus = set(corpus_ids)

    def run(keys, k):
        chunks = [("%s#%d" % (a, n), t) for a in keys for n, t in enumerate(split(text(a)))]
        bm = BM25([bigrams(t) for _, t in chunks])
        per, same20, outside = {}, 0, 0
        for i in ids:
            top = [chunks[j][0] for j in np.argsort(-bm.scores(bigrams(Q[i]["question"])), kind="stable")[:20]]
            same20 += top == RET[i]["bm25"]
            d = {c.split("#")[0] for c in top[:k]}
            outside += sum(c.split("#")[0] not in in_corpus for c in top[:k])
            ev = set(G[i]["evidence_ids"])
            h = len(ev & d)
            per[i] = (h > 0, h == len(ev), h / len(ev))
        return len(chunks), per, same20, outside

    n1, p1, s1, _ = run(sorted(corpus_ids), 8)
    ck("corpus chunks / stored top-20 BM25 rankings reproduced", (n1, s1), (6861, 975))
    n2, p2, _, out2 = run(full, 8)
    summ = lambda p: (round(100.0 * sum(v[0] for v in p.values()) / 975, 1), round(100.0 * sum(v[1] for v in p.values()) / 975, 1),
                      round(100.0 * sum(v[2] for v in p.values()) / 975, 1))
    ck("BM25 top-8 recall any, all, coverage: corpus", summ(p1), (84.7, 37.4, 57.3))
    ck("BM25 top-8 recall any, all, coverage: full period (chunks %d)" % n2, summ(p2), (80.0, 33.9, 53.3))
    print("top-8 chunks from outside the corpus: %d of %d" % (out2, 975 * 8))

    def key(a):
        m = re.match(r"(\d\d)-(윤)?(\d\d)-(\d\d|○○)\[(\d\d)\]", a)
        return (int(m.group(1)), int(m.group(3)), 1 if m.group(2) else 0, 99 if m.group(4) == "○○" else int(m.group(4)), int(m.group(5)))

    order = sorted(full, key=key)
    pos = {a: n for n, a in enumerate(order)}
    EV = {r["id"]: r for r in jl("data", "evidence.jsonl")}
    items = [i for i in ids if G[i]["category"] == "시간추론" and EV[i]["relation"] in ("직전", "직후")]
    flagged, clear, noname = [], [], []
    for i in items:
        m = re.match(r"^'?([가-힣]{2,})(?:\([^)]*\))?(?:이|가|은|는|의|에게|께서|와|과)\s", Q[i]["question"])
        if not m:
            noname.append(i)
            continue
        a, t = pos[EV[i]["anchor"]["source_id"]], pos[EV[i]["target"]["source_id"]]
        hits = [x for x in order[min(a, t) + 1:max(a, t)] if x not in in_corpus and m.group(1) in text(x)]
        (flagged if hits else clear).append(i)
        if hits:
            print("  flagged %s (%s): %s" % (i, m.group(1), ", ".join(hits)))
    ck("immediately-before/after items: total, clear, flagged, name not parsed", (len(items), len(clear), len(flagged), len(noname)), (72, 62, 6, 4))
    print("  name not parsed:", ", ".join(noname))
    print("failures:", BAD[0])
    sys.exit(1 if BAD[0] else 0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
