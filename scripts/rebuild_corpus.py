"""Rebuild the SejongQA retrieval corpus (4,273 articles) from a local copy of the translation.

The article texts are NOT redistributed. Obtain the modern Korean translation of the Sejong sillok
from the National Institute of Korean History (https://sillok.history.go.kr) under its terms of use,
and convert it to a JSONL file with one article per line:
    {"article_ref": "00-08-11[01]", "date_heading": "세종 즉위년 무술(1418) 8월 11일(무자) 양력 1418-09-10",
     "body_text": "<translated body without title, source note, classification and footnotes>"}
article_ref = YY-MM-DD[NN] (YY = regnal year, 00 = accession year; leap months as YY-윤MM-DD[NN]).

Usage:
    python scripts/rebuild_corpus.py my_articles.jsonl corpus/articles.csv
    python scripts/rebuild_corpus.py --restore-prompt corpus/articles.csv

The second form refills the two example articles masked in graphrag/prompts/extract_graph.txt
(<<ARTICLE 01-06-13[03]>>, <<ARTICLE 06-04-24[02]>>) from the rebuilt corpus, writes the restored
prompt next to it (extract_graph.restored.txt) and checks its SHA-256 against the prompt used in
the paper (prefix 5c62347f4685df93). The first form does the same automatically after rebuilding.

Each rebuilt text is  date_heading + "\\n\\n" + body_text + "\\n"  and is checked against
data/article_ids.tsv (length and SHA-256). Mismatches usually mean a different edition or
different removal of editorial material; the script reports them and still writes the file.
"""
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = os.path.join(HERE, "graphrag", "prompts", "extract_graph.txt")
PROMPT_SHA256_PREFIX = "5c62347f4685df93"
MASKED = ("01-06-13[03]", "06-04-24[02]")


def restore_prompt(texts, write=True):
    """texts: {article_id: rebuilt text}. Returns (sha256 hex of restored prompt, ok)."""
    with open(PROMPT, encoding="utf-8", newline="") as f:
        eg = f.read()
    for k in MASKED:
        eg = eg.replace("<<ARTICLE %s>>" % k, texts[k].strip().replace("\n", "\r\n"))  # prompt file uses CRLF
    h = hashlib.sha256(eg.encode("utf-8")).hexdigest()
    if write:
        with open(PROMPT.replace(".txt", ".restored.txt"), "w", encoding="utf-8", newline="") as f:
            f.write(eg)
    return h, h.startswith(PROMPT_SHA256_PREFIX)


def load_csv(path):
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r["text"] for r in csv.DictReader(f)}


def main(src, dst):
    want = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "article_ids.tsv"), encoding="utf-8"), delimiter="\t"):
        want[r["id"]] = (int(r["n_chars"]), r["sha256_text"])
    have = {}
    for line in open(src, encoding="utf-8"):
        a = json.loads(line)
        have.setdefault(a["article_ref"], a)
    rows, missing, mismatch = [], [], []
    for i in sorted(want):
        if i not in have:
            missing.append(i)
            continue
        text = have[i]["date_heading"] + "\n\n" + have[i]["body_text"] + "\n"
        if (len(text), hashlib.sha256(text.encode("utf-8")).hexdigest()) != want[i]:
            mismatch.append(i)
        rows.append({"id": i, "title": i, "text": text})
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "text"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print("written %d / %d articles; missing %d; hash mismatch %d" % (len(rows), len(want), len(missing), len(mismatch)))
    if missing[:10]:
        print("missing (first 10):", missing[:10])
    if mismatch[:10]:
        print("mismatch (first 10):", mismatch[:10])
    if all(k in {r["id"] for r in rows} for k in MASKED):
        h, ok = restore_prompt({r["id"]: r["text"] for r in rows})
        print("extract_graph prompt restored: sha256 %s... %s" % (h[:16], "matches the paper" if ok else "DOES NOT match"))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--restore-prompt":
        h, ok = restore_prompt(load_csv(sys.argv[2]))
        print("extract_graph prompt restored: sha256 %s... %s" % (h[:16], "matches the paper" if ok else "DOES NOT match"))
        sys.exit(0 if ok else 1)
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
