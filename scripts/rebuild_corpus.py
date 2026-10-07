"""Rebuild the SejongQA retrieval corpus (4,273 articles) from a local copy of the translation.

The article texts are not redistributed. Obtain the new modern Korean translation of the
Sejong Sillok (Institute for the Translation of Korean Classics, Korean Classics DB,
https://db.itkc.or.kr) under its terms of use, article by article from the provider's web
pages or Open API, and convert it to a JSONL file with one article per line:

    {"article_ref": "00-08-11[01]",
     "date_heading": "세종 즉위년 무술(1418) 8월 11일(무자)\\u00a0양력 1418-09-10",
     "body_text": "<translated body>"}

date_heading: the date line of the article. The day and "양력" (solar date) are separated
    by a no-break space (U+00A0), not an ordinary space. Articles without a known day use
    "세종 9년 정미(1427) 9월\\u00a0추록(追錄)\\u00a0양력 1427-08-00" (two U+00A0).
body_text: the translated body without title, source note, classification line and
    footnote markers or notes; paragraphs joined by a single "\\n", no trailing newline.

article_ref = YY-MM-DD[NN]: YY regnal year (00 = accession year, 1418), MM lunar month,
DD day, NN article number within the day. Examples:
    00-08-11[01]     accession year, month 8, day 11, first article
    04-윤12-26[02]   leap month: "윤" before the month
    09-08-○○[01]     day unknown: "○○" in place of the day

Usage:
    python scripts/rebuild_corpus.py my_articles.jsonl corpus/articles.csv
    python scripts/rebuild_corpus.py --restore-prompt corpus/articles.csv

Each rebuilt text is  date_heading + "\\n\\n" + body_text + "\\n"  and is checked against
data/article_ids.tsv (length and SHA-256). Mismatches usually mean a different edition or a
different removal of editorial material. The CSV is written in any case, but the script
exits with status 1 if any article is missing or does not match.

Prompt restore: graphrag/prompts/extract_graph.txt has two example articles masked as
<<ARTICLE 01-06-13[03]>> and <<ARTICLE 06-04-24[02]>>. The script refills them from the
rebuilt corpus, writes graphrag/prompts/extract_graph.restored.txt and checks its SHA-256
against the prompt used in the paper (prefix 5c62347f4685df93). The first form does this
automatically after rebuilding; the second does it from an existing corpus CSV.
Index with the restored prompt: copy it over extract_graph.txt, or point
extract_graph.prompt in graphrag/settings.yaml to it. The restored file contains article
text, so do not commit it.
"""
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = os.path.join(HERE, "graphrag", "prompts", "extract_graph.txt")
RESTORED = os.path.join(HERE, "graphrag", "prompts", "extract_graph.restored.txt")
PROMPT_SHA256_PREFIX = "5c62347f4685df93"
MASKED = ("01-06-13[03]", "06-04-24[02]")


def restore_prompt(texts, write=True):
    """texts: {article_id: rebuilt text}. Returns (sha256 hex of restored prompt, ok)."""
    # read bytes as stored; the released prompt uses CRLF line endings
    with open(PROMPT, encoding="utf-8", newline="") as f:
        eg = f.read()
    for k in MASKED:
        eg = eg.replace("<<ARTICLE %s>>" % k, texts[k].strip().replace("\n", "\r\n"))
    h = hashlib.sha256(eg.encode("utf-8")).hexdigest()
    if write:
        with open(RESTORED, "w", encoding="utf-8", newline="") as f:
            f.write(eg)
    return h, h.startswith(PROMPT_SHA256_PREFIX)


def report_prompt(h, ok):
    print("extract_graph prompt restored: sha256 %s... %s" % (h[:16], "matches the paper" if ok else "DOES NOT match"))
    if ok:
        print("written %s; index with this file (copy it over extract_graph.txt or point settings.yaml to it)"
              % os.path.relpath(RESTORED, HERE))
    else:
        print("error: expected sha256 prefix %s. Check that the two example articles match data/article_ids.tsv "
              "and that graphrag/prompts/extract_graph.txt still has CRLF line endings "
              "(git line-ending conversion changes the hash)." % PROMPT_SHA256_PREFIX)


def load_csv(path):
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r["text"] for r in csv.DictReader(f)}


def main(src, dst):
    want = {}
    with open(os.path.join(HERE, "data", "article_ids.tsv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            want[r["id"]] = (int(r["n_chars"]), r["sha256_text"])
    have = {}
    with open(src, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                a = json.loads(line)
                have.setdefault(a["article_ref"], a)  # first occurrence wins
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
    if missing:
        print("missing (first 10):", missing[:10])
    if mismatch:
        print("mismatch (first 10):", mismatch[:10])
        print("hint: the date heading uses U+00A0 before '양력'; see the docstring for the text format")
    ok = not missing and not mismatch
    if all(k in have for k in MASKED):
        h, p_ok = restore_prompt({r["id"]: r["text"] for r in rows})
        report_prompt(h, p_ok)
        ok = ok and p_ok
    return ok


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--restore-prompt":
        texts = load_csv(sys.argv[2])
        lost = [k for k in MASKED if k not in texts]
        if lost:
            sys.exit("error: %s not in %s" % (", ".join(lost), sys.argv[2]))
        h, ok = restore_prompt(texts)
        report_prompt(h, ok)
        sys.exit(0 if ok else 1)
    if len(sys.argv) != 3 or sys.argv[1].startswith("-"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help") else 2)
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
