"""Vanilla RAG retrieval (optional): dense top-10 over the GraphRAG text units.

Not needed to reproduce the paper's numbers: check_paper.py and score.py read the
saved results in runs/vanilla/ (retrieval.jsonl, answers.jsonl). This script shows how
those retrievals were made, for anyone who rebuilds the index and wants to rerun them.

Method: cosine similarity between the question embedding and the GraphRAG text-unit
embeddings (the same 4,600 text units and embedding model as Basic Search), top 10.
No graph, reranking or query rewriting.

Requirements:
    - an index of the rebuilt corpus (scripts/rebuild_corpus.py) made with
      Microsoft GraphRAG 2.7.2 and graphrag/settings.yaml; the output dir holds
      text_units.parquet and lancedb/ with the table "default-text_unit-text"
      (the table name follows the GraphRAG 2.7.2 layout)
    - Python packages: numpy, pandas, pyarrow, lancedb, openai
      (pip install numpy pandas pyarrow lancedb openai; any versions that read the
      GraphRAG 2.7.2 output should work)
    - question embeddings from OpenAI text-embedding-3-small (needs OPENAI_API_KEY
      and is billed), or a saved array passed with --query-npy

Usage:
    python scripts/vanilla_retrieval.py <graphrag output dir> data/questions.jsonl retrieval.jsonl

Each output line is {"id", "retrieved": ten "<article id>#tu<text-unit human_readable_id>"},
the format of runs/vanilla/retrieval.jsonl.
"""
import argparse
import json
import os
import sys

K = 10
MODEL = "text-embedding-3-small"
TABLE = "default-text_unit-text"


def main():
    ap = argparse.ArgumentParser(description="Vanilla RAG retrieval: dense top-10 over GraphRAG text units (optional).")
    ap.add_argument("graphrag_output", help="GraphRAG output dir with text_units.parquet and lancedb/")
    ap.add_argument("questions", help="question file, e.g. data/questions.jsonl")
    ap.add_argument("out", help="output JSONL, one line per question with the top-10 text units")
    ap.add_argument("--query-npy", default=None,
                    help="saved question embeddings (.npy, float, one row per question in questions file order); "
                         "skips the OpenAI embedding call")
    a = ap.parse_args()

    # heavy dependencies are imported here so that --help works without them
    import lancedb
    import numpy as np
    import pandas as pd
    with open(a.questions, encoding="utf-8") as f:
        qs = [json.loads(l) for l in f if l.strip()]
    tu = pd.read_parquet(os.path.join(a.graphrag_output, "text_units.parquet"))
    vec = lancedb.connect(os.path.join(a.graphrag_output, "lancedb")).open_table(TABLE).to_pandas()
    vmap = {r.id: np.asarray(r.vector, dtype=np.float32) for r in vec.itertuples()}

    # one row per text unit, labelled with its article id
    ids, rows = [], []
    for r in tu.itertuples():
        docs = list(r.document_ids)
        if len(docs) != 1:
            sys.exit("error: text unit %s spans %d articles; chunks must be split at article boundaries"
                     % (r.human_readable_id, len(docs)))
        if r.id not in vmap:
            sys.exit("error: no vector for text unit %s in %s" % (r.human_readable_id, TABLE))
        ids.append("%s#tu%d" % (docs[0], r.human_readable_id))
        rows.append(vmap[r.id])
    M = np.stack(rows)
    M /= np.linalg.norm(M, axis=1, keepdims=True)

    if a.query_npy:
        Q = np.load(a.query_npy).astype(np.float32)
    else:
        from openai import OpenAI
        client = OpenAI()
        Q = []
        for i in range(0, len(qs), 256):
            r = client.embeddings.create(model=MODEL, input=[q["question"] for q in qs[i:i + 256]])
            Q += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
        Q = np.asarray(Q, dtype=np.float32)
    if len(Q) != len(qs):
        sys.exit("error: %d question vectors for %d questions" % (len(Q), len(qs)))
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)

    S = Q @ M.T
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for qi, q in enumerate(qs):
            top = np.argsort(-S[qi], kind="stable")[:K]
            f.write(json.dumps({"id": q["id"], "retrieved": [ids[i] for i in top]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
