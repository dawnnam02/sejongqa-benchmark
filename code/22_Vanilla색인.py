"""Vanilla RAG index: splits each corpus article into chunks of at most 600 characters at paragraph
boundaries (the date heading is repeated on later chunks) and embeds them with text-embedding-3-small.
Needs an OpenAI API key for the embeddings.
"""
import csv
import hashlib
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "rag", "vanilla")
EMBED = os.environ.get("E5_MODEL", "intfloat/multilingual-e5-base")
PROVIDER = "openai"
OPENAI_MODEL = "text-embedding-3-small"
LIMIT = 600


def split(text, limit):
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
            buf = f"{buf}\n{p}".strip()
    if buf:
        chunks.append(buf)
    return [f"{head}\n\n{c}" for c in chunks] or [text]


def main():
    src = os.path.join(ROOT, "rag", "graphrag", "input", "articles.csv")
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    chunks = []
    for r in rows:
        for n, t in enumerate(split(r["text"], LIMIT)):
            chunks.append({"chunk_id": f"{r['id']}#{n}", "doc_id": r["id"], "text": t})
    os.makedirs(OUT, exist_ok=True)
    cpath = os.path.join(OUT, "chunks.jsonl")
    with open(cpath, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(cpath, "rb").read()).hexdigest()[:16]
    docs = {c["doc_id"] for c in chunks}
    no_head = sum(1 for c in chunks if not c["text"].startswith("세종"))

    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c["text"].partition("\n\n")[2])
    squash = lambda s: re.sub(r"\s+", "", s)
    lost = sum(1 for r in rows if squash(r["text"].partition("\n\n")[2]) != squash("".join(by_doc.get(r["id"], []))))
    t0 = time.time()
    import numpy as np
    usage_line = ""
    if PROVIDER == "openai":
        from openai import OpenAI
        key = next(re.match(r"\s*OPENAI_API_KEY\s*=\s*\"?([^\"\s]+)", l).group(1)
                   for l in open(os.path.join(ROOT, "rag", "graphrag", ".env"), encoding="utf-8-sig")
                   if re.match(r"\s*OPENAI_API_KEY\s*=", l))
        client = OpenAI(api_key=key)
        epath = os.path.join(OUT, f"{OPENAI_MODEL}_{h}.npy")
        vecs, used = [], 0
        if not os.path.exists(epath):
            for i in range(0, len(chunks), 256):
                r = client.embeddings.create(model=OPENAI_MODEL, input=[c["text"] for c in chunks[i:i + 256]])
                vecs += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
                used += r.usage.prompt_tokens
            arr = np.asarray(vecs, dtype=np.float32)
            np.save(epath, arr)
            usage_line = f"OpenAI {OPENAI_MODEL}: 입력 {used:,} 토큰 · ${used * 0.02 / 1e6:.4f}"
        arr = np.load(epath)
        checks_emb = [("벡터 수", arr.shape[0], len(chunks), arr.shape[0] == len(chunks)),
                      ("차원", arr.shape[1], 1536, arr.shape[1] == 1536)]
    else:
        from sentence_transformers import SentenceTransformer
        epath = os.path.join(OUT, f"e5_{h}.npy")
        if not os.path.exists(epath):
            m = SentenceTransformer(EMBED, device="cpu")
            np.save(epath, m.encode(["passage: " + c["text"] for c in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=True))
        arr = np.load(epath)
        checks_emb = [("벡터 수", arr.shape[0], len(chunks), arr.shape[0] == len(chunks))]
    rep = [f"Vanilla RAG 색인 보고 — tools/22_Vanilla색인.py (임베딩 {PROVIDER})", "",
           "[checks]  항목 | 실제 | 기대 | 일치",
           f"  문서 수 | {len(docs)} | {len(rows)} | {'○' if len(docs) == len(rows) else '✗'}",
           f"  {LIMIT}자 초과 조각 | {sum(len(c['text']) > LIMIT for c in chunks)} | 0 | {'○' if all(len(c['text']) <= LIMIT for c in chunks) else '✗'}",
           f"  날짜 머리글 없는 조각 | {no_head} | 0 | {'○' if no_head == 0 else '✗'}",
           f"  본문 손실 | {lost} | 0 | {'○' if lost == 0 else '✗'}"]
    rep += [f"  {a} | {b} | {c} | {'○' if ok else '✗'}" for a, b, c, ok in checks_emb]
    rep += ["", f"조각 {len(chunks)} · chunks.jsonl sha256 {h} · 임베딩 {os.path.basename(epath)} · {round(time.time() - t0)}초", usage_line]
    open(os.path.join(OUT, "색인보고.txt"), "w", encoding="utf-8").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
