"""Vanilla RAG retrieval: BM25 over character bigrams and dense retrieval (inner product), top 20 each,
fused by reciprocal rank fusion (k = 60); the top 8 chunks go to the reader. Writes retrieval.jsonl.
Needs an OpenAI API key for the query embeddings (not for the BM25-only variant).
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3공통 as C

ROOT = C.ROOT
VAN = os.path.join(ROOT, "rag", "vanilla")
K_BM25, K_DENSE, RRF_K, K_FINAL, K_RRF_SAVE = 20, 20, 60, 8, 20
MODEL = "text-embedding-3-small"


def bigrams(text):
    g = []
    for w in re.findall(r"[0-9A-Za-z가-힣一-鿿]+", text):
        g += [w[i:i + 2] for i in range(len(w) - 1)] or [w]
    return g


def topk(scores, k):
    return np.argsort(-scores, kind="stable")[:k]


def embed(client, texts, usage, models):
    r = client.embeddings.create(model=MODEL, input=texts)
    C.add_usage(usage, MODEL, r.usage.prompt_tokens, 0)
    models.add(r.model)
    return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]


def main():
    bm25_only = "--bm25-only" in sys.argv
    from rank_bm25 import BM25Okapi
    started, t0 = C.utc_now(), time.time()
    q_path = C.QUESTIONS
    qs = [{"id": r["id"], "question": r["question"]} for r in C.load_jsonl(q_path)]
    chunk_path = os.path.join(VAN, "chunks.jsonl")
    chunks = C.load_jsonl(chunk_path)
    chunks_sha = C.sha256(chunk_path)
    ids = [c["chunk_id"] for c in chunks]
    npy_path = os.path.join(VAN, f"{MODEL}_{chunks_sha[:16]}.npy")
    CM = Q = None
    usage, models, check5 = {}, set(), None
    if not bm25_only:
        if not os.path.exists(npy_path):
            sys.exit(f"조각 임베딩 {os.path.basename(npy_path)} 없음 — 이름의 해시가 지금 chunks.jsonl({chunks_sha[:16]})과 맞는 npy 가 없다. 중단")
        CM = np.load(npy_path)
        if CM.shape[0] != len(chunks):
            sys.exit(f"npy 행 {CM.shape[0]} ≠ 조각 {len(chunks)} — 중단")
        qcache = os.path.join(VAN, f"q_{C.sha256(q_path)[:16]}.npy")
        if os.path.exists(qcache):
            Q = np.load(qcache)
        else:
            from openai import OpenAI
            client = OpenAI(api_key=C.openai_key())
            texts = [q["question"] for q in qs]
            vecs = embed(client, texts[:5], usage, models)
            u = usage.get(MODEL, {})
            check5 = {"n": 5, "prompt_tokens": u.get("prompt", 0), "models_returned": sorted(models), "passed": u.get("prompt", 0) > 0}
            print(f"5문항 확인: {check5}", flush=True)
            if not check5["passed"]:
                sys.exit("5문항 확인 실패: 임베딩 토큰 기록 0 — 중단")
            for i in range(5, len(texts), 256):
                vecs += embed(client, texts[i:i + 256], usage, models)
            Q = np.asarray(vecs, dtype=np.float32)
            np.save(qcache, Q)
        if Q.shape[0] != len(qs):
            sys.exit(f"질의 임베딩 {Q.shape[0]} ≠ 문항 {len(qs)} — 캐시가 다른 질문 파일 것이다. 중단")
    bm = BM25Okapi([bigrams(c["text"]) for c in chunks])
    out_dir = os.path.join(ROOT, "rag", "runs", "v3_vanilla" + ("_bm25only" if bm25_only else ""))
    os.makedirs(out_dir, exist_ok=True)
    ret_path = os.path.join(out_dir, "retrieval.jsonl")
    with open(ret_path, "w", encoding="utf-8") as f:
        for qi, q in enumerate(qs):
            b = topk(bm.get_scores(bigrams(q["question"])), K_BM25)
            d = topk(CM @ Q[qi], K_DENSE) if Q is not None else []
            s = {}
            for lst in (b, d):
                for rank, i in enumerate(lst):
                    s[int(i)] = s.get(int(i), 0) + 1 / (RRF_K + rank + 1)
            fused = sorted(s, key=lambda i: (-s[i], i))[:K_RRF_SAVE]
            f.write(json.dumps({"id": q["id"], "bm25": [ids[i] for i in b], "dense": [ids[i] for i in d],
                                "rrf": [ids[i] for i in fused], "final": [ids[i] for i in fused[:K_FINAL]]}, ensure_ascii=False) + "\n")
    m = C.base_manifest("vanilla_retrieval" + ("_bm25only" if bm25_only else ""), os.path.abspath(__file__), q_path)
    m.update({"run_started": started, "run_finished": C.utc_now(), "seconds": round(time.time() - t0, 1),
              "inputs": {"chunks_sha256": chunks_sha, "n_chunks": len(chunks),
                         "npy_file": None if bm25_only else os.path.basename(npy_path),
                         "npy_sha256": None if bm25_only else C.sha256(npy_path),
                         "npy_shape": None if CM is None else list(CM.shape),
                         "retrieval_sha256": C.sha256(ret_path)},
              "model_requested": None if bm25_only else MODEL, "model_returned": (sorted(models)[0] if len(models) == 1 else sorted(models) or None),
              "temperature": None, "max_tokens": None,
              "top_k": {"bm25": K_BM25, "dense": 0 if bm25_only else K_DENSE, "rrf_k": RRF_K, "rrf_saved": K_RRF_SAVE, "final": K_FINAL},
              "system_prompt": None, "system_prompt_sha256": None, "concurrency": 1,
              "usage": usage, "usd": C.usd_of(usage), "check5": check5,
              "n_rows": len(qs), "retrieval": "BM25Okapi(글자 2-gram) + 밀집(내적) → RRF", "rerank": None,
              "query_embedding_cache": None if bm25_only else os.path.basename(qcache)})
    C.set_counts(m, len(qs), len(qs), 0)
    C.write_json(os.path.join(out_dir, "manifest_retrieval.json"), m)
    mp = os.path.join(out_dir, "manifest.json")
    old = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
    if not old or str(old.get("system", "")).startswith("vanilla_retrieval") or old.get("dry_run"):
        C.write_json(mp, m)
    print(json.dumps({k: m[k] for k in ("system", "n_questions", "questions_sha256", "inputs", "model_returned", "usage", "usd", "seconds")},
                     ensure_ascii=False))
    print(f"manifest 키 누락 {C.missing_keys(m) or 0}")


if __name__ == "__main__":
    main()
