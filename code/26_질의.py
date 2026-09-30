"""Reader runs for Closed-Book, Vanilla RAG, Gold Evidence and the GraphRAG-context re-read
(gpt-4.1-mini, temperature 0, max 100 output tokens). Writes answers.jsonl and manifest.json.
Needs an OpenAI API key.
"""
import argparse
import asyncio
import collections
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3공통 as C

ROOT = C.ROOT
EXP = C.EXP
CATS = ["인물판정", "시간추론", "멀티홉"]
MODEL = "gpt-4.1-mini"
TEMPERATURE, MAX_TOKENS, CONCURRENCY = 0, 100, 8
SYSTEM = ("당신은 조선 세종대 실록 기사에 관한 질문에 답한다.\n"
          "{ctx_rule}\n"
          "답변 형식: 짧은 답 하나. 단어나 짧은 구로만 답하고 설명·근거 표기는 쓰지 않는다. "
          "답할 수 없으면 '근거 부족'이라고만 쓴다.")
CTX_RULE = {"closed": "참고 자료는 주어지지 않는다. 알고 있는 지식만으로 답하라.",
            "vanilla": "아래 [자료]에 주어진 기사 조각만 근거로 답하라. 자료 밖의 지식을 쓰지 마라.",
            "gold": "아래 [자료]에 주어진 기사만 근거로 답하라. 자료 밖의 지식을 쓰지 마라.",
            "graphrag_ctx": "아래 [자료]에 주어진 표(개체·관계·보고서·기사 조각)만 근거로 답하라. 자료 밖의 지식을 쓰지 마라."}
TOP_K = 8
GOLD_TOKEN_LIMIT = 16000
EST_OUT_TOKENS = 15

FAMILY = {"하루": {"전날", "이튿날"}, "달": {"그달 초", "그달 말", "전월"},
          "해": {"전해", "이듬해", "익년", "후년", "그해 초", "그해 중순", "그해 말", "연초", "연말", "그해", "당해"},
          "순서": {"직전", "직후", "앞서", "이전에", "이후", "그 후", "장차", "기왕에"}}
ALIAS_RULES = [("표기 이형", ("표기이형", "표기 이형")),
               ("호·이칭·약칭", ("약칭", "이칭", "자호", "옛 이름", "아명")),
               ("관직·관계 호칭", ("관직호칭", "관계호칭", "칭호")),
               ("봉호·군호", ("봉호", "군호")),
               ("시호·묘호", ("시호", "묘호"))]
ALIAS_CLASSES = [c for c, _ in ALIAS_RULES]


def alias_class(alias_type):
    t = alias_type or ""
    return next((c for c, keys in ALIAS_RULES if any(k in t for k in keys)), None)


def relation_family(r):
    fam = r.get("family")
    by_rel = next((k for k, s in FAMILY.items() if r.get("relation") in s), None)
    return fam or by_rel, (fam and by_rel and fam != by_rel)


def build():
    os.makedirs(EXP, exist_ok=True)
    qs, gold, bad = [], [], collections.defaultdict(list)
    for cat in CATS:
        for r in C.load_jsonl(os.path.join(ROOT, "정답지", "v3", f"{cat}.jsonl")):
            ids = list(dict.fromkeys((r.get("source_ids") or []) + [s for c in r.get("chain") or [] for s in c.get("source_ids") or []]))
            g = {"id": r["id"], "category": cat, "sub": r.get("sub") or "", "pair_id": r.get("pair_id"),
                 "answer": r["answer"], "answer_aliases": r.get("answer_aliases") or [], "evidence_ids": ids,
                 "n_evidence": len(ids), "hops": r.get("hops"), "relation": r.get("relation"),
                 "relation_family": None, "alias_type": r.get("alias_type"), "alias_class": None}
            if cat == "시간추론":
                g["relation_family"], clash = relation_family(r)
                if g["relation_family"] not in FAMILY:
                    bad["시간추론 계열 없음"].append(r["id"])
                if clash:
                    bad["family 필드 ≠ 관계어 계열"].append(r["id"])
            if cat == "인물판정" and g["sub"] == "같은 사람":
                g["alias_class"] = alias_class(r.get("alias_type"))
                if not g["alias_class"]:
                    bad["alias_type 5부류 대응 없음"].append(f"{r['id']}:{r.get('alias_type')}")
            if not ids:
                bad["근거 기사 0"].append(r["id"])
            qs.append({"id": r["id"], "category": cat, "sub": r.get("sub") or "", "pair_id": r.get("pair_id"), "question": r["question"]})
            gold.append(g)
    for name, rows in (("questions", qs), ("gold", gold)):
        with open(os.path.join(EXP, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    leak = sorted({k for x in qs for k in x} - {"id", "category", "sub", "pair_id", "question"})
    by = collections.Counter(x["category"] for x in qs)
    pairs = lambda rows, key: dict(collections.Counter(g[key] for g in {g["pair_id"]: g for g in rows}.values()))
    print(f"문항 {len(qs)} {dict(by)} · questions.jsonl 허용 밖 키 {leak or 0} → {EXP}")
    print(f"  시간추론 계열(쌍) {pairs([g for g in gold if g['category'] == '시간추론'], 'relation_family')}")
    print(f"  인물판정 같은 사람 alias_class(쌍) {pairs([g for g in gold if g['sub'] == '같은 사람'], 'alias_class')}")
    mh = [g for g in gold if g["category"] == "멀티홉"]
    print(f"  멀티홉 {dict(collections.Counter(g['sub'] for g in mh))} · 근거 8편 초과 {collections.Counter(g['sub'] for g in mh if g['n_evidence'] > 8)}")
    print(f"  sha256 questions {C.sha256(C.QUESTIONS)[:16]} · gold {C.sha256(C.GOLD)[:16]}")
    for k, v in bad.items():
        print(f"  [문제] {k} {len(v)} {v[:10]}")
    if bad:
        sys.exit(1)


def stratified_sample(qs, n):
    gold = {g["id"]: g for g in C.load_jsonl(C.GOLD)}
    rng = random.Random(C.SEED)
    qids = {q["id"] for q in qs}
    pairs = collections.defaultdict(list)
    for q in qs:
        if q["category"] in ("인물판정", "시간추론") and q.get("pair_id"):
            pairs[q["pair_id"]].append(q["id"])
    full = {p: sorted(v) for p, v in pairs.items() if len(v) == 2}
    pick, strata = [], {}

    def take(keys, k, label):
        keys = sorted(keys)
        got = rng.sample(keys, min(k, len(keys)))
        strata[label] = {"population": len(keys), "picked": len(got)}
        return sorted(got)

    n_pairs_p, n_pairs_t, n_mh = round(n * 0.3 / 2), round(n * 0.3 / 2), n - 4 * round(n * 0.3 / 2)
    for sub, k in (("같은 사람", n_pairs_p // 2 + n_pairs_p % 2), ("다른 사람", n_pairs_p // 2)):
        for p in take([p for p, v in full.items() if gold[v[0]]["category"] == "인물판정" and gold[v[0]]["sub"] == sub], k, f"인물판정·{sub}(쌍)"):
            pick += full[p]
    fam_pairs = collections.defaultdict(list)
    for p, v in full.items():
        if gold[v[0]]["category"] == "시간추론":
            fam_pairs[gold[v[0]]["relation_family"]].append(p)
    tot = sum(len(v) for v in fam_pairs.values())
    quota = {f: n_pairs_t * len(v) / tot for f, v in fam_pairs.items()} if tot else {}
    alloc = {f: int(q) for f, q in quota.items()}
    for f in sorted(quota, key=lambda f: (-(quota[f] - alloc[f]), f))[: n_pairs_t - sum(alloc.values())]:
        alloc[f] += 1
    for f in sorted(fam_pairs):
        for p in take(fam_pairs[f], alloc[f], f"시간추론·{f}(쌍)"):
            pick += full[p]
    for sub in ("2hop", "3hop", "5hop", "7hop"):
        pick += take([q["id"] for q in qs if q["category"] == "멀티홉" and q["sub"] == sub], n_mh // 4, f"멀티홉·{sub}")
    pick = [i for i in pick if i in qids]
    order = [q["id"] for q in qs]
    pick = sorted(set(pick), key=order.index)
    return {"seed": C.SEED, "n": len(pick), "method": "인물 짝(같은/다른 반반) · 시간 짝(계열 비례) · 멀티홉 깊이별 같은 수",
            "questions_sha256": C.sha256(C.QUESTIONS), "gold_sha256": C.sha256(C.GOLD), "strata": strata, "ids": pick}


def load_articles():
    csv.field_size_limit(10 ** 9)
    return {r["id"]: r["text"] for r in csv.DictReader(open(C.ARTICLES, encoding="utf-8"))}


def contexts(system, qs, dry=False, retrieval=None):
    if system == "closed":
        return {q["id"]: ([], []) for q in qs}, {}, []
    if system == "gold":
        arts = load_articles()
        gold = {g["id"]: g for g in C.load_jsonl(C.GOLD)}
        miss = [q["id"] for q in qs if any(i not in arts for i in gold[q["id"]]["evidence_ids"])]
        return ({q["id"]: ([arts[i] for i in gold[q["id"]]["evidence_ids"] if i in arts], gold[q["id"]]["evidence_ids"]) for q in qs},
                {"gold_sha256": C.sha256(C.GOLD), "articles_sha256": C.sha256(C.ARTICLES)}, miss)
    if system == "vanilla":
        ret_path = os.path.join(ROOT, "rag", "runs", "v3_vanilla", "retrieval.jsonl")
        if retrieval:
            ret_path = os.path.join(ROOT, retrieval, "retrieval.jsonl")
        mock = os.path.join(ROOT, "rag", "runs", "v3_vanilla_bm25only", "retrieval.jsonl")
        if not os.path.exists(ret_path) and dry and os.path.exists(mock):
            print("[dry-run] v3_vanilla/retrieval.jsonl 없음 → 모의 v3_vanilla_bm25only 로 토큰만 잰다", file=sys.stderr)
            ret_path = mock
        if not os.path.exists(ret_path):
            sys.exit("먼저 tools/25_Vanilla검색.py 를 돌려라 (rag/runs/v3_vanilla/retrieval.jsonl 없음)")
        chunk_path = os.path.join(ROOT, "rag", "vanilla", "chunks.jsonl")
        chunks = {c["chunk_id"]: c for c in C.load_jsonl(chunk_path)}
        ret = {r["id"]: r for r in C.load_jsonl(ret_path)}
        out, miss = {}, []
        for q in qs:
            top = (ret.get(q["id"]) or {}).get("final", [])[:TOP_K]
            if len(top) != TOP_K:
                miss.append(q["id"])
            out[q["id"]] = ([chunks[c]["text"] for c in top], list(dict.fromkeys(chunks[c]["doc_id"] for c in top)))
        man = os.path.join(os.path.dirname(ret_path), "manifest_retrieval.json")
        rman = json.load(open(man, encoding="utf-8")) if os.path.exists(man) else {}
        return out, {"retrieval_file": os.path.relpath(ret_path, ROOT), "retrieval_sha256": C.sha256(ret_path),
                     "retrieval_manifest_sha256": C.sha256(man),
                     "chunks_sha256": C.sha256(chunk_path), "npy_sha256": (rman.get("inputs") or {}).get("npy_sha256"),
                     "retrieval_questions_sha256": rman.get("questions_sha256"), "rerank": rman.get("rerank"),
                     "retrieval_manifest": rman or None}, miss
    if system == "graphrag_ctx":
        cpath = os.path.join(ROOT, "rag", "runs", "v3_graphrag", "contexts.jsonl")
        ctx = {r["id"]: r for r in C.load_jsonl(cpath)}
        miss = [q["id"] for q in qs if q["id"] not in ctx]
        return ({q["id"]: ([ctx[q["id"]]["context_text"]], ctx[q["id"]].get("context_docs") or []) if q["id"] in ctx else ([], []) for q in qs},
                {"graphrag_contexts_sha256": C.sha256(cpath),
                 "graphrag_manifest_sha256": C.sha256(os.path.join(os.path.dirname(cpath), "manifest.json"))}, miss)
    raise ValueError(system)


def system_prompt(system):
    return SYSTEM.format(ctx_rule=CTX_RULE[system])


def messages(system, q, ctx):
    user = q["question"] if not ctx else "[자료]\n" + "\n\n---\n\n".join(ctx) + "\n\n[질문]\n" + q["question"]
    return [{"role": "system", "content": system_prompt(system)}, {"role": "user", "content": user}]


def manifest(system, out_dir, scope, inputs, extra):
    m = C.base_manifest(system, os.path.abspath(__file__))
    ok, err = C.answers_state(os.path.join(out_dir, "answers.jsonl"))
    ids = {q["id"] for q in scope}
    m.update(C.aggregate_logs([e for e in C.run_log(out_dir) if not e.get("dry_run")]))
    m.update({"gold_sha256": inputs.get("gold_sha256"), "inputs": inputs, "model_requested": MODEL,
              "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "top_k": TOP_K if system == "vanilla" else None,
              "system_prompt": system_prompt(system), "system_prompt_sha256": C.sha256_text(system_prompt(system)),
              "concurrency": CONCURRENCY, "n_questions": len(ids), "n_answered": len(ids & set(ok)),
              "n_errors": len(ids & err), "check5_passed": C.check5_passed(out_dir)})
    m.update(extra)
    return m


async def run(a):
    system = a.system
    qs = C.load_jsonl(C.QUESTIONS)
    out_dir = a.out_dir or os.path.join(ROOT, "rag", "runs", f"v3_{system}")
    sample = None
    if system == "graphrag_ctx":
        sample = stratified_sample(qs, a.sample)
        keep = set(sample["ids"])
        qs = [q for q in qs if q["id"] in keep]
    if a.limit:
        qs = qs[:a.limit]
    res = os.path.join(out_dir, "answers.jsonl")
    ok, err = C.answers_state(res)
    todo = [q for q in qs if q["id"] not in ok]
    retry = [q["id"] for q in todo if q["id"] in err]
    if a.check5:
        todo = todo[:5]
    ctx, inputs, miss = contexts(system, todo, a.dry_run, a.retrieval)
    if sample:
        inputs["sample_sha256"] = C.sha256_text(json.dumps(sample["ids"]))
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    toks = {q["id"]: sum(len(enc.encode(m["content"])) for m in messages(system, q, ctx[q["id"]][0])) for q in todo}
    over = sorted((i for i, t in toks.items() if t > GOLD_TOKEN_LIMIT), key=lambda i: -toks[i])
    summary = {"system": system, "out_dir": os.path.relpath(out_dir, ROOT), "scope": len(qs), "already_ok": len(set(ok) & {q["id"] for q in qs}),
               "todo": len(todo), "retry_errors": retry, "input_tokens": sum(toks.values()),
               "max_prompt_tokens": max(toks.values()) if toks else 0, "max_prompt_id": max(toks, key=toks.get) if toks else None,
               f"over_{GOLD_TOKEN_LIMIT}": [(i, toks[i]) for i in over],
               "est_usd": round(C.usd_of({MODEL: {"prompt": sum(toks.values()), "completion": EST_OUT_TOKENS * len(todo)}}), 4),
               "context_missing": miss}
    if a.dry_run:
        summary["sample_user_msg"] = messages(system, todo[0], ctx[todo[0]["id"]][0])[1]["content"][:300] if todo else ""
        if sample:
            summary["sample_strata"] = sample["strata"]
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        os.makedirs(out_dir, exist_ok=True)
        mp = os.path.join(out_dir, "manifest.json")
        real = os.path.exists(mp) and not json.load(open(mp, encoding="utf-8")).get("dry_run")
        m = manifest(system, out_dir, qs, inputs, {"dry_run": True, "dry_run_summary": {k: v for k, v in summary.items() if k != "sample_user_msg"}})
        m["run_started"] = m["run_finished"] = C.utc_now()
        C.write_json(os.path.join(out_dir, "dryrun.json") if real else mp, m)
        print(f"manifest 키 누락 {C.missing_keys(m) or 0} → {'dryrun.json' if real else 'manifest.json'}")
        return
    if miss:
        sys.exit(f"컨텍스트 없는 문항 {len(miss)} {miss[:5]} — 선행 산출물(retrieval·contexts)을 먼저 만든다")
    if system == "vanilla" and inputs.get("retrieval_questions_sha256") != C.sha256(C.QUESTIONS):
        sys.exit("retrieval.jsonl 이 지금 questions.jsonl 로 만든 것이 아니다(manifest_retrieval.json 해시 불일치) — 25 를 다시 돌린다")
    if not a.check5 and not C.check5_passed(out_dir):
        sys.exit("5문항 비용 확인 기록이 없다 — 먼저 `run " + system + " --check5` ")
    if over:
        print(f"[보고] 입력 {GOLD_TOKEN_LIMIT} 토큰 초과 {len(over)}문항 {over[:10]} (자르지 않고 그대로 보낸다)")
    os.makedirs(out_dir, exist_ok=True)
    if sample:
        C.write_json(os.path.join(out_dir, "sample.json"), sample)
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=C.openai_key())
    sem, lock, usage, models, fps = asyncio.Semaphore(CONCURRENCY), asyncio.Lock(), {}, set(), set()
    started, t0, n_ok = C.utc_now(), time.time(), [0]

    async def one(q):
        async with sem:
            text, docs = ctx[q["id"]]
            try:
                r = await client.chat.completions.create(model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                                         messages=messages(system, q, text))
                C.add_usage(usage, MODEL, r.usage.prompt_tokens, r.usage.completion_tokens)
                models.add(r.model)
                fps.add(getattr(r, "system_fingerprint", None))
                row = {"id": q["id"], "answer": (r.choices[0].message.content or "").strip(), "context_docs": docs}
                n_ok[0] += 1
            except Exception as e:
                row = {"id": q["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"}
            async with lock:
                with open(res, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(q) for q in todo))
    entry = {"phase": "check5" if a.check5 else "run", "started": started, "finished": C.utc_now(),
             "seconds": round(time.time() - t0, 1), "n_calls": len(todo), "n_ok": n_ok[0], "n_error": len(todo) - n_ok[0],
             "usage": usage, "usd": C.usd_of(usage), "models_returned": sorted(models),
             "system_fingerprints": sorted(x for x in fps if x)}
    if a.check5:
        entry["check5_passed"] = C.check5_verdict(usage, MODEL)
        per_q = entry["usd"] / max(n_ok[0], 1)
        entry["projected_usd_rest"] = round(per_q * (len(qs) - len(ok) - n_ok[0]), 4)
    C.append_log(out_dir, entry)
    C.compact_answers(res, [q["id"] for q in qs])
    m = manifest(system, out_dir, qs, inputs, {"sample_file": "sample.json" if sample else None})
    C.write_json(os.path.join(out_dir, "manifest.json"), m)
    print(json.dumps({k: entry[k] for k in entry if k != "system_fingerprints"}, ensure_ascii=False))
    print(f"manifest: 답함 {m['n_answered']}/{m['n_questions']} · 오류 {m['n_errors']} · 누적 ${m['usd']} · 반환 모델 {m['model_returned']}")
    if a.check5 and not entry["check5_passed"]:
        sys.exit("5문항 확인 실패: 토큰 기록이 0 이다 — 본 실행 중단")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("system", nargs="?", choices=["closed", "vanilla", "gold", "graphrag_ctx"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check5", action="store_true", help="5문항만 부르고 비용 기록을 확인한 뒤 멈춘다")
    ap.add_argument("--sample", type=int, default=200, help="graphrag_ctx 층화 표본 크기")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--retrieval", default=None, help="vanilla 검색 결과 폴더(ROOT 기준, 기본 rag/runs/v3_vanilla). 예: rag/runs/v3_vanilla_bm25only")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif not a.system:
        ap.error("run 에는 system 이 필요하다")
    else:
        asyncio.run(run(a))


if __name__ == "__main__":
    main()
