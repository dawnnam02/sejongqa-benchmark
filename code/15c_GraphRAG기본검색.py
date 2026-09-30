"""GraphRAG Basic search runs (Microsoft GraphRAG 2.7.2, native basic-search prompt, our answer format
passed as the response type). Same structure as 15_GraphRAG질의.py (Local search): questions only (no
answer file is read), the existing index is read but not rebuilt, and the evidence of a question is the
set of documents of the text units in its context. Stores answers, context article and text-unit IDs and
the contexts. Needs the index and an OpenAI API key.
"""
import argparse
import asyncio
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3공통 as C  # noqa: E402

RESPONSE_TYPE = "짧은 답 하나. 단어나 짧은 구로만 답하고 설명·근거 표기는 쓰지 않는다. 주어진 자료로 답할 수 없으면 '근거 부족'이라고만 쓴다."
CONCURRENCY = 4
MAX_USD_DEFAULT = 6.0

USAGE, RETURNED, FINGERPRINTS = {}, set(), set()


def make_logger():
    from litellm.integrations.custom_logger import CustomLogger

    class UsageLogger(CustomLogger):
        def _add(self, kwargs, response):
            slo = kwargs.get("standard_logging_object") or {}
            model = str(slo.get("model") or kwargs.get("model") or "unknown")
            p, c = slo.get("prompt_tokens"), slo.get("completion_tokens")
            if p is None:
                u = getattr(response, "usage", None)
                p = getattr(u, "prompt_tokens", 0) if u else 0
                c = getattr(u, "completion_tokens", 0) if u else 0
            C.add_usage(USAGE, model, p, c)
            full = kwargs.get("complete_streaming_response") or response
            rm = getattr(full, "model", None)
            if not rm and isinstance(slo.get("response"), dict):
                rm = slo["response"].get("model")
            if rm:
                RETURNED.add(str(rm))
            fp = getattr(full, "system_fingerprint", None)
            if fp:
                FINGERPRINTS.add(str(fp))

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self._add(kwargs, response_obj)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self._add(kwargs, response_obj)

    return UsageLogger()


def _usage_of(usage, model):
    return next((u for k, u in usage.items() if model and model in k), None)


def check5_verdict(rows, ctx_rows, usage, chat_model, emb_model, n_expected=5):
    """Smoke-test checks: n answers, n contexts with context documents, usage recorded, token estimate within 50%."""
    ok_rows = [r for r in rows if "error" not in r and str(r.get("answer") or "").strip()]
    ctx = list({c["id"]: c for c in ctx_rows if c["id"] in {r["id"] for r in rows}}.values())
    chat, emb = _usage_of(usage, chat_model), _usage_of(usage, emb_model)
    est = sum(r.get("est_prompt_tokens") or 0 for r in ok_rows)
    rec = chat["prompt"] if chat else 0
    ratio = rec / est if est else None
    n_empty = sum(1 for c in ctx if not c.get("context_docs"))
    crit = {
        "1_응답": {"pass": len(rows) == n_expected and len(ok_rows) == n_expected,
                  "detail": f"행 {len(rows)} · 정상 응답 {len(ok_rows)} / {n_expected}"},
        "2_contexts": {"pass": len(ctx) == n_expected and n_empty == 0,
                       "detail": f"contexts {len(ctx)}행 · context_docs 빈 행 {n_empty}"},
        "3_사용량": {"pass": bool(chat and chat["prompt"] > 0 and chat["completion"] > 0 and emb and emb["prompt"] > 0),
                   "detail": f"{chat_model} {chat} · {emb_model} {emb}"},
        "4_토큰추정": {"pass": ratio is not None and 0.5 <= ratio <= 1.5,
                    "detail": f"기록 {rec} / tiktoken 추정 {est} = {'—' if ratio is None else f'{ratio:.2f}'} (0.5~1.5)"},
    }
    return all(c["pass"] for c in crit.values()), crit


def build_engine(root, config):
    """Build the engine in the same order as graphrag.api.query.basic_search_streaming (graphrag 2.7.2)."""
    import pandas as pd
    from graphrag.config.embeddings import text_unit_text_embedding
    from graphrag.query.factory import get_basic_search_engine
    from graphrag.query.indexer_adapters import read_indexer_text_units
    from graphrag.utils.api import get_embedding_store, load_search_prompt
    units = pd.read_parquet(root / "output" / "text_units.parquet")
    vector_store_args = {k: s.model_dump() for k, s in config.vector_store.items()}
    store = get_embedding_store(config_args=vector_store_args, embedding_name=text_unit_text_embedding)
    prompt = load_search_prompt(config.root_dir, config.basic_search.prompt)
    engine = get_basic_search_engine(config=config, text_units=read_indexer_text_units(units), text_unit_embeddings=store,
                                     system_prompt=prompt, response_type=RESPONSE_TYPE, callbacks=[])
    short2docs = {str(h): list(d) for h, d in zip(units.human_readable_id, units.document_ids)}
    return engine, short2docs, prompt


def manifest(root, config, q_path, out_dir, scope_ids, prompt_text):
    m = C.base_manifest("graphrag", os.path.abspath(__file__), str(q_path))
    ok, err = C.answers_state(str(out_dir / "answers.jsonl"))
    m.update(C.aggregate_logs([e for e in C.run_log(str(out_dir)) if not e.get("dry_run")]))
    chat = config.models[config.basic_search.chat_model_id] if config else None
    bs = config.basic_search if config else None
    prompts = {os.path.basename(p): C.sha256(p) for p in sorted(glob.glob(str(root / "prompts" / "*.txt")))}
    m.update({"inputs": {"root": os.path.relpath(root, C.ROOT), "settings_sha256": C.sha256(str(root / "settings.yaml")),
                         "prompts_sha256": prompts, "stats_sha256": C.sha256(str(root / "output" / "stats.json")),
                         "parquet_sha256": {os.path.basename(p): C.sha256(p) for p in sorted(glob.glob(str(root / "output" / "*.parquet")))},
                         "contexts_sha256": C.sha256(str(out_dir / "contexts.jsonl"))},
              "model_requested": getattr(chat, "model", None), "temperature": getattr(chat, "temperature", None),
              "max_tokens": getattr(chat, "max_tokens", None),
              "embedding_model": getattr(config.models[bs.embedding_model_id], "model", None) if config else None,
              "top_k": None if bs is None else {"k": bs.k, "max_context_tokens": bs.max_context_tokens},
              "system_prompt": prompt_text, "system_prompt_sha256": C.sha256_text(prompt_text) if prompt_text else None,
              "response_type": RESPONSE_TYPE, "concurrency": CONCURRENCY, "search": "basic (native prompt)",
              "check5_passed": C.check5_passed(str(out_dir))})
    ids = set(scope_ids)
    C.set_counts(m, len(ids), len(ids & set(ok)), len(ids & err))
    return m


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("questions")
    ap.add_argument("out_dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check5", action="store_true", help="5문항만 부르고 비용 기록을 확인한 뒤 멈춘다")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=MAX_USD_DEFAULT, help="누적 비용 상한(넘으면 새 호출을 멈춘다)")
    a = ap.parse_args()
    root, q_path, out_dir = Path(a.root).resolve(), Path(a.questions).resolve(), Path(a.out_dir).resolve()
    questions = [{"id": r["id"], "question": r["question"]} for r in C.load_jsonl(str(q_path))]
    if a.limit:
        questions = questions[:a.limit]
    res_path = out_dir / "answers.jsonl"
    ok, err = C.answers_state(str(res_path))
    todo = [q for q in questions if q["id"] not in ok]
    retry = [q["id"] for q in todo if q["id"] in err]
    if a.check5:
        todo = todo[:5]
    has_output = (root / "output" / "text_units.parquet").exists()
    print(f"문항 {len(questions)} · 이미 답함 {len(set(ok) & {q['id'] for q in questions})} · 이번 {len(todo)} · 오류 재시도 {len(retry)} {retry[:10]}", flush=True)
    if a.dry_run:
        os.environ.setdefault("OPENAI_API_KEY", "sk-dry-run-no-call")
        from graphrag.config.load_config import load_config
        config = load_config(root)
        prompt = None
        if has_output:
            engine, short2docs, prompt = build_engine(root, config)
            print(f"엔진 구성 ○ ({type(engine).__name__}) · text unit {len(short2docs)} · 프롬프트 {len(prompt)}자 · "
                  f"context_builder_params {engine.context_builder_params} — 질의는 부르지 않음")
        m = manifest(root, config, q_path, out_dir, [q["id"] for q in questions], prompt)
        m["dry_run"] = True
        print(f"model_requested {m['model_requested']} · temperature {m['temperature']} · top_k {m['top_k']}")
        print(f"manifest 키 누락 {C.missing_keys(m) or 0} (dry-run 은 파일을 쓰지 않는다)")
        return
    if not has_output:
        sys.exit(f"{root / 'output'} 에 인덱스가 없다")
    if not a.check5 and not C.check5_passed(str(out_dir)):
        sys.exit("5문항 비용 확인 기록이 없다 — 먼저 --check5")
    os.environ.setdefault("OPENAI_API_KEY", C.openai_key())
    import litellm
    from graphrag.config.load_config import load_config
    litellm.callbacks = [make_logger()]
    config = load_config(root)
    engine, short2docs, prompt = build_engine(root, config)
    chat_model = config.models[config.basic_search.chat_model_id].model
    emb_model = config.models[config.basic_search.embedding_model_id].model
    prior_usd = sum(e.get("usd") or 0 for e in C.run_log(str(out_dir)) if not e.get("dry_run")) if out_dir.exists() else 0.0
    this_rows, this_ctx = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_path = out_dir / "contexts.jsonl"
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    started, t0, n_ok, stopped = C.utc_now(), time.time(), [0], [False]

    async def one(q):
        async with sem:
            if prior_usd + C.usd_of(USAGE) > a.max_usd:
                stopped[0] = True
                return
            t = time.time()
            ctx_row = None
            try:
                r = await engine.search(q["question"])
                records = r.context_data if isinstance(r.context_data, dict) else {}
                src = records.get("Sources")
                if src is None:
                    src = records.get("sources")
                short = [str(x) for x in src["source_id"]] if src is not None and len(src) else []
                docs = sorted({d for s in short for d in short2docs.get(s, [])})
                answer = str(r.response).strip()
                if not answer:
                    raise RuntimeError("빈 응답 — search() 가 LLM 오류를 삼켰을 수 있다(query.log 확인)")
                ctx_text = r.context_text if isinstance(r.context_text, str) else str(r.context_text)
                row = {"id": q["id"], "answer": answer, "context_docs": docs, "context_units": short,
                       "sec": round(time.time() - t, 1),
                       "est_prompt_tokens": (r.prompt_tokens_categories or {}).get("response", r.prompt_tokens) + len(engine.tokenizer.encode(q["question"])),
                       "est_context_tokens": len(engine.tokenizer.encode(ctx_text)),
                       "est_output_tokens": r.output_tokens}
                ctx_row = {"id": q["id"], "context_text": ctx_text, "context_docs": docs}
                n_ok[0] += 1
            except Exception as e:
                row = {"id": q["id"], "error": f"{type(e).__name__}: {str(e)[:300]}", "sec": round(time.time() - t, 1)}
            async with lock:
                this_rows.append(row)
                if ctx_row:
                    this_ctx.append(ctx_row)
                with open(res_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if ctx_row:
                    with open(ctx_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(ctx_row, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(q) for q in todo))
    n_called = len(this_rows)
    entry = {"phase": "check5" if a.check5 else "run", "started": started, "finished": C.utc_now(),
             "seconds": round(time.time() - t0, 1), "n_calls": n_called, "n_ok": n_ok[0], "n_error": n_called - n_ok[0],
             "usage": dict(USAGE), "usd": C.usd_of(USAGE), "models_returned": sorted(RETURNED), "system_fingerprints": sorted(FINGERPRINTS),
             "response_type": RESPONSE_TYPE, "search": "basic", "stopped_by_cost_cap": stopped[0], "max_usd": a.max_usd}
    if a.check5:
        entry["check5_passed"], crit = check5_verdict(this_rows, this_ctx, USAGE, chat_model, emb_model, n_expected=len(todo))
        entry["check5_criteria"] = crit
        entry["projected_usd_rest"] = round(entry["usd"] / max(n_ok[0], 1) * (len(questions) - len(ok) - n_ok[0]), 4)
        entry["projected_usd_total"] = round(prior_usd + entry["usd"] + entry["projected_usd_rest"], 4)
    C.append_log(str(out_dir), entry)
    C.compact_answers(str(res_path), [q["id"] for q in questions], side_files=[str(ctx_path)])
    m = manifest(root, config, q_path, out_dir, [q["id"] for q in questions], prompt)
    C.write_json(str(out_dir / "manifest.json"), m)
    print(json.dumps({k: entry.get(k) for k in ("phase", "n_calls", "n_ok", "n_error", "usage", "usd", "models_returned",
                                                "projected_usd_rest", "projected_usd_total", "stopped_by_cost_cap")}, ensure_ascii=False))
    print(f"manifest: 답함 {m['n_answered']}/{m['n_questions']} · 오류 {m['n_errors']} · 누적 ${m['usd']} · 반환 모델 {m['model_returned']}")
    if a.check5:
        for k, v in entry["check5_criteria"].items():
            print(f"  {k}: {'○' if v['pass'] else '✗'} {v['detail']}")
        if not entry["check5_passed"]:
            sys.exit("5문항 확인 실패 — 본 실행 중단")
    if stopped[0]:
        sys.exit(f"비용 상한 ${a.max_usd} 도달 — 멈춤")


if __name__ == "__main__":
    asyncio.run(main())
