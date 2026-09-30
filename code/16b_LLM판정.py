"""LLM-Eq judge (gpt-4.1-mini, temperature 1, three votes, majority). EM = 1 is accepted and abstentions
are rejected without a judge call. Writes judged.jsonl. Needs an OpenAI API key.
"""
import argparse
import asyncio
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3공통 as C

ROOT = C.ROOT
MODEL = "gpt-4.1-mini"
N_VOTES, TEMPERATURE, MAX_TOKENS, CONCURRENCY = 3, 1, 5, 8
PROMPT = """질문에 대한 모범 답과 응답이 주어진다. 응답이 모범 답과 같은 내용을 가리키면 '정답', 아니면 '오답'이라고 한 단어로만 답하라.
표기 차이(띄어쓰기, 한자 병기, 숫자 표기, 조사)는 무시한다. 모범 답의 다른 표기로 답한 것도 정답이다. 응답이 모범 답보다 넓거나 여러 후보를 나열하면 오답이다.

질문: {q}
모범 답: {gold}{alias}
응답: {pred}"""
LOG = "judge_log.jsonl"


def alias_text(aliases):
    return f" (다른 표기: {' · '.join(aliases)})" if aliases else ""


def load_questions(run):
    if os.path.basename(os.path.normpath(run)).startswith("v3_") or os.path.exists(os.path.join(run, "sample.json")):
        qfile = C.QUESTIONS
    else:
        qfile = os.path.join(ROOT, "rag", "data", "questions.jsonl")
    q = {r["id"]: r["question"] for r in C.load_jsonl(qfile)}
    for r in C.load_jsonl(os.path.join(ROOT, "정답지", "고난도_5hop_7hop.jsonl")):
        q.setdefault(r["id"], r["question"])
    return q, qfile


def free_verdict(s):
    if s.get("answered") is False:
        return ["미응답"], 0.0
    if s["em"] == 1.0:
        return ["EM"], 1.0
    if s["abstain"]:
        return ["기권"], 0.0
    return None


def judge_prompt(questions, s):
    return PROMPT.format(q=questions[s["id"]], gold=s["gold"], alias=alias_text(s.get("gold_aliases") or []), pred=s["pred"])


def manifest(run, qfile, scored, judged, extra):
    m = C.base_manifest("judge", os.path.abspath(__file__), qfile)
    m.update(C.aggregate_logs(C.load_jsonl(os.path.join(run, LOG))))
    m.update({"gold_sha256": C.sha256(C.GOLD), "inputs": {"run": os.path.relpath(run, ROOT), "scored_sha256": C.sha256(os.path.join(run, "scored.jsonl")),
                                                          "answers_sha256": C.sha256(os.path.join(run, "answers.jsonl"))},
              "model_requested": MODEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "top_k": None,
              "system_prompt": PROMPT, "system_prompt_sha256": C.sha256_text(PROMPT), "concurrency": CONCURRENCY, "n_votes": N_VOTES,
              "check5_passed": C.check5_passed(run, LOG)})
    C.set_counts(m, len(scored), sum(1 for j in judged if j.get("judge") is not None), sum(1 for j in judged if j.get("judge") is None))
    m.update(extra)
    return m


async def judge(a):
    run = os.path.abspath(a.run)
    scored = C.load_jsonl(os.path.join(run, "scored.jsonl"))
    if not scored:
        sys.exit(f"{run}/scored.jsonl 없음 — 먼저 tools/27_채점.py {a.run}")
    questions, qfile = load_questions(run)
    prev = {j["id"]: j for j in C.load_jsonl(os.path.join(run, "judged.jsonl"))}
    out, need = {}, []
    for s in scored:
        fv = free_verdict(s)
        p = prev.get(s["id"])
        if fv:
            out[s["id"]] = {**s, "votes": fv[0], "judge": fv[1]}
        elif p and p.get("judge") is not None and p.get("pred") == s["pred"] and p.get("gold") == s["gold"]:
            out[s["id"]] = {**s, "votes": p["votes"], "judge": p["judge"]}
        else:
            need.append(s)
    if a.check5:
        need_now = need[:5]
    else:
        need_now = need
    est_in = sum(len(judge_prompt(questions, s)) for s in need_now) * N_VOTES
    summary = {"run": os.path.relpath(run, ROOT), "scored": len(scored), "free": len(out), "need_calls": len(need),
               "this_time": len(need_now), "est_usd_upper": round(C.usd_of({MODEL: {"prompt": est_in, "completion": 2 * N_VOTES * len(need_now)}}), 4)}
    if a.dry_run:
        summary["prompt_example"] = judge_prompt(questions, next((s for s in need_now if s.get("gold_aliases")), need_now[0])) if need_now else ""
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        m = manifest(run, qfile, scored, list(out.values()), {"dry_run": True})
        print(f"manifest_judge 키 누락 {C.missing_keys(m) or 0} (dry-run 은 파일을 쓰지 않는다)")
        return
    if not a.check5 and not C.check5_passed(run, LOG):
        sys.exit("5문항 비용 확인 기록이 없다 — 먼저 --check5 ")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=C.openai_key())
    sem, usage, models, fps = asyncio.Semaphore(CONCURRENCY), {}, set(), set()
    started, t0 = C.utc_now(), time.time()

    async def one(s):
        votes = []
        async with sem:
            try:
                for _ in range(N_VOTES):
                    r = await client.chat.completions.create(model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                                             messages=[{"role": "user", "content": judge_prompt(questions, s)}])
                    C.add_usage(usage, MODEL, r.usage.prompt_tokens, r.usage.completion_tokens)
                    models.add(r.model)
                    fps.add(getattr(r, "system_fingerprint", None))
                    votes.append((r.choices[0].message.content or "").strip())
            except Exception as e:
                return {**s, "votes": votes, "judge": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        yes = sum("정답" in v and "오답" not in v for v in votes)
        return {**s, "votes": votes, "judge": float(yes * 2 > N_VOTES)}

    for j in await asyncio.gather(*(one(s) for s in need_now)):
        out[j["id"]] = j
    for s in need[len(need_now):]:
        out[s["id"]] = {**s, "votes": ["미판정"], "judge": None}
    judged = [out[s["id"]] for s in scored]
    with open(os.path.join(run, "judged.jsonl"), "w", encoding="utf-8") as f:
        for j in judged:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    entry = {"phase": "check5" if a.check5 else "run", "started": started, "finished": C.utc_now(), "seconds": round(time.time() - t0, 1),
             "n_calls": len(need_now) * N_VOTES, "n_error": sum(1 for j in judged if j.get("error")), "usage": usage, "usd": C.usd_of(usage),
             "models_returned": sorted(models), "system_fingerprints": sorted(x for x in fps if x)}
    if a.check5:
        entry["check5_passed"] = C.check5_verdict(usage, MODEL)
    C.append_log(run, entry, LOG)
    groups = collections.defaultdict(list)
    for j in judged:
        if j["judge"] is not None:
            groups[j.get("category")].append(j["judge"])
            groups["전체"].append(j["judge"])
    lines = [f"LLM 판정 — {os.path.basename(run)}  (판정 모델 {MODEL}, {N_VOTES}회 다수결, temperature {TEMPERATURE}, 별칭 전달)",
             "묶음 | 판정된 문항 | 판정 정확도"]
    lines += [f"{k} | {len(v)} | {sum(v) / len(v):.3f}" for k, v in sorted(groups.items(), key=lambda x: (x[0] == "전체", str(x[0])))]
    lines += ["", f"미판정(오류·대기) {sum(1 for j in judged if j['judge'] is None)} · 이번 사용량 {usage} · ${entry['usd']}"]
    open(os.path.join(run, "판정보고.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    C.write_json(os.path.join(run, "manifest_judge.json"), manifest(run, qfile, scored, judged, {}))
    print("\n".join(lines))
    if a.check5 and not entry["check5_passed"]:
        sys.exit("5문항 확인 실패: 토큰 기록이 0 이다 — 본 실행 중단")


def sample30(runs, out, n):
    pool = []
    for run in runs:
        questions, _ = load_questions(run)
        for j in C.load_jsonl(os.path.join(run, "judged.jsonl")):
            if j.get("judge") is not None and j["em"] != j["judge"]:
                pool.append({"run": os.path.basename(os.path.normpath(run)), "id": j["id"], "category": j.get("category"),
                             "question": questions.get(j["id"], ""), "gold": j["gold"], "gold_aliases": j.get("gold_aliases") or [],
                             "pred": j["pred"], "em": j["em"], "judge": j["judge"], "votes": j.get("votes")})
    pool.sort(key=lambda x: (x["run"], x["id"]))
    pick = sorted(random.Random(C.SEED).sample(pool, min(n, len(pool))), key=lambda x: (x["run"], x["id"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    md = [f"# 판정 검증 표본 — EM ≠ LLM 판정 {len(pick)}문항 (모집단 {len(pool)}, seed {C.SEED})", "",
          "정답·응답만 보고 손으로 판정한다(같음 ○ / 다름 ✗ / 판단 불가 ?). LLM 판정 결과는 같은 이름의 _key.jsonl 에 있다.", "",
          "| # | run | id | 질문 | 모범 답 (다른 표기) | 응답 | 손 판정 |", "|---|---|---|---|---|---|---|"]
    esc = lambda t: str(t).replace("|", "\\|").replace("\n", " ")
    for i, x in enumerate(pick, 1):
        md.append(f"| {i} | {x['run']} | {x['id']} | {esc(x['question'])} | {esc(x['gold'])}{esc(alias_text(x['gold_aliases']))} | {esc(x['pred'])} |  |")
    open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    key = os.path.splitext(out)[0] + "_key.jsonl"
    with open(key, "w", encoding="utf-8") as f:
        for i, x in enumerate(pick, 1):
            f.write(json.dumps({"no": i, **x}, ensure_ascii=False) + "\n")
    print(f"표본 {len(pick)} / 모집단 {len(pool)} → {out} · {key}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sample30":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("runs", nargs="+")
        ap.add_argument("--out", default=os.path.join(ROOT, "rag", "runs", "v3_채점", "판정검증_30.md"))
        ap.add_argument("--n", type=int, default=30)
        a = ap.parse_args()
        return sample30(a.runs, a.out, a.n)
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check5", action="store_true")
    asyncio.run(judge(ap.parse_args()))


if __name__ == "__main__":
    main()
