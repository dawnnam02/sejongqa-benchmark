"""Shared helpers for the query, judge and scoring scripts: hashing, run manifests, API key lookup
(environment variable OPENAI_API_KEY, or a .env file in the GraphRAG workspace), resumable answer files,
token usage and cost. No API calls.
"""
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "rag", "v3", "실험")
QUESTIONS = os.path.join(EXP, "questions.jsonl")
GOLD = os.path.join(EXP, "gold.jsonl")
ARTICLES = os.path.join(ROOT, "rag", "graphrag", "input", "articles.csv")
FROZEN = os.path.join(ROOT, "rag", "v3", "동결.json")
ENV_FILE = os.path.join(ROOT, "rag", "graphrag", ".env")
PACKAGES = ("openai", "graphrag", "litellm", "rank_bm25", "numpy", "tiktoken", "pandas")
SEED = 20260929

REQUIRED_KEYS = ("system", "run_started", "run_finished", "code_sha256", "questions_sha256", "gold_sha256", "frozen_ref",
                 "corpus_sha256", "inputs", "model_requested", "model_returned", "system_fingerprint", "temperature",
                 "max_tokens", "top_k", "system_prompt", "system_prompt_sha256", "concurrency", "usage", "usd", "seconds",
                 "n_questions", "n_answered", "n_errors", "packages")


def sha256(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def packages():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def frozen_ref():
    return sha256(FROZEN)


def openai_key():
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k
    if os.path.exists(ENV_FILE):
        for line in open(ENV_FILE, encoding="utf-8-sig"):
            m = re.match(r"\s*OPENAI_API_KEY\s*=\s*\"?([^\"\s]+)", line)
            if m:
                return m.group(1)
    raise SystemExit("OPENAI_API_KEY 없음 (환경변수 또는 rag/graphrag/.env)")


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def write_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def is_ok(row):
    return "error" not in row and row.get("answer") is not None


def answers_state(path):
    ok, err = {}, set()
    for r in load_jsonl(path):
        if is_ok(r):
            ok.setdefault(r["id"], r)
        else:
            err.add(r["id"])
    err |= {r["id"] for r in load_jsonl(os.path.join(os.path.dirname(path), "errors.jsonl"))}
    return ok, err - set(ok)


def compact_answers(path, order_ids, side_files=()):
    rows = load_jsonl(path)
    ok, errors = {}, []
    for r in rows:
        if is_ok(r):
            ok.setdefault(r["id"], r)
        else:
            errors.append(r)
    order = [i for i in order_ids if i in ok] + [i for i in ok if i not in set(order_ids)]
    with open(path, "w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(ok[i], ensure_ascii=False) + "\n")
    if errors:
        with open(os.path.join(os.path.dirname(path), "errors.jsonl"), "a", encoding="utf-8") as f:
            for r in errors:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in side_files:
        last = {}
        for r in load_jsonl(sp):
            last[r["id"]] = r
        with open(sp, "w", encoding="utf-8") as f:
            for i in order:
                if i in last:
                    f.write(json.dumps(last[i], ensure_ascii=False) + "\n")
    return len(ok), len(errors)


def add_usage(total, model, prompt, completion, calls=1):
    m = total.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
    m["prompt"] += prompt or 0
    m["completion"] += completion or 0
    m["calls"] += calls
    return total


PRICE = {"gpt-4.1-mini": (0.40, 1.60), "text-embedding-3-small": (0.02, 0.0)}


def usd_of(usage):
    t = 0.0
    for model, u in usage.items():
        key = next((k for k in PRICE if k in model), None)
        if key:
            t += u["prompt"] * PRICE[key][0] / 1e6 + u["completion"] * PRICE[key][1] / 1e6
    return round(t, 6)


def run_log(out_dir):
    return load_jsonl(os.path.join(out_dir, "run_log.jsonl"))


def append_log(out_dir, entry, log_name="run_log.jsonl"):
    with open(os.path.join(out_dir, log_name), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def check5_passed(out_dir, log_name="run_log.jsonl"):
    return any(e.get("phase") == "check5" and e.get("check5_passed") for e in load_jsonl(os.path.join(out_dir, log_name)))


def check5_verdict(usage, chat_model):
    u = next((v for k, v in usage.items() if chat_model in k), None)
    return bool(u and u["prompt"] > 0 and u["completion"] > 0)


def aggregate_logs(entries):
    usage, models, fps = {}, set(), set()
    for e in entries:
        for m, u in (e.get("usage") or {}).items():
            add_usage(usage, m, u.get("prompt"), u.get("completion"), u.get("calls", 0))
        models.update(e.get("models_returned") or [])
        fps.update(x for x in e.get("system_fingerprints") or [] if x)
    one = lambda s: None if not s else (sorted(s)[0] if len(s) == 1 else sorted(s))
    return {"run_started": min((e["started"] for e in entries if e.get("started")), default=None),
            "run_finished": max((e["finished"] for e in entries if e.get("finished")), default=None),
            "usage": usage, "usd": round(sum(e.get("usd") or 0 for e in entries), 6),
            "seconds": round(sum(e.get("seconds") or 0 for e in entries), 1),
            "model_returned": one(models), "system_fingerprint": one(fps), "invocations": len(entries)}


def base_manifest(system, script, questions_path=QUESTIONS):
    m = {k: None for k in REQUIRED_KEYS}
    m.update({"system": system, "code_sha256": sha256(script), "code_file": os.path.relpath(script, ROOT),
              "common_sha256": sha256(os.path.abspath(__file__)), "questions_sha256": sha256(questions_path),
              "frozen_ref": frozen_ref(), "corpus_sha256": sha256(ARTICLES), "inputs": {}, "usage": {}, "usd": 0.0,
              "packages": packages(), "dry_run": False})
    return m


def set_counts(m, n_questions, n_done, n_errors):
    m.update({"n_questions": n_questions, "n_answered": n_done, "n_errors": n_errors})
    return m


def missing_keys(m):
    return [k for k in REQUIRED_KEYS if k not in m]
