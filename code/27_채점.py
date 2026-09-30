"""Scorer used for the paper: EM, F1, LLM-Eq, Pair EM, abstention, article-level evidence recall and
coverage, bootstrap 95% CIs (2,000 resamples, seed 20260929, pairs or items as units, stratified by
category) and paired differences between systems. Group names (Korean labels) seed the resampling.

    python 27_채점.py RUN_DIR [RUN_DIR ...]      python 27_채점.py --selftest
"""
import argparse
import collections
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3공통 as C

ROOT = C.ROOT
B = 2000
SEED = C.SEED
RETRIEVAL = {"vanilla", "graphrag", "graphrag_ctx"}
KINDS = {"closed", "gold", "vanilla", "graphrag", "graphrag_ctx"}
PAIR_CATS = ("인물판정", "시간추론")
FAM_ORDER = ["하루", "달", "해", "순서"]
ALIAS_ORDER = ["관직·관계 호칭", "봉호·군호", "호·이칭·약칭", "표기 이형", "시호·묘호"]
Q_METRICS = ["em", "f1", "judge", "abstain", "ev_any", "ev_all", "ev_cov"]
P_METRICS = ["pair_em", "pair_judge"]


def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[一-鿿]", "", s)
    return re.sub(r"[\s\W_]+", "", s).lower()


def f1(p, g):
    p, g = list(norm(p)), list(norm(g))
    if not p or not g:
        return float(p == g)
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not c:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)


def depth_label(sub):
    m = re.match(r"(\d+)hop", sub or "")
    if not m:
        return sub
    n = int(m.group(1))
    return f"{n}-page chain ({n - 1} relation{'s' if n - 1 != 1 else ''})"


def system_kind(run, override=None):
    name = os.path.basename(os.path.normpath(run))
    if override and name in override:
        return override[name]
    mp = os.path.join(run, "manifest.json")
    if os.path.exists(mp):
        s = str(json.load(open(mp, encoding="utf-8")).get("system") or "")
        if s in KINDS:
            return s
    low = name.lower()
    for key, kind in (("graphrag_ctx", "graphrag_ctx"), ("graphrag", "graphrag"), ("local", "graphrag"),
                      ("closed", "closed"), ("gold", "gold"), ("vanilla", "vanilla")):
        if key in low:
            return kind
    sys.exit(f"{run}: 시스템 종류를 알 수 없다 — --system {name}=closed|gold|vanilla|graphrag|graphrag_ctx")


def evidence(E, Cx, kind):
    if not Cx and kind not in RETRIEVAL:
        return None, None, None
    if not E:
        return None, None, None
    inter = len(E & Cx)
    return float(inter > 0), float(E <= Cx), inter / len(E)


def score(run, gold_rows, kind):
    sp = os.path.join(run, "sample.json")
    scope = set(json.load(open(sp, encoding="utf-8"))["ids"]) if os.path.exists(sp) else None
    ans, _ = C.answers_state(os.path.join(run, "answers.jsonl"))
    judged = {j["id"]: j for j in C.load_jsonl(os.path.join(run, "judged.jsonl"))}
    rows = []
    for g in gold_rows:
        gid = g["id"]
        if scope is not None and gid not in scope:
            continue
        a = ans.get(gid)
        pred = str(a["answer"]) if a else ""
        golds = [g["answer"]] + list(g.get("answer_aliases") or [])
        golds = [x for x in golds if norm(x)] or golds
        E, Cx = set(g.get("evidence_ids") or []), set((a or {}).get("context_docs") or [])
        ev_any, ev_all, ev_cov = evidence(E, Cx, kind)
        j = judged.get(gid)
        rows.append({"id": gid, "category": g["category"], "sub": g.get("sub") or "", "pair_id": g.get("pair_id"),
                     "relation_family": g.get("relation_family"), "alias_class": g.get("alias_class"),
                     "n_evidence": g.get("n_evidence", len(E)), "answered": a is not None, "pred": pred,
                     "gold": g["answer"], "gold_aliases": list(g.get("answer_aliases") or []),
                     "em": float(any(norm(pred) == norm(x) for x in golds)) if a else 0.0,
                     "f1": max(f1(pred, x) for x in golds) if a else 0.0,
                     "judge": j.get("judge") if j and j.get("pred", pred) == pred else None,
                     "abstain": float("근거 부족" in pred),
                     "ctx_empty": (not Cx) if kind in RETRIEVAL else None,
                     "ev_any": ev_any, "ev_all": ev_all, "ev_cov": ev_cov})
    with open(os.path.join(run, "scored.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def group_rows(rows):
    G = collections.OrderedDict()

    def add(name, pred):
        v = [r for r in rows if pred(r)]
        if v:
            G[name] = v
    add("인물판정", lambda r: r["category"] == "인물판정")
    add("인물판정 · 같은 사람", lambda r: r["category"] == "인물판정" and r["sub"] == "같은 사람")
    for c in ALIAS_ORDER:
        add(f"인물판정 · 같은 사람 · {c}", lambda r, c=c: r["category"] == "인물판정" and r["alias_class"] == c)
    add("인물판정 · 다른 사람", lambda r: r["category"] == "인물판정" and r["sub"] == "다른 사람")
    add("시간추론", lambda r: r["category"] == "시간추론")
    for f in FAM_ORDER:
        add(f"시간추론 · {f}", lambda r, f=f: r["category"] == "시간추론" and r["relation_family"] == f)
    add("멀티홉", lambda r: r["category"] == "멀티홉")
    subs = sorted({r["sub"] for r in rows if r["category"] == "멀티홉"}, key=lambda s: (int(re.sub(r"\D", "", s) or 0), s))
    for s in subs:
        add(f"멀티홉 · {depth_label(s)}", lambda r, s=s: r["category"] == "멀티홉" and r["sub"] == s)
    add("전체", lambda r: True)
    return G


class Boot:

    def __init__(self, rows, name):
        cl = collections.OrderedDict()
        for r in sorted(rows, key=lambda r: (r["category"], r["pair_id"] or r["id"], r["id"])):
            key = r["pair_id"] if r["category"] in PAIR_CATS and r["pair_id"] else r["id"]
            cl.setdefault((r["category"], key), []).append(r)
        self.keys = list(cl)
        members = list(cl.values())
        self.sums, self.cnts = {}, {}
        for m in Q_METRICS:
            vals = [[x[m] for x in c if x[m] is not None] for c in members]
            self.sums[m] = np.array([sum(v) for v in vals], dtype=float)
            self.cnts[m] = np.array([len(v) for v in vals], dtype=float)
        for m, base in (("pair_em", "em"), ("pair_judge", "judge")):
            ok = [len(c) == 2 and c[0]["category"] in PAIR_CATS and all(x[base] is not None for x in c) for c in members]
            self.sums[m] = np.array([float(all(x[base] for x in c)) if o else 0.0 for c, o in zip(members, ok)])
            self.cnts[m] = np.array([1.0 if o else 0.0 for o in ok])
        rng = np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8"))])
        strata = collections.OrderedDict()
        for i, (cat, _) in enumerate(self.keys):
            strata.setdefault(cat, []).append(i)
        parts = [np.asarray(ix)[rng.integers(0, len(ix), size=(B, len(ix)))] for ix in strata.values()]
        self.idx = np.concatenate(parts, axis=1) if parts else np.zeros((B, 0), dtype=int)

    def point(self, m):
        c = self.cnts[m].sum()
        return float(self.sums[m].sum() / c) if c else None

    def dist(self, m):
        c = self.cnts[m][self.idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, self.sums[m][self.idx].sum(1) / np.where(c > 0, c, 1), np.nan)

    def ci(self, m, d=None):
        d = self.dist(m) if d is None else d
        if np.all(np.isnan(d)):
            return None
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return float(lo), float(hi)


def summarize(rows):
    out = collections.OrderedDict()
    for name, v in group_rows(rows).items():
        bt = Boot(v, name)
        s = {"n": len(v), "answered": sum(r["answered"] for r in v),
             "n_pairs": int(bt.cnts["pair_em"].sum()),
             "judge_complete": all(r["judge"] is not None for r in v),
             "ctx_empty": sum(1 for r in v if r["ctx_empty"]),
             "em1_any0": sum(1 for r in v if r["em"] == 1 and r["ev_any"] == 0),
             "em0_all1": sum(1 for r in v if r["em"] == 0 and r["ev_all"] == 1),
             "over8": sum(1 for r in v if (r["n_evidence"] or 0) > 8)}
        for m in Q_METRICS + P_METRICS:
            s[m] = bt.point(m)
            s[m + "_ci"] = bt.ci(m) if s[m] is not None else None
        out[name] = s
    return out


def paired(rows_a, rows_b, groups=None):
    ids = {r["id"] for r in rows_a} & {r["id"] for r in rows_b}
    ra = [r for r in rows_a if r["id"] in ids]
    rb = [r for r in rows_b if r["id"] in ids]
    ga, gb = group_rows(ra), group_rows(rb)
    res = collections.OrderedDict()
    for name in ga:
        if groups and name not in groups:
            continue
        A, Bt = Boot(ga[name], name), Boot(gb[name], name)
        assert A.keys == Bt.keys and np.array_equal(A.idx, Bt.idx)
        both_judged = all(r["judge"] is not None for r in ga[name] + gb[name])
        is_pair = ga[name][0]["category"] in PAIR_CATS and name != "전체"
        ms = [("EM", "em")] + ([("짝 EM", "pair_em")] if is_pair else [])
        if both_judged:
            ms += [("LLM-Eq", "judge")] + ([("짝 LLM-Eq", "pair_judge")] if is_pair else [])
        if name == "전체" and any(r["ev_cov"] is not None for r in ga[name]) and any(r["ev_cov"] is not None for r in gb[name]):
            ms.append(("coverage", "ev_cov"))
        for label, m in ms:
            pa, pb = A.point(m), Bt.point(m)
            if pa is None or pb is None:
                continue
            d = A.dist(m) - Bt.dist(m)
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            res[(name, label)] = {"n": len(ga[name]), "diff": pa - pb, "ci": (float(lo), float(hi))}
    return res, len(ids)


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def fmt_ci(v, m):
    if v.get(m) is None:
        return "—"
    c = v.get(m + "_ci")
    return f"{v[m]:.3f} [{c[0]:.3f}, {c[1]:.3f}]" if c else f"{v[m]:.3f}"


def report(run, kind, rows, s, gold_sha):
    name = os.path.basename(os.path.normpath(run))
    lines = [f"v3 채점 — {name}  (시스템 {kind}{' · 검색형: 컨텍스트가 비면 근거 회수 0' if kind in RETRIEVAL else ''})",
             f"gold.jsonl sha256 {str(gold_sha)[:16]} · 문항 {len(rows)} · 응답 {sum(r['answered'] for r in rows)} · "
             f"bootstrap {B}회 seed {SEED} (짝 단위·카테고리 층화)", "",
             "묶음 | 문항(응답) | EM [95% CI] | F1 | LLM-Eq [95% CI] | 짝 LLM-Eq [95% CI] | 짝 EM [95% CI] | 짝 수 | "
             "근거 any | 근거 all | coverage [95% CI] | 기권"]
    for k, v in s.items():
        lines.append(f"{k} | {v['n']}({v['answered']}) | {fmt_ci(v, 'em')} | {fmt(v['f1'])} | {fmt_ci(v, 'judge')} | "
                     f"{fmt_ci(v, 'pair_judge')} | {fmt_ci(v, 'pair_em')} | {v['n_pairs'] or '—'} | {fmt(v['ev_any'])} | "
                     f"{fmt(v['ev_all'])} | {fmt_ci(v, 'ev_cov')} | {fmt(v['abstain'])}")
    t = s["전체"]
    lines += ["", f"LLM-Eq 판정 {'모든 문항에 있음' if t['judge_complete'] else '없음 또는 일부(주 지표는 EM 으로 읽는다)'}",
              f"검색형 컨텍스트 빈 문항 {t['ctx_empty'] if kind in RETRIEVAL else '해당 없음'} · EM=1&any=0 {t['em1_any0']} · EM=0&all=1 {t['em0_all1']}",
              "근거 8편 초과 문항(Vanilla@8 로는 all 불가능): " + " · ".join(f"{k.replace('멀티홉 · ', '')} {v['over8']}" for k, v in s.items() if k.startswith("멀티홉 · "))]
    return "\n".join(lines) + "\n"


def compare_md(table, rows_by, kinds):
    names = list(table)
    head = "| 묶음 | " + " | ".join(f"{n} ({len(rows_by[n])})" for n in names) + " |"
    sep = "|---|" + "---|" * len(names)
    keys = list(collections.OrderedDict((k, 1) for t in table.values() for k in t))

    def tab(title, cell, only=None):
        out = [f"## {title}", "", head, sep]
        for k in keys:
            if only and not only(k):
                continue
            out.append(f"| {k} | " + " | ".join(cell(table[n][k], n) if k in table[n] else "—" for n in names) + " |")
        return out + [""]

    main = lambda v, n: f"EM {fmt_ci(v, 'em')}"
    pair = lambda v, n: ((f"{fmt_ci(v, 'pair_em')} (LLM-Eq {fmt(v['pair_judge'])})" if v["judge_complete"] else f"{fmt_ci(v, 'pair_em')}")
                         if v["n_pairs"] else "—")
    is_pair = lambda k: k.startswith(PAIR_CATS)
    md = ["# v3 비교표 — tools/27_채점.py", "",
          f"bootstrap {B}회 · seed {SEED} · 재표집 단위 짝(인물·시간)/문항(멀티홉), 카테고리 층화. 열 이름 옆 괄호 = 채점 문항 수(graphrag_ctx 는 표본).",
          "주 지표는 EM(95% CI). LLM-Eq 는 보조. 짝 정확도는 EM 기준, 괄호는 LLM-Eq 기준.", ""]
    md += tab("주 지표 EM [95% CI]", main)
    md += tab("짝 정확도 EM [95% CI] (괄호 LLM-Eq)", pair, is_pair)
    md += tab("LLM-Eq [95% CI] (보조)", lambda v, n: fmt_ci(v, "judge"))
    md += tab("F1", lambda v, n: fmt(v["f1"]))
    md += tab("근거 회수 any / all / coverage [95% CI] (검색형은 컨텍스트가 비면 0)",
              lambda v, n: "—" if v["ev_cov"] is None else f"{fmt(v['ev_any'])} / {fmt(v['ev_all'])} / {fmt_ci(v, 'ev_cov')}")
    md += tab("기권률 ('근거 부족')", lambda v, n: fmt(v["abstain"]))
    md += ["## 시스템 간 paired 차이 (A − B) [95% CI] — 공통 문항, 같은 재표집", "",
           "| A − B | 공통 문항 | 묶음 | 지표 | 차이 [95% CI] |", "|---|---|---|---|---|"]
    focus = {"인물판정", "인물판정 · 같은 사람", "인물판정 · 다른 사람", "시간추론", "멀티홉", "전체"} | {k for k in keys if k.startswith("멀티홉 · ")}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            res, n_common = paired(rows_by[a], rows_by[b], focus)
            for (g, label), v in res.items():
                md.append(f"| {a} − {b} | {n_common} | {g} | {label} | {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] |")
    md += ["", "## 근거·검증 수", "", "| 시스템 | 종류 | 검색형 컨텍스트 빈 문항 | EM=1 & any=0 | EM=0 & all=1 | 근거 8편 초과(멀티홉) |", "|---|---|---|---|---|---|"]
    for n in names:
        t = table[n]["전체"]
        md.append(f"| {n} | {kinds[n]} | {t['ctx_empty'] if kinds[n] in RETRIEVAL else '—'} | {t['em1_any0']} | {t['em0_all1']} | {t['over8']} |")
    return "\n".join(md) + "\n"


def depth_csv(table):
    fmt = lambda x: "" if x is None else f"{x:.3f}"
    lines =["system,depth,n,main_metric,main,main_lo,main_hi,em,em_lo,em_hi,coverage"]
    for n, t in table.items():
        for k, v in t.items():
            if not k.startswith("멀티홉 · "):
                continue
            mm = "em"
            c, e = v.get(mm + "_ci") or (None, None), v.get("em_ci") or (None, None)
            lines.append(",".join([n, '"' + k.replace("멀티홉 · ", "") + '"', str(v["n"]), "LLM-Eq" if mm == "judge" else "EM",
                                   fmt(v[mm]), fmt(c[0]), fmt(c[1]), fmt(v["em"]), fmt(e[0]), fmt(e[1]), fmt(v["ev_cov"])]))
    return "\n".join(lines) + "\n"


def run_all(runs, gold_path=C.GOLD, table_path=None, override=None, quiet=False):
    gold_rows = C.load_jsonl(gold_path)
    gold_sha = C.sha256(gold_path)
    table, rows_by, kinds = collections.OrderedDict(), {}, {}
    for run in runs:
        name = os.path.basename(os.path.normpath(run))
        kinds[name] = kind = system_kind(run, override)
        rows_by[name] = rows = score(run, gold_rows, kind)
        table[name] = s = summarize(rows)
        text = report(run, kind, rows, s, gold_sha)
        open(os.path.join(run, "채점보고.txt"), "w", encoding="utf-8", newline="\n").write(text)
        if not quiet:
            print(text)
    if len(runs) > 1:
        table_path = table_path or os.path.join(ROOT, "rag", "runs", "v3_비교표.md")
        md = compare_md(table, rows_by, kinds)
        open(table_path, "w", encoding="utf-8", newline="\n").write(md)
        cdir = os.path.join(os.path.dirname(table_path), "v3_채점")
        os.makedirs(cdir, exist_ok=True)
        open(os.path.join(cdir, "그림2_depth.csv"), "w", encoding="utf-8", newline="\n").write(depth_csv(table))
        if not quiet:
            print(md)
    return table, rows_by


def _write(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def selftest():
    fails = []

    def check(label, cond, got=None):
        print(f"  {'○' if cond else '✗'} {label}" + ("" if cond else f"   (값 {got})"))
        if not cond:
            fails.append(label)

    tmp = tempfile.mkdtemp(prefix="27selftest_")
    try:
        G = [
            {"id": "P1a", "category": "인물판정", "sub": "같은 사람", "pair_id": "P1", "answer": "이비(李裶)", "answer_aliases": ["경녕군"], "evidence_ids": ["d1"], "alias_class": "봉호·군호"},
            {"id": "P1b", "category": "인물판정", "sub": "같은 사람", "pair_id": "P1", "answer": "500마리", "answer_aliases": [], "evidence_ids": ["d1", "d2"], "alias_class": "봉호·군호"},
            {"id": "P2a", "category": "인물판정", "sub": "다른 사람", "pair_id": "P2", "answer": "우의정", "answer_aliases": [], "evidence_ids": ["d3"]},
            {"id": "P2b", "category": "인물판정", "sub": "다른 사람", "pair_id": "P2", "answer": "좌의정", "answer_aliases": [], "evidence_ids": ["d4"]},
            {"id": "T1a", "category": "시간추론", "sub": "", "pair_id": "T1", "answer": "옹진 진군 충당", "answer_aliases": ["옹진의 진군에 충당"], "evidence_ids": ["d5", "d6"], "relation_family": "순서"},
            {"id": "T1b", "category": "시간추론", "sub": "", "pair_id": "T1", "answer": "단천 관노", "answer_aliases": [], "evidence_ids": ["d5", "d7"], "relation_family": "순서"},
            {"id": "M1", "category": "멀티홉", "sub": "2hop", "pair_id": None, "answer": "이지강", "answer_aliases": [], "evidence_ids": ["d8", "d9"]},
            {"id": "M2", "category": "멀티홉", "sub": "7hop", "pair_id": None, "answer": "황희", "answer_aliases": [], "evidence_ids": [f"e{i}" for i in range(9)]},
        ]
        for g in G:
            g["n_evidence"] = len(g["evidence_ids"])
        gp = os.path.join(tmp, "gold.jsonl")
        _write(gp, G)

        def mk(name, answers, judged=None):
            d = os.path.join(tmp, name)
            os.makedirs(d)
            _write(os.path.join(d, "answers.jsonl"), answers)
            if judged:
                _write(os.path.join(d, "judged.jsonl"), judged)
            return d

        right = {g["id"]: g["answer"] for g in G}
        ov = {}
        print("1) 정답을 응답으로 → EM 1.0 · 짝 EM 1.0 (별칭·한자 병기 표기 포함)")
        alt = dict(right, P1a="경녕군", T1a="옹진의 진군에 충당", M1="이지강(李之剛)")
        d = mk("perfect", [{"id": i, "answer": a, "context_docs": next(g["evidence_ids"] for g in G if g["id"] == i)} for i, a in alt.items()])
        ov["perfect"] = "gold"
        t, rows = run_all([d], gp, override=ov, quiet=True)
        s = t["perfect"]
        check("전체 EM = 1.0", s["전체"]["em"] == 1.0, s["전체"]["em"])
        check("인물판정 짝 EM = 1.0", s["인물판정"]["pair_em"] == 1.0, s["인물판정"]["pair_em"])
        check("시간추론 짝 EM = 1.0", s["시간추론"]["pair_em"] == 1.0, s["시간추론"]["pair_em"])
        check("Gold(컨텍스트 = 근거) coverage = 1.0", s["전체"]["ev_cov"] == 1.0, s["전체"]["ev_cov"])

        print("2) 빈 응답 → 0")
        d = mk("empty", [{"id": i, "answer": "", "context_docs": []} for i in right])
        ov["empty"] = "closed"
        s = run_all([d], gp, override=ov, quiet=True)[0]["empty"]
        check("전체 EM = 0", s["전체"]["em"] == 0.0, s["전체"]["em"])
        check("전체 F1 = 0", s["전체"]["f1"] == 0.0, s["전체"]["f1"])
        check("짝 EM = 0", s["인물판정"]["pair_em"] == 0.0 and s["시간추론"]["pair_em"] == 0.0)
        check("닫힌 책: 근거 회수 해당 없음(None)", s["전체"]["ev_any"] is None and s["전체"]["ev_cov"] is None, s["전체"]["ev_any"])

        print("3) 짝 한쪽만 맞음 → 그 짝 0")
        half = dict(right, P1b="499마리")
        d = mk("half", [{"id": i, "answer": a, "context_docs": []} for i, a in half.items()],
               judged=[{"id": i, "pred": a, "judge": 0.0 if i in ("P1b", "P2b") else 1.0} for i, a in half.items()])
        ov["half"] = "closed"
        s = run_all([d], gp, override=ov, quiet=True)[0]["half"]
        check("같은 사람 짝 EM = 0 (P1 한쪽만 맞음)", s["인물판정 · 같은 사람"]["pair_em"] == 0.0, s["인물판정 · 같은 사람"]["pair_em"])
        check("인물판정 짝 EM = 0.5 (P2 는 둘 다 맞음)", s["인물판정"]["pair_em"] == 0.5, s["인물판정"]["pair_em"])
        check("인물판정 문항 EM = 0.75", s["인물판정"]["em"] == 0.75, s["인물판정"]["em"])
        check("판정 기준 짝(P1b·P2b 판정 오답) 인물 = 0.0", s["인물판정"]["pair_judge"] == 0.0, s["인물판정"]["pair_judge"])
        check("alias_class 묶음(봉호·군호)이 있다", "인물판정 · 같은 사람 · 봉호·군호" in s)
        check("시간추론 계열 묶음(순서)이 있다", "시간추론 · 순서" in s)
        check("깊이 표기 '7-page chain (6 relations)'", "멀티홉 · 7-page chain (6 relations)" in s and "멀티홉 · 2-page chain (1 relation)" in s)
        check("근거 8편 초과 1(M2)", s["전체"]["over8"] == 1, s["전체"]["over8"])

        print("4) 검색형에서 컨텍스트가 비면 any = all = coverage = 0 (None 아님), error 행은 미응답")
        rows = [{"id": i, "answer": a, "context_docs": []} for i, a in right.items() if i != "M2"]
        rows[0] = {"id": rows[0]["id"], "error": "RateLimitError"}
        d = mk("retr_empty", rows)
        ov["retr_empty"] = "vanilla"
        t, rb = run_all([d], gp, override=ov, quiet=True)
        s = t["retr_empty"]
        check("전체 any = 0", s["전체"]["ev_any"] == 0.0, s["전체"]["ev_any"])
        check("전체 all = 0", s["전체"]["ev_all"] == 0.0, s["전체"]["ev_all"])
        check("전체 coverage = 0", s["전체"]["ev_cov"] == 0.0, s["전체"]["ev_cov"])
        check("모든 문항이 분모에 있다(8/8 값 있음)", all(r["ev_cov"] == 0.0 for r in rb["retr_empty"]))
        check("error 행·빠진 행은 미응답(응답 6)", s["전체"]["answered"] == 6, s["전체"]["answered"])
        check("컨텍스트 빈 문항 8", s["전체"]["ctx_empty"] == 8, s["전체"]["ctx_empty"])

        print("5) coverage 알려진 값: E={d8,d9}, C={d8} → any 1 · all 0 · coverage 0.5")
        d = mk("cov", [{"id": "M1", "answer": "이지강", "context_docs": ["d8", "x"]}])
        ov["cov"] = "vanilla"
        rows = run_all([d], gp, override=ov, quiet=True)[1]["cov"]
        m1 = next(r for r in rows if r["id"] == "M1")
        check("M1 any 1 · all 0 · coverage 0.5", (m1["ev_any"], m1["ev_all"], m1["ev_cov"]) == (1.0, 0.0, 0.5), (m1["ev_any"], m1["ev_all"], m1["ev_cov"]))
        check("EM=1&any=0 = 0 (M1 은 any 1), 컨텍스트 없는 7문항도 분모에 있음(any 0)",
              sum(1 for r in rows if r["ev_any"] == 0) == 7 and sum(1 for r in rows if r["em"] == 1 and r["ev_any"] == 0) == 0)

        print("6) bootstrap 재현성 · paired 차이")
        r1 = summarize(rb["retr_empty"])
        r2 = summarize(rb["retr_empty"])
        check("seed 고정 두 번 → CI 같음", json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True))
        check("CI 폭 > 0 (전체 EM)", r1["전체"]["em_ci"][0] < r1["전체"]["em_ci"][1], r1["전체"]["em_ci"])
        res, _ = paired(rb["retr_empty"], rb["retr_empty"])
        check("같은 시스템끼리 paired 차이 = 0 [0, 0]", all(v["diff"] == 0 and v["ci"] == (0.0, 0.0) for v in res.values()))
        dp, dh = os.path.join(tmp, "perfect"), os.path.join(tmp, "half")
        t, rb2 = run_all([dp, dh], gp, table_path=os.path.join(tmp, "비교표.md"), override=ov, quiet=True)
        res, n = paired(rb2["perfect"], rb2["half"])
        v = res[("전체", "EM")]
        check("perfect − half 전체 EM 차이 = 0.125 (1/8)", abs(v["diff"] - 0.125) < 1e-12, v["diff"])
        check("paired CI 가 차이를 포함", v["ci"][0] <= v["diff"] <= v["ci"][1], v["ci"])

        print("7) 채점보고·비교표 바이트 재현")
        b1 = [open(p, "rb").read() for p in (os.path.join(dh, "채점보고.txt"), os.path.join(tmp, "비교표.md"))]
        run_all([dp, dh], gp, table_path=os.path.join(tmp, "비교표.md"), override=ov, quiet=True)
        b2 = [open(p, "rb").read() for p in (os.path.join(dh, "채점보고.txt"), os.path.join(tmp, "비교표.md"))]
        check("다시 돌려도 바이트 단위로 같다", b1 == b2)

        if os.path.exists(C.GOLD):
            print(f"8) 실제 gold.jsonl({C.sha256(C.GOLD)[:16]})에 정답을 응답으로 → 1.0")
            real = C.load_jsonl(C.GOLD)
            d = mk("real", [{"id": g["id"], "answer": g["answer"], "context_docs": g["evidence_ids"]} for g in real])
            ov["real"] = "gold"
            s = run_all([d], C.GOLD, override=ov, quiet=True)[0]["real"]
            check(f"전체 EM = 1.0 ({s['전체']['n']}문항)", s["전체"]["em"] == 1.0, s["전체"]["em"])
            check("인물·시간 짝 EM = 1.0", s["인물판정"]["pair_em"] == 1.0 and s["시간추론"]["pair_em"] == 1.0)
            check("인물 alias_class 5부류 묶음 모두 있음", all(f"인물판정 · 같은 사람 · {c}" in s for c in ALIAS_ORDER))
            check("시간 계열 4묶음 모두 있음", all(f"시간추론 · {f}" in s for f in FAM_ORDER))
            print("   짝 수: " + " · ".join(f"{k} {v['n_pairs']}" for k, v in s.items() if v["n_pairs"] and k.startswith(PAIR_CATS)))
            d = mk("real_empty", [{"id": g["id"], "answer": "", "context_docs": []} for g in real])
            ov["real_empty"] = "vanilla"
            s = run_all([d], C.GOLD, override=ov, quiet=True)[0]["real_empty"]
            check("실제 gold · 빈 응답·빈 컨텍스트(검색형) → EM 0 · coverage 0", s["전체"]["em"] == 0.0 and s["전체"]["ev_cov"] == 0.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nselftest {'통과' if not fails else '실패 ' + str(len(fails))}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--table", default=None)
    ap.add_argument("--gold", default=C.GOLD)
    ap.add_argument("--system", nargs="*", default=[], help="run폴더이름=종류 (closed|gold|vanilla|graphrag|graphrag_ctx)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.runs:
        ap.error("run 폴더를 하나 이상 주거나 --selftest")
    run_all(a.runs, a.gold, a.table, dict(x.split("=", 1) for x in a.system))


if __name__ == "__main__":
    main()
