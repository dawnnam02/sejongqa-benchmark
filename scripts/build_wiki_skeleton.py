"""Build the wiki skeleton (python scripts/build_wiki_skeleton.py WIKI_ENTRIES_DIR [OUT_JSONL] [--check ...]).

The LLM-written wiki is not released. This script turns its entry pages (one Markdown file per page)
into an identifier-level skeleton with one JSON line per page:

  page          page title (front-matter `title`)
  page_type     the first of 인물 / 사건 / 장소 / 제도 / 기물 / 문헌 found in the front-matter tags
  tags          all front-matter tags (topic labels such as 사법, 외교)
  aliases       name tokens from the "이명" (other names) line of the person-identification box
  caution_names name tokens from the "주의 인물" (persons not to confuse) line of the same box
  timeline      rows of the sections whose heading contains 연표 (timeline): {"date", "ids"}
  links         target pages of [[wiki links]] in order of first appearance (the page itself excluded)
  sections      every heading in order: {"level", "heading", "ids"}; text before the first
                sub-heading is {"level": 1, "heading": "_lead"}
  cited_ids     all article identifiers cited on the page, sorted

No sentence of the wiki is copied. Prose, quotations, table cells other than the date column, link
display texts, the explanations in the identification box and descriptive headings are dropped. Only
page titles, tags, name tokens, dates, generic heading labels and article identifiers are kept.

Name tokens (`aliases`, `caution_names`). The field value is read together with its continuation
lines. Anything that does not satisfy one of rules 1-3 is left out (when in doubt, a token is dropped).
  0. Removed first: code spans (`...`), quoted passages ("...", *"..."*, “...”), corner-bracket spans
     (「...」, 『...』) and article identifiers. Link markup [[T|D]] is read as its target T.
  1. Hanja-anchored name: Hangul syllables immediately followed by a parenthesis that opens with
     Hanja, e.g. 강상례(姜尙禮), 원숙(元肅, ...), 상선감 좌소감(尙膳監左少監). The Hangul part is the
     last k syllables before the parenthesis, where k is the number of Hanja characters (the first
     variant if several are separated by '/'); spaces inside are kept. The token is kept only if
     k >= 2, the k syllables start at a word boundary, and the parenthesis is not followed by the
     possessive 의 (a modifier such as 대마주(對馬州)의 ...). The same rule applies inside
     parentheses of the form (이제 李褆), giving 이제(李褆). Output form: Hangul(Hanja).
  2. Bold name: a bold span (**...**) with parenthesized text removed is split on · / , → and 또는.
     The span is used only if every piece is name-like: one word of 2-8 Hangul syllables, or two
     words of at least 2 syllables each whose first word does not end in a particle
     (은 는 이 가 을 를 의 와 과 로 에 도 만) and whose last word does not end in a verb ending
     (다 고 며 서 면 게 니); no piece may contain a word from STOP (kinship terms and descriptive
     words). Pieces with a word equal to the Hangul part of a rule-1
     name are skipped. In `caution_names`, a piece that is not a
     wiki-link target must also begin with a common Korean surname (SURNAMES), so that office
     titles are not taken for persons.
  3. Listed names: a run of three or more items joined by '·', each item 2-4 Hangul syllables not
     ending in a particle or 다, delimited by spaces, punctuation or the ends of the field
     (e.g. 김녕·김종서·김습·...). A line break directly after '·' continues the run.
  4. Tokens in NOT_NAMES (common nouns that pass the rules above in this wiki) are dropped. The page's
     own title is dropped from `aliases`, and from `caution_names` unless it carries Hanja.

Heading labels. Each heading is cleaned (markup, identifiers, symbols, list numbering and
parenthesized Hanja removed) and split into a date label and a remainder. The date label is a leading
part before a dash, or a trailing parenthesis, that passes the date test below. The remainder is kept
only if the same remainder heads sections on at least three pages (a generic label such as 연표,
참고, 평가, 전개, 관력). The output heading is "remainder date", the remainder alone, the date alone,
or null. Descriptive headings therefore appear as null or as their date label only.

Date test (timeline rows and heading date labels): after removing markup and symbols, the text
consists only of digits, punctuation, calendar words (즉위년 년 월 일 윤 무렵 모일 봄 여름 가을 겨울 미상),
sexagenary-year names and king names, and contains a digit. The first table cell (or the leading bold
text of a list item) of a timeline row is kept as its date if it passes; otherwise the date is null.
Rows with neither a date nor an identifier are dropped.

Checks (--check EVIDENCE_JSONL): (a) no string value is sentence-like (20 or more characters ending
in a verb ending or particle); (b) no string value other than a name token shares a 10-character
window (spaces removed) with an evidence quote. Name tokens that share such a window are proper
nouns and are listed for information.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

TYPES = ["인물", "사건", "장소", "제도", "기물", "문헌"]
ID_RE = re.compile(r"\d{2}-(?:윤)?\d{2}-\d{2}\[\d{2}\]")
LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
HAN = "一-鿿㐀-䶿豈-﫿"
HANGUL = "가-힣"
PARTICLE_END = tuple("은는이가을를의와과로에도만")
VERB_END = tuple("다요까라나냐지고며자")
PIECE_VERB_END = tuple("다고며서면게니")
STOP = ["별개", "인물", "혼동", "주의", "구분", "구별", "동일", "한자", "원문", "표기", "없음", "모두", "함께",
        "나란히", "반드시", "특히", "관계", "사람", "확인", "아버지", "어머니", "아들", "아우", "형제", "조카",
        "숙부", "할아버지", "손자", "외조", "사위", "서형", "관직", "통칭", "봉호", "시호", "국역", "각주",
        "기사", "이름", "호칭", "이표기", "직명", "지칭", "무관", "실명"]
NOT_NAMES = {"전시", "유학", "얼자", "사인", "이복형", "황보", "한자", "여덟"}
SURNAMES = set("김이박최정강조윤장임한오서신권황안송유류홍전고문양손배백허남심노하곽성차주우구민나진지엄채원"
               "천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예경봉복목형피두감음"
               "빈동온호범팽승간갈견") | {"황보", "남궁", "선우", "제갈"}
KINGS = ["태조", "정종", "태종", "세종", "우왕", "창왕", "공양왕"]
GAN, JI = "갑을병정무기경신임계", "자축인묘진사오미신유술해"
GANZHI = sorted({GAN[i % 10] + JI[i % 12] for i in range(60)})
DATE_WORDS = ["즉위년", "무렵", "모일", "여름", "가을", "겨울", "미상", "봄", "년", "월", "일", "윤"]


def strip_markup(s):
    s = LINK_RE.sub(r"\1", s)
    s = re.sub(r"[*`_]", "", s)
    s = re.sub(r"[★☆⚠️◆●■▶→←↔✓✔①②③④⑤⑥⑦⑧⑨⑩]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def remove_quoted(s):
    s = re.sub(r"`[^`]*`", " ", s)
    s = re.sub(r"\*\"[^\"]*\"\*", " ", s)
    s = re.sub(r"\"[^\"]*\"", " ", s)
    s = re.sub(r"“[^”]*”", " ", s)
    s = re.sub(r"「[^」]*」", " ", s)
    s = re.sub(r"『[^』]*』", " ", s)
    return ID_RE.sub(" ", s)


def is_hangul(c):
    return bool(re.match("[" + HANGUL + "]", c))


def hanja_names(t):
    out = []
    t = LINK_RE.sub(r"\1", t).replace("**", "")
    for m in re.finditer(r"\(([" + HAN + r"]+)(?:[/·][" + HAN + r"]+)*(?=[,)\s—/·])", t):
        k = len(m.group(1))
        if k < 2:
            continue
        close = t.find(")", m.end())
        if close >= 0 and t[close + 1:close + 2] == "의":
            continue
        pre = t[:m.start()]
        syl, i = 0, len(pre)
        while i > 0 and syl < k:
            c = pre[i - 1]
            if is_hangul(c):
                syl += 1
            elif c != " " or syl == 0:
                break
            i -= 1
        if syl != k or (i > 0 and is_hangul(pre[i - 1])):
            continue
        out.append(re.sub(r"\s+", " ", pre[i:].strip()) + "(" + m.group(1) + ")")
    for m in re.finditer(r"\(([" + HANGUL + r"]{2,4}) ([" + HAN + r"]{2,4})\)", t):
        if len(m.group(1)) == len(m.group(2)):
            out.append(m.group(1) + "(" + m.group(2) + ")")
    return out


def piece_ok(piece):
    words = piece.split()
    if not words or len(words) > 2 or not all(re.fullmatch("[" + HANGUL + "]{1,8}", w) for w in words):
        return False
    if len(words) == 1 and len(words[0]) < 2:
        return False
    if len(words) == 2 and (min(len(w) for w in words) < 2 or words[0].endswith(PARTICLE_END)):
        return False
    if words[-1].endswith(PIECE_VERB_END) or any(st in piece for st in STOP):
        return False
    return True


def bold_names(raw, anchored, caution):
    out = []
    anchored_hangul = {a.split("(")[0] for a in anchored}
    for span in re.findall(r"\*\*(.+?)\*\*", raw):
        link_targets = {x.strip() for x in LINK_RE.findall(span)}
        span = re.sub(r"\([^)]*\)", " ", LINK_RE.sub(r"\1", span))
        span = re.sub(r"[‘’'\"]", "", span)
        pieces = [strip_markup(p).strip(" .:;-—") for p in re.split(r"\s*(?:[·/,→]|또는)\s*", span)]
        pieces = [p for p in pieces if p]
        if not pieces or not all(piece_ok(p) for p in pieces):
            continue
        for p in pieces:
            if any(w in anchored_hangul for w in p.split()):
                continue
            if caution and p not in link_targets and not (p[:2] in SURNAMES or p[0] in SURNAMES):
                continue
            out.append(p)
    return out


def listed_names(t):
    t = strip_markup(t)
    out = []
    pat = r"(?:(?<=^)|(?<=[\s,.(]))((?:[" + HANGUL + r"]{2,4}·){2,}[" + HANGUL + r"]{2,4})(?=$|[\s,.)])"
    for m in re.finditer(pat, t):
        items = m.group(1).split("·")
        if all(not x.endswith(PARTICLE_END + ("다",)) for x in items):
            out.extend(items)
    return out


def name_tokens(value, caution):
    raw = remove_quoted(value)
    anchored = hanja_names(raw)
    toks = anchored + bold_names(raw, anchored, caution) + listed_names(raw)
    seen, out = set(), []
    for x in toks:
        x = re.sub(r"\s+", " ", x).strip()
        if x and x not in seen and x.split("(")[0] not in NOT_NAMES:
            seen.add(x)
            out.append(x)
    return out


def box_field(lines, label):
    for i, l in enumerate(lines):
        m = re.match(r"^>\s*-\s*\*\*" + label + r"\*\*\s*:(.*)", l)
        if m:
            val = m.group(1).strip()
            j = i + 1
            while j < len(lines) and lines[j].startswith(">") and not re.match(r"^>\s*-", lines[j]) \
                    and lines[j].strip() != ">":
                nxt = lines[j][1:].strip()
                val = val + nxt if val.endswith("·") else val + " " + nxt
                j += 1
            return val
    return None


def clean_date(c):
    c = strip_markup(c).strip(" —-:,")
    if not c or not re.search(r"\d", c) or len(c) > 30:
        return None
    rest = c
    for w in sorted(KINGS + GANZHI + DATE_WORDS, key=len, reverse=True):
        rest = rest.replace(w, "")
    return c if re.fullmatch(r"[\d\s.·~\-,()○]*", rest) else None


def split_heading(h):
    """Return (remainder, date label) of a heading."""
    h = ID_RE.sub(" ", h)
    h = strip_markup(h)
    h = re.sub(r"\(\s*[" + HAN + r"·,\s]+\)", " ", h)
    h = re.sub(r"\(\s*[,·\s]*\)|\(§[^)]*\)", " ", h)
    h = re.sub(r"^\s*(?:\d+\s*[.)]|[가-하]\.)\s+", "", h)
    h = re.sub(r"\s+", " ", h).strip(" —-:,·")
    date = None
    parts = re.split(r"\s+[—–-]\s+", h, maxsplit=1)
    if len(parts) == 2 and clean_date(parts[0]):
        date, h = clean_date(parts[0]), parts[1]
    elif len(parts) == 2 and clean_date(parts[1]):
        date, h = clean_date(parts[1]), parts[0]
    else:
        m = re.search(r"\(([^()]*)\)\s*$", h)
        if m and clean_date(m.group(1)):
            date, h = clean_date(m.group(1)), h[:m.start()]
        elif clean_date(h):
            date, h = clean_date(h), ""
    rest = re.sub(r"\s+", " ", h).strip(" —-:,·")
    return rest, date


def parse_page(path):
    text = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    fm, body = {}, text
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        body = text[m.end():]
        for line in m.group(1).split("\n"):
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    title = fm.get("title") or os.path.splitext(os.path.basename(path))[0]
    tags = [t.strip() for t in fm.get("tags", "").strip("[]").split(",") if t.strip()]
    lines = body.split("\n")

    alias_v, caution_v = box_field(lines, "이명"), box_field(lines, "주의 인물")
    aliases = [a for a in (name_tokens(alias_v, False) if alias_v else []) if a.split("(")[0] != title]
    caution = [c for c in (name_tokens(caution_v, True) if caution_v else []) if c != title]

    sections, timeline = [], []
    cur = {"level": 1, "raw": None, "ids": []}
    in_tl = in_code = False
    for l in lines:
        if l.startswith("```"):
            in_code = not in_code
        hm = None if in_code else re.match(r"^(#{1,6})\s+(.*)$", l)
        if hm:
            if len(hm.group(1)) == 1:
                cur["ids"] += ID_RE.findall(hm.group(2))
                continue
            sections.append(cur)
            cur = {"level": len(hm.group(1)), "raw": hm.group(2), "ids": ID_RE.findall(hm.group(2))}
            in_tl = "연표" in hm.group(2)
            continue
        ids = ID_RE.findall(l)
        cur["ids"] += ids
        if in_tl and l.strip():
            date = None
            if l.lstrip().startswith("|"):
                cells = [c.strip() for c in l.strip().strip("|").split("|")]
                if re.fullmatch(r"[-:\s]*", cells[0]):
                    continue
                date = clean_date(cells[0])
            else:
                bm = re.match(r"^\s*-\s*\*\*(.+?)\*\*", l)
                if bm:
                    date = clean_date(bm.group(1))
            if date or ids:
                timeline.append({"date": date, "ids": list(dict.fromkeys(ids))})
    sections.append(cur)
    links = [x for x in dict.fromkeys(t.strip() for t in LINK_RE.findall(body)) if x != title]
    return {"page": title, "page_type": next((t for t in tags if t in TYPES), None), "tags": tags,
            "aliases": aliases, "caution_names": caution, "timeline": timeline, "links": links,
            "sections": sections, "cited_ids": sorted(set(ID_RE.findall(body)))}


def finish_sections(rows):
    pages_by_rest = collections.defaultdict(set)
    for r in rows:
        for s in r["sections"]:
            if s["raw"] is not None:
                pages_by_rest[split_heading(s["raw"])[0]].add(r["page"])
    generic = {k for k, v in pages_by_rest.items() if k and len(v) >= 3}
    for r in rows:
        out = []
        for s in r["sections"]:
            ids = list(dict.fromkeys(s["ids"]))
            if s["raw"] is None:
                if ids:
                    out.append({"level": 1, "heading": "_lead", "ids": ids})
                continue
            rest, date = split_heading(s["raw"])
            keep = rest if rest in generic else ""
            label = " ".join(x for x in (keep, date) if x) or None
            out.append({"level": s["level"], "heading": label, "ids": ids})
        r["sections"] = out
    return generic


def strings(o, key=None):
    if isinstance(o, str):
        yield key, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from strings(v, k)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v, key)


def check(rows, evidence_path, n=10):
    sent = [s for r in rows for _, s in strings(r)
            if len(s) >= 20 and re.search("[" + HANGUL + "]", s)
            and s.endswith(VERB_END + PARTICLE_END + ("니다", "였다", "했다"))]
    print("(a) sentence-like strings:", len(sent))
    grams = set()
    for l in open(evidence_path, encoding="utf-8"):
        if l.strip():
            for e in json.loads(l).get("evidence", []):
                q = re.sub(r"\s+", "", e.get("quote", ""))
                grams.update(q[i:i + n] for i in range(len(q) - n + 1))

    def hit(s):
        s = re.sub(r"\s+", "", s)
        return any(s[i:i + n] in grams for i in range(len(s) - n + 1))

    names = ("aliases", "caution_names")
    other = sorted({s for r in rows for k, s in strings(r) if k not in names and not ID_RE.fullmatch(s) and hit(s)})
    nm = sorted({s for r in rows for k, s in strings(r) if k in names and hit(s)})
    print(f"(b) non-name strings sharing a {n}-character window with an evidence quote:", len(other), other[:10])
    print(f"    name tokens sharing such a window (proper nouns, information only): {len(nm)}")
    return len(sent) + len(other)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("entries_dir", help="folder with the wiki entry pages (*.md)")
    ap.add_argument("out", nargs="?", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "wiki_skeleton", "wiki_skeleton.jsonl"))
    ap.add_argument("--check", metavar="EVIDENCE_JSONL", help="run the text checks against data/evidence.jsonl")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.entries_dir, "*.md")))
    if not files:
        sys.exit("no *.md files in the given folder")
    rows = sorted((parse_page(f) for f in files), key=lambda r: r["page"])
    generic = finish_sections(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    titles = {r["page"] for r in rows}
    secs = [s for r in rows for s in r["sections"]]
    print("pages", len(rows), dict(collections.Counter(r["page_type"] for r in rows)))
    print("aliases", sum(len(r["aliases"]) for r in rows), "on", sum(bool(r["aliases"]) for r in rows), "pages;",
          "caution_names", sum(len(r["caution_names"]) for r in rows), "on",
          sum(bool(r["caution_names"]) for r in rows), "pages")
    print("timeline rows", sum(len(r["timeline"]) for r in rows), "on", sum(bool(r["timeline"]) for r in rows),
          "pages; with date", sum(bool(t["date"]) for r in rows for t in r["timeline"]))
    print("links", sum(len(r["links"]) for r in rows), "to existing pages",
          sum(x in titles for r in rows for x in r["links"]))
    print("sections", len(secs), "with a label", sum(bool(s["heading"]) for s in secs),
          "generic labels", len(generic))
    print("page-id pairs", sum(len(r["cited_ids"]) for r in rows),
          "distinct cited ids", len({i for r in rows for i in r["cited_ids"]}))
    if a.check and check(rows, a.check):
        sys.exit(1)


if __name__ == "__main__":
    main()
