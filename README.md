# SejongQA (v3, frozen 2026-09-29)

SejongQA is a Korean historical question answering benchmark of 975 test-only questions over the modern Korean translation of the *Veritable Records of King Sejong* (Sejong Sillok). All questions, answers and the retrieval corpus are in Korean.
- **Alias** (75 contrastive pairs, 150 questions): the same question about one person asked under two different names or titles; both answers must be the same.
- **Confusable** (75 pairs, 150 questions): the same question about two different people who may be confused (identical or similar names, a shared surname and period, similar titles); the answers must differ. Alias and Confusable together are the **Identity** questions.
- **Temporal** (150 pairs, 300 questions): the same anchor event asked with two temporal relation expressions; the target article and the answer differ.
- **Relation Chain** (375 questions): questions built by sequentially connecting facts across 2, 3, 5 or 7 wiki documents.

This package accompanies the paper *SejongQA: A Korean Historical Question Answering Benchmark for Evaluating Person Identity and Temporal Reasoning*. It contains everything needed to check the paper's numbers. The source article texts are not included (see Corpus).

## Checking the paper's numbers
Python 3 (tested with 3.14). No API keys or network access are needed.
```
python scripts/final_paper_checks.py  # McNemar tests (Section IV-B) and the 141 of 150 Identity pairs (Section III-B)
python scripts/test_score.py          # scorer rules and EM numerators
python scripts/verify_paper.py        # Table I, Table II point values, corpus counts, chain-length EM, costs, index and graph checks
python scripts/paired_ci.py           # confidence intervals and the gpt-4.1 Gold-context run (needs numpy)
python scripts/evidence_arrival.py    # Table III and the evidence-delivery split of Section IV-C
python scripts/vi_recompute.py        # graph linkage counts (diagnostics not reported in the final paper)
python scripts/paper_diagnostics.py   # Identity pair counts, the 200-question re-read, other diagnostics
python scripts/full_period_check.py my_articles.jsonl   # full-period checks (needs numpy and a local copy of the translation)
python scripts/score.py runs/gold/answers.jsonl --judged runs/gold/judged.jsonl
```
Each script prints the recomputed value next to the expected value and exits with a non-zero status on a mismatch. All of them pass on this release.

The older scripts were written for earlier drafts of the paper. Their printed labels use the earlier names and section numbers (e.g., "Gold" or "Gold Evidence" for Gold-context, "different person" for Confusable, "Multi-hop" for Relation Chain, "Section V/VI" for Section IV), and they also check runs that the final paper does not report (Closed-Book, Vanilla RAG, BM25 only). The values themselves are unchanged. Where each number of the final paper is checked:

| Final paper | Value(s) | Script (printed label) |
|---|---|---|
| III-A | 4,273 articles, 2,214,948 characters, 773 evidence articles | `verify_paper.py` (corpus characters; gold articles) |
| III-B, Table I | counts, mean length and mean evidence per type; Temporal families 62 / 76 / 90 / 72 | `verify_paper.py` (Table I …; Temporal items by family) |
| III-B | 141 of 150 Identity pairs differ only in the name and particle | `final_paper_checks.py` |
| IV-B, Table II | EM by type, Pair EM, abstention, LLM-Eq for Gold-context, Local and Basic | `verify_paper.py` (gold / graphrag / graphrag_basic EM …) |
| IV-B | Local vs. Basic: McNemar p = 0.084 (all), p = 0.012 (Identity) | `final_paper_checks.py` |
| IV-B | 200-question re-read: EM 14.0 → 14.5 | `paper_diagnostics.py` (EM %: native, re-read) |
| IV-B, Table II note | gpt-4.1 Gold-context: EM 80.9, Pair EM 88.7 / 74.0 | `paired_ci.py` (gold_gpt41) |
| IV-C, Table III | Evidence Hit Rate 31.6 / 37.2; all delivered 12.7 / 12.4; Temporal both 5.0 / 5.7; 124 questions, EM 74.2 vs. 9.5 (Gold-context 91.9 vs. 72.2) | `evidence_arrival.py` (Local / Basic: any, all, Temporal both, …) |
| IV-C | 21 Alias pairs with one side correct for Local (5 for Gold-context) | `paper_diagnostics.py` (same-person pairs with exactly one side EM) |
| IV-C | Relation Chain EM, 2 → 7 documents: Local 30.4 → 3.5, Gold-context 90.2 → 42.4 | `verify_paper.py` (Multi-hop EM 2-page -> 7-page) |

## Terminology
| Final paper | File labels | Earlier drafts |
|---|---|---|
| Alias | `인물판정`, sub `같은 사람` | same person, Alias |
| Confusable | `인물판정`, sub `다른 사람` | different person, Homonym |
| Temporal; day / month / year / sequence | `시간추론`; `하루` / `달` / `해` / `순서` | Temporal; day / month / year / order |
| Relation Chain; k-document chain | `멀티홉`; `2hop` … `7hop` | Multi-hop; k-page chain |
| Gold-context | `runs/gold/` | Gold Evidence, Oracle |
| GraphRAG Local / Basic | `runs/graphrag/` / `runs/graphrag_basic/` | GraphRAG Local / Basic search |
| Evidence Hit Rate | `ev_any` in the scorer | evidence recall, "any gold article reaches the reader" |
| evidence articles (gold articles) | `evidence_ids` | gold articles |

## Layout
```
README.md  LICENSE.md  MANIFEST.json
data/
  questions.jsonl            975 items: id, category, sub, pair_id, question (give only the question to systems)
  answers.jsonl              answer, answer_aliases, evidence_ids, n_evidence, relation, relation_family, alias_class
  evidence.jsonl             short quotes supporting each answer; temporal anchor/target; relation chains over wiki documents
  article_ids.tsv            the 4,273 corpus article IDs with text length and SHA-256 (no text)
  author_sample.tsv          IDs, category and verdict of the 100-item stratified sample reviewed by the first author
                             during construction (an earlier review stage; see Provenance and validation)
  temporal_prev_month.tsv    month-adjacency check of the 19 previous-month items
  corpus_stats.json          corpus size, Vanilla RAG chunk count and length limit, hub-node size
  validation_summary.json    verdict counts of re-solving without the answer key (first pass and isolated re-solving)
  ledger_ids.tsv             the construction ledger at identifier level: all 11,275 articles read for the wiki
                             and whether each was incorporated (4,213 yes, 7,062 no); titles and reasons are not included
runs/
  gold/                      Gold-context (all evidence articles in full, no retrieval)          reported in the paper
  graphrag/                  GraphRAG Local search                                              reported in the paper
  graphrag_basic/            GraphRAG Basic search (added after the main runs)                  reported in the paper
  graphrag_ctx/              re-read of 200 stored Local contexts with the Gold-context prompt  reported in the paper
  gold_gpt41/                Gold-context re-read with gpt-4.1 (same prompt and output cap)     reported in the paper
  closed/ vanilla/ vanilla_bm25/   Closed-Book, Vanilla RAG and BM25-only runs; released, not reported in the final paper
  judge_manifest.json        LLM-Eq judge settings and prompt
  each run: answers.jsonl, judged.jsonl (LLM-Eq votes), manifest.json
  vanilla/retrieval.jsonl, manifest_retrieval.json   BM25, dense and fused rankings (chunk IDs)
  graphrag/context_index.jsonl                       per question: context article IDs, report, entity,
                                                     relationship and text-unit IDs (no text)
  graphrag_basic/answers.jsonl                       answer, context article IDs and text-unit IDs (no text)
  graphrag_ctx/sample.json                           the 200 IDs and the sampling strata
graph/                       GraphRAG index at identifier level: entities, relationships, text units (no descriptions)
graphrag/                    settings.yaml, prompts/, index_stats.json
diagnostics/                 identity, temporal, relation-chain and graph diagnostics (see Released diagnostics)
prompts/                     reader prompts (Closed-Book, Vanilla RAG, Gold-context) and the LLM-Eq judge prompt
wiki_skeleton/
  wiki_skeleton.jsonl        the wiki's 176 documents at identifier level (no wiki text; see Wiki skeleton)
scripts/                     score.py, final_paper_checks.py, test_score.py, verify_paper.py, paired_ci.py,
                             evidence_arrival.py, vi_recompute.py, paper_diagnostics.py, full_period_check.py,
                             rebuild_corpus.py, build_wiki_skeleton.py
code/                        retrieval, query, judge and scoring code used for the runs (see Re-running the systems)
```

## Format rules
- **Identity pairs** ask the same question about a person under two references; they do not ask whether the names denote the same person. In 141 of the 150 pairs the two questions differ only in the person's name and the attached particle; in the other 9 (PJ-007, PJ-009, PJ-010, PJ-037, PJ-038, PJ-146, PJ-147, PJ-149, PJ-150) the referring expression or part of the context also differs.
- **Temporal pairs** share the anchor article and differ in the relation expression; in 34 of 150 pairs the attached verb ending also differs. Questions give no date or year; the anchor is identified from the event description. Definitions: the previous / following day = the article dated one day earlier / later; early / late in that month = days 1–10 / 21–end of the anchor's lunar month; early / middle / late in that year = months 1–3 / 4–9 / 10–12 of the same regnal year; previous year / following year / the year after next = regnal year −1 / +1 / +2; previous month = the lunar month immediately before the anchor's, an intercalary month counting as its own (a 62-day window only screened candidates; `data/temporal_prev_month.tsv`); immediately before / after = the adjacent article on the same matter in the corpus. No item was made when no suitable article existed in the target interval or when more than one answer was plausible.
- **Relation Chain**: a k-document chain connects k wiki documents through k − 1 relations; the question names only the starting entity. Chain length is a construction attribute, not a count of reasoning steps or of required evidence articles.
- **Questions** are at most 50 characters (70 for 5-document and 90 for 7-document chains) and contain no dates, era names or sexagenary years.

## Scoring
- **EM (primary)**: NFKC, then remove parenthesized text, Hanja, punctuation and spaces, and lowercase; compare with the answer and every alias (e.g., 남지(南智) and 남지 are the same answer). Aliases were fixed with the questions before any run (the frozen `data/answers.jsonl` hash is in `MANIFEST.json`). Aliases that become empty are ignored, so a Hanja-only answer cannot match.
- **Pair EM**: over the 150 Identity and 150 Temporal pairs, a pair counts only if both questions are correct.
- **Abstention**: an answer containing `근거 부족` ("insufficient evidence") is an abstention and counts as wrong.
- **LLM-Eq (supplementary, not verified accuracy)**: gpt-4.1-mini, three votes at temperature 1, majority decision; it sees the question, the reference answer with aliases and the prediction, not the evidence. EM = 1 is accepted and abstentions are rejected without a judge call.
- **Evidence Hit Rate**: share of questions for which at least one evidence article is among the articles passed to the answer generation model. Evidence articles are all articles cited when the question was built; whether each one is necessary is not distinguished.
- **Known limitations**: unit suffixes are not normalized ("1422년" ≠ "1422"); the range sign is removed ("4~5일" = "45일").

## Systems and records
- Answer generation and judge: gpt-4.1-mini (returned version gpt-4.1-mini-2025-04-14, temperature 0 for answers); embeddings: text-embedding-3-small. GraphRAG calls gpt-4.1-mini through LiteLLM, which did not record a dated version.
- **Gold-context** uses the reader prompt in `prompts/reader_gold.txt` with a 100-token output cap. **GraphRAG Local and Basic** use the library's native prompts with the same answer instruction passed as the response type and no output cap; the exact prompts are in `runs/*/manifest.json`. Local: community level 2, 12,000-token context, community proportion 0.25. **GraphRAG Basic**: vector search over the index's 1,200-token text units (k = 10, 12,000-token context), no graph. Basic was added after the main runs and is reported regardless of direction.
- **Gold-context with gpt-4.1** (`runs/gold_gpt41/`, returned version gpt-4.1-2025-04-14): same prompt, temperature and 100-token cap; EM 80.9% (gpt-4.1-mini 74.7%). Added after the main runs.
- Run manifests keep models, temperature, output cap, retrieval and search settings, prompts and their hashes, number of questions, token totals and cost. The prompt files in `prompts/` use CRLF line endings; the exact strings are in the manifests.
- `graphrag_ctx/manifest.json` field `sample_ids_sha256` is the SHA-256 of `json.dumps(sample["ids"])` (the JSON-serialized ID list), not of the file bytes.
- `graphrag/settings.yaml`: comments removed; functional settings identical to the run (the hash of the file used in the run is `settings_sha256_at_run` in `runs/graphrag/manifest.json`; both hashes are in `MANIFEST.json`).
- `graphrag/prompts/extract_graph.txt`: the two example articles are masked as `<<ARTICLE id>>`; `python scripts/rebuild_corpus.py --restore-prompt corpus/articles.csv` restores them and checks the hash of the prompt used (prefix `5c62347f4685df93`).

## Released diagnostics not reported in the final paper
These were reported in earlier drafts and remain checked by the scripts; the final paper does not rely on them.
- **Other runs**: Closed-Book, Vanilla RAG (BM25 + dense, fused) and BM25 only (`runs/closed/`, `runs/vanilla/`, `runs/vanilla_bm25/`).
- **Graph diagnostics** (`diagnostics/`, `graph/`; `scripts/vi_recompute.py`, `scripts/paper_diagnostics.py`, `scripts/verify_paper.py`). For Identity, a name node is a node whose title equals the Hangul form of the questioned name with any parenthesized Hanja removed (e.g., 이비(李裶) → 이비), or that form followed by the same Hanja; people distinguished only by Hanja therefore map to one node. For Relation Chain, chain entities are matched by exact string to the wiki document title. Missing spellings and aliases are not inferred. `diagnostics/answer_string.tsv` and `basic_answer_string.tsv` flag whether the gold answer string appears in the GraphRAG contexts (Local 469 questions, EM 36.7%; Basic 355, EM 41.7%).
- `diagnostics/graph_checks/` holds judgments by LLM annotators (Claude agents, not human annotators) against the source articles for 90 sampled graph elements and 8 illustrative candidates, and a second, non-blind pass over 35 of them by another Claude agent. Of the 20 sampled same-name merge candidates (from 23 flagged), 15 put distinct referents in one node. Fields are in Korean: 판정 = verdict, 근거_인용 = quoted evidence, 기사ID = article IDs, 설명 = explanation, 확신 = confidence, 표시 = tags; `table_iv_candidates.jsonl` refers to a table of an earlier draft.
- **Gap decomposition** and **community reports**: in GraphRAG Local, community reports entered 809 of the 975 contexts (EM 16.9% with them, 21.7% without); an observation, not a controlled comparison (`runs/graphrag/context_index.jsonl`, `report_ids`).

## Corpus
- **Source**: the new modern Korean translation of the Sejong Sillok (신역 조선왕조실록, 2022–) by the Institute for the Translation of Korean Classics (ITKC, 한국고전번역원), read in the Korean Classics DB (https://db.itkc.or.kr), not the Classical Chinese original. It differs in wording from the older translation shown by default on sillok.history.go.kr (e.g., 01-12-07[03]: "양 500마리를 특별히 내려 주어" in the ITKC translation vs "양 5백 마리를 하사하여"); article identifiers are the same in both.
- **Selection**: an LLM agent read all 11,275 articles from the accession year through regnal year 9 (1418–1427) while writing the wiki. The construction ledger records, by lookup, whether each article is cited in the wiki (4,213 incorporated, 7,062 not; `data/ledger_ids.tsv`). The retrieval corpus is the union of the articles cited in the wiki (4,264) and the incorporated articles (4,213) = 4,273 articles, 2,214,948 characters; 4,245 fall within the period and 28 later articles are cited in the wiki's scope notes. All 773 evidence articles lie within the period. The generated wiki is not part of the retrieval corpus and was not given to any system.
- **Full-period checks** (`scripts/full_period_check.py`): over all 11,275 articles of the period (plus the 28 later corpus articles), BM25 top-8 evidence recall falls from 84.7% to 80.0% (no QA run). For the 72 immediately-before/after items, a screen by the questioned person's Hangul name finds out-of-corpus articles between anchor and target for 6 items and cannot parse the name for 4; these 10 items were read by an LLM agent (one reader, not blind). No answer changed (TR-003b: over the full period the adjacent article is 08-02-10[02] rather than the corpus one; the answer, 찬성, is the same). The name screen cannot see articles that name the person by an alias.
- **Rebuilt text**: `date_heading + "\n\n" + body_text + "\n"`; translator titles, source notes, classification tags and footnotes removed; parentheticals inside the body kept. Rebuild with `python scripts/rebuild_corpus.py my_articles.jsonl corpus/articles.csv`, which checks every text against `data/article_ids.tsv`.

## Items referenced in the paper
Identifiers in `data/questions.jsonl` (PJ = Identity pair, TR = Temporal pair, MH = Relation Chain item).

| Paper | Items | What |
|---|---|---|
| Fig. 2 | PJ-073, PJ-130, TR-074 | contrastive pair examples (Hyoryeonggun / Yi Bo; Sim In-bong / Sim Jing; the previous / following day) |
| III-B | PJ-073 | Alias example in the text (Hyoryeonggun / Yi Bo) |
| III-B | PJ-007, PJ-009, PJ-010, PJ-037, PJ-038, PJ-146, PJ-147, PJ-149, PJ-150 | the 9 Identity pairs that differ beyond the name and particle |
| IV-C | PJ-003 | Alias pair Gil Jae / Yaeun: evidence delivered and answered correctly for one name only (Local) |
| README only | PJ-078, PJ-148, PJ-149, PJ-150 | the four Confusable pairs with identical Hangul names (PJ-149: identical Hanja as well) |

## Provenance and validation
- **Wiki**: built with LLM-Wiki by an LLM agent running Claude Opus 5 (the authors' record of the model; no run log is released): 176 interconnected wiki documents, every statement citing article identifiers. Its 8,103 quoted passages were checked against the cited articles by normalized string matching (punctuation and Hanja glosses normalized), and every passage was found. The wiki was used only to find candidate questions and relevant articles, not as the source of the answers. The wiki text and the item-drafting prompts are not released (its identifier-level skeleton is; see Wiki skeleton).
- **Items**: drafting, revision and re-solving by Claude Code sub-agents running Claude Opus 5.5 (claude-opus-5-5).
- **Validation** against the source articles in three stages: automatic checks (schema, quotes verbatim after NFKC and whitespace normalization, date leakage, length); independent re-solving from the evidence articles without the answer key by agents other than the item writers (blinding self-reported; verdict counts in `data/validation_summary.json`), with disagreements revised against the sources and re-solved; and human review of all 975 items in the final benchmark by multiple reviewers, who checked the questions, answers and supporting quotations against the source articles. During construction the first author also reviewed a 100-item stratified sample (`data/author_sample.tsv`; 15 Identity pairs, 15 Temporal pairs, 40 Relation Chain items; all judged correct). The full human review is not released as a separate file.
- The items, answers and aliases were frozen (2026-09-29 15:24 KST) before any QA system was run, and none was removed afterwards.
- **Scope**: no professional historian reviewed the items. All 975 items are test-only evaluation items; there is no training or development split, and no retrieval, index or prompt setting was tuned on them (five items served as a pipeline smoke test).

## Wiki skeleton
`wiki_skeleton/wiki_skeleton.jsonl` lets readers follow an item's `source_pages` into the wiki's structure without the wiki text. One line per wiki document (176: 114 person, 42 event, 12 place, 5 institution, 2 object, 1 text):
- `page` (title), `page_type` (인물 person, 사건 event, 장소 place, 제도 institution, 기물 object, 문헌 text) and `tags` (topic labels);
- `aliases` and `caution_names`: name tokens only, from the "other names" (이명) and "persons not to confuse" (주의 인물) lines of the person documents (100 tokens on 38 documents; 375 on 102 documents), usually as Hangul(Hanja), e.g. 이비(李裶). One document can carry several of these attributes. The explanations on those lines are dropped. Tokens are extracted by fixed rules stated in `scripts/build_wiki_skeleton.py`, and a doubtful token is left out, so the lists are incomplete by design; a caution name may also be a relative mentioned on the line rather than a namesake;
- `timeline`: the rows of the timeline sections (1,179 rows on 101 documents) as date + article IDs, without the event descriptions;
- `links`: target documents of the wiki links (3,011; 2,793 to documents in the skeleton; the rest point to wiki pages outside the entry set);
- `sections`: each heading's level and cited article IDs (2,805 sections). A heading is shown only as a generic label that recurs on three or more documents (e.g., 연표, 평가, 참고) and/or its date label; descriptive headings are `null` or reduced to their date;
- `cited_ids`: all article IDs cited in the document (3,095 distinct IDs, all in `data/article_ids.tsv`; the corpus count of 4,264 wiki-cited articles also includes the wiki's topic, timeline and index pages, which are not in the skeleton).

**Not included**: any sentence of the wiki (descriptions, evaluations, interpretations), quotations of the translation, link display texts and descriptive headings. An automatic check found no string of 20 or more characters ending in a verb ending or particle, and no non-name string sharing a 10-character window with an evidence quote; the only such overlaps are proper nouns in the name lists (e.g., 온인공용순효 대왕(溫仁恭勇順孝大王)) and three dates.

**Linking items to the skeleton**: every document name in `data/evidence.jsonl` `source_pages` (2,273 references to 169 documents, all 975 items) is a `page` in the skeleton, and relation-chain `chain` steps name their `from`, `to` and `page` the same way. 2,263 of the 2,450 item–article pairs (`source_ids`) are cited in one of the item's source documents; of the other 187, 176 are cited in another document and 11 in none. Example:
```
import json
wiki = {r["page"]: r for r in map(json.loads, open("wiki_skeleton/wiki_skeleton.jsonl", encoding="utf-8"))}
ev = {r["id"]: r for r in map(json.loads, open("data/evidence.jsonl", encoding="utf-8"))}
for p in ev["PJ-001a"]["source_pages"]:
    print(p, wiki[p]["aliases"], [s for s in wiki[p]["sections"] if "01-12-07[03]" in s["ids"]])
```
The file was produced with `python scripts/build_wiki_skeleton.py WIKI_ENTRIES_DIR --check data/evidence.jsonl` from the unreleased wiki.

## Re-running the systems
`code/` contains the scripts that produced the runs: `22_Vanilla색인.py` (chunks and embeddings), `25_Vanilla검색.py` (BM25, dense, fusion), `26_질의.py` (reader runs, including Gold-context), `15_GraphRAG질의.py` (GraphRAG Local search), `15c_GraphRAG기본검색.py` (GraphRAG Basic search), `16b_LLM판정.py` (LLM-Eq judge), `27_채점.py` (scorer used for the paper) and `v3공통.py` (shared helpers). Comments were removed or shortened to English; the code is otherwise as run, and `MANIFEST.json` records the hashes of the original files. The scripts expect the original working layout (a `tools/` folder next to `rag/`), a rebuilt corpus, and an **OpenAI API key** (`OPENAI_API_KEY`); GraphRAG runs also need Microsoft GraphRAG 2.7.2 and an index built with `graphrag/settings.yaml` and `graphrag/prompts/`. Indexing cost US$45.89 in our run; LLM extraction is not deterministic, so a rebuilt index will differ, and `MANIFEST.json` records the hashes of our index files. None of this is needed to check the paper's numbers.

## Quoted source text
`data/evidence.jsonl` and `diagnostics/graph_checks/` contain short quotations from the new modern Korean translation of the Sejong Sillok, included only to document the evidence for each answer and judgment. The translation is © Institute for the Translation of Korean Classics (한국고전번역원), 2022–, and is provided in the Korean Classics DB (db.itkc.or.kr) under that provider's terms of use. The quotations are short excerpts (typically one sentence) used for research documentation, and the licenses of this package do not cover them. Full articles are not redistributed: obtain them from the provider under its terms of use and rebuild the corpus with `scripts/rebuild_corpus.py`. If a rights holder objects, the `quote` fields will be removed and the article identifiers kept.

## License
See `LICENSE.md`. Annotations and metadata: CC BY-NC 4.0. Code (`scripts/`, `code/`): MIT. Neither license covers the quoted source text, and the source translation is not redistributed.
