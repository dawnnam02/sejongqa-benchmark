# SejongQA (v3, frozen 2026-09-29)

SejongQA is a Korean-language benchmark of 975 questions over the modern Korean translation of the *Veritable Records of King Sejong* (Sejong sillok). All questions, answers and the retrieval corpus are in Korean.
- **Identity** (150 pairs): one attribute question asked under two names that denote the same person (75 pairs) or two different people (75 pairs).
- **Temporal** (150 pairs): the same event-anchored question under two temporal relation words, with different answers.
- **Multi-hop** (375 items): questions following relation chains across two to seven wiki pages.

This package accompanies the paper *Same Name, Different Day: Korean Historical QA from an LLM-Built Wiki and Where GraphRAG Loses the Evidence*. It contains everything needed to check the paper's numbers. The source article texts are not included (see Corpus).

> **Review copy.** Author information is not included in this repository; see the paper.

## Checking the paper's numbers
Python 3 (tested with 3.14). No API keys or network access are needed.
```
python scripts/test_score.py        # scorer rules, Table III EM numerators, Section VI counts
python scripts/verify_paper.py      # Tables I-III point values, Sections III-VI counts, costs, index and graph checks
python scripts/paired_ci.py         # all confidence intervals in the paper (needs numpy)
python scripts/evidence_arrival.py  # Table IV (gold articles in the context) and the Gold-solved re-count of Section V
python scripts/vi_recompute.py      # Section VI linkage counts
python scripts/full_period_check.py my_articles.jsonl   # Section VII full-period checks (needs numpy and a local copy of the translation)
python scripts/score.py runs/gold/answers.jsonl --judged runs/gold/judged.jsonl
```
Each script prints the recomputed value next to the expected value and exits with a non-zero status on a mismatch. Expected values are the paper's; details that the paper summarizes (e.g., per-sample graph-check counts, token and cost breakdowns) are documented in this README and in the run manifests.

## Layout
```
README.md  LICENSE.md  MANIFEST.json
data/
  questions.jsonl            975 items: id, category, sub, pair_id, question (give only the question to systems)
  answers.jsonl              answer, answer_aliases, evidence_ids, n_evidence, relation, relation_family, alias_class
  evidence.jsonl             short quotes supporting each answer; temporal anchor/target; multi-hop page chains
  article_ids.tsv            the 4,273 corpus article IDs with text length and SHA-256 (no text)
  author_sample.tsv          IDs, category and verdict of the 100-item sample reviewed by the first author
  temporal_prev_month.tsv    month-adjacency check of the 19 previous-month items
  corpus_stats.json          corpus size, Vanilla RAG chunk count and length limit, hub-node size
  validation_summary.json    verdict counts of re-solving without the answer key (first pass and isolated re-solving)
  ledger_ids.tsv             the construction ledger at identifier level: all 11,275 articles read for the wiki
                             and whether each was used (4,213 yes, 7,062 no); titles and reasons are not included
runs/
  closed/ vanilla/ vanilla_bm25/ graphrag/ graphrag_basic/ gold/   the six systems of Table III
                                          (vanilla_bm25 = BM25 only; graphrag = Local search;
                                          graphrag_basic = Basic search; both added after the main runs:
                                          vanilla_bm25 and graphrag_basic)
  graphrag_ctx/                           re-read of 200 stored GraphRAG contexts with the common instruction
  judge_manifest.json                     LLM-Eq judge settings and prompt
  each run: answers.jsonl, judged.jsonl (LLM-Eq votes), manifest.json
  vanilla/retrieval.jsonl, manifest_retrieval.json   BM25, dense and fused rankings (chunk IDs)
  graphrag/context_index.jsonl                       per question: context article IDs, report, entity,
                                                     relationship and text-unit IDs (no text)
  graphrag_basic/answers.jsonl                       answer, context article IDs and text-unit IDs (no text)
  graphrag_ctx/sample.json                           the 200 IDs and the sampling strata
graph/                       GraphRAG index at identifier level: entities, relationships, text units (no descriptions)
graphrag/                    settings.yaml, prompts/, index_stats.json
diagnostics/
  identity_items.tsv, identity_pairs.tsv, temporal.tsv, multihop.tsv, answer_string.tsv   Section VI tables
  vanilla_answer_string.tsv  answer-string flag for the 73 Vanilla misses with all gold articles retrieved
  basic_answer_string.tsv    answer-string flag for the GraphRAG Basic contexts (Table IV note)
  graph_checks/              LLM-annotator judgments of sampled graph elements and 8 candidate examples;
                             auto_flags.json lists the automatically flagged nodes
prompts/                     reader prompts (Closed-Book, Vanilla RAG, Gold Evidence) and the LLM-Eq judge prompt
scripts/                     score.py, test_score.py, verify_paper.py, paired_ci.py, evidence_arrival.py, vi_recompute.py,
                             full_period_check.py, rebuild_corpus.py
code/                        retrieval, query, judge and scoring code used for the runs (see Re-running the systems)
```

## Format rules
- **Identity pairs** ask the same attribute question under two names; they do not ask whether the names denote the same person.
- **Temporal pairs** differ only in the relation word; in 34 of 150 pairs the attached verb ending also differs. Definitions: the day before / after = the article dated one day earlier / later; early/late in that month = days 1–10 / 21–end of the anchor's lunar month; early/middle/late in that year = months 1–3 / 4–9 / 10–12 of the same regnal year; previous month = the lunar month immediately before the anchor's, an intercalary month counting as its own (a 62-day window only screened candidates; `data/temporal_prev_month.tsv`); immediately before/after = the adjacent article on the same matter in the corpus.
- **Multi-hop**: a k-page chain has k − 1 relations between wiki pages; this is a construction attribute, not a count of reasoning steps.
- **Questions** are at most 50 characters (70 for 5-page and 90 for 7-page chains) and contain no dates, era names or sexagenary years.
- Korean labels in the files: 인물판정 = Identity, 시간추론 = Temporal, 멀티홉 = Multi-hop; 같은 사람 / 다른 사람 = same / different person; 하루, 달, 해, 순서 = day, month, year, order; 2hop … 7hop = 2- … 7-page chains.

## Scoring
- **EM (primary)**: NFKC, then remove parenthesized text, Hanja, punctuation and spaces, and lowercase; compare with the answer and every alias. Aliases that become empty are ignored, so a Hanja-only answer cannot match.
- **Pair EM**: over the 300 Identity and Temporal pairs, a pair counts only if both items are correct.
- **Abstention**: an answer containing `근거 부족` ("insufficient evidence") is an abstention and counts as wrong.
- **LLM-Eq (secondary, not verified accuracy)**: gpt-4.1-mini, three votes at temperature 1; it sees the question, the gold answer with aliases and the prediction, not the source. EM = 1 is accepted and abstentions are rejected without a judge call.
- **Known limitations**: unit suffixes are not normalized ("1422년" ≠ "1422"); the range sign is removed ("4~5일" = "45일").

## Systems and records
- Reader and judge: gpt-4.1-mini (returned version gpt-4.1-mini-2025-04-14); embeddings: text-embedding-3-small. GraphRAG calls gpt-4.1-mini through LiteLLM, which did not record a dated version.
- **BM25 only** passes the top 8 BM25 chunks of the Vanilla RAG ranking to the reader. **GraphRAG Basic** is the library's Basic search over the same index: vector search over its text units (k = 10, 12,000-token context) with the native prompt, and no graph. Both were added after the main runs and are reported regardless of direction.
- GraphRAG Local contexts: community reports entered 809 of the 975 contexts (83.0%); EM was 16.9% with them and 21.7% without (an observation, not a controlled comparison; `runs/graphrag/context_index.jsonl`, `report_ids`).
- Run manifests keep the fields needed for the paper: models, temperature, output cap, retrieval and search settings, prompts and their hashes, number of questions, token totals and cost where the paper reports them. The prompt files in `prompts/` use CRLF line endings; the exact strings are in the manifests.
- `graphrag_ctx/manifest.json` field `sample_ids_sha256` is the SHA-256 of `json.dumps(sample["ids"])` (the JSON-serialized ID list), not of the file bytes.
- `graphrag/settings.yaml`: comments removed; functional settings identical to the run (the hash of the file used in the run is `settings_sha256_at_run` in `runs/graphrag/manifest.json`; both hashes are in `MANIFEST.json`).
- `graphrag/prompts/extract_graph.txt`: the two example articles are masked as `<<ARTICLE id>>`; `python scripts/rebuild_corpus.py --restore-prompt corpus/articles.csv` restores them and checks the hash of the prompt used (prefix `5c62347f4685df93`).

## Graph diagnostics
- Section VI is recomputed from the identifier-level tables in `diagnostics/` and `graph/`. For Identity, a name node is a node whose title equals the Hangul form of the questioned name with any parenthesized Hanja removed (e.g., 이비(李裶) → 이비), or that form followed by the same Hanja; people distinguished only by Hanja therefore map to one node, and a name matching several nodes counts as present if any of them matches. For Multi-hop, chain entities are matched by exact string to the wiki page title. Missing spellings and aliases are not inferred. "Same node" means that the node linked to a gold article's text unit is itself in the context's entity table. The answer-string checks need the context texts and are released as per-question flags.
- `diagnostics/graph_checks/` holds the judgments of LLM annotators (Claude agents) against the source articles for 90 sampled graph elements and 8 illustrative candidates (6 confirmed; the paper cites Jo Chi and the two Pak Sil in Section VI), and a second, non-blind pass over 35 of them by another Claude agent. Of the 20 sampled same-name merge candidates (from 23 flagged), 15 put distinct referents in one node. Fields are in Korean: 판정 = verdict, 근거_인용 = quoted evidence (from the source article; in a few Table IV candidates, from the GraphRAG entity description being checked), 기사ID = article IDs, 설명 = explanation, 확신 = confidence, 표시 = tags; `table_iv_line` is the English line drafted for Table IV.

## Corpus
- **Source**: the modern Korean translation of the Sejong sillok published by the National Institute of Korean History (https://sillok.history.go.kr), not the Classical Chinese original.
- **Selection**: articles cited in an LLM-written wiki (4,264) ∪ articles used in its construction ledger (4,213; `data/ledger_ids.tsv`) = 4,273 articles, 2,214,948 characters; 4,245 fall within the accession year through regnal year 9, 28 are later. All 773 gold evidence articles lie in the earlier period.
- **Full-period checks** (`scripts/full_period_check.py`): over all 11,275 articles of the period (plus the 28 later corpus articles), BM25 top-8 evidence recall falls from 84.7% to 80.0% (no QA run; the other systems were not re-indexed). For the 72 immediately-before/after items, a screen by the questioned person's Hangul name finds out-of-corpus articles between anchor and target for 6 items and cannot parse the name for 4; these 10 items were read by an LLM agent (one reader, not blind). No answer changed (see TR-003b below). The name screen cannot see articles that name the person by an alias.
- **Rebuilt text**: `date_heading + "\n\n" + body_text + "\n"`; translator titles, source notes, classification tags and footnotes removed; parentheticals inside the body kept. Rebuild with `python scripts/rebuild_corpus.py my_articles.jsonl corpus/articles.csv`, which checks every text against `data/article_ids.tsv`.

## Items referenced in the paper
Identifiers in `data/questions.jsonl` (PJ = Identity pair, TR = Temporal pair, MH = Multi-hop item). The paper cites only PJ-148 and PJ-149 by identifier; the others are counted there and listed here.

| Paper | Items | What |
|---|---|---|
| Table II, Fig. 1(b) | PJ-130, TR-074 | example pairs (different persons; the day before / the day after) |
| III-B | PJ-009, PJ-010, PJ-037, PJ-038, PJ-146 | the person is referred to by a descriptive phrase |
| III-B | PJ-007, PJ-147, PJ-150 | other wording also changes between the two questions |
| III-B, VI | PJ-149 | two men with the same Hangul and Hanja name (the two Pak Sil) |
| III-B | PJ-078, PJ-148, PJ-149, PJ-150 | the four different-person pairs with identical Hangul names |
| III-B | PJ-061 to PJ-068, PJ-078, PJ-148, PJ-150 | identical Hangul names that differ only in the parenthesized Hanja |
| III-B | PJ-127 | the only era-derived name in a question (the Yongle Emperor) |
| VI | PJ-078, PJ-148, PJ-150 | different-person pairs distinguished only by Hanja, mapped to one node |
| VI | PJ-148, PJ-149, PJ-150 | different-person pairs whose names map to one node holding both gold articles |
| VI | PJ-065a, PJ-065b, TR-133a | no gold answer or alias of two or more characters for the answer-string check |
| VI | PJ-148 | both gold articles attach to the merged Jo Chi node |
| VII | TR-003a, TR-003b, TR-015a, TR-027a, TR-028a, TR-028b | flagged by the full-period name screen; read by an LLM agent; no answer changed |
| VII | TR-004a, TR-004b, TR-058a, TR-058b | name not parsed by the screen; read by an LLM agent; no answer changed |
| README only | TR-003b | over the full period the adjacent article is 08-02-10[02] rather than the corpus one; the answer (찬성) is the same |

## Provenance and validation
- **Wiki**: written by an LLM agent running Claude Opus 5 (the authors' record of the model; no run log is released). Its person pages list aliases and confusable persons, with dated entries and links between pages, all citing article identifiers. The wiki and the item-drafting prompts are not released; items are validated against the cited articles, not against the wiki.
- **Items**: drafting, revision and re-solving by Claude Code sub-agents running Claude Opus 5.5 (claude-opus-5-5).
- **Validation**: automatic checks (schema, quotes verbatim after NFKC and whitespace normalization, date leakage, length); re-solving from the gold articles without the answer key (blinding self-reported) by agents other than the item writers (verdict counts in `data/validation_summary.json`; this tests answerability from the gold articles, not corpus-wide uniqueness); and a 100-item stratified sample reviewed by the first author (IDs in `data/author_sample.tsv`; 15 Identity pairs, 15 Temporal pairs, 40 Multi-hop items; random within strata, the second Identity draw favoring distinct alias types; all judged correct).
- **Human judgment**: the first author set the construction rules and reviewed the 100-item sample (not blind to the evidence and, for Identity and Multi-hop, the re-solver's answers); disagreements from re-solving were checked and revised by the drafting agents. The items were frozen (2026-09-29 15:24 KST) before any QA system was run, and none was removed afterwards.
- **Scope**: no professional historian reviewed the items, and the validation records do not establish error-free annotation. All 975 items are evaluation items; no retrieval, index or prompt setting was tuned on them (five items served as a pipeline smoke test).

## Re-running the systems
`code/` contains the scripts that produced the runs: `22_Vanilla색인.py` (chunks and embeddings), `25_Vanilla검색.py` (BM25, dense, fusion), `26_질의.py` (reader runs), `15_GraphRAG질의.py` (GraphRAG Local search), `16b_LLM판정.py` (LLM-Eq judge), `27_채점.py` (scorer used for the paper), `v3공통.py` (shared helpers) and `15c_GraphRAG기본검색.py` (GraphRAG Basic search). Comments were removed or shortened to English; the code is otherwise as run, and `MANIFEST.json` records the hashes of the original files. The scripts expect the original working layout (a `tools/` folder next to `rag/`), a rebuilt corpus, and an **OpenAI API key** (`OPENAI_API_KEY`); GraphRAG runs also need Microsoft GraphRAG 2.7.2 and an index built with `graphrag/settings.yaml` and `graphrag/prompts/`. Indexing cost US$45.89 in our run; LLM extraction is not deterministic, so a rebuilt index will differ, and `MANIFEST.json` records the hashes of our index files. None of this is needed to check the paper's numbers.

## Quoted source text
`data/evidence.jsonl` and `diagnostics/graph_checks/` contain short quotations from the modern Korean translation of the Sejong sillok, included only to document the evidence for each answer and judgment. The Korean translation is © Sejong the Great Memorial Society (세종대왕기념사업회) and is provided online by the National Institute of Korean History (sillok.history.go.kr); on that site the Korea Open Government License (KOGL) mark accompanies the classical Chinese original, not the translation. The quotations are short excerpts (typically one sentence) used for research documentation, and the licenses of this package do not cover them. Full articles are not redistributed: obtain them from the provider under its terms of use and rebuild the corpus with `scripts/rebuild_corpus.py`. If a rights holder objects, the `quote` fields will be removed and the article identifiers kept.

## License
See `LICENSE.md`. Annotations and metadata: CC BY-NC 4.0. Code (`scripts/`, `code/`): MIT. Neither license covers the quoted source text, and the source translation is not redistributed.
