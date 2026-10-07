# SejongQA: An Evidence-Grounded Diagnostic Benchmark for Korean Historical Question Answering

SejongQA is a test-only Korean historical question answering benchmark of 975 questions built from the modern Korean translation of the *Veritable Records of King Sejong* (Sejong Sillok). Each question comes with an answer, answer aliases, gold evidence articles (gold articles) and supporting quotations, so that answer accuracy and gold article retrieval can be analyzed together by question type. There is no training or development split. Questions, answers and source texts are in Korean.

| Type | Questions | Description |
|---|---|---|
| Alias | 150 (75 pairs) | the same question about one person under two names or titles; both answers are the same |
| Confusable | 150 (75 pairs) | the same question frame about two easily confused people; the answers differ |
| Temporal | 300 (150 pairs) | the same anchor article with two different temporal expressions; the target articles and answers differ |
| Relation Chain | 375 | facts connected across 2, 3, 5 or 7 wiki pages (92/98/100/85 questions); only the starting entity is named |

Alias and Confusable together form the Identity dimension.

## Construction

- **Source.** 11,275 articles of the modern Korean translation of the Sejong Sillok by the Institute for the Translation of Korean Classics (ITKC), Korean Classics DB (https://db.itkc.or.kr), from Sejong's accession year through his ninth regnal year (regnal years 0–9, 1418–1427).
- **Article selection.** Before wiki construction, short routine articles (regular rites, royal outings and portents recorded as short articles, and short reports of appointments or punishments only) were excluded by rules based on repeated expressions and length. The selection used article content only and was made before question generation. The 7,030 excluded IDs are listed in `data/excluded_article_ids.tsv`.
- **Wiki.** Following the LLM Wiki pattern (A. Karpathy, https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), 176 interconnected wiki pages were built from the remaining articles with Claude Opus 5. Each statement records its supporting article IDs, and all wiki quotations were checked against the cited articles. The wiki served only to find candidate questions and relevant articles; it is not released.
- **Candidate questions.** Claude Opus 5.5 agents drafted candidates from the alternate names, confusable people, dates and page relations recorded in the wiki (`prompts/question_generation_rules.md`).
- **LLM verification.** A separate Claude Opus 5.5 agent that had not drafted the question answered it from the question and gold articles only. Disagreements were resolved against the source articles and the revised question was verified again (`prompts/independent_solving_protocol.md`).
- **Human review.** Two reviewers checked the question, answer and supporting quotations of every question against the source articles. 25 questions were excluded, leaving 975.
- **Freezing.** Questions, answers and answer aliases were fixed before any evaluation run and were not changed afterwards.

## Retrieval corpus

The retrieval corpus consists of the 4,273 articles that are cited in the wiki or recorded in its construction log (`data/article_ids.tsv`): the 4,245 articles from 1418–1427 that remained after selection (11,275 − 7,030) and 28 articles from regnal years 10–32. All 773 gold articles fall within 1418–1427. The wiki was not given to any evaluated system; all retrieval indexes were built from these articles.

Article texts are not redistributed. To rebuild the corpus:

1. Obtain the articles from the Korean Classics DB under its terms of use and convert them to a JSONL file with one article per line (`article_ref`, `date_heading`, `body_text`; the exact format is in the docstring of `scripts/rebuild_corpus.py`).
2. Run `python scripts/rebuild_corpus.py my_articles.jsonl corpus/articles.csv`. Every text is checked against the length and SHA-256 in `data/article_ids.tsv`.
3. The two example articles in `graphrag/prompts/extract_graph.txt` are masked. The rebuild step (or `python scripts/rebuild_corpus.py --restore-prompt corpus/articles.csv`) writes `graphrag/prompts/extract_graph.restored.txt` and checks it against the hash of the prompt used in the paper. Before indexing, use the restored file as `extract_graph.txt` (or point `extract_graph.prompt` in `settings.yaml` to it). The restored file contains source text; do not commit it.
4. To rebuild the GraphRAG index, place `articles.csv` in the `input/` folder of a GraphRAG 2.7.2 project that uses `graphrag/settings.yaml` and `graphrag/prompts/`, and run `graphrag index`. This calls the OpenAI API.

## Contents

```
data/
  questions.jsonl            975 questions (give only the question text to a system)
  answers.jsonl              answers, aliases, gold article IDs, Temporal expression category
  evidence.jsonl             supporting quotations, Temporal anchor/target articles, Relation Chain steps
  article_ids.tsv            the 4,273 corpus article IDs with text length and SHA-256 (no text)
  excluded_article_ids.tsv   the 7,030 article IDs excluded before wiki construction (no text)
  identity_names.tsv         the two person references of each Identity pair
prompts/
  reader_closed.txt          answer prompt, Parametric Knowledge
  reader_gold.txt            answer prompt, Gold-context (also used for graphrag_ctx and gold_gpt41)
  reader_vanilla.txt         answer prompt, Vanilla RAG
  question_generation_rules.md
  independent_solving_protocol.md
graphrag/
  settings.yaml              Microsoft GraphRAG 2.7.2 settings used for indexing and search
  prompts/                   indexing prompts and the Local/Basic Search prompts
runs/                        execution records: answers.jsonl and manifest.json per condition
  closed/                    Parametric Knowledge (question only)
  gold/                      Gold-context (full text of all gold articles, no retrieval)
  vanilla/                   Vanilla RAG; retrieval.jsonl lists the retrieved text units
  graphrag/                  GraphRAG Local Search
  graphrag_basic/            GraphRAG Basic Search
  graphrag_ctx/              200 stored Local Search contexts re-answered with the Gold-context prompt; sample.json
  gold_gpt41/                Gold-context with gpt-4.1 as the answer generator (Table II, note c)
scripts/
  check_paper.py             recomputes every number in the paper from these files
  score.py                   EM, Pair EM and abstention by question type
  rebuild_corpus.py          rebuilds the corpus and restores the masked prompt examples
  vanilla_retrieval.py       optional: reruns Vanilla RAG retrieval on a rebuilt GraphRAG index
                             (needs numpy, pandas, pyarrow, lancedb, openai and an OpenAI key)
```

The folder `closed` and the label `Closed-Book` in the scripts correspond to Parametric Knowledge in the paper; `graphrag` and `graphrag_basic` correspond to Local Search and Basic Search.

## Data fields

Labels in the data files are in Korean. Article IDs have the form `YY-MM-DD[NN]`: regnal year (`00` = accession year), lunar month, day and article number within the day. Leap months are written `YY-윤MM-DD[NN]` (e.g., `04-윤12-26[02]`), and one article whose day is not recorded is written `09-08-○○[01]`. Question IDs start with `PJ` (Identity), `TR` (Temporal) or `MH2`/`MH3`/`MH5`/`MH7` (Relation Chain); paired questions end in `a` and `b`.

| Korean label | English |
|---|---|
| `category`: `인물판정` / `시간추론` / `멀티홉` | Identity / Temporal / Relation Chain |
| `sub`: `같은 사람` / `다른 사람` | Alias / Confusable (empty for Temporal) |
| `sub`: `2hop` / `3hop` / `5hop` / `7hop` | 2-, 3-, 5-, 7-document chain |
| `relation_family`: `하루` / `달` / `해` / `순서` | day / month / year / order |
| `relation`: `전날` / `이튿날` | previous day / following day |
| `relation`: `그달 초` / `그달 말` / `전월` | early in the same month / late in the same month / previous month |
| `relation`: `그해 초` / `그해 중순` / `그해 말` | early / middle / late in the same year |
| `relation`: `전해` / `이듬해` / `후년` | previous year / following year / two years later |
| `relation`: `직전` / `직후` | immediately before / immediately after |

**questions.jsonl:** `id`, `category`, `sub`, `pair_id` (null for Relation Chain), `question`.

**answers.jsonl:** `id`, `category`, `sub`, `pair_id`, `answer`, `answer_aliases` (accepted alternative forms), `evidence_ids` (gold article IDs), `relation_family` (Temporal only). The fields `n_evidence`, `hops`, `relation`, `alias_type` and `alias_class` are not used by the paper or the scripts. They are kept so that the file stays byte-identical to the frozen answer key whose hash is recorded in the run manifests.

**evidence.jsonl:** `id`, `source_ids` (same as `evidence_ids`), `evidence` [{`source_id`, `quote`}]. Identity items add `alias_type`. Temporal items add `relation`, `family`, `anchor` {`source_id`, `event`} and `target` {`source_id`, `event`}. Relation Chain items add `hops` and `chain` [{`step`, `from`, `relation`, `to`, `source_ids`, `quote`}] with k−1 steps for a k-document chain. Quotations match the source articles up to whitespace; `event`, `from`, `relation` and `to` are short descriptions written by the authors.

**article_ids.tsv:** `id`, `n_chars`, `sha256_text`. **excluded_article_ids.tsv:** `id`. **identity_names.tsv:** `pair`, `sub`, `name_a`, `name_b`.

## Evaluation settings

All conditions use gpt-4.1-mini (gpt-4.1-mini-2025-04-14) at temperature 0 as the answer generator, and each condition was run once.

- **Parametric Knowledge:** the question only.
- **Gold-context:** the full text of the gold articles, without retrieval.
- **Vanilla RAG:** top-10 text chunks by embedding similarity, from the same 4,600 chunks as the GraphRAG index; no graph, reranking or query rewriting.
- **Local Search / Basic Search:** Microsoft GraphRAG 2.7.2. Basic Search retrieves the top-10 chunks without the graph.

| Setting | Value |
|---|---|
| Embedding model | text-embedding-3-small |
| Chunks | 1,200 tokens, 100-token overlap |
| GraphRAG index | built with gpt-4.1-mini; entity types person, official position, government office, place, country, event, institutional system, document |
| Local Search | community level 2, text-unit proportion 0.5, community-report proportion 0.25 |
| Context limit | 12,000 tokens (Local and Basic Search) |
| Output limit | 100 tokens for Parametric Knowledge, Gold-context and Vanilla RAG; none for Local and Basic Search |
| Prompts | Parametric Knowledge, Gold-context and Vanilla RAG share one answer instruction (`prompts/`); Local and Basic Search use the default GraphRAG search prompts with the same answer instruction as the response type |

The indexing prompts (`extract_graph`, `summarize_descriptions`, `community_report_graph`) were adapted to Korean output and to the source; the search prompts are the GraphRAG defaults. Search methods not used in the paper (Global, DRIFT) and claim extraction are still configured in `settings.yaml`, but their prompt files are not included; GraphRAG falls back to its built-in prompts for them. The settings and numbers above are also recorded in each `runs/*/manifest.json`.

## Metrics

- **Exact Match (EM).** The output and the references are normalized (Unicode NFKC; parenthesized text, Chinese characters, punctuation and whitespace removed; English lowercased). An answer is correct if it equals the answer or any alias after normalization, e.g., `남지(南智)` = `남지`.
- **Pair EM.** The share of contrastive pairs in which both questions are correct, computed separately for the 150 Identity pairs and the 150 Temporal pairs.
- **Abstention.** Systems were instructed to answer `근거 부족` ("insufficient evidence") when the evidence was insufficient. Such answers are counted and scored as incorrect.
- **Retrieval Hit Rate.** The share of all questions for which at least one gold article is retrieved and included in the answer generator's input. An article counts as retrieved if any of its chunks enters the input. The scripts also report the share of questions with all gold articles retrieved and, for Temporal questions, with both the anchor and target articles retrieved.

## Run manifests

Each `runs/*/manifest.json` records the model, temperature, output limit, prompt or prompt hash, retrieval settings and run date. `questions_sha256` and `answers_sha256` are the SHA-256 of the frozen question file and answer key, i.e., the bytes of `data/questions.jsonl` and `data/answers.jsonl` as released (CRLF line endings; `.gitattributes` keeps them unchanged). In `runs/gold_gpt41/manifest.json`, `run_answers_sha256` is the SHA-256 of that run's own `answers.jsonl`. `graphrag/settings.yaml` is the configuration used in the runs with its comments removed, so its SHA-256 differs from `settings_sha256_at_run` in the GraphRAG manifests. The GraphRAG prompt files match the `prompts_sha256` values in `runs/graphrag/manifest.json`; `extract_graph.txt` matches only after the masked examples are restored. Information unrelated to the experiments, such as API costs, was removed from the public copy. The `smoke_test` note is kept as provenance: the first five questions were answered in a smoke-test call with identical settings, and those answers are part of the 975.

## Checking the paper

Python 3, standard library only; no API key or network access is needed.

```
python scripts/check_paper.py
python scripts/score.py runs/gold/answers.jsonl
```

`check_paper.py` prints each recomputed value next to the value in the paper (Tables I–III, corpus counts, the 200-question re-read, the gpt-4.1 run and the failure-pattern counts) and exits with a non-zero status on any mismatch. `score.py` prints EM, Pair EM and abstention by question type for one run.

## Quoted source text

`data/evidence.jsonl` contains short quotations from the ITKC translation, included only to document the evidence for each answer. They are not covered by the licenses of this package (see `LICENSE.md`).

## License

Data, annotations and run records: CC BY-NC 4.0. Code (`scripts/`): MIT. `graphrag/prompts/` is adapted from Microsoft GraphRAG and remains under the MIT License. Details are in `LICENSE.md`.

## Citation

```bibtex
@inproceedings{kim2027sejongqa,
  title     = {SejongQA: An Evidence-Grounded Diagnostic Benchmark for Korean Historical Question Answering},
  author    = {Kim, Hyonam and Choi, Seungyun},
  booktitle = {Proceedings of the International Conference on Electronics, Information, and Communication (ICEIC)},
  year      = {2027},
  note      = {To appear}
}
```

## Contact

Hyonam Kim, Chung-Ang University (dawnnam@cau.ac.kr)
