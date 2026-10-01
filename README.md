# SejongQA

SejongQA is a test-only Korean historical question answering benchmark of 975 questions over the modern Korean translation of the *Veritable Records of King Sejong* (Sejong Sillok), accession year through regnal year 9 (1418–1427). It accompanies the paper *SejongQA: A Korean Historical Question Answering Benchmark for Evaluating Person Identity and Temporal Reasoning*.

| Type | Questions | Description |
|---|---|---|
| Alias | 150 (75 pairs) | the same question about one person under two names or titles; both answers must be the same |
| Confusable | 150 (75 pairs) | the same question about two different people who may be confused; the answers must differ |
| Temporal | 300 (150 pairs) | the same anchor event with two temporal relation expressions; the target article and answer differ |
| Relation Chain | 375 | facts connected across 2, 3, 5 or 7 wiki documents; only the starting entity is named |

Alias and Confusable are the Identity questions. Questions, answers and the retrieval corpus are in Korean. There is no training or development split.

## Contents
```
data/
  questions.jsonl        id, category, sub, pair_id, question (give only the question to a system)
  answers.jsonl          answer, answer_aliases, evidence_ids (evidence articles), relation_family, alias_class
  evidence.jsonl         supporting quotations; Temporal anchor and target articles; Relation Chain steps
  article_ids.tsv        the 4,273 retrieval-corpus article IDs with text length and SHA-256 (no text)
  identity_names.tsv     the two person references of each Identity pair
prompts/                 Gold-context answer prompt; LLM-Eq judge prompt
graphrag/                Microsoft GraphRAG 2.7.2 settings.yaml and prompts used for indexing and search
runs/                    execution records: answers.jsonl, judged.jsonl (LLM-Eq votes), manifest.json
  gold/                  Gold-context (all evidence articles in full, no retrieval)
  graphrag/              GraphRAG Local Search
  graphrag_basic/        GraphRAG Basic Search
  graphrag_ctx/          200 stored Local contexts re-read with the Gold-context prompt and output limit
  gold_gpt41/            Gold-context with gpt-4.1 as the answer generation model
  judge_manifest.json    LLM-Eq judge settings
scripts/
  score.py               scorer (EM, Pair EM, LLM-Eq, abstention)
  check_paper.py         recomputes every number in the paper from these files
  rebuild_corpus.py      rebuilds the corpus from a local copy of the translation and checks it against article_ids.tsv
```
File labels: `인물판정` = Identity (`같은 사람` = Alias, `다른 사람` = Confusable); `시간추론` = Temporal (`하루`, `달`, `해`, `순서` = day, month, year, sequence); `멀티홉` = Relation Chain (`2hop` … `7hop` = 2- … 7-document chains).

## Checking the paper
Python 3, standard library only; no API keys or network access.
```
python scripts/check_paper.py
python scripts/score.py runs/graphrag/answers.jsonl --judged runs/graphrag/judged.jsonl
```
`check_paper.py` prints each recomputed value next to the paper's value (Tables I–III, corpus counts, McNemar tests, the 200-question re-read, the gpt-4.1 run and the failure-pattern counts) and exits with a non-zero status on a mismatch.

## Scoring
- **EM**: NFKC, then remove parenthesized text, Chinese characters, punctuation and whitespace, and lowercase; correct if equal to the answer or any alias (e.g., 남지(南智) = 남지). Answers and aliases were frozen with the questions (2026-09-29) before any run.
- **Pair EM**: a pair counts only if both questions are correct.
- **Abstention**: an answer containing `근거 부족` ("insufficient evidence") counts as wrong.
- **LLM-Eq** (supplementary): gpt-4.1-mini, three votes at temperature 1, majority decision; it sees the question, the answer with aliases and the response, not the evidence.
- **Evidence Hit Rate**: share of questions for which at least one evidence article is among the articles passed to the answer generation model.

## Corpus
The source is the new modern Korean translation of the Sejong Sillok by the Institute for the Translation of Korean Classics (ITKC), Korean Classics DB (https://db.itkc.or.kr). The retrieval corpus is the 4,273 articles (2,214,948 characters) cited in the LLM-Wiki or recorded as incorporated during construction; the generated wiki was not given to any system. Article texts are not redistributed: obtain them from the provider under its terms of use and run `python scripts/rebuild_corpus.py my_articles.jsonl corpus/articles.csv`, which checks every text against `data/article_ids.tsv`. The two example articles in `graphrag/prompts/extract_graph.txt` are masked; `python scripts/rebuild_corpus.py --restore-prompt corpus/articles.csv` restores them.

## Quoted source text
`data/evidence.jsonl` contains short quotations from the ITKC translation (© Institute for the Translation of Korean Classics, 2022–), included only to document the evidence for each answer. The licenses of this package do not cover them. If a rights holder objects, the `quote` fields will be removed and the article identifiers kept.

## License
See `LICENSE.md`. Annotations and metadata: CC BY-NC 4.0. Code (`scripts/`): MIT.
